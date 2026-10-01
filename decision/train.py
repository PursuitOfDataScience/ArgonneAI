"""Fine-tune argonne-4.5-base-ctx13568 into the Argonne decision model.

    torchrun --standalone --nproc_per_node=8 decision/train.py --data PACKED --out RUN_DIR
    python decision/train.py --data PACKED --out RUN_DIR                  # one GPU

Batches: right-padded, length-grouped micro-batches under a padded-token budget, cut once per epoch from a
world-size-independent shuffle. An optimizer step is world x accum micro-batches of similar cost, and its loss is
the mean over every example in the step, so a step of twenty 13k-token documents and a step of eight hundred
short questions weigh each example the same. Targets are the record's soft distribution when it has one
(a graded similarity, an annotator fraction), else one-hot; log loss is strictly proper, which is
what makes the output probabilities calibratable.

PORTABLE CHECKPOINTS (a run can move between a 4-GPU job, an 8-GPU node and one GPU):
progress is the set of micro-batches consumed this epoch plus a global count, the LR schedule is a function of
that count, and optimizer state is saved per PARAMETER NAME by whichever rank owns it, so any world size can
reload it. One trainer at a time: --lock_file is refreshed every step (a launcher refuses to start while it is
fresh), and --yield_file makes a running trainer save and exit at the next step so a bigger allocation can take
over.
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.modeling import DecisionHead, DecisionModel  # noqa: E402

PAD = 151643
BASE = os.environ.get("DECISION_BASE", "PursuitOfDataScience/argonne-4.5-base-ctx13568")


# ---- data ------------------------------------------------------------------------------------------------------
class Packed:
    def __init__(self, d):
        self.dir = d
        self.tokens = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r")
        self.index = np.load(os.path.join(d, "index.npy"))
        self.opt_pos = np.load(os.path.join(d, "opt_pos.npy"))
        self.soft = np.load(os.path.join(d, "soft.npy"))
        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)
        self.lengths = self.index[:, 1]

    def __len__(self):
        return len(self.index)

    def collate(self, idxs, device, pad_multiple=64):
        rows = self.index[idxs]
        t = int(rows[:, 1].max())
        t = ((t + pad_multiple - 1) // pad_multiple) * pad_multiple
        k = int(rows[:, 4].max())
        b = len(idxs)
        ids = np.full((b, t), PAD, dtype=np.int64)
        opt = np.zeros((b, k), dtype=np.int64)
        mask = np.zeros((b, k), dtype=bool)
        tgt = np.zeros((b, k), dtype=np.float32)
        for j, (st, n, ans, os_, no, lab, typ, src, ss, tr) in enumerate(rows):
            ids[j, :n] = self.tokens[st:st + n]
            opt[j, :no] = self.opt_pos[os_:os_ + no]
            mask[j, :no] = True
            if ss >= 0:
                tgt[j, :no] = self.soft[ss:ss + no]
            else:
                tgt[j, lab] = 1.0
        to = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
        return {"input_ids": to(ids), "opt_pos": to(opt), "opt_mask": to(mask), "ans_pos": to(rows[:, 2].copy()),
                "target": to(tgt), "label": to(rows[:, 5].copy()), "type": to(rows[:, 6].copy()),
                "source": to(rows[:, 7].copy()), "real_tokens": int(rows[:, 1].sum())}


def make_micros(lengths, idx_pool, micro_tokens, micro_max_ex, rng, mega=4096):
    """Length-grouped micro-batches: shuffle, sort within mega-chunks by length, cut under the padded budget."""
    order = idx_pool[rng.permutation(len(idx_pool))]
    micros = []
    for c in range(0, len(order), mega):
        chunk = order[c:c + mega]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        cur, cur_max = [], 0
        for i in chunk:
            L = int(lengths[i])
            m = max(cur_max, L)
            if cur and (m * (len(cur) + 1) > micro_tokens or len(cur) >= micro_max_ex):
                micros.append(cur)
                cur, m = [], L
            cur.append(int(i))
            cur_max = m
        if cur:
            micros.append(cur)
    return micros


def epoch_micros(data, args, epoch):
    """The epoch's micro-batches: identical on every rank and for every world size."""
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    return make_micros(data.lengths, np.arange(len(data)), args.micro_tokens, args.micro_max_ex, rng)


def build_steps(micros, remaining, per, lengths, salt):
    """Group the remaining micro indices into steps of `per` micro-batches of similar cost, in shuffled order.
    The tail that cannot fill a step is returned separately (skipped this epoch, counted as consumed)."""
    rem = sorted(remaining)
    cost = np.array([int(lengths[micros[i]].max()) * len(micros[i]) for i in rem]) if rem else np.zeros(0)
    rem = [rem[i] for i in np.argsort(cost, kind="stable")]
    n_full = len(rem) // per * per
    steps = [rem[i:i + per] for i in range(0, n_full, per)]
    order = np.random.default_rng(salt).permutation(len(steps))
    return [steps[i] for i in order], rem[n_full:]


# ---- distributed -----------------------------------------------------------------------------------------------
def setup():
    cuda = torch.cuda.is_available()
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank, world, local = 0, 1, 0
    if cuda:
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    return rank, world, torch.device("cpu")


def allsum(x, device):
    t = torch.tensor(x, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t)
    return t.tolist()


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ---- metrics ---------------------------------------------------------------------------------------------------
def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, data, idxs, args, device, rank, world):
    """Loss / accuracy / ECE per question type on a fixed subset, sharded over ranks."""
    model.eval()
    mine = np.asarray(idxs[rank::world], dtype=np.int64)
    micros = make_micros(data.lengths, mine, args.micro_tokens, args.micro_max_ex, np.random.default_rng(0)) if len(mine) else []
    stats = {t: [0.0, 0.0, 0.0] for t in range(3)}
    confs = {t: [] for t in range(3)}
    corrs = {t: [] for t in range(3)}
    for m in micros:
        b = data.collate(m, device)
        with autocast(device):
            logits = model(b["input_ids"], b["opt_pos"], b["opt_mask"], b["ans_pos"])
        logp = F.log_softmax(logits.float(), -1)
        loss = -(b["target"] * logp.masked_fill(b["target"] == 0, 0.0)).sum(-1)
        conf, pred = logp.exp().max(-1)
        ok = (pred == b["label"]).float()
        for t in range(3):
            sel = b["type"] == t
            if sel.any():
                stats[t][0] += loss[sel].sum().item()
                stats[t][1] += ok[sel].sum().item()
                stats[t][2] += sel.sum().item()
                confs[t] += conf[sel].tolist()
                corrs[t] += ok[sel].tolist()
    flat = allsum([v for t in range(3) for v in stats[t]], device)
    if dist.is_initialized():
        gathered = [None] * world
        dist.all_gather_object(gathered, (confs, corrs))
    else:
        gathered = [(confs, corrs)]
    out = {}
    for t, name in enumerate(["choice", "score", "noul"]):
        ls, c, n = flat[3 * t:3 * t + 3]
        if n:
            out[name] = {"loss": ls / n, "acc": c / n, "n": int(n),
                         "ece": ece([x for g in gathered for x in g[0][t]], [x for g in gathered for x in g[1][t]])}
    model.train()
    return out


# ---- portable optimizer state ------------------------------------------------------------------------------------
def inner_opt(opt):
    return opt.optim if hasattr(opt, "optim") else opt


def save_opt_shard(opt, model, path):
    """This rank's optimizer state keyed by parameter NAME (only the parameters it owns under ZeRO)."""
    inner = inner_opt(opt)
    names = {id(p): n for n, p in model.named_parameters()}
    shard = {}
    for g in inner.param_groups:
        for p in g["params"]:
            st = inner.state.get(p)
            if st:
                shard[names[id(p)]] = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    torch.save(shard, path)


def load_opt_shards(opt, model, files, device):
    """Fill this rank's optimizer state from shards written at ANY world size: keep only owned parameters."""
    inner = inner_opt(opt)
    mine = {id(q) for g in inner.param_groups for q in g["params"]}
    owned = {n: p for n, p in model.named_parameters() if id(p) in mine}
    loaded = 0
    for f in files:
        shard = torch.load(f, map_location="cpu", mmap=True)
        for n, st in shard.items():
            p = owned.get(n)
            if p is None:
                continue
            inner.state[p] = {k: (v.to(p.device) if torch.is_tensor(v) and k != "step" else v) for k, v in st.items()}
            loaded += 1
        del shard
    if hasattr(opt, "optim"):
        opt.state.clear()     # ZeRO's wrapper state is never read; keep it empty as in a fresh run
    return loaded, len(owned)


# ---- main ------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--init", default=None, help="start from a decision model dir instead of the base")
    ap.add_argument("--tiny", type=int, default=0, help="CPU tests: a random 2-layer trunk instead of the base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps in THIS process")
    ap.add_argument("--micro_tokens", type=int, default=16384)
    ap.add_argument("--micro_max_ex", type=int, default=128)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--zero", type=int, default=1)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--head_rank", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_max", type=int, default=3000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every_min", type=float, default=30)
    ap.add_argument("--wall_time", type=int, default=0, help="seconds; save and exit 5 min before it")
    ap.add_argument("--lock_file", default=None)
    ap.add_argument("--yield_file", default=None)
    ap.add_argument("--save_final_on_max_steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    if not args.tiny:
        from decision.modeling import local_dir
        args.base = local_dir(args.base)

    rank, world, device = setup()
    main_proc = rank == 0
    torch.manual_seed(args.seed)
    t_start = time.time()
    log = (lambda *a: print(*a, flush=True)) if main_proc else (lambda *a: None)

    train = Packed(os.path.join(args.data, "train"))
    val = Packed(os.path.join(args.data, "val"))
    per = world * args.accum
    micros = epoch_micros(train, args, 0)
    total_micros = int(len(micros) * args.epochs)
    log(f"[train] world={world} per_step={per} micro-batches | train {len(train):,} ex / {train.meta['tokens']:,} tok "
        f"| {len(micros):,} micro-batches/epoch, {total_micros:,} total ~ {total_micros // per:,} steps | val {len(val):,}")

    if args.tiny:
        from model import ArgonneConfig, ArgonneModel
        cfg = ArgonneConfig(vocab_size=151669, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, intermediate_size=128, max_position_embeddings=16384, block_size=16384,
                            tie_word_embeddings=True, qk_norm=True, v_norm=True, sandwich_norm=True, logit_softcap=15.0,
                            rope_theta=1e6, mlp_type="swiglu")
        torch.manual_seed(0)
        model = DecisionModel(ArgonneModel(cfg), DecisionHead(64, 16))
    elif args.init:
        model = DecisionModel.load(args.init, dtype=torch.float32)
    else:
        model = DecisionModel.from_base(args.base, rank=args.head_rank, dtype=torch.float32)
    model.grad_ckpt = bool(args.grad_ckpt)
    model.to(device).train()
    log(f"[train] params {sum(p.numel() for p in model.parameters()):,} (head {sum(p.numel() for p in model.head.parameters()):,})")

    decay, no_decay, head = [], [], []
    for n, p in model.named_parameters():
        (head if n.startswith("head.") else decay if (p.ndim >= 2 and "embed" not in n) else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": args.wd, "base_lr": args.lr},
              {"params": no_decay, "weight_decay": 0.0, "base_lr": args.lr},
              {"params": head, "weight_decay": 0.0, "base_lr": args.head_lr}]
    ddp = model
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None, gradient_as_bucket_view=True)
    if world > 1 and args.zero:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=torch.optim.AdamW, lr=args.lr,
                                      betas=(0.9, 0.95), weight_decay=args.wd)
        opt.param_groups[0]["base_lr"] = args.lr
        for g in groups[1:]:
            opt.add_param_group(dict(g, lr=g["base_lr"]))
    else:
        opt = torch.optim.AdamW([dict(g, lr=g["base_lr"]) for g in groups], betas=(0.9, 0.95))

    def lr_mult(done):
        frac = done / max(1, total_micros)
        if frac < args.warmup_frac:
            return (frac + 1e-9) / args.warmup_frac
        prog = (frac - args.warmup_frac) / max(1e-9, 1 - args.warmup_frac)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    # resume (any world size)
    ckpt = os.path.join(args.out, "ckpt")
    epoch, consumed, done, step, saved_per = 0, set(), 0, 0, per
    if os.path.exists(os.path.join(ckpt, "state.json")):
        with open(os.path.join(ckpt, "state.json")) as f:
            st = json.load(f)
        sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu", mmap=True)
        model.load_state_dict(sd)
        del sd
        n_loaded, n_owned = load_opt_shards(opt, model, sorted(glob.glob(os.path.join(ckpt, "opt_rank*.pt"))), device)
        epoch, consumed, done, step = st["epoch"], set(st["consumed"]), st["done"], st["step"]
        saved_per = st.get("per", -1)
        log(f"[train] resumed: step {step}, epoch {epoch}, {done:,}/{total_micros:,} micro-batches done, written at "
            f"world {st['world']}, running at world {world}; rank 0 restored optimizer state for {n_loaded}/{n_owned} params")
        if epoch > 0:
            micros = epoch_micros(train, args, epoch)
    # The epoch's step order is a function of (seed, epoch, per). A resume at the same per replays that exact
    # order minus the finished steps (bit-identical continuation); a resume at a different world size regroups
    # whatever micro-batches are left.
    steps, tail = build_steps(micros, set(range(len(micros))), per, train.lengths, args.seed * 7919 + epoch * 131 + per)
    if consumed:
        if all(set(s) <= consumed or not (set(s) & consumed) for s in steps) and saved_per == per:
            steps = [s for s in steps if not (set(s) & consumed)]
        else:
            steps, tail = build_steps(micros, set(range(len(micros))) - consumed, per, train.lengths,
                                      args.seed * 7919 + epoch * 131 + per + len(consumed))

    eval_idx = np.random.default_rng(0).permutation(len(val))[:args.eval_max].tolist()

    def save_ckpt():
        tmp = ckpt + ".tmp"
        if main_proc:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
        barrier()
        save_opt_shard(opt, model, os.path.join(tmp, f"opt_rank{rank}.pt"))
        if main_proc:
            torch.save(model.state_dict(), os.path.join(tmp, "model.pt"))
            with open(os.path.join(tmp, "state.json"), "w") as f:
                json.dump({"step": step, "epoch": epoch, "consumed": sorted(consumed), "done": done, "world": world, "per": per,
                           "total_micros": total_micros, "args": vars(args),
                           "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        barrier()
        if main_proc:
            shutil.rmtree(ckpt, ignore_errors=True)     # latest-only: the new one is complete before the old goes
            os.replace(tmp, ckpt)
            log(f"[train] checkpoint saved at step {step} ({done:,}/{total_micros:,} micro-batches)")
        barrier()

    def touch_lock():
        if main_proc and args.lock_file:
            with open(args.lock_file, "w") as f:
                f.write(f"{os.environ.get('PBS_JOBID', os.environ.get('SLURM_JOB_ID', '?'))} {os.uname().nodename} "
                        f"world={world} step={step} {time.strftime('%H:%M:%S')}\n")

    touch_lock()
    stagger = float(os.environ.get("DECISION_TEST_RANK_STAGGER", "0"))
    if stagger:                  # tests only: give the ranks different clocks, the condition that desynced 7673053
        time.sleep(rank * stagger)
    agg = [0.0, 0.0, 0.0, 0.0]
    t_log = time.time()
    last_save = time.time()
    steps_here = 0
    exit_reason = None
    si = 0
    while done < total_micros:
        if si >= len(steps):                      # epoch boundary: the unfillable tail counts as consumed
            done += len(tail)
            if done >= total_micros:
                break
            epoch += 1
            consumed = set()
            micros = epoch_micros(train, args, epoch)
            steps, tail = build_steps(micros, set(range(len(micros))), per, train.lengths, args.seed * 7919 + epoch * 131 + per)
            si = 0
            continue
        spec = steps[si]
        n_step = sum(len(micros[i]) for i in spec)
        mine = spec[rank * args.accum:(rank + 1) * args.accum]
        mult = lr_mult(done)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        for j, mi in enumerate(mine):
            b = train.collate(micros[mi], device)
            ctx = ddp.no_sync() if (world > 1 and j < len(mine) - 1) else _null()
            with ctx:
                with autocast(device):
                    logits = ddp(b["input_ids"], b["opt_pos"], b["opt_mask"], b["ans_pos"])
                logp = F.log_softmax(logits.float(), -1)
                per_ex = -(b["target"] * logp.masked_fill(b["target"] == 0, 0.0)).sum(-1)
                (per_ex.sum() * world / n_step).backward()
            with torch.no_grad():
                agg[0] += per_ex.sum().item()
                agg[1] += (logp.argmax(-1) == b["label"]).sum().item()
                agg[2] += len(micros[mi])
                agg[3] += b["real_tokens"]
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        consumed |= set(spec)
        done += per
        step += 1
        steps_here += 1
        si += 1
        touch_lock()

        if step % args.log_every == 0 or done >= total_micros:
            s = allsum(agg, device)
            dt = time.time() - t_log
            mem = ""
            if device.type == "cuda":
                mem_r = torch.cuda.memory_reserved(device) / 2**30
                mem_t = torch.cuda.get_device_properties(device).total_memory / 2**30
                mem = (f" | HBM {mem_r:.1f}/{mem_t:.1f}GiB ({100 * mem_r / mem_t:.0f}%) | peak "
                       f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB")
            log(f"step {step} | loss {s[0] / max(s[2], 1):.4f} | acc {s[1] / max(s[2], 1):.3f} | lr {opt.param_groups[0]['lr']:.2e} "
                f"| gnorm {float(gnorm):.2f} | ex/step {s[2] / args.log_every:.0f} | tok/s {s[3] / dt:,.0f} "
                f"({s[3] / dt / world:,.0f}/GPU){mem} | {100 * done / total_micros:.1f}% | {(time.time() - t_start) / 60:.1f} min")
            agg = [0.0, 0.0, 0.0, 0.0]
            t_log = time.time()
        if args.eval_every and (step % args.eval_every == 0 or done >= total_micros):
            ev = evaluate(model, val, eval_idx, args, device, rank, world)
            log(f"eval step {step} | " + " | ".join(
                f"{k}: loss {v['loss']:.4f} acc {v['acc']:.3f} ece {v['ece']:.3f} n {v['n']}" for k, v in ev.items()))
            t_log = time.time()     # the eval's wall time is not training time: keep it out of the next tok/s
        if done >= total_micros:
            break
        # EVERY time-based decision is made on rank 0 and shared, never evaluated per rank: each rank's clock and
        # start time differ by seconds, so a per-rank "20 minutes since the last save?" sent one rank into
        # save_ckpt()'s barrier while the others ran the next step's collectives. That desync hung a run
        # until the NCCL watchdog killed it (10 min) and lost the steps since the last save.
        flags = [0.0, 0.0, 0.0]
        if main_proc:
            flags[0] = float(bool(args.yield_file and os.path.exists(args.yield_file)))
            flags[1] = float(bool(args.wall_time and time.time() - t_start > args.wall_time - 300))
            flags[2] = float((time.time() - last_save) / 60 > args.save_every_min)
        flags = allsum(flags, device)
        if flags[0]:
            exit_reason = "yield requested"
        elif flags[1]:
            exit_reason = "wall time"
        elif args.max_steps and steps_here >= args.max_steps:
            exit_reason = "max_steps"
        if exit_reason in ("yield requested", "wall time") or (not exit_reason and flags[2]):
            save_ckpt()
            last_save = time.time()
        if exit_reason:
            log(f"[train] {exit_reason} at step {step}: exiting")
            break

    finished = done >= total_micros
    log(f"[train] {'finished' if finished else 'stopped'}: {min(done, total_micros):,}/{total_micros:,} micro-batches "
        f"({100 * min(done, total_micros) / total_micros:.1f}%), step {step}, {(time.time() - t_start) / 60:.1f} min")
    if finished or (exit_reason == "max_steps" and args.save_final_on_max_steps):
        final = os.path.join(args.out, "final")
        if main_proc:
            model.save(final, extra_config={"trained_steps": step, "micro_batches": done, "base": args.init or args.base,
                                            "data": args.data, "args": vars(args)},
                       tokenizer_dir=None if args.tiny else (args.init or args.base))
            log(f"[train] final model saved to {final}")
            if finished:
                shutil.rmtree(ckpt, ignore_errors=True)
        barrier()
    if main_proc and args.lock_file and os.path.exists(args.lock_file):
        os.remove(args.lock_file)
    if dist.is_initialized():
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
