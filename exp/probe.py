"""argonne4.5 probe harness — 3x H200 DDP, <=30 min per arm.

One arm = one JSON config. Trains the argonne4.5 candidate on the production 50/30/20
edu/math/code mixture for a FIXED TOKEN BUDGET, then evaluates pure next-token CE on the
production held-out val sets. Emits throughput + HBM so systems arms and quality arms use the
same harness.

Invariants (each one is a lesson from a previous Argonne campaign):
  I1  Quality arms are compared ISO-TOKEN, never iso-wall. Wall time is a resource, not a metric.
  I2  Eval is PURE next-token CE: forward WITHOUT labels, compute CE from logits. MTP / z-loss
      must never leak into the reported number (arch-sweep invariant #4).
  I3  The reported node name is part of the record. Cross-GPU comparison is what forced the
      KEY FINDING 14 retraction and the Finding C retraction, and this cluster mixes H100/H200.
  I4  The source-sampling RNG is seeded identically on every rank and is independent of the
      micro-batch, so every arm sees the same source sequence.
  I5  A wall-guard trip still writes a record, marked invalid. A missing result is
      indistinguishable from a crash otherwise.
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ArgonneConfig, ArgonneModel

VOCAB = 151680                      # len(Qwen3 tokenizer) padded to a mult of 128 for fp8 lm_head
EOS = 151643                        # Qwen3 <|endoftext|>; the flat bins use it as the doc separator.
# Measured density in the first 20M tokens of each source: edu 1 doc / 1,008 tok,
# finemath 1 / 1,505, code 1 / 1,875 -> a block-2048 window spans ~1.1-2.0 documents, so the
# packed batch really is letting tokens attend across document boundaries.
DATA = "/project/rcc/youzhi/data/argonne4_pretrain"
SOURCES = [("edu_flat.bin", 50), ("finemath_flat.bin", 30), ("code_flat.bin", 20)]
VALS = [("edu", "val_edu.bin"), ("math", "val_math.bin"), ("code", "val_code.bin")]


def load_bin(path):
    with open(path, "rb") as f:
        if np.frombuffer(f.read(256 * 4), dtype=np.int32)[0] != 20240801:
            raise ValueError(f"bad magic: {path}")
    return np.memmap(path, dtype=np.uint32, mode="r", offset=256 * 4)


class Src:
    """One flat .bin, sharded across ranks (offset rank*B*T, stride B*T*world)."""

    def __init__(self, path, B, T, rank, world, start=0):
        self.t, self.B, self.T = load_bin(path), B, T
        self.rank, self.world = rank, world
        self.base = start
        self.pos = start + rank * B * T

    def next(self):
        n = self.B * self.T + 1
        if self.pos + n > len(self.t):
            self.pos = self.base + self.rank * self.B * self.T
        buf = torch.from_numpy(self.t[self.pos:self.pos + n].astype(np.int64))
        self.pos += self.B * self.T * self.world
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)


class Mix:
    """Weighted multi-source sampler. I4: rng carries NO rank term, so all ranks agree."""

    def __init__(self, B, T, rank, world, seed=1337):
        self.s = [Src(os.path.join(DATA, p), B, T, rank, world) for p, _ in SOURCES]
        w = np.array([w for _, w in SOURCES], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self.counts = [0, 0, 0]

    def next(self):
        i = int(self.rng.choice(len(self.s), p=self.p))
        self.counts[i] += 1
        return self.s[i].next()


@torch.no_grad()
def eval_ce(model, path, B, T, device, windows=64):
    """I2: pure next-token CE. Forward WITHOUT labels so no auxiliary term can leak in.

    Called at MULTIPLE T (see eval_lengths). Evaluating past the training block is the only way
    to see length generalisation, which is the entire reason the NoPE-global layout was proposed:
    CE at the training length is blind to it. Batch is chosen so B*T is constant, keeping the
    fp32 logit transient the same size at every length.
    """
    model.eval()
    toks = load_bin(path)
    tot, n = 0.0, 0
    for w in range(windows):
        off = w * B * T
        if off + B * T + 1 > len(toks):
            break
        buf = torch.from_numpy(toks[off:off + B * T + 1].astype(np.int64))
        x = buf[:-1].view(B, T).to(device)
        y = buf[1:].view(B, T).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        tot += ce.item() * y.numel(); n += y.numel()
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    main_proc = rank == 0
    t_job = time.time()
    deadline = t_job + float(cfg.get("wall_guard_sec", 1500))

    torch.manual_seed(cfg.get("seed", 444))
    np.random.seed(cfg.get("seed", 444))

    T = int(cfg.get("block_size", 2048))
    B = int(cfg["micro_batch"])
    accum = int(cfg["grad_accum"])
    eff = B * accum * T * world
    budget = int(cfg["train_tokens"])
    steps = max(1, budget // eff)

    arch = {k: v for k, v in cfg.items() if k in {
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "qk_norm", "v_norm", "sandwich_norm", "z_loss_weight",
        "logit_softcap", "rope_theta", "tie_word_embeddings", "mtp_horizon", "mtp_loss_weight",
        "interleaved_local_attention", "local_attention_window",
        "attn_pattern", "sliding_window_size", "nope_global", "attn_gate", "mlp_type",
        "mtp_module_layers", "doc_mask"}}
    mc = ArgonneConfig(vocab_size=VOCAB, max_position_embeddings=T, use_flash_attention=True,
                       loss_chunk_size=int(cfg.get("loss_chunk_size", 0)), **arch)
    mc._keep_in_fp32_modules = []
    model = ArgonneModel(mc).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    if cfg.get("grad_checkpointing", 1):
        model.set_gradient_checkpointing(True)
        model.checkpoint_stride = int(cfg.get("checkpoint_stride", 1))

    if cfg.get("fp8", 0):
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        f8 = Float8LinearConfig(enable_fsdp_float8_all_gather=False)
        convert_to_float8_training(
            model, config=f8,
            module_filter_fn=lambda m, fqn: ("lm_head" not in fqn or cfg.get("fp8_lm_head", 1))
            and getattr(m, "in_features", 16) % 16 == 0 and getattr(m, "out_features", 16) % 16 == 0)
    t_compile0 = time.time()
    if cfg.get("torch_compile", 1):
        model = torch.compile(model, mode=cfg.get("compile_mode", "default"))
    model = DDP(model, device_ids=[local])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 6e-4),
                            betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    warm, cd = int(cfg.get("warmup", 0.02 * steps)) or 1, cfg.get("cooldown_frac", 0.15)
    cds = max(1, int(cd * steps))

    def lr_l(s):
        if s < warm:
            return s / warm
        if s < steps - cds:
            return 1.0
        return 1.0 - min(1.0, (s - (steps - cds)) / cds) * 0.9
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)

    mix = Mix(B, T, rank, world, seed=cfg.get("data_seed", 1337))
    use_doc = bool(cfg.get("doc_mask", False))
    if main_proc:
        print(f"[probe] {cfg['id']} params={nparams:,} steps={steps} eff_batch={eff:,} "
              f"block={T} micro={B}x{accum} world={world} node={socket.gethostname()}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses, t_first, t_steady, tok_steady, aborted = [], None, None, 0, False
    STEADY_FROM = min(10, max(2, steps // 10))
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for m in range(accum):
            x, y = mix.next()
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            # Document ids for intra-document masking: doc_id[i] = how many EOS tokens occur
            # strictly BEFORE position i, so the separator belongs to the document it terminates.
            # Ids only need to be unique within a row; each row is an independent window.
            kw = {}
            if use_doc:
                is_eos = (x == EOS)
                kw["document_ids"] = torch.cumsum(is_eos.long(), dim=1) - is_eos.long()
            sync = (m == accum - 1)
            with model.no_sync() if not sync else torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, labels=y, **kw).loss / accum
                loss.backward()
            acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 0.4))
        opt.step(); sched.step()
        losses.append(acc)
        if s == 0:
            t_first = time.time()
        if s == STEADY_FROM:
            torch.cuda.synchronize(); t_steady = time.time(); tok_steady = 0
        elif t_steady is not None:
            tok_steady += eff
        # Live progress, so a stalled or slower-than-budget arm is visible from squeue-time rather
        # than only at the end. Also the only place the wall-guard ETA is observable mid-run.
        if main_proc and (s % 25 == 0 or s == steps - 1) and t_steady is not None and s > STEADY_FROM:
            el = time.time() - t_steady
            tps_now = tok_steady / max(1e-9, el)
            eta = (steps - 1 - s) * eff / max(1.0, tps_now)
            print(f"[probe] step {s}/{steps} loss={acc:.4f} tok/s={tps_now:,.0f} "
                  f"eta={eta/60:.1f}min guard_left={(deadline - time.time())/60:.1f}min "
                  f"hbm={torch.cuda.max_memory_allocated()/torch.cuda.get_device_properties(local).total_memory:.1%}",
                  flush=True)
        if time.time() > deadline:
            aborted = True
            if main_proc:
                print(f"[probe] WALL GUARD at step {s}/{steps}", flush=True)
            break
    torch.cuda.synchronize()
    t_train_end = time.time()
    done_steps = len(losses)
    tps = tok_steady / max(1e-9, t_train_end - t_steady) if t_steady else 0.0
    hbm = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(local).total_memory

    base = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
    ces, len_ces = {}, {}
    if main_proc and not aborted:
        for name, f in VALS:
            ces[name] = eval_ce(base, os.path.join(DATA, f), max(1, 8), T, dev)
        # Length generalisation: same held-out text, longer windows. Anything past the training
        # block T is extrapolation. B*T held at 16,384 so the logit transient does not grow.
        for L in cfg.get("eval_lengths", []):
            if L == T:
                len_ces[str(L)] = float(np.mean(list(ces.values())))
                continue
            try:
                vals = [eval_ce(base, os.path.join(DATA, f), max(1, 16384 // L), L, dev, windows=16)
                        for _, f in VALS]
                len_ces[str(L)] = float(np.mean(vals))
            except Exception as e:
                len_ces[str(L)] = f"ERR {type(e).__name__}"

    if main_proc:
        rec = dict(
            id=cfg["id"], status="wall_guard" if aborted else "ok", valid=not aborted,
            node=socket.gethostname(), gpu=torch.cuda.get_device_name(local), world=world,
            params=nparams, steps_planned=steps, steps_done=done_steps,
            tokens=done_steps * eff, eff_batch=eff, block=T, micro_batch=B, grad_accum=accum,
            train_loss_ema=float(np.mean(losses[-20:])) if losses else None,
            ce=ces, tgt=float(np.mean(list(ces.values()))) if ces else None,
            tgt_by_length=len_ces,
            tokens_per_sec=round(tps, 1), hbm_frac=round(hbm, 4),
            startup_sec=round(t_first - t_job, 1) if t_first else None,
            compile_and_first_step_sec=round(t_first - t_compile0, 1) if t_first else None,
            train_sec=round(t_train_end - (t_steady or t_job), 1),
            job_sec=round(time.time() - t_job, 1), src_counts=mix.counts, config=cfg)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
        print("[probe] RESULT " + json.dumps({k: rec[k] for k in
              ("id", "status", "params", "tokens", "tgt", "ce", "tokens_per_sec",
               "hbm_frac", "startup_sec", "job_sec")}), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
