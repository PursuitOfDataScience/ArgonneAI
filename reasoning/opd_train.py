#!/usr/bin/env python3
"""ON-POLICY DISTILLATION: per-token reverse KL to a teacher, on the STUDENT's own rollouts.

WHY THIS EXISTS, and why it is not a thirteenth imitation arm.
Every post-training lever run on argonne4-think so far is a LIKELIHOOD objective on a set of
sequences somebody else picked: stage-C CoT-SFT, the verify tier, RFT/STaR rounds 1-2,
distillation from 3.5-think, distillation from Llama-3.1-8B. Their combined effect is exactly
what a likelihood objective predicts -- pass@8 rose 62.4 -> 69.0 while greedy sat at ~43 and
`acc|ANSWERED` never left ~50% (3.5-think: 70%). Two structural reasons:

  1. **The fuel is capped by the student's own correctness.** `fail_taxonomy.py` on 93,912
     on-policy rollouts of the current best checkpoint: 44.1% of training problems are NEVER
     solved in 8 samples, and only 23.3% of rollouts are correct. RFT can train on the 23% and
     must throw the other 77% away. That is why round 2 saturated.
  2. **Imitating an off-policy teacher does not transfer.** Llama-3.1-8B solves 45.7% of the same
     problems (3x the student) and distilling its text was measured at 39.24 vs the 39.21
     baseline -- a null, with `acc|ANSWERED` DOWN to 46.1%. The student learned a style it cannot
     execute. Off-policy sequence imitation is the wrong channel.

Per-token reverse KL on the student's own traces fixes both. The traces come from the student, so
there is no distribution shift; the supervision is the teacher's full next-token distribution at
every state the student actually visits, so a WRONG trace is just as informative as a right one --
the 77% stops being waste. Reverse KL is mode-seeking, which is the property this line needs: the
failure is a diffuse argmax (greedy 43 under pass@8 69), and mode-seeking sharpens one mode
instead of spreading mass over eight. Discount zero: each token is graded on its own, as in
Thinking Machines' on-policy distillation recipe.

WHAT MAKES IT POSSIBLE HERE. argonne4 kept the Qwen3 tokenizer, so a Qwen3 teacher and the
student assign IDENTICAL ids to identical text -- verified, not assumed: tokenizer vocab (151,643
entries), merges, all 26 added tokens, and the `<think>`/`</think>`/`<|im_end|>` ids are equal
between `think_combo` and `Qwen3-4B-Thinking-2507`, and a real 329-token trace tokenises to the
same id sequence under both. So ONE token sequence can be fed to both models and the
distributions are directly comparable, position by position. (The teacher's 151,936-wide head is
sliced to the student's 151,669 and renormalised; those extra ids are reserved tokens the student
has no output for.)

The teacher is only ever run FORWARD, never sampled. vLLM arch support is therefore irrelevant --
a teacher vLLM 0.11.2 cannot serve is still usable here.

  python reasoning/opd_train.py \
      --student /project/rcc/youzhi/models/a4_think_final/think_combo \
      --model_def model.py \
      --teacher /project/rcc/youzhi/toxic-models/Qwen/Qwen3-4B-Thinking-2507 \
      --rollouts /project/rcc/youzhi/data/a4_dpo/a4_dpo_all.jsonl \
      --out /project/rcc/youzhi/models/a4_think_final/think_opd
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_cotsft():
    """cot-sft.py has a hyphen in its name, so it cannot be imported normally.

    Reuse its loaders rather than copying them: this model needs manual construction to re-tie
    lm_head (AutoModelForCausalLM silently fails to) and its rotary embedding rebuilt at every
    layer. Divergence between two copies of that sequence is exactly the class of bug that has
    cost this line whole runs.
    """
    path = os.path.join(RDIR, "cot-sft.py")
    spec = importlib.util.spec_from_file_location("cot_sft_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cot_sft_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

LABEL_ORDER = ["correct", "wrong", "unclosed", "no_answer"]


def build_rows(rollouts, tok, build_ids, max_seq_len, per_problem, labels_keep,
               eos_id, seed):
    """One row per rollout: prompt ids + the trace the STUDENT actually generated.

    Stratified by label so a batch contains both states the student got right and states it got
    wrong -- the wrong ones are where the teacher has something to say. NOTHING is filtered for
    quality on purpose: unlike RFT, a bad trace is not waste here, it is the most informative
    state the teacher can be asked about.
    """
    by_q = defaultdict(list)
    with open(rollouts) as f:
        for line in f:
            r = json.loads(line)
            by_q[(r["pool"], r["question"])].append(r)

    rng = random.Random(seed)
    rows, stat = [], Counter()
    for (pool, q), rs in sorted(by_q.items()):
        buckets = defaultdict(list)
        for r in rs:
            if r["label"] in labels_keep:
                buckets[r["label"]].append(r)
        for v in buckets.values():
            rng.shuffle(v)
        # round-robin over labels so every problem contributes a MIX, not 3 copies of one mode
        picked, i = [], 0
        while len(picked) < per_problem:
            avail = [L for L in LABEL_ORDER if buckets.get(L)]
            if not avail:
                break
            L = avail[i % len(avail)]
            picked.append(buckets[L].pop())
            i += 1
        p_ids = build_ids(tok, q)
        for r in picked:
            tr = r["trace"]
            c_ids = tok.encode(tr, add_special_tokens=False)
            # A generation that ended on its own emitted <|im_end|>, which vLLM strips from the
            # text. An `unclosed` trace hit the token cap instead, so it has no terminator to
            # learn. Getting this backwards would train the model to stop mid-sentence.
            if r["label"] != "unclosed":
                c_ids = c_ids + [eos_id]
            if len(p_ids) + len(c_ids) > max_seq_len:
                stat["drop_too_long"] += 1
                continue
            if len(c_ids) < 8:
                stat["drop_too_short"] += 1
                continue
            rows.append({"ids": p_ids + c_ids, "n_prompt": len(p_ids),
                         "label": r["label"], "pool": pool})
            stat[f"keep_{r['label']}"] += 1
    rng.shuffle(rows)
    return rows, stat


def make_micro_batches(rows, max_batch_tokens, seed, window=256):
    """Group rows into micro-batches by PADDED TOKEN COUNT, not row count.

    Two reasons, both measured on this line rather than assumed:
      * MEMORY. The KD term materialises several [rows, T, 151669] fp32 tensors. With a row-count
        batch, peak HBM is set by the longest row that happens to land together -- an unlucky batch
        of four 1024-token traces OOMs 80% of the way through a run that had been fine for an hour.
        A token budget makes the peak a constant the probe can actually verify.
      * THROUGHPUT. These traces average 344 tokens with a p95 of 611, so a fixed row count either
        wastes the card on short batches or risks the long ones. Length-grouped batching was worth
        1.74x on a4's SFT for exactly this reason.
    Rows are shuffled, sorted inside windows (so a batch is length-homogeneous), packed, and then
    the resulting micro-batches are shuffled again so step order stays random.
    """
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    batches = []
    for w in range(0, len(idx), window):
        chunk = sorted(idx[w:w + window], key=lambda i: len(rows[i]["ids"]))
        cur, cur_max = [], 0
        for i in chunk:
            L = len(rows[i]["ids"])
            m = max(cur_max, L)
            if cur and (len(cur) + 1) * m > max_batch_tokens:
                batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(batch, pad_id):
    T = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.bool)     # True = a completion token
    is_corr = torch.zeros(len(batch), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"], dtype=torch.long)
        mask[i, b["n_prompt"]:n] = True
        is_corr[i] = (b["label"] == "correct")
    return ids, mask, is_corr


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def kd_loss(student_logits, teacher_logits, tgt_mask, V, teacher_temp):
    """Per-token reverse KL, KL(student || teacher), averaged over masked positions.

    Reverse (not forward) KL on purpose: it is mode-seeking. Forward KL would ask the student's
    1.04B of capacity to cover every mode a 4B teacher has, which is the mass-spreading this
    model already suffers from (greedy 43 under pass@8 69).
    """
    sl = student_logits[:, :-1, :V].float()
    tl = teacher_logits[:, :-1, :V].float()
    if teacher_temp != 1.0:
        tl = tl / teacher_temp
    log_ps = F.log_softmax(sl, dim=-1)
    log_pt = F.log_softmax(tl, dim=-1)          # renormalised over the SHARED vocab
    ps = log_ps.exp()
    per_tok = (ps * (log_ps - log_pt)).sum(-1)   # [B, T-1]
    m = tgt_mask[:, 1:].float()
    n = m.sum().clamp(min=1.0)
    return (per_tok * m).sum() / n


def ce_loss(student_logits, ids, tgt_mask, row_mask):
    """Plain next-token CE on selected rows -- the anchor term, off by default."""
    logits = student_logits[:, :-1, :]
    tgt = ids[:, 1:]
    m = tgt_mask[:, 1:] & row_mask.unsqueeze(1)
    if m.sum() == 0:
        return student_logits.sum() * 0.0
    lab = tgt.masked_fill(~m, -100)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           lab.reshape(-1), ignore_index=-100)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--model_def", default="model.py")
    ap.add_argument("--tokenizer_path", default="")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-problem", type=int, default=3)
    ap.add_argument("--labels", nargs="*", default=["correct", "wrong", "unclosed", "no_answer"])
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--rope-theta", type=float, default=1000000.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-batch-tokens", type=int, default=8192,
                    help="padded tokens per micro-batch; sets peak HBM (see make_micro_batches)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.0,
                    help="CE on gold-verified rows only. 0 = pure on-policy distillation.")
    ap.add_argument("--teacher-temp", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--stats-out", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    cot = _load_cotsft()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from clean_eval import build_ids

    tok_path = a.tokenizer_path or a.student
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    eos_id = cot.detect_eos_from_template(tok) or tok.eos_token_id
    print(f"[opd] tokenizer {tok_path}  len={len(tok)}  eos={eos_id}", flush=True)

    # ---- teacher: forward-only, bf16, frozen -------------------------------------------
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        a.teacher, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True)
    teacher.config.use_cache = False
    teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ttok = AutoTokenizer.from_pretrained(a.teacher)
    print(f"[opd] teacher {os.path.basename(a.teacher)}  "
          f"{sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B  "
          f"vocab={teacher.config.vocab_size}  loaded in {time.time() - t0:.0f}s", flush=True)

    # HARD GATE on the assumption the whole method rests on. If the two tokenizers ever disagree
    # on a single id, every KL term is computed against the teacher's distribution for a
    # DIFFERENT token and the run is silently meaningless.
    probe = ("Natalia sold clips to 48 friends in April, and then she sold half as many "
             "clips in May. How many clips did Natalia sell altogether?\n"
             "<think>\n48 / 2 = 24, so 48 + 24 = 72.\n</think>\n\nThe answer is $\\boxed{72}$.")
    ia = tok.encode(probe, add_special_tokens=False)
    ib = ttok.encode(probe, add_special_tokens=False)
    if ia != ib:
        raise SystemExit(f"FATAL tokenizer mismatch: student {len(ia)} ids vs teacher {len(ib)} "
                         "-- per-token KD is only defined under identical tokenisation")
    for t in ("<think>", "</think>", "<|im_end|>"):
        if tok.convert_tokens_to_ids(t) != ttok.convert_tokens_to_ids(t):
            raise SystemExit(f"FATAL special-token id mismatch on {t!r}")
    print(f"[opd] tokenizer identity verified on a {len(ia)}-token probe "
          f"(+ <think>/</think>/<|im_end|> ids)", flush=True)

    # ---- student ----------------------------------------------------------------------
    model_module = cot.import_model_definition(a.model_def)
    ArgonneConfig = getattr(model_module, "ArgonneConfig")
    ArgonneModel = getattr(model_module, "ArgonneModel")
    RotaryEmbedding = getattr(model_module, "RotaryEmbedding")
    cfg_d = json.load(open(os.path.join(a.student, "config.json")))
    cfg_d = {k: v for k, v in cfg_d.items() if not k.startswith("_")}
    sd = cot.load_hf_state_dict(a.student)
    for key in ("embed_tokens.weight", "lm_head.weight"):
        if key in sd:
            cfg_d["vocab_size"] = int(sd[key].shape[0])
            break
    config = ArgonneConfig(**cfg_d)
    model = ArgonneModel(config)
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.tie_weights()
    V = int(model.config.vocab_size)
    print(f"[opd] student {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B  vocab={V}  "
          f"missing={len(miss)} unexpected={len(unexp)}  "
          f"tied={model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}",
          flush=True)
    model.config.rope_theta = a.rope_theta
    model.config.max_position_embeddings = max(a.max_seq_len, int(cfg_d.get("max_position_embeddings", 0) or 0))
    model.config.block_size = model.config.max_position_embeddings
    cot.replace_rotary_embeddings(model, RotaryEmbedding, a.rope_theta,
                                  model.config.max_position_embeddings)
    model.config.use_flash_attention = True
    for blk in model.blocks:
        if hasattr(blk, "attn") and hasattr(blk.attn, "use_flash_attention"):
            blk.attn.use_flash_attention = True
    model.config.use_cache = False
    model.config.loss_chunk_size = 0          # KD needs logits; the chunked-CE path returns None
    model.gradient_checkpointing_enable()
    model.to(dev)
    model.train()
    del sd

    # ---- data ------------------------------------------------------------------------
    rows, dstat = build_rows(a.rollouts, tok, build_ids, a.max_seq_len, a.per_problem,
                             set(a.labels), eos_id, a.seed)
    print(f"[opd] rows={len(rows):,}  " + "  ".join(f"{k}={v:,}" for k, v in sorted(dstat.items())),
          flush=True)
    if not rows:
        raise SystemExit("FATAL no training rows")
    mean_comp = sum(len(r["ids"]) - r["n_prompt"] for r in rows) / len(rows)
    print(f"[opd] mean completion tokens {mean_comp:.0f}  mean total {sum(len(r['ids']) for r in rows) / len(rows):.0f}",
          flush=True)

    mb = make_micro_batches(rows, a.max_batch_tokens, a.seed)
    seq_per_mb = sum(len(b) for b in mb) / len(mb)
    pad_frac = 1.0 - sum(sum(len(rows[i]["ids"]) for i in b) for b in mb) / \
        sum(len(b) * max(len(rows[i]["ids"]) for i in b) for b in mb)
    steps_per_epoch = len(mb) // a.grad_accum
    total_steps = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    print(f"[opd] micro-batches={len(mb):,}  {seq_per_mb:.1f} seq/micro  "
          f"{a.max_batch_tokens} tok budget  padding {pad_frac * 100:.1f}%  "
          f"accum={a.grad_accum}  eff {seq_per_mb * a.grad_accum:.0f} seq/step  "
          f"steps/epoch={steps_per_epoch}  total={total_steps}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / max(1, a.warmup)
        prog = (s - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    pad_id = eos_id
    hist = []
    step = 0
    t_start = time.time()
    micro_i = 0
    # `n` counts every micro-step (loss terms); `diag_n` counts only the micro-steps the
    # diagnostics ran on. Dividing the diagnostics by `n` under grad_accum>1 under-reports them by
    # exactly that factor -- which is what made the 6-step probe read 38% agreement against the
    # real run's 77% on the same models.
    accum = {"kd": 0.0, "ce": 0.0, "n": 0, "diag_n": 0, "argmax_agree": 0.0, "close_ps": 0.0,
             "close_pt": 0.0, "close_n": 0}

    for ep in range(a.epochs):
        epoch_mb = mb if ep == 0 else make_micro_batches(rows, a.max_batch_tokens, a.seed + ep)
        for group in epoch_mb:
            batch = [rows[i] for i in group]
            ids, cmask, is_corr = collate(batch, pad_id)
            ids, cmask, is_corr = ids.to(dev), cmask.to(dev), is_corr.to(dev)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits = teacher(input_ids=ids).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_out = model(input_ids=ids)
            s_logits = s_out.logits

            L_kd = kd_loss(s_logits, t_logits, cmask, V, a.teacher_temp)
            L = a.kd_weight * L_kd
            L_ce = torch.zeros((), device=dev)
            if a.ce_weight > 0:
                L_ce = ce_loss(s_logits, ids, cmask, is_corr)
                L = L + a.ce_weight * L_ce
            (L / a.grad_accum).backward()

            # Diagnostics run on the FIRST micro-step of each accumulation group (micro_i is still
            # pre-increment here, so this fires exactly once per optimizer step): they need
            # another pass over a [B, T, 151669] tensor and paying that every micro-step would
            # slow the run for numbers only read in the log.
            if micro_i % a.grad_accum == 0:
                with torch.no_grad():
                    m = cmask[:, 1:]
                    nm = int(m.sum())
                    if nm:
                        accum["diag_n"] += 1
                        sl = s_logits[:, :-1, :V].detach()
                        tl = t_logits[:, :-1, :V]
                        accum["argmax_agree"] += float(((sl.argmax(-1) == tl.argmax(-1)) & m).sum()) / nm
                        # THE LENGTH RISK, measured rather than hoped: at the position where the
                        # student actually closed its think block, does a long-CoT teacher agree
                        # it should close? Every arm on this line that lengthened traces LOST, so
                        # a teacher that systematically demotes `</think>` is a run to kill early.
                        ci = tok.convert_tokens_to_ids("</think>")
                        close_pos = (ids[:, 1:] == ci) & m
                        if close_pos.any():
                            accum["close_ps"] += float(
                                (sl[..., ci] - sl.logsumexp(-1))[close_pos].exp().mean())
                            accum["close_pt"] += float(
                                (tl[..., ci] - tl.logsumexp(-1))[close_pos].exp().mean())
                            accum["close_n"] += 1
            accum["kd"] += float(L_kd)
            accum["ce"] += float(L_ce)
            accum["n"] += 1
            micro_i += 1

            if micro_i % a.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or step == 1:
                    n = max(1, accum["n"])
                    dn = max(1, accum["diag_n"])
                    cn = max(1, accum["close_n"])
                    print(f"[opd] step {step}/{total_steps}  revKL {accum['kd'] / n:.4f}  "
                          f"ce {accum['ce'] / n:.4f}  agree {accum['argmax_agree'] / dn * 100:.1f}%  "
                          f"p(</think>) student {accum['close_ps'] / cn:.3f} teacher "
                          f"{accum['close_pt'] / cn:.3f}  gnorm {float(gn):.2f}  "
                          f"lr {lr_at(step):.2e}  "
                          f"HBM {torch.cuda.max_memory_allocated() / 2**30:.1f}G  "
                          f"{(time.time() - t_start) / 60:.1f}min", flush=True)
                    hist.append({"step": step, "revKL": accum["kd"] / n, "ce": accum["ce"] / n,
                                 "agree": accum["argmax_agree"] / dn,
                                 "p_close_student": accum["close_ps"] / cn,
                                 "p_close_teacher": accum["close_pt"] / cn})
                    accum = {k: (0.0 if isinstance(v, float) else 0) for k, v in accum.items()}
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    print(f"[opd] done {step} steps in {(time.time() - t_start) / 60:.1f} min", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.to(torch.bfloat16)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    cpath = os.path.join(a.out, "config.json")
    c = json.load(open(cpath))
    c["eos_token_id"] = 151645       # the deployed stop token; 151643 never terminates a chat turn
    c["dtype"] = "bfloat16"
    c.pop("auto_map", None)
    json.dump(c, open(cpath, "w"), indent=2)
    # build_ids() renders the chat template, so a checkpoint without one silently evaluates on a
    # different prompt than it trained on. save_pretrained normally writes it; copy if it did not.
    for f in ("chat_template.jinja",):
        p = os.path.join(a.out, f)
        src = os.path.join(tok_path, f)
        if not os.path.exists(p) and os.path.exists(src):
            import shutil
            shutil.copy(src, p)
            print(f"[opd] copied {f} from {tok_path}", flush=True)
    # The soup builder and the vLLM port both index weights BY NAME. A save that renames or drops
    # a tensor loads as a fresh-init model and scores like noise, which reads as "the method
    # failed". Compare against the checkpoint we started from.
    from safetensors.torch import load_file
    src_keys = set(cot.load_hf_state_dict(a.student).keys())
    new_keys = set(load_file(os.path.join(a.out, "model.safetensors")).keys())
    if src_keys != new_keys:
        print(f"WARNING key-set drift: only-in-source={sorted(src_keys - new_keys)[:6]} "
              f"only-in-saved={sorted(new_keys - src_keys)[:6]}", flush=True)
    else:
        print(f"[opd] tensor key set identical to the source checkpoint ({len(new_keys)} tensors)",
              flush=True)
    print(f"[opd] saved -> {a.out}  eos={c['eos_token_id']}", flush=True)
    open(os.path.join(a.out, ".opd_complete"), "w").write(str(step))

    if a.stats_out:
        json.dump({"rows": len(rows), "data_stat": dstat, "steps": step,
                   "hist": hist, "args": vars(a)}, open(a.stats_out, "w"), indent=1)
        print(f"[opd] wrote {a.stats_out}", flush=True)


if __name__ == "__main__":
    main()
