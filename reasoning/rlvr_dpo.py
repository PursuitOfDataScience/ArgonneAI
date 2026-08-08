#!/usr/bin/env python3
"""RLVR-DPO for Argonne-3.0-think (§22 lever #3) — de-risked, with an HONEST gate.

WHY DPO AND NOT MORE GRPO: GRPO's group-relative advantage dies when all G rollouts share a
reward (§9/§20: ~2.6% solve-rate -> most groups all-wrong -> zero gradient). DPO needs only ONE
positive and ONE negative per prompt, which pass@32 supplies. Reward is EXTERNAL gold (not a
base-limited learned verifier, which failed in §22i).

TWO DE-RISKING CHOICES (from the 2026-07-09 adversarial audit):
  1. **correct-vs-WRONG pairs only.** The corpus also has correct-vs-UNCLOSED pairs, but those
     teach *closure* — which budget-forcing already gives free at deploy (2.5->7.5%). Training on
     them would let DPO optimize the easily-separable FORMAT feature and reproduce the GRPO
     reward-proxy trap (moves the proxy, zero held-out gain).
  2. **step-verify the CHOSEN trace.** A trace can reach the gold answer *through* a verified-wrong
     arithmetic step ("lucky-via-wrong-step"). Rewarding those teaches bad reasoning. We drop any
     chosen trace containing a detectably-wrong inline `a op b = c`.

CONTAMINATION NOTE (critical, 2026-07-09): the pairs derive from GSM8K problems, and GSM8K is
contaminated for every Argonne think model (CoT-SFT saw ~94% of GSM8K test; STaR saw the exact
eval problems). So this trains on GSM8K but **must be graded with `clean_eval.py` on SVAMP/ASDiv**,
which appear in no training mix and are disjoint from the pairs.

Right-padding is safe: ArgonneModel forces attention_mask=None (pure causal), so a real token
never attends to trailing pads; the loss is masked to response tokens (same argument as grpo.py).
"""
import argparse
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401  (registers argonne2)
except ModuleNotFoundError:
    pass

from star_generate import extract_boxed, norm  # noqa: E402

EQ = re.compile(r"(-?\d+(?:\.\d+)?)\s*([-+*/×xX])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def _val(a, op, b):
    op = op.lower()
    if op == "+": return a + b
    if op == "-": return a - b
    if op in ("*", "×", "x"): return a * b
    if op == "/": return a / b if b else None
    return None


def has_bad_arith(text):
    """True if the trace contains a detectably-wrong inline `a op b = c` step."""
    for m in EQ.finditer(text):
        try:
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        v = _val(a, op, b)
        if v is None:
            continue
        if abs(v - c) > 1e-6:
            return True
    return False


def build_pairs(tok, data_path, max_len, neg_kinds, step_verify, log):
    from datasets import load_from_disk
    ds = load_from_disk(data_path)
    ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")

    pairs = []
    n_kind_drop = n_lucky_drop = n_len_drop = 0
    for r in ds:
        if r.get("neg_kind") not in neg_kinds:
            n_kind_drop += 1
            continue
        ch, rj = r["chosen"], r["rejected"]
        q = ch[0]["content"]
        c_txt, r_txt = ch[-1]["content"].strip(), rj[-1]["content"].strip()
        if not c_txt or not r_txt or c_txt == r_txt:
            continue
        if step_verify and has_bad_arith(c_txt):
            n_lucky_drop += 1          # gold reached THROUGH a wrong arithmetic step
            continue
        p_ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=True)
        if hasattr(p_ids, "keys"):
            p_ids = p_ids["input_ids"]
        if len(p_ids) > 0 and isinstance(p_ids[0], (list, tuple)):
            p_ids = p_ids[0]
        p_ids = [int(x) for x in p_ids]
        c_ids = p_ids + tok.encode(c_txt, add_special_tokens=False) + [eos]
        r_ids = p_ids + tok.encode(r_txt, add_special_tokens=False) + [eos]
        if len(c_ids) > max_len or len(r_ids) > max_len:
            n_len_drop += 1
            continue
        pairs.append((p_ids, c_ids, r_ids))
    log(f"[dpo] pairs kept={len(pairs)}  dropped: wrong_kind={n_kind_drop} "
        f"lucky_wrong_step={n_lucky_drop} too_long={n_len_drop}  (neg_kinds={sorted(neg_kinds)})")
    return pairs


def _pad(seqs, pad_id, device):
    m = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (m - len(s)) for s in seqs], dtype=torch.long, device=device)
    real = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs], dtype=torch.long, device=device)
    return ids, real


def resp_logps(model, ids, real, plens):
    """Sum log p over RESPONSE positions only. attention_mask is NOT passed (Argonne forces
    causal-only); right-padding makes that safe, and `real` masks pads out of the loss."""
    out = model(ids)
    logits = out.logits[:, :-1, :]
    tgt = ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    B, Lm1 = tok_logp.shape
    pos = torch.arange(1, Lm1 + 1, device=ids.device).unsqueeze(0).expand(B, -1)
    plen = torch.tensor(plens, device=ids.device).unsqueeze(1)
    mask = (pos >= plen) & (real[:, 1:] == 1)
    return (tok_logp * mask).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--data", default="/project/rcc/youzhi/data/star_dpo_soup_r1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--neg-kinds", nargs="+", default=["wrong"],
                    help="'wrong' only by default; add 'unclosed' to include closure pairs (NOT advised)")
    ap.add_argument("--no-step-verify", dest="step_verify", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    fh = open(args.log, "a") if args.log else None

    def log(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    log("=" * 72)
    log(f"RLVR-DPO  model={args.model}")
    log(f"  beta={args.beta} lr={args.lr} epochs={args.epochs} "
        f"eff_batch={args.micro_bs*args.grad_accum} step_verify={args.step_verify}")
    log("=" * 72)

    pairs = build_pairs(tok, args.data, args.max_len, set(args.neg_kinds), args.step_verify, log)
    if not pairs:
        log("ERROR: no pairs after filtering"); sys.exit(1)

    log(f"[dpo] loading policy (fp32) + frozen ref (bf16) from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=torch.float32, low_cpu_mem_usage=True).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None
    ref = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                               dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    steps_per_epoch = max(1, len(pairs) // (args.micro_bs * args.grad_accum))
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    log(f"[dpo] pairs={len(pairs)} steps/epoch={steps_per_epoch} total={total}")

    t0 = time.time()
    opt_step = micro = 0
    acc = {"loss": 0.0, "margin_acc": 0.0, "chosen_lp": 0.0, "rej_lp": 0.0, "n": 0}
    for ep in range(args.epochs):
        order = list(range(len(pairs)))
        random.Random(args.seed + ep).shuffle(order)
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(order) - args.micro_bs + 1, args.micro_bs):
            bp = [pairs[j] for j in order[i:i + args.micro_bs]]
            plens = [len(p) for p, _, _ in bp]
            c_ids, c_real = _pad([c for _, c, _ in bp], pad_id, "cuda")
            r_ids, r_real = _pad([r for _, _, r in bp], pad_id, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pol_c = resp_logps(model, c_ids, c_real, plens)
                pol_r = resp_logps(model, r_ids, r_real, plens)
            with torch.no_grad():
                ref_c = resp_logps(ref, c_ids, c_real, plens)
                ref_r = resp_logps(ref, r_ids, r_real, plens)
            logits = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -F.logsigmoid(logits).mean()
            (loss / args.grad_accum).backward()

            acc["loss"] += loss.item()
            acc["margin_acc"] += (logits > 0).float().mean().item()
            # likelihood displacement watch: DPO can raise the MARGIN while pushing chosen DOWN
            acc["chosen_lp"] += (pol_c - ref_c).mean().item()
            acc["rej_lp"] += (pol_r - ref_r).mean().item()
            acc["n"] += 1
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 5 == 0 or opt_step == 1:
                    n = max(acc["n"], 1)
                    log(f"[dpo] ep{ep} step {opt_step}/{total} loss {acc['loss']/n:.4f} "
                        f"margin_acc {acc['margin_acc']/n:.3f} "
                        f"d_chosen {acc['chosen_lp']/n:+.3f} d_rej {acc['rej_lp']/n:+.3f} "
                        f"{(time.time()-t0)/60:.1f}min")
                    acc = {k: 0.0 for k in acc}

    model.config.use_cache = True
    os.makedirs(args.out, exist_ok=True)
    log(f"[dpo] saving -> {args.out}")
    model.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    src_ct = Path(args.model) / "chat_template.jinja"
    if src_ct.exists():
        shutil.copy(src_ct, Path(args.out) / "chat_template.jinja")
    log(f"[dpo] DONE in {(time.time()-t0)/60:.1f} min")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
