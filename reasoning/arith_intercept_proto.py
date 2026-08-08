#!/usr/bin/env python3
"""Offline diagnostic that BOUNDS the arithmetic-interception lever (§22b #5) before building it.

The documented root cause (§10) is: correct multi-step PROCEDURES with wrong elementary
ARITHMETIC FACTS executed inside them (writes `2(l+w)` correctly, then computes `8+3=7`).
Lever #5 ("arithmetic-interception decoding") proposes to watch generation for a completed
`a op b =` and overwrite the RHS with the Python-true value. Its ceiling is: *of the traces
that get the wrong final answer, how many contain a detectable wrong inline arithmetic step?*

This script samples K traces/problem (reusing star_generate's KV-cached sampler + verifier)
and splits WRONG-but-closed traces into:
  * INTERCEPTABLE : >=1 inline `a op b = c` step is arithmetically wrong  -> #5 could fix it
  * STRUCTURAL    : every inline step is correct but the final answer is wrong
                    (wrong operand / wrong procedure / missing step)       -> #5 cannot help

It also reports, over ALL closed traces, the fraction that contain any checkable `a op b =`
step (feasibility gate: if the model rarely writes explicit inline equations, interception has
little surface to act on), and the "lucky" rate (CORRECT final answer despite >=1 wrong step —
the poison that motivates step-verified filtering, lever #7).

No weights changed, no training — pure measurement. ~1 GPU-hr for a few hundred problems.
"""

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass

from star_generate import extract_boxed, norm, load_problems, batched_sample, autofit_k  # noqa: E402

# Inline binary op: digit(.digit) OP digit(.digit) = digit(.digit).
# `*`,`×` -> multiply; a bare `x` is EXCLUDED (would false-match algebra variables).
EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([-+*/×])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
_TOL = 1e-6


def _apply(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op in ("*", "×"):
        return a * b
    if op == "/":
        return a / b if b != 0 else None
    return None


def step_errors(text):
    """Return (n_checkable, n_wrong) for inline `a op b = c` steps in `text`."""
    n_check = n_wrong = 0
    for m in EQ_RE.finditer(text):
        a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        val = _apply(a, op, b)
        if val is None:
            continue
        n_check += 1
        if abs(val - c) > _TOL:
            n_wrong += 1
    return n_check, n_wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--target-hbm", type=float, default=0.0,
                    help="auto-fit K to fill this HBM fraction (0=off, use fixed --k); OOM-safe.")
    ap.add_argument("--max-k", type=int, default=256, help="cap for --target-hbm auto-fit.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    fh = open(args.log, "a") if args.log else None

    def out(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True)
    model.to("cuda"); model.eval()

    problems = load_problems(args.source, args.n_problems, seed=args.seed)
    eff_k = args.k
    if args.target_hbm > 0:
        longest_q = max(problems, key=lambda p: len(p[0]))[0]
        penc = tok.apply_chat_template(
            [{"role": "user", "content": longest_q}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
        pid = penc["input_ids"] if hasattr(penc, "keys") else penc
        eff_k = autofit_k(model, pid.to("cuda"), eos_id=eos_id, target_frac=args.target_hbm,
                          max_k=args.max_k, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p)
    out("=" * 70)
    out(f"arith-interception proto  {_dt.datetime.now().isoformat(timespec='seconds')}")
    out(f"model={args.model_path}")
    out(f"n={len(problems)} K={eff_k} source={args.source} temp={args.temperature}")
    out("=" * 70)

    # counters over CLOSED traces
    closed = wrong = correct = 0
    wrong_interceptable = wrong_structural = 0
    closed_with_eq = 0
    lucky = 0  # correct final answer but >=1 wrong step
    t0 = _dt.datetime.now()

    for pi, (q, gold, tier) in enumerate(problems):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        while True:
            try:
                batch = ids.repeat(eff_k, 1)
                with torch.autocast("cuda", dtype=dtype):
                    gens = batched_sample(
                        model, batch, max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if eff_k <= 1:
                    raise
                eff_k = max(1, eff_k // 2)
                out(f"  [oom] halved K -> {eff_k} at problem {pi}")
        for g in gens:
            text = tok.decode(g, skip_special_tokens=True)
            if "</think>" not in text:
                continue  # unclosed: not this lever's target (that's budget-forcing #4)
            closed += 1
            pred = extract_boxed(text)
            n_check, n_wrong = step_errors(text)
            if n_check > 0:
                closed_with_eq += 1
            if pred is not None and pred == gold:
                correct += 1
                if n_wrong > 0:
                    lucky += 1
            else:
                wrong += 1
                if n_wrong > 0:
                    wrong_interceptable += 1
                else:
                    wrong_structural += 1

        if pi == 0 or (pi + 1) % 25 == 0:
            el = (_dt.datetime.now() - t0).total_seconds()
            out(f"  [{pi+1}/{len(problems)}] closed={closed} wrong={wrong} "
                f"(interceptable={wrong_interceptable} structural={wrong_structural}) "
                f"lucky={lucky} | {el/(pi+1):.2f}s/prob")

    out("\n  " + "-" * 66)
    out(f"  closed traces               : {closed}")
    out(f"  of closed, contain any `a op b =`: {closed_with_eq} "
        f"({100*closed_with_eq/max(closed,1):.1f}%)   <- feasibility surface for #5")
    out(f"  correct / wrong (closed)    : {correct} / {wrong}")
    if wrong:
        out(f"  wrong INTERCEPTABLE (>=1 bad step): {wrong_interceptable} "
            f"({100*wrong_interceptable/wrong:.1f}% of wrong)   <- CEILING of arith-interception #5")
        out(f"  wrong STRUCTURAL (all steps ok)  : {wrong_structural} "
            f"({100*wrong_structural/wrong:.1f}% of wrong)   <- #5 CANNOT fix (procedure/operand)")
    out(f"  'lucky' correct (>=1 bad step but right answer): {lucky} "
        f"({100*lucky/max(correct,1):.1f}% of correct)   <- poison for STaR/SFT (motivates #7)")
    out("  " + "-" * 66)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
