#!/usr/bin/env python3
"""Fast vLLM labeled-rollout generator + RLVR-DPO pair builder (contamination-safe).

Replaces the slow HF path in `star_generate.py` (~56 s/problem, 30h for 2000 problems) with the
validated vLLM port (~4.5 s/problem) — a ~12x speedup that makes a properly-powered RLVR-DPO /
STaR corpus cheap.

TWO BUGS THIS FIXES vs star_generate.py:
  1. **split filter.** `star_generate.load_problems()` reads a shard holding GSM8K train+test
     POOLED and never filters `split`. Here `--split train` is enforced, so generation never
     touches GSM8K test.
  2. **train/eval overlap.** star_gen used the DEFAULT seed=0 + START=0, the exact shuffle
     `eval_math.py` evaluates on (200/200 overlap). Here you pass an explicit `--seed`/`--start`,
     and the honest gate is `clean_eval.py` (SVAMP/ASDiv) which shares no problems at all.

Outputs:
  --all-out   HF dataset of EVERY rollout with label {correct,wrong,unclosed,no_answer} + gold
  --dpo-out   RLVR-DPO pairs, already de-risked:
                * correct-vs-WRONG only (correct-vs-unclosed teaches closure, which budget-forcing
                  gives free -> the GRPO reward-proxy trap)
                * CHOSEN traces step-verified: drop "lucky-via-wrong-step" positives (~35% of
                  them reach gold THROUGH a verified-wrong `a op b = c`)
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star_generate import extract_boxed, norm  # noqa: E402

GSM8K = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
EQ = re.compile(r"(-?\d+(?:\.\d+)?)\s*([-+*/×xX])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def _val(a, op, b):
    op = op.lower()
    if op == "+": return a + b
    if op == "-": return a - b
    if op in ("*", "×", "x"): return a * b
    if op == "/": return a / b if b else None
    return None


def has_bad_arith(text):
    for m in EQ.finditer(text):
        try:
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        v = _val(a, op, b)
        if v is not None and abs(v - c) > 1e-6:
            return True
    return False


def load_gsm8k(split, n, start, seed):
    probs = []
    for ln in open(GSM8K):
        o = json.loads(ln)
        if split != "all" and o.get("split") != split:
            continue
        g = extract_boxed(o["answer"])
        if g is not None:
            probs.append((o["question"], g))
    random.Random(seed).shuffle(probs)
    probs = probs[start:]
    return probs[:n] if n and n > 0 else probs


def label(text, gold):
    pred = extract_boxed(text)
    if "</think>" not in text:
        return "unclosed", pred
    if pred is None:
        return "no_answer", pred
    return ("correct" if pred == gold else "wrong"), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--n-problems", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234, help="NOT 0 — keep disjoint from the old seed-0 runs")
    ap.add_argument("--all-out", default=None)
    ap.add_argument("--dpo-out", default=None)
    ap.add_argument("--max-neg-per-q", type=int, default=2)
    ap.add_argument("--max-pos-per-q", type=int, default=2)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    fh = open(args.log, "a") if args.log else None

    def out(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    probs = load_gsm8k(args.split, args.n_problems, args.start, args.seed)
    out("=" * 72)
    out(f"vLLM rollouts  model={args.model}")
    out(f"  gsm8k split={args.split} n={len(probs)} start={args.start} seed={args.seed} K={args.k}")
    out("=" * 72)

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_new_tokens)

    def ids(q):
        e = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                    add_generation_prompt=True, enable_thinking=True)
        if hasattr(e, "keys"):
            e = e["input_ids"]
        if len(e) > 0 and isinstance(e[0], (list, tuple)):
            e = e[0]
        return [int(x) for x in e]

    outs = llm.generate([TokensPrompt(prompt_token_ids=ids(q)) for q, _ in probs], sp)

    fm = Counter()
    all_rows = []
    by_q = defaultdict(lambda: {"correct": [], "wrong": [], "unclosed": [], "no_answer": []})
    n_solved = 0
    for (q, gold), o in zip(probs, outs):
        got = False
        for c in o.outputs:
            t = c.text.strip()
            try:
                lab, pred = label(t, gold)
            except Exception:
                lab, pred = "no_answer", None   # never let one trace kill the job

            fm[lab] += 1
            by_q[q][lab].append(t)
            if lab == "correct":
                got = True
            all_rows.append({"question": q, "trace": t, "label": lab,
                             "pred": pred or "", "gold": gold})
        if got:
            n_solved += 1

    n = len(probs)
    tot = sum(fm.values())
    out(f"\n  pass@{args.k}            : {100*n_solved/n:.2f}%  ({n_solved}/{n} problems)")
    out(f"  single-sample acc    : {100*fm['correct']/max(tot,1):.2f}%")
    out(f"  label dist           : {dict(fm)}")

    if args.all_out:
        from datasets import Dataset
        Dataset.from_list(all_rows).save_to_disk(args.all_out)
        out(f"  saved {len(all_rows)} labeled rollouts -> {args.all_out}")

    if args.dpo_out:
        from datasets import Dataset
        rng = random.Random(args.seed)
        pairs, n_lucky = [], 0
        for q, g in by_q.items():
            if not g["correct"] or not g["wrong"]:
                continue
            # step-verified positives, shortest first (compress pass@1, don't reward length)
            pos = [t for t in g["correct"] if not has_bad_arith(t)]
            n_lucky += len(g["correct"]) - len(pos)
            if not pos:
                continue
            pos = sorted(pos, key=len)[:args.max_pos_per_q]
            negs = rng.sample(g["wrong"], min(args.max_neg_per_q, len(g["wrong"])))
            for c in pos:
                for r in negs:
                    if c.strip() == r.strip():
                        continue
                    pairs.append({
                        "chosen": [{"role": "user", "content": q},
                                   {"role": "assistant", "content": c}],
                        "rejected": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": r}],
                        "neg_kind": "wrong"})
        rng.shuffle(pairs)
        Dataset.from_list(pairs).save_to_disk(args.dpo_out)
        out(f"  DPO pairs (correct-vs-wrong, step-verified): {len(pairs)}  "
            f"(dropped {n_lucky} lucky-via-wrong-step positives)")
        out(f"  saved -> {args.dpo_out}")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
