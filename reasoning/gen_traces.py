#!/usr/bin/env python3
"""v8 distillation trace generator (§29) — model-agnostic, diversity-preserving.

Serves BOTH tiers of the v8 plan from one script:
  • TEACHER tier:     --model Qwen/Qwen3-4B --think 0   (concise solutions → wrapped in <think>)
  • SELF-ANCHOR tier: --model <v3 x_v6v2_040> --think 1 (v3's OWN <think> traces, kept as-is)

Both sample the model on CONTAMINATION-SAFE train problems (gsm8k split==train), keep only
verified-correct (extract_boxed == gold), and dedup to up to --keep-per-problem GENUINELY-DISTINCT
traces per problem (by a digits-masked step signature) — the diversity mechanism the v7 pilot lacked
(v7 kept 1 greedy trace/problem → student mode-collapse → self-cons −6.25, §26). Output rows are
canonicalized to the deployed student format and re-verified.

Run via the validated vLLM path (register() applies the transformers-5.x tokenizer shim, needed even
for native Qwen3). Output: a HF Dataset of {question, gold, source, trace}.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed, norm  # noqa: E402
from clean_eval import build_ids               # noqa: E402

GSM = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
MATH = "/project/rcc/youzhi/data/nlile_hendrycks-MATH-benchmark"

SOLVE_CONCISE = (
    "Solve this problem step by step, but keep it CONCISE (at most about 6 short steps). "
    "Do the arithmetic carefully. End with a single final line exactly of the form: "
    "The answer is \\boxed{{N}}.\n\nProblem: {q}"
)


def load_problems(sources, n, seed=0):
    import random as _r
    probs = []
    srcs = sources.split(",")
    if "gsm8k_train" in srcs:
        for ln in open(GSM):
            o = json.loads(ln)
            if o.get("split") != "train":
                continue
            g = extract_boxed(o["answer"])
            if g is not None:
                probs.append((o["question"], g, "gsm8k_train"))
    if "math_l13" in srcs:
        from datasets import load_from_disk
        for r in load_from_disk(MATH)["train"]:
            if int(r.get("level", 9)) > 3:
                continue
            cleaned = str(r["answer"]).strip().replace("$", "").replace(",", "").replace(" ", "")
            if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
                continue  # numeric-only (sympy-equiv is a Phase-B refinement)
            g = norm(cleaned)
            if g is not None:
                probs.append((r["problem"], g, "math_l13"))
    _r.Random(seed).shuffle(probs)
    return probs[:n] if n and n > 0 else probs


def step_signature(text):
    """Digits-masked, whitespace-collapsed skeleton — distinct signatures = distinct solution shapes."""
    t = text.split("</think>")[0]
    t = re.sub(r"\d+", "N", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:400]


def canonicalize(trace_text, gold, think_mode):
    """Return a deployed-format student trace or None."""
    if think_mode:
        # v3 self-anchor: already <think>…</think>… — keep iff closed + boxed==gold
        if "</think>" not in trace_text:
            return None
        i = trace_text.rfind("</think>")
        think = trace_text[:i + len("</think>")]
        content = think + f"\n\nThe answer is $\\boxed{{{gold}}}$."
    else:
        # teacher concise solution → wrap steps in <think>
        sol = trace_text.strip()
        j = sol.rfind("The answer is")
        steps = sol[:j].strip() if j > 0 else sol
        if not steps:
            steps = sol
        content = f"<think>\n{steps}\n</think>\n\nThe answer is $\\boxed{{{gold}}}$."
    if extract_boxed(content) != gold:
        return None
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", type=int, default=0, help="1 for a reasoning model (v3 self-anchor)")
    ap.add_argument("--sources", default="gsm8k_train")
    ap.add_argument("--n-problems", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-trace-chars", type=int, default=2200)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import Dataset
    import datetime as _dt

    probs = load_problems(args.sources, args.n_problems, args.seed)
    print(f"[gen] model={Path(args.model).name} think={args.think} problems={len(probs)} "
          f"n={args.n_samples} keep={args.keep_per_problem} T={args.temperature}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    q_text = SOLVE_CONCISE if not args.think else None
    prompts = [TokensPrompt(prompt_token_ids=build_ids(
        tok, (q_text.format(q=q) if q_text else q), think=bool(args.think))) for q, _, _ in probs]
    if args.temperature <= 0:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[im_end])
    else:
        sp = SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_tokens=args.max_tokens, stop_token_ids=[im_end])
    t0 = _dt.datetime.now()
    outs = llm.generate(prompts, sp)
    dt = (_dt.datetime.now() - t0).total_seconds()

    rows = []
    n_solved = 0
    for (q, gold, src), o in zip(probs, outs):
        seen_sig = set()
        kept = 0
        solved = False
        for cand in o.outputs:
            if extract_boxed(cand.text) != gold:
                continue
            solved = True
            content = canonicalize(cand.text, gold, args.think)
            if content is None or len(content) > args.max_trace_chars:
                continue
            sig = step_signature(content)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            rows.append({"question": q, "gold": gold, "source": src, "trace": content})
            kept += 1
            if kept >= args.keep_per_problem:
                break
        n_solved += int(solved)

    Dataset.from_list(rows).save_to_disk(args.out)
    by_src = defaultdict(int)
    for r in rows:
        by_src[r["source"]] += 1
    print(f"[gen] done in {dt:.1f}s | solved {n_solved}/{len(probs)} ({100*n_solved/max(len(probs),1):.1f}%) "
          f"| kept {len(rows)} distinct traces {dict(by_src)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
