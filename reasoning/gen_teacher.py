#!/usr/bin/env python3
"""External-TEACHER distillation generation (§25-motivated): use Qwen3-4B (which solves clean
grade-school problems at ~94%, §25) as a teacher to produce CORRECT short worked solutions on
gsm8k-TRAIN, for CoT-SFT into v7.

WHY. §25 showed v3's pass@K candidates are rankable to ~75% by a strong external reasoner, i.e. the
wall is v3's own generation/self-verification (base capability). Self-STaR saturates (§8/§21) because
you can only imitate successes you already produce. Distilling a STRONGER teacher's correct traces is
the higher-EV single-card lever: v3 imitates Qwen's solutions, not its own thin ones.

CONTAMINATION-SAFE. Solves gsm8k split=="train" ONLY (the pooled TEST rows were the contamination,
§23); the held-out judge stays clean SVAMP/ASDiv (disjoint). Same methodology as v6's gsm8k_train_short.

Teacher decode: Qwen3-4B NON-thinking (enable_thinking=False), "solve concisely, end with \\boxed{}".
Keep a trace iff its boxed answer == gold AND it is short (termination-safe, matches v6's short-only
diet). Output rows = {question, gold, solution} -> Dataset; build_mix_v7 wraps them into <think> traces.

Run via vLLM (fast + fills HBM). register() is called for the transformers-5.x tokenizer shim (needed
even for native Qwen3-4B, §25 review).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed, norm  # noqa: E402
from clean_eval import build_ids               # noqa: E402

GSM = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"

SOLVE = (
    "Solve this problem step by step, but keep it CONCISE (at most about 6 short steps). "
    "Do the arithmetic carefully. End with a single final line exactly of the form: "
    "The answer is \\boxed{{N}}.\n\nProblem: {q}"
)


def load_gsm_train():
    probs = []
    for ln in open(GSM):
        o = json.loads(ln)
        if o.get("split") != "train":
            continue
        gold = extract_boxed(o["answer"])
        if gold is not None:
            probs.append((o["question"], gold))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="Qwen/Qwen3-4B")   # the teacher
    ap.add_argument("--out", default="/project/rcc/youzhi/data/teacher_qwen_gsm")
    ap.add_argument("--n-problems", type=int, default=0)     # 0 = all train (7473)
    ap.add_argument("--max-tokens", type=int, default=400)   # short solutions only
    ap.add_argument("--max-sol-chars", type=int, default=1400)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import vllm_argonne
    vllm_argonne.register()   # tokenizer shim (needed even for native Qwen3-4B)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import Dataset
    import datetime as _dt

    probs = load_gsm_train()
    if args.n_problems and args.n_problems > 0:
        probs = probs[:args.n_problems]
    print(f"[teacher] gsm8k-train problems: {len(probs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.verifier, trust_remote_code=True)
    llm = LLM(model=args.verifier, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    # NON-thinking teacher: concise, terminating; keep only correct+short.
    prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, SOLVE.format(q=q), think=False))
               for q, _ in probs]
    sp = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_tokens)
    t0 = _dt.datetime.now()
    outs = llm.generate(prompts, sp)
    dt = (_dt.datetime.now() - t0).total_seconds()

    rows = []
    n_correct = n_short = 0
    for (q, gold), o in zip(probs, outs):
        sol = o.outputs[0].text.strip()
        pred = extract_boxed(sol)
        if pred is None or pred != gold:
            continue
        n_correct += 1
        if len(sol) > args.max_sol_chars:
            continue
        n_short += 1
        rows.append({"question": q, "gold": gold, "solution": sol})

    Dataset.from_list(rows).save_to_disk(args.out)
    print(f"[teacher] solved {len(probs)} in {dt:.1f}s | correct {n_correct} "
          f"({100*n_correct/max(len(probs),1):.1f}%) | correct+short kept {n_short} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
