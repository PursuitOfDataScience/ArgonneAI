#!/usr/bin/env python3
"""Self-consistency inference for argonne3.0-think (§22 cheap win).

The deployed model answers a math question with GREEDY pass@1 ~3-6%, but pass@256 ~82%: the
right answer is usually in the sample set, the model just can't pick it. Self-consistency is
the cheapest picker: sample K traces, DISCARD the ~half that never close </think> or have no
boxed answer, then MAJORITY-VOTE the rest. Measured ~14% on GSM8K vs ~3% greedy — a 2-4x gain,
zero training. This is the deployable artifact (no gold needed) plus a --grade mode to measure.

Optional --think-budget N force-closes </think> at N tokens (s1-style) so the ~half-that-loop
still contribute a vote — stacks with voting. Reuses eval_math.sample_batch (budget-aware,
KV-cached) + star_generate's verifier. No weights changed.
"""
import argparse
import json
import sys
from collections import Counter
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

from eval_math import sample_batch, CLOSE_STR  # budget-aware KV-cached sampler
from star_generate import extract_boxed, norm, load_problems  # verifier + loader


@torch.inference_mode()
def vote(model, tok, question, *, eos_id, k, max_new_tokens, temperature, top_k, top_p,
         think_budget=0, close_ids=None, dtype=torch.bfloat16):
    """Return (answer, confidence, votes, n_scorable). answer is the majority-voted \\boxed value
    over the closed+boxed samples (None if none are scorable)."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    with torch.autocast("cuda", dtype=dtype):
        gens = sample_batch(model, tok, ids.repeat(k, 1), max_new_tokens=max_new_tokens,
                            eos_id=eos_id, do_sample=True, temperature=temperature,
                            top_k=top_k, top_p=top_p, think_budget=think_budget,
                            close_ids=close_ids or [])
    votes = Counter()
    for g in gens:
        text = tok.decode(g, skip_special_tokens=True)
        if "</think>" not in text:
            continue
        pred = extract_boxed(text)
        if pred is not None:
            votes[pred] += 1
    if not votes:
        return None, 0.0, votes, 0
    ans, cnt = votes.most_common(1)[0]
    n_scorable = sum(votes.values())
    return ans, cnt / n_scorable, votes, n_scorable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--k", type=int, default=16, help="samples per question to vote over")
    ap.add_argument("--think-budget", type=int, default=0, help="force-close </think> at N tokens (0=off)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    # deploy: answer these questions (one per line); grade: measure on GSM8K/MATH
    ap.add_argument("--questions-file", default=None, help="deploy mode: a .txt of questions (one/line)")
    ap.add_argument("--question", default=None, help="deploy mode: a single question")
    ap.add_argument("--grade", action="store_true", help="measure accuracy vs gold on --source")
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    fh = open(args.log, "a") if args.log else None

    def out(*p):
        line = " ".join(str(x) for x in p)
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
    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False) if args.think_budget else []

    vkw = dict(eos_id=eos_id, k=args.k, max_new_tokens=args.max_new_tokens,
               temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
               think_budget=args.think_budget, close_ids=close_ids, dtype=dtype)

    if args.grade:
        import datetime as _dt
        problems = load_problems(args.source, args.n_problems, seed=args.seed)
        out("=" * 60)
        out(f"self-consistency GRADE  {_dt.datetime.now().isoformat(timespec='seconds')}")
        out(f"model={args.model_path} K={args.k} think_budget={args.think_budget or 'off'} "
            f"src={args.source} n={len(problems)}")
        out("=" * 60)
        n_vote_correct = n_pass_k = n_first = n_scorable_q = 0
        for i, (q, gold, tier) in enumerate(problems):
            ans, conf, votes, ns = vote(model, tok, q, **vkw)
            if ans is not None:
                n_scorable_q += 1
                if ans == gold:
                    n_vote_correct += 1
            if gold in votes:          # pass@k proxy: gold appeared among closed+boxed samples
                n_pass_k += 1
            if (i + 1) % 25 == 0:
                out(f"  [{i+1}/{len(problems)}] vote-acc={100*n_vote_correct/(i+1):.1f}% "
                    f"pass@{args.k}(scorable)={100*n_pass_k/(i+1):.1f}%")
        n = len(problems)
        out(f"\n  MAJORITY-VOTE accuracy : {100*n_vote_correct/n:.2f}%  ({n_vote_correct}/{n})")
        out(f"  gold-in-samples (pass@{args.k}): {100*n_pass_k/n:.2f}%  ({n_pass_k}/{n})")
        out(f"  questions with any scorable vote: {n_scorable_q}/{n}")
    else:
        qs = []
        if args.question:
            qs = [args.question]
        elif args.questions_file:
            qs = [ln.strip() for ln in open(args.questions_file) if ln.strip()]
        else:
            out("deploy mode needs --question or --questions-file"); sys.exit(2)
        for q in qs:
            ans, conf, votes, ns = vote(model, tok, q, **vkw)
            out(f"Q: {q}")
            out(f"A: {ans}   (confidence {conf:.0%} over {ns} votes; top-3 {votes.most_common(3)})")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
