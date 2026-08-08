"""vLLM-backed best-of-N reranking (§22 step-change) — the fast, HBM-filling payoff harness.

Uses the VALIDATED vLLM port (reasoning/vllm_argonne.py) for BOTH phases via continuous
batching (fills the card + 10-50x faster than the old per-prompt loop):
  --mode generate : policy samples K candidates for every problem in ONE llm.generate (n=K),
                    save {question, gold, candidates}. Prints self-consistency + pass@K for free.
  --mode rerank   : verifier scores every closed+boxed candidate (1-token gen, top-logprobs of
                    'Yes'/'No'), batched across ALL candidates. Reports best-of-N vs majority
                    vs single-sample vs pass@K on the same samples.

Run in 2 processes (generate, then rerank) so the two engines don't co-reside.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star_generate import extract_boxed, norm, load_problems  # noqa: E402
from build_verifier_data import PROMPT_TMPL  # noqa: E402


def build_ids(tok, text, think=True):
    enc = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=think)
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    if len(enc) > 0 and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(x) for x in enc]


def run_generate(args):
    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.policy, trust_remote_code=True)
    probs = load_problems(args.source, args.n_problems, seed=args.seed)
    llm = LLM(model=args.policy, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_new_tokens)
    prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, q, think=True)) for q, _, _ in probs]
    import datetime as _dt
    t0 = _dt.datetime.now()
    outs = llm.generate(prompts, sp)
    dt = (_dt.datetime.now() - t0).total_seconds()
    rows = []
    for (q, gold, tier), o in zip(probs, outs):
        rows.append({"question": q, "gold": gold,
                     "candidates": [co.text for co in o.outputs]})
    json.dump(rows, open(args.gen_out, "w"))
    # quick metrics from candidates (self-consistency is free here)
    n = len(rows)
    n_pass = n_maj = 0
    tot = corr = 0
    for r in rows:
        closed = [extract_boxed(c) for c in r["candidates"] if "</think>" in c]
        closed = [(p) for p in closed if p is not None]
        tot += len(r["candidates"])
        corr += sum(1 for c in r["candidates"] if "</think>" in c and extract_boxed(c) == r["gold"])
        if any(p == r["gold"] for p in closed):
            n_pass += 1
        v = Counter(closed)
        if v and v.most_common(1)[0][0] == r["gold"]:
            n_maj += 1
    print("=" * 60)
    print(f"[generate] {n} problems x K={args.k} in {dt:.1f}s "
          f"({args.k*n/dt:.0f} samples/s) -> {args.gen_out}")
    print(f"  single-sample acc : {100*corr/max(tot,1):.2f}%")
    print(f"  self-consistency  : {100*n_maj/n:.2f}%  (majority vote)")
    print(f"  pass@{args.k:<11}: {100*n_pass/n:.2f}%  (ceiling)")
    print("=" * 60)


def run_rerank(args):
    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.verifier, trust_remote_code=True)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    rows = json.load(open(args.gen_out))

    # build verifier prompts for every closed+boxed candidate
    vprompts, meta = [], []
    for ri, r in enumerate(rows):
        for ci, cand in enumerate(r["candidates"]):
            if "</think>" not in cand:
                continue
            pred = extract_boxed(cand)
            if pred is None:
                continue
            vp = build_ids(tok, PROMPT_TMPL.format(q=r["question"], sol=cand), think=False)
            vprompts.append(TokensPrompt(prompt_token_ids=vp))
            meta.append((ri, ci, pred))

    llm = LLM(model=args.verifier, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    import datetime as _dt
    t0 = _dt.datetime.now()
    vouts = llm.generate(vprompts, sp)
    dt = (_dt.datetime.now() - t0).total_seconds()

    scores = {}  # (ri, ci) -> (p_yes, pred)
    for (ri, ci, pred), o in zip(meta, vouts):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        ly = lp[yes_id].logprob if yes_id in lp else None
        ln = lp[no_id].logprob if no_id in lp else None
        if ly is None and ln is None:
            p_yes = 0.0
        elif ly is None:
            p_yes = 0.0
        elif ln is None:
            p_yes = 1.0
        else:
            p_yes = math.exp(ly) / (math.exp(ly) + math.exp(ln))
        scores[(ri, ci)] = (p_yes, pred)

    n = len(rows)
    n_bon = n_maj = n_pass = 0
    tot = corr = 0
    for ri, r in enumerate(rows):
        gold = r["gold"]
        closed = [(ci, extract_boxed(c)) for ci, c in enumerate(r["candidates"])
                  if "</think>" in c and extract_boxed(c) is not None]
        tot += len(r["candidates"])
        corr += sum(1 for c in r["candidates"] if "</think>" in c and extract_boxed(c) == gold)
        if any(p == gold for _, p in closed):
            n_pass += 1
        votes = Counter(p for _, p in closed)
        if votes and votes.most_common(1)[0][0] == gold:
            n_maj += 1
        best, bs = None, -1.0
        for ci, p in closed:
            s = scores.get((ri, ci), (0.0, p))[0]
            if s > bs:
                bs, best = s, p
        if best == gold:
            n_bon += 1

    print("=" * 64)
    print(f"vLLM best-of-N rerank | policy={Path(args.gen_out).name} "
          f"verifier={Path(args.verifier).name}")
    print(f"  scored {len(vprompts)} candidates in {dt:.1f}s ({len(vprompts)/max(dt,1):.0f}/s)")
    print(f"  single-sample acc      : {100*corr/max(tot,1):.2f}%")
    print(f"  majority vote (self-c) : {100*n_maj/n:.2f}%  ({n_maj}/{n})")
    print(f"  BEST-OF-N (verifier)   : {100*n_bon/n:.2f}%  ({n_bon}/{n})")
    print(f"  pass@K (ceiling)       : {100*n_pass/n:.2f}%  ({n_pass}/{n})")
    print(f"  verifier lift vs vote  : {100*(n_bon-n_maj)/n:+.2f} pts")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["generate", "rerank"])
    ap.add_argument("--policy", default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--verifier", default="/project/rcc/youzhi/models/instruct/verifier_soup_r1")
    ap.add_argument("--gen-out", default="report/vllm_bon_candidates.json")
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "generate":
        run_generate(args)
    else:
        run_rerank(args)


if __name__ == "__main__":
    main()
