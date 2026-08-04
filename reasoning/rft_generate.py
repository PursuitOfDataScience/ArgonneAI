#!/usr/bin/env python3
"""Rejection-sampling / STaR fuel generator for the argonne-3.5-think reasoning line.

WHY (2026-08-02, §33). The shipped `blend_a085` sits at clean greedy 65.0/73.0 with pass@8
90.7/92.3 -- a ~20pt floor-to-ceiling gap. On the 3.0 base every lever aimed at that gap died of
SIGNAL STARVATION (§9/§20: a binary reward at ~2% accuracy leaves most groups with no gradient).
At 40-65% single-sample accuracy the informative regime is finally available, so the cheapest
correct move is rejection-sampling fine-tuning (RFT/STaR): sample K, keep the GOLD-VERIFIED
correct traces, train on them.

What this adds over `vllm_rollouts.py` (gsm8k-only):
  * any pool `effort_probe.load_pool` knows (gsm8k_train, math_train_easy/hard) -- TRAIN splits only
  * **difficulty-aware keep counts.** A problem solved 15/16 times teaches almost nothing; one
    solved 2/16 is exactly the pass@K capability that is not yet the floor. Keep-per-problem is
    therefore a function of c/K (--keep-easy/-mid/-hard), which is the whole point of the corpus.
  * **step-signature dedupe** so K near-identical traces for one problem don't crowd the mix
    (the §29 diversity-collapse lesson: homogenised SFT fuel sharpens sampling and costs
    self-consistency).
  * **arithmetic step-verification of positives** (`has_bad_arith`): ~35% of gold-reaching traces
    pass THROUGH a wrong `a op b = c`; training on those teaches lucky-wrong procedure.
  * canonicalises the close to the deployed `</think>\\n\\nThe answer is $\\boxed{N}$.`

Writes JSONL (shardable across GPUs), not an HF dataset, so N GPUs can generate in parallel and
`build_mix_rft.py` merges. Also writes a stats JSON with the correct-count histogram per pool.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EQ = re.compile(r"(-?\d+(?:\.\d+)?)\s*([-+*/×xX])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
CLOSE_FMT = "\n</think>\n\nThe answer is $\\boxed{%s}$."


def _val(a, op, b):
    op = op.lower()
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op in ("*", "×", "x"):
        return a * b
    if op == "/":
        return a / b if b else None
    return None


def has_bad_arith(text):
    """True if any explicit `a op b = c` in the trace is arithmetically wrong."""
    for m in EQ.finditer(text):
        try:
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        v = _val(a, op, b)
        if v is not None and abs(v - c) > 1e-6:
            return True
    return False


def step_signature(text):
    """Coarse fingerprint of the REASONING PATH: the ordered list of its equations.

    Two traces with the same equation sequence are the same derivation in different words --
    keeping both adds tokens and no diversity. Falls back to a normalised-word signature when a
    trace shows no explicit arithmetic (common on MATH).
    """
    eqs = ["%s%s%s=%s" % (m.group(1), m.group(2).lower(), m.group(3), m.group(4))
           for m in EQ.finditer(text)]
    if eqs:
        return "|".join(eqs)
    words = re.findall(r"[a-z]+", text.lower())
    return " ".join(words[:40])


def canonicalize(trace, gold):
    """Force the deployed close. Returns (content, think_text) or None."""
    t = trace.strip()
    if not t.startswith("<think>"):
        t = "<think>\n" + t.lstrip("\n")
    i = t.find("</think>")
    if i < 0:
        return None
    think = t[:i].rstrip("\n")
    return think + CLOSE_FMT % gold, think[len("<think>"):]


def is_degenerate(think_text, gold, min_tok, tok, require_gold, min_eqs):
    """Reject traces that reach gold WITHOUT deriving it.

    This filter is not optional. The first smoke run of this script kept, as a top pick,

        <think>\\n</think>\\n\\nThe answer is $\\boxed{50}$.

    -- an EMPTY think block on a problem the model solved 2/8 times. Shortest-first selection
    actively seeks these out: an empty-think lucky guess is always the shortest "correct"
    candidate. Training on them teaches the model to skip reasoning and guess, which is the exact
    opposite of the goal, and it would have looked like a data win (loss drops fast on 14-token
    rows). Requiring the gold value to APPEAR IN THE THINK BLOCK is the cheap structural test for
    "the reasoning produced this answer" rather than "the suffix asserted it".
    """
    n = len(tok.encode(think_text, add_special_tokens=False))
    if n < min_tok:
        return "too_short"
    if require_gold:
        g = str(gold).strip()
        gg = g[:-2] if g.endswith(".0") else g          # norm() renders 50 as "50.0" sometimes
        if g not in think_text and gg not in think_text:
            return "gold_not_derived"
    if min_eqs and len(EQ.findall(think_text)) < min_eqs:
        return "too_few_equations"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--n-per-pool", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=1234,
                    help="NOT 0 -- keeps the fuel disjoint from the seed-0 eval shuffles")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-tok", type=int, default=768,
                    help="length ceiling, in tokens, on a KEPT trace (v6's termination pressure)")
    ap.add_argument("--keep-easy", type=int, default=1, help="keeps when c/K > 0.75")
    ap.add_argument("--keep-mid", type=int, default=2, help="keeps when 0.25 < c/K <= 0.75")
    ap.add_argument("--keep-hard", type=int, default=3, help="keeps when c/K <= 0.25")
    ap.add_argument("--no-step-verify", action="store_true")
    ap.add_argument("--min-think-tok", type=int, default=48,
                    help="reject a trace whose <think> block is shorter than this (anti-degenerate)")
    ap.add_argument("--no-require-gold", action="store_true",
                    help="disable the 'gold value must appear inside <think>' derivation test")
    ap.add_argument("--min-eqs", type=int, default=0,
                    help="require at least this many explicit `a op b = c` steps in <think>")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--all-jsonl-out", default=None,
                    help="every labeled rollout (fuel for RLVR-DPO / a verifier)")
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    from effort_probe import load_pool, make_llm, prompt_ids
    from star_generate import extract_boxed, norm  # noqa: F401  (norm used via load_pool)

    # ---------------------------------------------------------------- problems
    probs, origin = [], []
    npp = args.n_per_pool if len(args.n_per_pool) == len(args.pools) else args.n_per_pool * len(args.pools)
    for pool, n in zip(args.pools, npp):
        p = load_pool(pool, n, seed=args.seed)
        probs += p
        origin += [pool] * len(p)
    keep_idx = [i for i in range(len(probs)) if i % args.n_shards == args.shard]
    probs = [probs[i] for i in keep_idx]
    origin = [origin[i] for i in keep_idx]

    print("=" * 78)
    print(f"RFT GENERATE  model={args.model}")
    print(f"  pools={args.pools}  n={len(probs)} (shard {args.shard}/{args.n_shards})  K={args.k}"
          f"  T={args.temperature}")
    print("=" * 78, flush=True)

    llm, tok = make_llm(args.model, args.max_model_len, args.gpu_util, seed=args.seed)
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_new_tokens, seed=args.seed)
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(tok, q)) for q, _ in probs], sp)

    # ------------------------------------------------------------------ select
    fh = open(args.jsonl_out, "w")
    fa = open(args.all_jsonl_out, "w") if args.all_jsonl_out else None
    fm = Counter()
    hist = defaultdict(Counter)          # pool -> Counter(correct_count)
    kept_by_pool = Counter()
    drop = Counter()
    n_kept = 0
    for (q, gold), pool, o in zip(probs, origin, outs):
        cands = []
        for cand in o.outputs:
            t = cand.text.strip()
            pred = extract_boxed(t)
            closed = "</think>" in t
            if not closed:
                lab = "unclosed"
            elif pred is None:
                lab = "no_answer"
            elif pred == gold:
                lab = "correct"
            else:
                lab = "wrong"
            fm[lab] += 1
            if fa:
                fa.write(json.dumps({"question": q, "pool": pool, "trace": t, "label": lab,
                                     "pred": pred or "", "gold": gold}) + "\n")
            if lab == "correct":
                cands.append(t)
        c = len(cands)
        hist[pool][c] += 1
        if c == 0:
            drop["no_positive"] += 1
            continue
        frac = c / args.k
        budget = (args.keep_easy if frac > 0.75 else
                  args.keep_mid if frac > 0.25 else args.keep_hard)

        # shortest-first AMONG NON-DEGENERATE traces: v6's lesson is that short closed traces are
        # what raises greedy, but shortest-overall picks empty-think guesses (see is_degenerate).
        cands.sort(key=len)
        seen_sig = set()
        taken = 0
        for t in cands:
            if taken >= budget:
                break
            if not args.no_step_verify and has_bad_arith(t):
                drop["bad_arith"] += 1
                continue
            sig = step_signature(t)
            if sig in seen_sig:
                drop["dup_steps"] += 1
                continue
            can = canonicalize(t, gold)
            if can is None:
                drop["uncanonicalizable"] += 1
                continue
            content, think_text = can
            bad = is_degenerate(think_text, gold, args.min_think_tok, tok,
                                not args.no_require_gold, args.min_eqs)
            if bad:
                drop[bad] += 1
                continue
            ntok = len(tok.encode(content, add_special_tokens=False))
            if ntok > args.max_tok:
                drop["too_long"] += 1
                continue
            seen_sig.add(sig)
            taken += 1
            n_kept += 1
            kept_by_pool[pool] += 1
            fh.write(json.dumps({
                "messages": [{"role": "user", "content": q},
                             {"role": "assistant", "content": content}],
                "tier": f"rft_{pool}", "num_tokens": ntok,
                "c": c, "k": args.k, "gold": gold, "pool": pool}) + "\n")
    fh.close()
    if fa:
        fa.close()

    # ------------------------------------------------------------------- stats
    print(f"\n  rollout format mix: {dict(fm)}")
    print(f"  kept traces: {n_kept}   by pool: {dict(kept_by_pool)}")
    print(f"  drops: {dict(drop)}")
    for pool, h in hist.items():
        n = sum(h.values())
        K = args.k
        sig = sum(v for c, v in h.items() if 0 < c < K)
        dead = h[0]
        sat = h[K]
        print(f"  [{pool}] n={n}  single-acc {100*sum(c*v for c,v in h.items())/(n*K):.2f}%"
              f"  signal {100*sig/n:.1f}%  dead {100*dead/n:.1f}%  saturated {100*sat/n:.1f}%")
    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        json.dump({"fm": dict(fm), "kept": n_kept, "kept_by_pool": dict(kept_by_pool),
                   "drops": dict(drop),
                   "hist": {p: {str(k): v for k, v in sorted(h.items())} for p, h in hist.items()},
                   "args": vars(args)}, open(args.stats_out, "w"), indent=1)
    print(f"\n[jsonl] {args.jsonl_out}  ({n_kept} rows)")


if __name__ == "__main__":
    main()
