#!/usr/bin/env python3
"""OFFLINE trace selection from a labeled-rollout corpus (`rft_generate.py --all-jsonl-out`).

WHY SEPARATE FROM GENERATION. Generation is the GPU-expensive half (~25 min x 3 H200 for 11.5k
problems x K=16) and selection is pure CPU. Keeping them apart means a selection-policy change --
a stricter degeneracy filter, a different difficulty weighting, a different keep budget -- costs
seconds instead of a regeneration, so the policy can actually be swept. It also means a bug in the
selection policy (there was one: see `is_degenerate`) does not cost the rollouts.

Selection policy, in order:
  1. gold-verified correct only
  2. no arithmetically-wrong `a op b = c` step  (~35% of gold-reaching traces fail this)
  3. not degenerate: >= --min-think-tok tokens of thinking AND the gold value must appear inside
     <think> (proves the reasoning derived it rather than the suffix asserting it)
  4. distinct step-signature within a problem (diversity; §29's collapse lesson)
  5. length cap --max-tok (v6's termination pressure)
  6. difficulty-aware keep budget from c/K: --keep-hard for c/K<=0.25, --keep-mid, --keep-easy
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rft_generate import canonicalize, has_bad_arith, is_degenerate, step_signature  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-jsonl", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="/project/rcc/youzhi/models/a35_reason/blend_a085")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--max-tok", type=int, default=768)
    ap.add_argument("--min-think-tok", type=int, default=48)
    ap.add_argument("--no-require-gold", action="store_true")
    ap.add_argument("--min-eqs", type=int, default=0)
    ap.add_argument("--no-step-verify", action="store_true")
    ap.add_argument("--keep-easy", type=int, default=1)
    ap.add_argument("--keep-mid", type=int, default=2)
    ap.add_argument("--keep-hard", type=int, default=3)
    ap.add_argument("--target", default="all", choices=["all", "mode_wrong"],
                    help="'mode_wrong' keeps ONLY problems whose majority answer is wrong while at "
                         "least one sample is right -- the exact pass@K-to-pass@1 conversion set")
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    by_q = defaultdict(lambda: {"pool": None, "gold": None, "k": 0, "correct": [],
                                "votes": Counter()})
    fm = Counter()
    for p in args.all_jsonl:
        for ln in open(p):
            o = json.loads(ln)
            q = o["question"]
            e = by_q[q]
            e["pool"] = o.get("pool", "?")
            e["gold"] = o["gold"]
            e["k"] += 1
            fm[o["label"]] += 1
            if o["label"] in ("correct", "wrong") and o.get("pred"):
                e["votes"][o["pred"]] += 1
            if o["label"] == "correct":
                e["correct"].append(o["trace"])
    print(f"rollouts: {sum(fm.values())}  labels={dict(fm)}  problems={len(by_q)}")

    if args.target == "mode_wrong":
        # The failure this campaign is trying to fix is not "the model cannot do it" (pass@32 is
        # 97-98% on the clean sets) but "its MODE is wrong". Greedy decoding returns the mode, so
        # the problems worth training on are exactly those where the majority vote misses while
        # some sample hits. Selecting on c/K alone only correlates with that.
        keep, drop_mode = {}, Counter()
        for q, e in by_q.items():
            if not e["correct"]:
                drop_mode["no_positive"] += 1
                continue
            top = e["votes"].most_common(1)[0][0] if e["votes"] else None
            if top is not None and top == e["gold"]:
                drop_mode["mode_already_right"] += 1
                continue
            keep[q] = e
        print(f"target=mode_wrong: {len(keep)}/{len(by_q)} problems kept  {dict(drop_mode)}")
        by_q = keep

    drop = Counter()
    kept_by_pool = Counter()
    hist = defaultdict(Counter)
    ck_kept = []
    fh = open(args.jsonl_out, "w")
    n_kept = 0
    for q, e in by_q.items():
        c, K, gold, pool = len(e["correct"]), e["k"], e["gold"], e["pool"]
        hist[pool][c] += 1
        if c == 0:
            drop["no_positive"] += 1
            continue
        frac = c / max(K, 1)
        budget = (args.keep_easy if frac > 0.75 else
                  args.keep_mid if frac > 0.25 else args.keep_hard)
        cands = sorted(e["correct"], key=len)
        seen, taken = set(), 0
        for t in cands:
            if taken >= budget:
                break
            if not args.no_step_verify and has_bad_arith(t):
                drop["bad_arith"] += 1
                continue
            sig = step_signature(t)
            if sig in seen:
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
            seen.add(sig)
            taken += 1
            n_kept += 1
            kept_by_pool[pool] += 1
            ck_kept.append(frac)
            fh.write(json.dumps({
                "messages": [{"role": "user", "content": q},
                             {"role": "assistant", "content": content}],
                "tier": f"rft_{pool}", "num_tokens": ntok,
                "c": c, "k": K, "gold": gold, "pool": pool}) + "\n")
        if taken == 0:
            drop["all_candidates_rejected"] += 1
    fh.close()

    print(f"kept {n_kept} traces  by pool {dict(kept_by_pool)}")
    print(f"drops {dict(drop)}")
    if ck_kept:
        print(f"kept mean c/K = {sum(ck_kept)/len(ck_kept):.3f}  "
              f"(share from c/K<=0.25: {100*sum(1 for x in ck_kept if x <= .25)/len(ck_kept):.1f}%)")
    for pool, h in hist.items():
        n = sum(h.values())
        K = max(max(h.keys()), 1)
        sig = sum(v for c, v in h.items() if 0 < c < K)
        print(f"  [{pool}] problems={n}  with-positive {100*(n-h[0])/n:.1f}%  signal {100*sig/n:.1f}%")
    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        json.dump({"fm": dict(fm), "kept": n_kept, "kept_by_pool": dict(kept_by_pool),
                   "drops": dict(drop), "args": vars(args),
                   "hist": {p: {str(k): v for k, v in sorted(h.items())} for p, h in hist.items()}},
                  open(args.stats_out, "w"), indent=1)
    print(f"[jsonl] {args.jsonl_out}")


if __name__ == "__main__":
    main()
