#!/usr/bin/env python3
"""Merge on-policy RFT/STaR traces (from `rft_generate.py`) into a CoT-SFT mix.

DESIGN CONSTRAINTS carried over from §32, deliberately kept as the ONLY moving part:
  * base mix stays `cot_sft_mix_v6` VERBATIM and training stays 1 epoch from the `dpo` checkpoint
    -- §32b measured 2 epochs of v6 as a regression (greedy -3.0/-5.0) and a35_bigsft measured
    "more CoT data" as widening the gap, so the amount of off-policy data is NOT the variable here.
  * the added tier is capped as a SHARE of the final mix (`--rft-share`), because the failure mode
    of every previous data change on this line was diet imbalance (§29 diversity collapse, §18
    zero-sum trade), not too little data.
  * per-problem keep counts already encode difficulty weighting upstream (rft_generate).

So the single variable vs the shipped recipe is: *does adding the model's OWN verified traces on
hard-but-solvable train problems convert pass@K into pass@1?*
"""
import argparse
import json
import random
from collections import Counter

from datasets import Dataset, load_from_disk

DATA = "/project/rcc/youzhi/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"{DATA}/cot_sft_mix_v6")
    ap.add_argument("--rft-jsonl", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rft-share", type=float, default=0.20,
                    help="target share of the FINAL mix taken by RFT rows")
    ap.add_argument("--max-per-problem", type=int, default=0,
                    help="extra global cap on kept traces per question (0 = keep upstream budget)")
    ap.add_argument("--hard-first", action="store_true", default=True,
                    help="when subsampling to hit --rft-share, drop the EASIEST (highest c/k) first")
    ap.add_argument("--drop-tiers", nargs="*", default=None,
                    help="base tiers to REMOVE (diet mode) before adding the new tier")
    ap.add_argument("--balance-tiers", action="store_true",
                    help="when subsampling to --rft-share, take an equal quota from each tier")
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    base = load_from_disk(args.base)
    base = base["train"] if hasattr(base, "keys") and "train" in base else base
    base_rows = [dict(r) for r in base]
    print(f"base {args.base}: {len(base_rows)} rows")
    if args.drop_tiers:
        # DIET mode, as opposed to additive mode. §32's large win (greedy 25.7->62.3) came from
        # REPLACING the diet (v3 -> v6), not from appending a tier; an additive 20% tier is a small
        # perturbation of the same diet, which is what the null Arm A measured.
        drop = set(args.drop_tiers)
        n0 = len(base_rows)
        base_rows = [r for r in base_rows if r["tier"] not in drop]
        print(f"  dropped tiers {sorted(drop)}: {n0} -> {len(base_rows)} rows")

    rft = []
    seen = set()
    for p in args.rft_jsonl:
        n0 = len(rft)
        for ln in open(p):
            o = json.loads(ln)
            key = (o["messages"][0]["content"], o["messages"][1]["content"])
            if key in seen:
                continue
            seen.add(key)
            rft.append(o)
        print(f"  + {p}: {len(rft)-n0} new rows")
    print(f"rft pool: {len(rft)} rows  tiers={Counter(r['tier'] for r in rft).most_common()}")

    if args.max_per_problem:
        by_q = Counter()
        keep = []
        rft.sort(key=lambda r: (r.get("c", 0), r["num_tokens"]))
        for r in rft:
            q = r["messages"][0]["content"]
            if by_q[q] >= args.max_per_problem:
                continue
            by_q[q] += 1
            keep.append(r)
        print(f"  per-problem cap {args.max_per_problem}: {len(rft)} -> {len(keep)}")
        rft = keep

    # solve  n_rft / (n_base + n_rft) = share
    if args.rft_share > 0:
        target = int(args.rft_share * len(base_rows) / (1 - args.rft_share))
    else:
        target = 0
    if len(rft) > target and args.balance_tiers:
        # Equal quota per tier. Needed for the verify tier, whose three flavours carry different
        # lengths: a plain shortest-first cut would silently keep almost only `verify_confirm`
        # and the arm would no longer be testing what it claims to test.
        groups = {}
        for r in rft:
            groups.setdefault(r["tier"], []).append(r)
        for g in groups.values():
            g.sort(key=lambda r: (r.get("c", 0) / max(r.get("k", 1), 1), r["num_tokens"]))
        keep, i = [], 0
        while len(keep) < target and any(len(g) > i for g in groups.values()):
            for t in sorted(groups):
                if len(keep) < target and len(groups[t]) > i:
                    keep.append(groups[t][i])
            i += 1
        print(f"  balanced cut to {len(keep)}: {Counter(r['tier'] for r in keep).most_common()}")
        rft = keep
    elif len(rft) > target:
        if args.hard_first:
            # keep the informative tail: lowest c/k first (hard-but-solvable), then shortest
            rft.sort(key=lambda r: (r.get("c", 0) / max(r.get("k", 1), 1), r["num_tokens"]))
        else:
            rng.shuffle(rft)
        dropped = rft[target:]
        rft = rft[:target]
        print(f"  share {args.rft_share:.2f} -> keep {len(rft)}, drop {len(dropped)} "
              f"(dropped mean c/k = "
              f"{sum(d.get('c',0)/max(d.get('k',1),1) for d in dropped)/max(len(dropped),1):.3f})")
    print(f"  kept mean c/k = {sum(r.get('c',0)/max(r.get('k',1),1) for r in rft)/max(len(rft),1):.3f}")

    rows = base_rows + [{"messages": r["messages"], "tier": r["tier"],
                         "num_tokens": r["num_tokens"]} for r in rft]
    rng.shuffle(rows)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.out)

    tiers = Counter(r["tier"] for r in rows)
    tot = len(rows)
    print(f"\n=== {args.out}: {tot} rows ===")
    for t, c in tiers.most_common():
        print(f"  {100*c/tot:5.1f}%  {t:<24} {c}")
    toks = [r["num_tokens"] for r in rows]
    print(f"  tokens: mean {sum(toks)/len(toks):.0f}  p50 {sorted(toks)[len(toks)//2]}  "
          f"p95 {sorted(toks)[int(.95*len(toks))]}  max {max(toks)}")


if __name__ == "__main__":
    main()
