#!/usr/bin/env python3
"""Build the cumulative STaR SFT dataset (round 2).

Recipe (matches round 1's star_sft_v1, scaled up):
  - STaR traces: star_correct_v1 (round 1) + star_correct_v2 (round 2),
    deduped, upsampled UPSAMPLE x  -> tier "star".
  - Anchor: a stratified-proportional sample of cot_sft_mix_v2 (keeps the
    model's general/format competence; round 1 used 5000 and held the
    no-think quadrants flat).
Shuffle and save. Only `messages` is consumed by cot-sft.py; `tier` is for
diagnostics.
"""
import argparse, random
from collections import Counter
from datasets import load_from_disk, Dataset, concatenate_datasets

STAR_V1 = "/project/rcc/youzhi/data/star_correct_v1"
STAR_V2 = "/project/rcc/youzhi/data/star_correct_v2"
MIX = "/project/rcc/youzhi/data/cot_sft_mix_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/rcc/youzhi/data/star_sft_v2")
    ap.add_argument("--star-dirs", nargs="+", default=[STAR_V1, STAR_V2],
                    help="one or more STaR correct-trace dirs to pool (deduped).")
    ap.add_argument("--mix", default=MIX, help="anchor mix dir (keeps general/format competence).")
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--anchor", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # --- STaR traces (cumulative, deduped on the assistant text) ---
    star = concatenate_datasets([load_from_disk(d) for d in args.star_dirs])
    seen, uniq = set(), []
    for r in star:
        key = r["messages"][-1]["content"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append({"messages": r["messages"], "tier": "star"})
    print(f"STaR unique traces: {len(uniq)} (from {len(star)} rows)")

    star_up = []
    for _ in range(args.upsample):
        star_up += uniq
    print(f"STaR upsampled x{args.upsample}: {len(star_up)}")

    # --- Anchor: stratified-proportional sample of the anchor mix ---
    mix = load_from_disk(args.mix)
    by_tier = {}
    for i, t in enumerate(mix["tier"]):
        by_tier.setdefault(t, []).append(i)
    total = len(mix)
    anchor_idx = []
    for t, idxs in by_tier.items():
        k = round(args.anchor * len(idxs) / total)
        anchor_idx += rng.sample(idxs, min(k, len(idxs)))
    rng.shuffle(anchor_idx)
    anchor_idx = anchor_idx[:args.anchor]
    anchor = [{"messages": mix[i]["messages"], "tier": mix[i]["tier"]} for i in anchor_idx]
    print(f"Anchor sample: {len(anchor)} {dict(Counter(a['tier'] for a in anchor))}")

    rows = star_up + anchor
    rng.shuffle(rows)
    print(f"TOTAL: {len(rows)}  STaR%={100*len(star_up)/len(rows):.1f}")
    Dataset.from_list(rows).save_to_disk(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
