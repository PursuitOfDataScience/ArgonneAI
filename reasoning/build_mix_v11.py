#!/usr/bin/env python
"""build_mix_v11.py -- v6's mix with the LENGTH STARVATION of the hard tiers undone.

Why this exists (2026-08-03, §34).  Two independent length limits were both capping the
argonne-3.5-think reasoning diet, and neither was intentional at the value it took:

 1. `build_mix_v6.py` filters every v3-derived row to `num_tokens <= 768` ("short-only
    ceiling (termination pressure)").  That filter was aimed at §23e's non-termination
    problem, which §32a then measured as SOLVED (no_answer 53.7% -> 1.3%, budget-forcing
    adds exactly 0.00).  Measured cost of leaving it in: the filter admits only
    651 of 12,000 `hard_strict` rows and 303 of 4,620 `med_openmath` rows -- a 95% cut
    of precisely the two tiers that carry long multi-step derivation.
 2. `cot-sft.py --max_think_tokens` defaults to 128 and `a35_cot.sh` / `run_arm.sh` never
    pass it, so 31.3% of v6's reasoning rows (80-89% of the four hard math tiers) had
    their <think> span cut mid-derivation before training.  Fixed separately by passing
    `--max_think_tokens 0`; that fix is what makes THIS rebuild worth doing, since
    untruncated 768-token traces are still short traces.

So v11 raises the row-length ceiling to 1792 (and gsm8k-train to 1024) and lifts the
per-tier caps on the three tiers the old ceiling was starving, while holding the general
anchors, the arithmetic drill and the procedure drills at their EXACT v6 sizes -- those
protect the 4-quadrant no-think axis (§18/§32) and the one-step arithmetic gate (§33s),
and moving them would confound the comparison.

Deliberately confounded, and stated so in the write-up: the three hard tiers get both
LONGER and MORE rows, because the same 768 filter caused both.  Separating the two would
cost a seed, and §33p says a single seed cannot resolve anything on this recipe anyway.

Usage:  python build_mix_v11.py [--out DIR] [--max-tok 1792] [--gsm-max-tok 1024]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_generate import extract_boxed  # noqa: E402

V3 = "/project/rcc/youzhi/data/cot_sft_mix_v3"
GSM = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
SEED = 20260803

# Held at v6 values on purpose (general axis + drills), vs RAISED for the starved hard tiers.
V6_CAPS = {
    "direct_tulu": 8000, "gen_ultrachat": 3000, "synth_arith": 2500, "med_math": 2000,
    "ms_algebra": 1200, "ms_series": 1200, "ms_geometry": 1200, "ms_divisors": 1290,
    "hq_opus": 800, "med_openmath": 300, "hard_strict": 600,
}
V11_CAPS = dict(V6_CAPS, hard_strict=2400, med_openmath=1400, hq_opus=2000)
GSM_TIER_CAP = 4400          # ~= v6's realized 4,338, so tier SIZE is not the variable here
GSM_UPSAMPLE = 2


def canonicalize_gsm(answer: str):
    """Reconstruct a curated gsm8k answer to end with the deployed boxed close."""
    gold = extract_boxed(answer)
    if gold is None:
        return None
    i = answer.rfind("</think>")
    think = answer[:i + len("</think>")] if i >= 0 else "<think>\n" + answer.strip() + "\n</think>"
    content = think + f"\n\nThe answer is $\\boxed{{{gold}}}$."
    return content if extract_boxed(content) == gold else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/rcc/youzhi/data/cot_sft_mix_v11")
    ap.add_argument("--max-tok", type=int, default=1792)
    ap.add_argument("--gsm-max-tok", type=int, default=1024)
    ap.add_argument("--caps", default="v11", choices=["v6", "v11"],
                    help="v6 reproduces the old per-tier caps at the new length ceiling")
    args = ap.parse_args()
    caps = V11_CAPS if args.caps == "v11" else V6_CAPS
    rng = random.Random(SEED)

    print(f"Loading v3 anchor <- {V3}  (max_tok={args.max_tok})")
    by_tier: dict[str, list] = {}
    for r in load_from_disk(V3):
        if r["num_tokens"] > args.max_tok or r["tier"] not in caps:
            continue
        by_tier.setdefault(r["tier"], []).append(
            {"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]})

    rows = []
    for tier, cap in caps.items():
        pool = by_tier.get(tier, [])
        rng.shuffle(pool)
        kept = pool[:cap]
        rows += kept
        flag = "  (POOL EXHAUSTED)" if len(pool) <= cap else ""
        print(f"  {tier:<16} kept {len(kept):>5} of {len(pool):>6} available{flag}")

    print(f"Building gsm8k-TRAIN tier <- {GSM}  (max_tok={args.gsm_max_tok})")
    # see build_mix_v6: canonicalize_gsm's `extract_boxed(content) == gold` compares the generated
    # text to itself, so it cannot catch a solution whose final answer is wrong -- 4.66% are.
    from effort_probe import gsm8k_gold_map
    auth = gsm8k_gold_map()
    gsm, n_badgold = [], 0
    for ln in open(GSM):
        o = json.loads(ln)
        if o.get("split") != "train" or o.get("num_tokens", 10 ** 9) > args.gsm_max_tok:
            continue
        if "</think>" not in o["answer"]:
            continue
        content = canonicalize_gsm(o["answer"])
        if content is None:
            continue
        a = auth.get(o["question"].strip())
        if a is not None and extract_boxed(content) != a:
            n_badgold += 1
            continue
        gsm.append({"messages": [{"role": "user", "content": o["question"]},
                                 {"role": "assistant", "content": content}],
                    "tier": "gsm8k_train_short", "num_tokens": o.get("num_tokens", 0)})
    print(f"  gsm8k unique canonicalized: {len(gsm)}  (upsample x{GSM_UPSAMPLE}, cap {GSM_TIER_CAP})"
          f"  [dropped {n_badgold} whose generated answer disagrees with GSM8K gold]")
    gsm = (gsm * GSM_UPSAMPLE)
    rng.shuffle(gsm)
    rows += gsm[:GSM_TIER_CAP]

    rng.shuffle(rows)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.out)
    comp = Counter(ds["tier"])
    total_tok = sum(ds["num_tokens"])
    print(f"\nTOTAL v11: {len(ds)} rows / {total_tok:,} tokens -> {args.out}")
    for t in sorted(comp):
        print(f"  {t:<20}{comp[t]:>7}  ({100 * comp[t] / len(ds):.1f}%)")
    print(f"  max num_tokens in mix: {max(ds['num_tokens'])}")


if __name__ == "__main__":
    main()
