#!/usr/bin/env python3
"""Build cot_sft_mix_v6 = a SHORT-ONLY CoT mix that attacks the #1 deployable failure
(non-termination: ~57% of traces never close </think>) + adds correct grade-school PROCEDURE.

Design (2026-07-10, from the clean-eval findings + the lever-search workflow):
- The model already HAS the capability (clean pass@32 ~73%, self-cons ~40-50%) but OVER-THINKS
  and won't terminate. Budget-forcing patches this at decode (~2x greedy); v6 teaches NATIVE
  termination by training ONLY on SHORT (<=768 tok) closed-correct traces. v3's easy_gsm8k had
  NO length filter -> it reinforced over-thinking; every long tier is dropped here.
- Adds a contamination-SAFE grade-school PROCEDURE tier: gsm8k_main_curated filtered to
  split=="train" ONLY (the pooled TEST rows were the contamination), verified closed+boxed,
  <=512 tok, upsampled. Answer traces are CANONICALIZED to the deployed
  `</think>\n\nThe answer is $\boxed{N}$.` close (some curated traces end "Exact Answer: N").
- Drops v3's easy_gsm8k (train+test pooled = contaminated). Eval stays SVAMP/ASDiv (disjoint),
  so training on gsm8k-TRAIN is clean methodology (contamination memo rule #2).
- Keeps v3's proven general/no-think anchor (direct_tulu) + procedure drills (synth_arith, ms_*,
  med_math) + a little ultrachat/opus/openmath/strict, ALL filtered to <=768 tok.

Output: /project/rcc/youzhi/data/cot_sft_mix_v6  (same schema as v3: messages, tier, num_tokens).
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_generate import extract_boxed, norm  # noqa: E402

V3 = "/project/rcc/youzhi/data/cot_sft_mix_v3"
GSM = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v6"
SEED = 20260710
MAX_TOK = 768                 # short-only ceiling (termination pressure)
GSM_MAX_TOK = 512             # gsm8k-train teacher tier ceiling
GSM_UPSAMPLE = 3
rng = random.Random(SEED)

# v3 tiers to keep (<=768 tok), with per-tier caps. easy_gsm8k DROPPED (contaminated).
V3_CAPS = {
    "direct_tulu": 8000,      # no-think general anchor (protect the 4-quadrant no-think axis)
    "gen_ultrachat": 3000,    # general with-think
    "synth_arith": 2500,      # single-fact arithmetic drill
    "med_math": 2000,         # MATH L1-3
    "ms_algebra": 1200, "ms_series": 1200, "ms_geometry": 1200, "ms_divisors": 1290,  # procedure
    "hq_opus": 800, "med_openmath": 300, "hard_strict": 600,
}


def canonicalize_gsm(answer):
    """Reconstruct a curated gsm8k answer to end with the deployed boxed close. Return (content, gold)
    or None if it can't be verified."""
    gold = extract_boxed(answer)
    if gold is None:
        return None
    i = answer.rfind("</think>")
    think = answer[:i + len("</think>")] if i >= 0 else "<think>\n" + answer.strip() + "\n</think>"
    content = think + f"\n\nThe answer is $\\boxed{{{gold}}}$."
    if extract_boxed(content) != gold:
        return None
    return content, gold


def main():
    print(f"Loading v3 anchor <- {V3}")
    v3 = load_from_disk(V3)
    by_tier = {}
    for r in v3:
        if r["num_tokens"] > MAX_TOK:
            continue
        if r["tier"] not in V3_CAPS:
            continue
        by_tier.setdefault(r["tier"], []).append(
            {"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]})
    rows = []
    for tier, cap in V3_CAPS.items():
        pool = by_tier.get(tier, [])
        rng.shuffle(pool)
        rows += pool[:cap]
    print("  v3-short kept:", dict(Counter(r["tier"] for r in rows)))

    print(f"Building gsm8k-TRAIN short procedure tier <- {GSM}")
    gsm = []
    for ln in open(GSM):
        o = json.loads(ln)
        if o.get("split") != "train":
            continue
        if o.get("num_tokens", 10 ** 9) > GSM_MAX_TOK:
            continue
        if "</think>" not in o["answer"]:
            continue
        res = canonicalize_gsm(o["answer"])
        if res is None:
            continue
        content, _ = res
        gsm.append({"messages": [{"role": "user", "content": o["question"]},
                                 {"role": "assistant", "content": content}],
                    "tier": "gsm8k_train_short", "num_tokens": o.get("num_tokens", 0)})
    print(f"  gsm8k_train_short unique: {len(gsm)}  (upsample x{GSM_UPSAMPLE})")
    rows += gsm * GSM_UPSAMPLE

    rng.shuffle(rows)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(OUT)
    print(f"\nTOTAL v6: {len(ds)} -> {OUT}")
    comp = Counter(ds["tier"])
    for t in sorted(comp):
        print(f"  {t:<20}{comp[t]:>7}  ({100*comp[t]/len(ds):.1f}%)")
    print(f"  max num_tokens in mix: {max(ds['num_tokens'])}")


if __name__ == "__main__":
    main()
