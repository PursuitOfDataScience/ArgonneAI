#!/usr/bin/env python3
"""Build cot_sft_mix_v5 = cot_sft_mix_v2 anchor + AMPLIFIED synthetic multi-step tier.

Overnight pure-SFT experiment (2026-07-06). GRPO plateaued (reward-proxy: format up,
accuracy flat) and is generation-bound (HBM idle most of each step). A pure CoT-SFT run
keeps the GPU HBM-full continuously AND is the highest-leverage lever we've found (data).

This reuses build_mix_v3's correct-by-construction multi-step generators (algebra / series /
geometry / divisors — the four §10 residual families) but amplifies PER_FAMILY 5000 -> 12000
(48k multi-step vs 20k), so the model drills the exact multi-step chains it fails on, with
every trace re-verified by the RLVR \\boxed extractor. Anchor = all of mix v2 (zero-sum-diet
guard: keep general/no-think intact). No self-generation -> no HBM-light generation phase.
Output: cot_sft_mix_v5.
"""

import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mix_v3 as v3  # reuse FAMILIES, build_family, generators, extract/verify

PER_FAMILY = 12000
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v5"


def main():
    tok = AutoTokenizer.from_pretrained(v3.TOK, trust_remote_code=True)
    print(f"Loading anchor mix v2 <- {v3.V2}")
    v2 = load_from_disk(v3.V2)
    v2_rows = [{"messages": r["messages"], "tier": r["tier"],
                "num_tokens": r["num_tokens"]} for r in v2]
    print(f"  v2: {len(v2_rows)} rows {dict(Counter(r['tier'] for r in v2_rows))}")

    print(f"Generating AMPLIFIED multi-step tier (PER_FAMILY={PER_FAMILY}):")
    ms_rows = []
    for name, gen in v3.FAMILIES.items():
        ms_rows += v3.build_family(name, gen, PER_FAMILY, tok)

    all_rows = v2_rows + ms_rows
    v3.rng.shuffle(all_rows)
    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(OUT)
    print(f"\nTOTAL: {len(ds)} -> {OUT}")
    print("composition:", dict(Counter(ds["tier"])))


if __name__ == "__main__":
    main()
