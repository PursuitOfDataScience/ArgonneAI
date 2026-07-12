#!/usr/bin/env python3
"""Build cot_sft_mix_v4 = cot_sft_mix_v3 + upsampled CONCISE no-think tier.

Why (2026-07-04, §18): the soup-base downstream run (`think_soup`) aced math
(10/10 both modes, all four §10 residuals) but REGRESSED on general chat --
5/10 no-think, 4/10 with-CoT -- with non-terminating LOOPS on simple general
prompts (grammar -> "conjunction" x11, oxygen trace never closes). The
diagnostic proved the pre-CoT model `dpo_soup` was general-HEALTHY (~7-8/10,
concise, no loops), so the CoT stage broke it. Root cause: mix v3 is 70%
long-`<think>` rows and only 30% direct/no-think (the single `direct_tulu`
tier). On the (more plastic) soup base that long-trace dominance over-generalized
into "always reason at length", i.e. looping on general questions. The SAME v3
kept general on the OLD base (mix3 8/10) -- so 30% direct was borderline-enough
there but insufficient for the soup base.

Fix = shift the balance toward CONCISE no-think WITHOUT touching the math tiers
(protect the hard-won 10/10 math): keep ALL of v3 and add extra copies of the
`direct_tulu` rows so direct/no-think goes ~30% -> ~56% (think 70% -> ~44%).
Additive-only: every math tier (incl. the ms_* residual families that install
the §10 procedures) is preserved intact. Re-run ONLY the CoT stage from
`dpo_soup` on this mix (LR 1e-5, theta=1e6) -> think_soup_v4.

Output: cot_sft_mix_v4, save_to_disk.
"""

import random
from collections import Counter
from datasets import Dataset, load_from_disk

V3 = "/project/rcc/youzhi/data/cot_sft_mix_v3"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v4"
UPSAMPLE_TIER = "direct_tulu"   # the concise no-think general tier
EXTRA_COPIES = 2                # +2 copies => 3x total exposure => direct ~30%->~56%
SEED = 20260704
rng = random.Random(SEED)

COLS = ("messages", "tier", "num_tokens")


def main():
    print(f"Loading v3 <- {V3}")
    v3 = load_from_disk(V3)
    rows = [{k: r[k] for k in COLS} for r in v3]
    base = Counter(r["tier"] for r in rows)
    print(f"  v3: {len(rows)} rows {dict(base)}")

    direct = [r for r in rows if r["tier"] == UPSAMPLE_TIER]
    if not direct:
        raise SystemExit(f"tier {UPSAMPLE_TIER!r} not found in v3")
    print(f"  upsampling tier {UPSAMPLE_TIER!r}: {len(direct)} rows x{EXTRA_COPIES} extra")

    all_rows = rows + [dict(r) for _ in range(EXTRA_COPIES) for r in direct]
    rng.shuffle(all_rows)

    # Report the think/direct balance (the whole point of v4).
    def is_think(r):
        return any("<think>" in (m.get("content") or "")
                   for m in r["messages"] if m["role"] == "assistant")
    nt = len(all_rows)
    think = sum(1 for r in all_rows if is_think(r))
    print(f"\nTOTAL: {nt} rows | <think> {think} ({100*think/nt:.0f}%) "
          f"direct {nt-think} ({100*(nt-think)/nt:.0f}%)")
    print("composition:", dict(Counter(r["tier"] for r in all_rows)))

    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(OUT)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
