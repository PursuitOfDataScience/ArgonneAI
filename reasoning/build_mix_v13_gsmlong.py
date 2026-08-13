#!/usr/bin/env python
"""Trade gsm8k UPSAMPLING for gsm8k COVERAGE: raise the 512-token cap instead of repeating 2,464 rows.

THE OBSERVATION. `build_mix_v6.py` builds its gsm8k tier with `GSM_MAX_TOK = 512` and then
`rows += gsm * GSM_UPSAMPLE` with upsample 3. Against the curated train split (7,473 rows):

    <=512  (usable today)   2,464        <-- tripled into ~4,338 mix rows
    513-1536 (excluded)     3,399        <-- more DISTINCT problems than the tier currently has
    >1536                   1,610

So the tier shows the model the same 2,464 problems three times while 3,399 different problems sit
unused, excluded on length alone. For generalisation that is the wrong trade: distinct problems carry
new information, repeats carry none. And the pool this targets is the one that matters most --
GSM-Plus is adversarially perturbed GSM8K and is where the model is weakest (42.00) and where the §36
data fix moved the most (+14.00).

WHY THE CAP EXISTED. Same reason as the 768 cap in `build_mix_v12_long` (see that file): termination
pressure, decided while the §34 truncation defect was active. This script is the same hypothesis
applied to the tier with the most to gain.

WHAT IT BUILDS. Starts from a base mix (default: the v12_long mix, so the two length-cap changes
compose) and REPLACES its `gsm8k_train_short` tier with one built at a higher cap and lower upsample,
holding the tier's row count roughly constant so total composition barely moves -- otherwise this
would be testing "more gsm8k" rather than "more DISTINCT gsm8k", which is the actual hypothesis.

Contamination: filters `split == "train"` exactly as v6 does, so no test item can enter. Verified
separately in §36 that the resulting tier is 4,338/4,338 train, 0 test.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_generate import extract_boxed  # noqa: E402

GSM = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
BASE = "/project/rcc/youzhi/data/cot_sft_mix_v12_long"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v13_gsmlong"
SEED = 20260804


from effort_probe import gsm8k_gold_map          # noqa: E402  (module-level config above)

_auth = gsm8k_gold_map()
n_badgold = [0]


def canonicalize_gsm(answer):
    """Identical to build_mix_v6.canonicalize_gsm -- reconstruct to end with the deployed boxed close."""
    gold = extract_boxed(answer)
    if gold is None:
        return None
    i = answer.rfind("</think>")
    think = answer[:i + len("</think>")] if i >= 0 else "<think>\n" + answer.strip() + "\n</think>"
    content = think + f"\n\nThe answer is $\\boxed{{{gold}}}$."
    if extract_boxed(content) != gold:
        return None
    return content, gold


def build_tier(cap, upsample):
    rows, n_test = [], 0
    for ln in open(GSM):
        o = json.loads(ln)
        if o.get("split") != "train":
            n_test += 1
            continue
        if o.get("num_tokens", 10 ** 9) > cap:
            continue
        if "</think>" not in o["answer"]:
            continue
        res = canonicalize_gsm(o["answer"])
        if res is None:
            continue
        # see build_mix_v6: canonicalize_gsm's check is self-consistent and cannot catch a
        # generated solution whose final answer is simply wrong (4.66% of them are)
        a = _auth.get(o["question"].strip())
        if a is not None and res[1] != a:
            n_badgold[0] += 1
            continue
        rows.append({"messages": [{"role": "user", "content": o["question"]},
                                  {"role": "assistant", "content": res[0]}],
                     "tier": "gsm8k_train_short", "num_tokens": o.get("num_tokens", 0)})
    print(f"  cap {cap}: {len(rows)} distinct train problems (skipped {n_test} non-train, "
          f"{n_badgold[0]} wrong-gold), x{upsample} -> {len(rows) * upsample} rows")
    return rows * upsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cap", type=int, default=1536, help="new gsm8k token ceiling (was 512)")
    ap.add_argument("--upsample", type=int, default=1,
                    help="1 by default: the point is coverage, not repetition")
    a = ap.parse_args()

    from datasets import Dataset, load_from_disk

    base = load_from_disk(a.base)
    base = base["train"] if hasattr(base, "keys") and "train" in base else base
    kept = [{"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]}
            for r in base if r["tier"] != "gsm8k_train_short"]
    dropped = len(base) - len(kept)
    print(f"base {a.base}: {len(base)} rows; removing the old gsm8k tier ({dropped} rows)")

    print("old tier, for reference:")
    build_tier(512, 3)
    print("new tier:")
    new = build_tier(a.cap, a.upsample)

    rows = kept + new
    random.Random(SEED).shuffle(rows)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(a.out)
    n = ds["num_tokens"]
    print(f"\nwrote {a.out}: {len(ds)} rows (base was {len(base)})")
    print(f"  num_tokens max {max(n)}   >768: {sum(1 for x in n if x > 768)}")
    print(f"  tiers: {Counter(ds['tier']).most_common()}")
    print(f"\n⚠️TRAIN WITH --max_seq_length >= {max(n) + 128}.")


if __name__ == "__main__":
    main()
