#!/usr/bin/env python
"""Re-admit the LONG reasoning traces that build_mix_v6.py drops, and see if termination survives.

THE ARGUMENT. `build_mix_v6.py:71` drops every source row over 768 tokens — deliberately, to create
"termination pressure", because the defining failure of the 3.0 line was traces that never closed
`</think>`. That worked: §32 turned no_answer from ~55% into ~1.5%.

But it was decided while the §34 loader defect was active, i.e. on a pipeline that ALSO severed every
surviving trace at 128 think-tokens mid-derivation. Non-termination was therefore being fought on two
fronts at once, and the 768 cap got credit for a problem the truncation was partly causing. With traces
now preserved whole (§35/§36) the cap may be paying for nothing — and it is expensive, because it is
not a uniform filter. Measured against `cot_sft_mix_v3` (113,341 rows):

    tier            total   <=768   769-1536   >1536
    hard_strict     12000     651       1819    9530      <-- 94.6% of the HARD tier excluded
    med_openmath     4620     303        813    3504      <-- 93.4% excluded
    hq_opus          2300    1408        782     110
    med_math         5729    5627        102       0

The two hardest tiers are almost absent from training purely because hard problems need long
derivations. That is a plausible cap on multi-step and competition-math ability, and math500 (39.18)
and GSM-Plus (42.00) are exactly where the model is weakest.

WHAT THIS BUILDS. The shipped mix, plus the 769-1536 band of the MATH/REASONING tiers only. General
tiers (direct_tulu, gen_ultrachat) are NOT extended: their long rows would shift the general/math
composition at the same time and confound the test — §36's whole lesson is that composition shifts are
what cost instruction-following. easy_gsm8k stays dropped (contaminated).

The >1536 band is left out on purpose: it needs max_seq >= 2048 to train without re-creating the very
truncation §34 was about, and 9,530 of those rows would double the mix. One variable at a time.

⚠️Rows added here come from `med_openmath` and `hard_strict`, which are the tiers §36 identified as the
source of math500 leakage. So this runs `pool_decontam`-equivalent filtering on the ADDED rows before
writing, against the full eval pools.

TRAIN IT AT max_seq_length >= 1664. At 1024 the added rows would be truncated by the collator, which
is the §34 defect wearing a different hat.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

V3 = "/project/rcc/youzhi/data/cot_sft_mix_v3"
BASE = "/project/rcc/youzhi/data/cot_sft_mix_v6_gen"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v12_long"
SEED = 20260804
# math / reasoning tiers only -- general tiers are held fixed so composition is not a second variable
LONG_TIERS = ("hard_strict", "med_openmath", "hq_opus", "med_math",
              "ms_algebra", "ms_series", "ms_geometry", "ms_divisors", "synth_arith")


def norm_q(s):
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", s)


def toks(s):
    return set(norm_q(s).split())


def user_text(row):
    for m in row["messages"]:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default=V3)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--lo", type=int, default=769, help="lower bound of the re-admitted band")
    ap.add_argument("--hi", type=int, default=1536, help="upper bound of the re-admitted band")
    ap.add_argument("--decontam-threshold", type=float, default=0.70)
    ap.add_argument("--pools", nargs="+",
                    default=["svamp", "asdiv", "mawps", "gsmplus", "math500"])
    a = ap.parse_args()

    from datasets import Dataset, load_from_disk

    base = load_from_disk(a.base)
    base = base["train"] if hasattr(base, "keys") and "train" in base else base
    print(f"base mix: {len(base)} rows  ({a.base})")

    v3 = load_from_disk(a.v3)
    v3 = v3["train"] if hasattr(v3, "keys") and "train" in v3 else v3
    print(f"v3 source: {len(v3)} rows")

    # already-present questions, so re-admitting cannot duplicate a row the base mix already has
    have = {norm_q(user_text(r)) for r in base}

    cand = []
    for r in v3:
        if r["tier"] not in LONG_TIERS:
            continue
        n = r.get("num_tokens", 0)
        if not (a.lo <= n <= a.hi):
            continue
        if norm_q(user_text(r)) in have:
            continue
        cand.append({"messages": r["messages"], "tier": r["tier"], "num_tokens": n})
    print(f"candidates in the {a.lo}-{a.hi} band, math tiers, not already present: {len(cand)}")
    for t, c in Counter(x["tier"] for x in cand).most_common():
        print(f"    {t:<18} {c}")

    # ---- decontaminate the ADDED rows against every eval pool (these tiers are the leak sources)
    from clean_eval import load_clean
    ctok = [toks(user_text(x)) for x in cand]
    df = Counter()
    for t in ctok:
        df.update(t)
    ubiq = {w for w, c in df.items() if c > len(ctok) * 0.10} if ctok else set()
    index = defaultdict(list)
    for i, t in enumerate(ctok):
        for w in t:
            if w not in ubiq:
                index[w].append(i)

    dirty = {}
    for pool in a.pools:
        hits = 0
        for q, _ in load_clean(pool, 0, seed=0):
            jt = toks(q)
            cs = Counter()
            for w in jt:
                if w in index:
                    cs.update(index[w])
            for i in cs:
                j = len(jt & ctok[i]) / len(jt | ctok[i])
                if j >= a.decontam_threshold and j > dirty.get(i, (0, ""))[0]:
                    dirty[i] = (j, pool)
                    hits += 1
        print(f"  decontam vs {pool:<9} -> {hits} hit(s) at J>={a.decontam_threshold}")
    print(f"  dropping {len(dirty)} contaminated candidate(s)")

    add = [x for i, x in enumerate(cand) if i not in dirty]
    rows = [{"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]}
            for r in base] + add
    random.Random(SEED).shuffle(rows)

    ds = Dataset.from_list(rows)
    ds.save_to_disk(a.out)
    n = ds["num_tokens"]
    print(f"\nwrote {a.out}: {len(ds)} rows (+{len(add)} vs base)")
    print(f"  num_tokens: max {max(n)}  p50 {sorted(n)[len(n)//2]}  "
          f">768: {sum(1 for x in n if x > 768)}")
    print(f"  tiers: {Counter(ds['tier']).most_common()}")
    print(f"\n⚠️TRAIN WITH --max_seq_length >= {max(n) + 128} or the added rows get truncated, "
          f"which is exactly the §34 defect.")


if __name__ == "__main__":
    main()
