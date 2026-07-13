#!/usr/bin/env python3
"""§29 Phase-B mix — cot_sft_mix_v8 (~50k rows, ALL ≤640 tok). Validated by the Phase-A STOP-GATE:
the diversity mechanism (self-anchor + teacher-M2) HOLDS self-consistency where the v7-replica
regressed −6 (xA1_30: greedy +5.0 / self-cons −0.75 vs v3 at n=400).

Tiers (per the adversarial-panel plan): v6 termination/procedure backbone + the SCALED teacher-math
tier (Qwen3-4B M≤2 on gsm8k-train + MATH-L1-3) + the self_anchor tier (v3's OWN verified traces —
the self-cons preserver). NO code/tool tiers (measured null, §26). NO gsm8k_train_short ×3 upsample
(v6's upsampling is itself a diversity killer). All ≤640 tok (termination pressure).
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from datasets import Dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))

V6 = "/project/rcc/youzhi/data/cot_sft_mix_v6"
D = "/project/rcc/youzhi/data"
TOK_SRC = "/project/rcc/youzhi/models/instruct/x_v6v2_040"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v8"
SEED = 20260712
MAX_TOK = 640

# v6-backbone tier caps (de-upsample gsm8k_train_short to ×1)
V6_CAPS = {
    "direct_tulu": 14500,       # ~29% no-think general anchor
    "ms_algebra": 1200, "ms_series": 1200, "ms_geometry": 1200, "ms_divisors": 1290,  # ~13% procedure
    "synth_arith": 3000,        # ~6%
    "med_math": 2500,           # ~5% MATH L1-3
    "gen_ultrachat": 3200, "hq_opus": 800, "hard_strict": 600, "med_openmath": 300,   # ~8% with-think general
    "gsm8k_train_short": 100000,  # take all UNIQUE (de-upsampled) — small
}
TEACHER_CAP = 4700              # ~15% (the plan's MODEST share; probe validated the mechanism at ~10%,
#                                 do NOT jump toward 32% — teacher share is the homogenizer)
SELF_ANCHOR_CAP = 3500         # ~7.5% self-cons preserver (all ~2.4k available kept)


def make_ntok(tok):
    def ntok(msgs):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return len(enc)
    return ntok


def trace_rows(path, tier, cap, ntok, rng, balance_source=False):
    ds = load_from_disk(path)
    pools = {}
    for r in ds:
        msgs = [{"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["trace"]}]
        nt = ntok(msgs)
        if nt > MAX_TOK:
            continue
        pools.setdefault(r.get("source", "x"), []).append(
            {"messages": msgs, "tier": tier, "num_tokens": nt})
    rows = []
    if balance_source and len(pools) > 1:
        per = cap // len(pools)
        for s, p in pools.items():
            rng.shuffle(p); rows += p[:per]
    else:
        for p in pools.values():
            rows += p
    rng.shuffle(rows)
    return rows[:cap]


def main():
    rng = random.Random(SEED)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_SRC, trust_remote_code=True)
    ntok = make_ntok(tok)

    print(f"Loading v6 backbone <- {V6}")
    v6 = load_from_disk(V6)
    by_tier = {}
    for r in v6:
        if r["num_tokens"] > MAX_TOK or r["tier"] not in V6_CAPS:
            continue
        by_tier.setdefault(r["tier"], []).append(
            {"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]})
    # de-upsample gsm8k_train_short -> unique
    if "gsm8k_train_short" in by_tier:
        seen, uniq = set(), []
        for r in by_tier["gsm8k_train_short"]:
            c = r["messages"][-1]["content"]
            if c not in seen:
                seen.add(c); uniq.append(r)
        by_tier["gsm8k_train_short"] = uniq
    rows = []
    for tier, cap in V6_CAPS.items():
        pool = by_tier.get(tier, [])
        rng.shuffle(pool)
        rows += pool[:cap]

    rows += trace_rows(f"{D}/teacher_v8_full", "teacher_math", TEACHER_CAP, ntok, rng, balance_source=True)
    rows += trace_rows(f"{D}/self_anchor_v8_full", "self_anchor", SELF_ANCHOR_CAP, ntok, rng)

    rng.shuffle(rows)
    Dataset.from_list(rows).save_to_disk(OUT)
    comp = Counter(r["tier"] for r in rows)
    print(f"\nTOTAL v8: {len(rows)} -> {OUT}")
    for t in sorted(comp):
        print(f"  {t:<20}{comp[t]:>7} ({100*comp[t]/len(rows):.1f}%)")
    print(f"  max num_tokens: {max(r['num_tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
