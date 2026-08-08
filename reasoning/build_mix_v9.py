#!/usr/bin/env python3
"""v9 mix — cot_sft_mix_v9. BREVITY self-distillation (§30).

Hypothesis (from the n=500 frontier audit): the dominant greedy loss is unclosed/no-answer
(15-30% of greedy attempts never emit an answer — thinking loops past the 256 budget at temp 0).
v6/v7/v8 all TRADED because the TEACHER tier homogenized the answer distribution (self-cons down).

v9 removes the homogenizer and makes the reasoning signal ENTIRELY v3's OWN verified-correct traces
that CLOSE within the 256-token eval budget (self_anchor_v9_short). Own-correct + short =>
(a) trains concise termination (attack unclosed) while (b) preserving the answer distribution
(hold self-cons). Keep the v6 NO-THINK general anchor to hold the guardrail. NO teacher, NO code/tool.
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
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v9"
SEED = 20260713
MAX_TOK = 640

# v6-backbone tiers KEPT (general anchor + procedure/format + with-think general). gsm8k_train_short
# is DROPPED — the fresh short self-anchor replaces it with model-own concise traces.
V9_CAPS = {
    "direct_tulu": 14500,       # no-think general anchor (hold guardrail)
    "ms_algebra": 1000, "ms_series": 1000, "ms_geometry": 1000, "ms_divisors": 1000,  # procedure/format
    "synth_arith": 2000,        # arithmetic
    "med_math": 2000,           # MATH L1-3 (with-think coverage)
    "gen_ultrachat": 3000, "hq_opus": 800, "hard_strict": 600, "med_openmath": 300,   # with-think general
}
SELF_ANCHOR_SHORT_CAP = 100000   # take ALL fresh short v3 traces (the dominant brevity signal)


def make_ntok(tok):
    def ntok(msgs):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return len(enc)
    return ntok


def trace_rows(path, tier, cap, ntok, rng):
    ds = load_from_disk(path)
    rows = []
    for r in ds:
        msgs = [{"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["trace"]}]
        nt = ntok(msgs)
        if nt > MAX_TOK:
            continue
        rows.append({"messages": msgs, "tier": tier, "num_tokens": nt})
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
        if r["num_tokens"] > MAX_TOK or r["tier"] not in V9_CAPS:
            continue
        by_tier.setdefault(r["tier"], []).append(
            {"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]})
    rows = []
    for tier, cap in V9_CAPS.items():
        pool = by_tier.get(tier, [])
        rng.shuffle(pool)
        rows += pool[:cap]

    rows += trace_rows(f"{D}/self_anchor_v9_short", "self_anchor_short", SELF_ANCHOR_SHORT_CAP, ntok, rng)

    rng.shuffle(rows)
    Dataset.from_list(rows).save_to_disk(OUT)
    comp = Counter(r["tier"] for r in rows)
    print(f"\nTOTAL v9: {len(rows)} -> {OUT}")
    for t in sorted(comp):
        print(f"  {t:<20}{comp[t]:>7} ({100*comp[t]/len(rows):.1f}%)")
    print(f"  max num_tokens: {max(r['num_tokens'] for r in rows)}")
    print(f"  mean num_tokens: {sum(r['num_tokens'] for r in rows)/len(rows):.0f}")


if __name__ == "__main__":
    main()
