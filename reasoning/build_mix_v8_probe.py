#!/usr/bin/env python3
"""§29 Phase-A probe mixes — Arm0 (v7-replica control) vs Arm1 (diversity hypothesis).

Both = a shared v6 backbone (gsm8k_train_short DE-upsampled to ×1 — v6's ×3 is itself a diversity
killer, per the plan) subsampled to ~BACKBONE rows, PLUS a teacher tier of the SAME size:
  --arm 0 : teacher_greedy_v8probe   (Qwen greedy n=1 — reproduces v7's mode-collapse)
  --arm 1 : teacher_m2_v8probe (M=2 distinct) + self_anchor_v8probe (v3's own verified traces)
The ONLY difference is the diversity mechanism → a clean test of "can it hold self-consistency."
All rows ≤640 tok. Output: cot_sft_mix_v8probe_arm{0,1}.
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
SEED = 20260712
MAX_TOK = 640
BACKBONE = 13000
TEACHER_CAP = 1600      # same in both arms
SELF_ANCHOR_CAP = 1200  # arm1 only


def make_ntok(tok):
    def ntok(msgs):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                      enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return len(enc)
    return ntok


def trace_rows(ds_path, tier, cap, ntok, rng):
    ds = load_from_disk(ds_path)
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


def backbone_rows(ntok, rng):
    v6 = load_from_disk(V6)
    by_tier = {}
    for r in v6:
        if r["num_tokens"] > MAX_TOK:
            continue
        by_tier.setdefault(r["tier"], []).append(
            {"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]})
    # de-upsample gsm8k_train_short (v6 ×3) -> unique by assistant content
    if "gsm8k_train_short" in by_tier:
        seen, uniq = set(), []
        for r in by_tier["gsm8k_train_short"]:
            c = r["messages"][-1]["content"]
            if c in seen:
                continue
            seen.add(c); uniq.append(r)
        by_tier["gsm8k_train_short"] = uniq
    rows = [r for rs in by_tier.values() for r in rs]
    rng.shuffle(rows)
    return rows[:BACKBONE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=int, required=True, choices=[0, 1])
    args = ap.parse_args()
    rng = random.Random(SEED + args.arm)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_SRC, trust_remote_code=True)
    ntok = make_ntok(tok)

    rows = backbone_rows(ntok, rng)
    if args.arm == 0:
        rows += trace_rows(f"{D}/teacher_greedy_v8probe", "teacher_greedy", TEACHER_CAP, ntok, rng)
    else:
        rows += trace_rows(f"{D}/teacher_m2_v8probe", "teacher_m2", TEACHER_CAP, ntok, rng)
        rows += trace_rows(f"{D}/self_anchor_v8probe", "self_anchor", SELF_ANCHOR_CAP, ntok, rng)
    rng.shuffle(rows)
    out = f"{D}/cot_sft_mix_v8probe_arm{args.arm}"
    Dataset.from_list(rows).save_to_disk(out)
    comp = Counter(r["tier"] for r in rows)
    print(f"ARM{args.arm}: {len(rows)} rows -> {out}")
    for t in sorted(comp):
        print(f"  {t:<20}{comp[t]:>6} ({100*comp[t]/len(rows):.1f}%)")
    print(f"  max num_tokens: {max(r['num_tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
