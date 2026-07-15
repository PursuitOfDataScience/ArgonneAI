#!/usr/bin/env python3
"""Combine the Qwen0.5B STaR-r2 self-traces with a cot_sft_mix_v3 anchor subsample.
Schema matches cot.py's build_examples: {messages, num_tokens, tier}.
Fresh self-traces ~25% (reinforce correct concise reasoning), v3 subsample ~75% (hold general +
diversity -> avoid the v9-style homogenization). No upsampling."""
import os
import random
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

STAR = "/project/rcc/youzhi/data/star_qwen05b_r2"
V3 = "/project/rcc/youzhi/data/cot_sft_mix_v3"
TOK = "/project/rcc/youzhi/models/reason_Qwen1.5-0.5B/think"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_qwen_star_r2"
V3_KEEP = int(os.environ.get("V3_KEEP", "18000"))
SEED = 20260714


def main():
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)

    def ntok(msgs):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return len(enc)

    star = load_from_disk(STAR)
    star_rows = []
    for r in star:
        msgs = [{"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["trace"]}]
        star_rows.append({"messages": msgs, "num_tokens": ntok(msgs), "tier": "star_r2"})
    star_ds = Dataset.from_list(star_rows)

    v3 = load_from_disk(V3)
    idx = list(range(len(v3)))
    random.Random(SEED).shuffle(idx)
    v3_sub = v3.select(idx[:V3_KEEP])
    # keep only the columns cot.py needs
    keep = ["messages", "num_tokens", "tier"]
    v3_sub = v3_sub.remove_columns([c for c in v3_sub.column_names if c not in keep])

    combined = concatenate_datasets([star_ds, v3_sub]).shuffle(seed=SEED)
    combined.save_to_disk(OUT)
    from collections import Counter
    comp = Counter(combined["tier"])
    print(f"TOTAL {len(combined)} -> {OUT}  (star_r2={len(star_ds)}, v3_sub={len(v3_sub)})")
    print("  top tiers:", dict(sorted(comp.items(), key=lambda x: -x[1])[:8]))


if __name__ == "__main__":
    main()
