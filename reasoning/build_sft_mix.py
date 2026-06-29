#!/usr/bin/env python3
"""Build a stratified ~21k CoT-SFT mix to A/B against the current think model.

Same total size as the current run (cot_filtered_3500_openended_strict, 21,177),
but a CALIBRATED distribution instead of 100% hard OpenR1/codeforces:
  - easy/short verified  (gsm8k_main_curated: short <think> + \\boxed answer)
  - medium math, verified (OpenMathReasoning cot; MATH levels 1-3)
  - high-quality mixed    (nohurry Opus-4.6 reasoning, filtered)
  - hard (ceiling)        (the current run's strict set)
  - direct / no-think     (tulu-3 general instruct, NO <think>) -> teach gating

Every example is normalized to the chat `messages` schema cot-sft.py expects.
Reasoning targets carry <think>...</think>; direct targets are plain (trained
via cot-sft.py --allow_non_reasoning 1). Output saved with save_to_disk.
"""

import json
import os
import random

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

BASE = "/project/rcc/youzhi/data"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v2"
TOK = "/project/rcc/youzhi/models/instruct/dpo_ckpts"
MAX_TOKENS = 4000          # leave headroom under the 4096 ctx (matches current run)
SEED = 20260612
rng = random.Random(SEED)

# v2: ~115k, rebalanced. v1 (21k) improved math-CoT (0->3/10) but regressed
# general no-think (8->6) because direct/general was only 12%. v2 adds a
# synthetic-arithmetic tier (drills fact execution, short calibrated traces),
# keeps easy-verified math prominent, and pushes general to ~43% (tulu = direct
# no-think anchor for the regression; ultrachat = non-math WITH think for the
# flat general-CoT quadrant).
TARGETS = {
    "synth_arith": 15000,
    "easy_gsm8k": 8500,
    "med_openmath": 22000,
    "med_math": 6000,
    "hq_opus": 2300,
    "hard_strict": 12000,
    "direct_tulu": 34000,
    "gen_ultrachat": 15000,
}

tokenizer = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)


def n_tokens(messages):
    """Token count of the rendered conversation (prompt + target)."""
    try:
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=True,
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        # apply_chat_template(tokenize=True) without return_tensors -> list
        return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
    except Exception:
        return 10 ** 9


def ok(messages):
    if not messages or messages[-1]["role"] != "assistant":
        return False
    if not messages[-1]["content"].strip():
        return False
    return n_tokens(messages) <= MAX_TOKENS


def collect(tier, target, gen):
    """Pull adapted examples from generator `gen` until `target` pass filters."""
    kept = []
    seen = 0
    for messages in gen:
        seen += 1
        if not ok(messages):
            continue
        kept.append({"messages": messages, "tier": tier})
        if len(kept) >= target:
            break
    print(f"  {tier:14s}: kept {len(kept):5d} / target {target:5d}  (scanned {seen})")
    return kept


# ---- per-source adapters (yield `messages`) --------------------------------

def gen_gsm8k():
    path = f"{BASE}/gsm8k_main_curated/shards/shard_00000.jsonl"
    lines = open(path).readlines()
    rng.shuffle(lines)
    for ln in lines:
        o = json.loads(ln)
        yield [{"role": "user", "content": o["question"]},
               {"role": "assistant", "content": o["answer"]}]


def gen_openmath():
    d = f"{BASE}/nvidia_OpenMathReasoning_curated/cot/shards"
    shards = sorted(os.listdir(d))[:2]
    rows = []
    for s in shards:
        rows += open(os.path.join(d, s)).readlines()
    rng.shuffle(rows)
    for ln in rows:
        o = json.loads(ln)
        sol = o.get("generated_solution", "")
        if "<think>" not in sol:
            continue
        yield [{"role": "user", "content": o["problem"]},
               {"role": "assistant", "content": sol}]


def gen_math():
    ds = load_from_disk(f"{BASE}/nlile_hendrycks-MATH-benchmark")["train"]
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for i in idxs:
        r = ds[i]
        if int(r.get("level", 9)) > 3:
            continue
        target = (f"<think>\n{r['solution'].strip()}\n</think>\n\n"
                  f"The answer is $\\boxed{{{r['answer']}}}$.")
        yield [{"role": "user", "content": r["problem"]},
               {"role": "assistant", "content": target}]


def gen_opus():
    ds = load_from_disk(f"{BASE}/nohurry_Opus-4.6-Reasoning-3000x-filtered")["train"]
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for i in idxs:
        r = ds[i]
        target = f"<think>\n{r['thinking'].strip()}\n</think>\n\n{r['solution'].strip()}"
        yield [{"role": "user", "content": r["problem"]},
               {"role": "assistant", "content": target}]


def gen_strict():
    ds = load_from_disk(f"{BASE}/cot_filtered_3500_openended_strict")
    ds = ds["train"] if hasattr(ds, "keys") else ds
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for i in idxs:
        msgs = [{"role": m["role"], "content": m["content"]} for m in ds[i]["messages"]]
        yield msgs


def gen_synth_arith():
    """Synthetic arithmetic with SHORT correct <think> traces. Drills the
    fact-execution the model fails at, and teaches calibrated (short) reasoning
    on easy problems. Correct by construction."""
    p_add = ["What is {a} + {b}?", "Compute {a} + {b}.", "Find the sum of {a} and {b}.", "{a} plus {b} equals what?"]
    p_sub = ["What is {a} - {b}?", "Compute {a} - {b}.", "Subtract {b} from {a}.", "{a} minus {b} equals what?"]
    p_mul = ["What is {a} times {b}?", "Compute {a} \\times {b}.", "Find the product of {a} and {b}.", "Multiply {a} by {b}."]
    p_div = ["What is {a} divided by {b}?", "Compute {a} / {b}.", "Divide {a} by {b}."]
    while True:
        kind = rng.choice(["add", "sub", "mul", "div", "two"])
        if kind == "add":
            a, b = rng.randint(2, 999), rng.randint(2, 999); r = a + b
            q = rng.choice(p_add).format(a=a, b=b); t = f"{a} + {b} = {r}"
        elif kind == "sub":
            a, b = rng.randint(2, 999), rng.randint(1, 999)
            if b > a: a, b = b, a
            r = a - b
            q = rng.choice(p_sub).format(a=a, b=b); t = f"{a} - {b} = {r}"
        elif kind == "mul":
            a, b = rng.randint(2, 99), rng.randint(2, 20); r = a * b
            q = rng.choice(p_mul).format(a=a, b=b); t = f"{a} \\times {b} = {r}"
        elif kind == "div":
            b = rng.randint(2, 20); r = rng.randint(2, 50); a = b * r
            q = rng.choice(p_div).format(a=a, b=b); t = f"{a} / {b} = {r}"
        else:
            a, b, c = rng.randint(2, 200), rng.randint(2, 200), rng.randint(1, 100)
            s = a + b; r = s - c
            q = f"What is {a} + {b} - {c}?"; t = f"First, {a} + {b} = {s}. Then {s} - {c} = {r}"
        content = f"<think>\n{t}.\n</think>\n\nThe answer is $\\boxed{{{r}}}$."
        yield [{"role": "user", "content": q},
               {"role": "assistant", "content": content}]


def gen_ultrachat():
    ds = load_from_disk(f"{BASE}/HuggingFaceH4_ultrachat_200k_cot_natural_v2")
    d = ds["train"] if hasattr(ds, "keys") else ds
    idxs = list(range(len(d)))
    rng.shuffle(idxs)
    for i in idxs:
        msgs = [{"role": m["role"], "content": m["content"]} for m in d[i]["messages"]]
        # Truncate to the last assistant turn that actually has a <think> block,
        # so the supervised target carries the reasoning signal.
        last = -1
        for j, m in enumerate(msgs):
            if m["role"] == "assistant" and "<think>" in m["content"]:
                last = j
        if last < 0:
            continue
        yield msgs[:last + 1]


def gen_tulu():
    ds = load_from_disk(f"{BASE}/allenai_tulu-3-sft-mixture")["train"]
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for i in idxs:
        msgs = ds[i]["messages"]
        if len(msgs) != 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        if any("<think>" in m["content"] for m in msgs):
            continue
        yield [{"role": "user", "content": msgs[0]["content"]},
               {"role": "assistant", "content": msgs[1]["content"]}]


GENS = {
    "synth_arith": gen_synth_arith,
    "easy_gsm8k": gen_gsm8k, "med_openmath": gen_openmath, "med_math": gen_math,
    "hq_opus": gen_opus, "hard_strict": gen_strict, "direct_tulu": gen_tulu,
    "gen_ultrachat": gen_ultrachat,
}


def main():
    print(f"Building stratified SFT mix -> {OUT}")
    all_rows = []
    for tier, target in TARGETS.items():
        all_rows += collect(tier, target, GENS[tier]())
    rng.shuffle(all_rows)

    # num_tokens column helps cot-sft.py's cheap pre-filter.
    for r in all_rows:
        r["num_tokens"] = n_tokens(r["messages"])

    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(OUT)
    print(f"\nTOTAL: {len(ds)} examples saved to {OUT}")
    from collections import Counter
    print("composition:", dict(Counter(ds["tier"])))


if __name__ == "__main__":
    main()
