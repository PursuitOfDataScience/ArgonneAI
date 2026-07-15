"""
Stage 4 -- CoT-SFT on cot_sft_mix_v3 (§4/§6/§10 of the doc).

THE step that makes it a reasoning model. cot_sft_mix_v3 is the SAME data used for the
Argonne `think_mix3` / `think_finemath` / `think_test` runs -- so this is a clean A/B:
identical CoT data, the only change is the base (Llama vs Argonne). Rows are stored as
tokenizer-agnostic `messages`, so we just re-render them with the Llama chat template.

The mix deliberately blends <think>...</think> reasoning rows with direct (no-think)
rows (the §6 zero-sum-diet balance), so the model learns both modes -- which the
4-quadrant eval then measures. We DROP (not truncate) over-length rows so every trace
keeps its closing </think> + \\boxed answer intact.
"""

from __future__ import annotations

import os
import sys
import time
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_tokenizer, render_chat, train_lm, BASE_MODEL  # noqa: E402

COT_DATA = os.environ.get("COT_DATA", "/project/rcc/youzhi/data/cot_sft_mix_v3")


def build_examples(tok, max_examples, max_len):
    # Load columns ONCE (sequential = fast) then index in memory with a Python-shuffled
    # order. ds.shuffle()+row iteration random-accesses the project FS and is far too slow;
    # shuffling here is also needed because the mix is tier-ORDERED (multi-step tier last).
    ds = load_from_disk(COT_DATA)
    msgs_col, nt_col, tier_col = ds["messages"], ds["num_tokens"], ds["tier"]
    idx = list(range(len(msgs_col)))
    random.Random(0).shuffle(idx)
    out, kept_tiers, dropped = [], Counter(), 0
    for i in idx:
        if len(out) >= max_examples:
            break
        # cheap pre-filter on the stored (Qwen) token count: skip long rows BEFORE the
        # expensive render (keeps the build fast). Exact drop still applies after render.
        if nt_col[i] > max_len:
            dropped += 1
            continue
        msgs = msgs_col[i]
        if not msgs:
            continue
        rc = render_chat(tok, msgs, max_len + 1)
        if rc is None:
            continue
        ids, labels = rc
        if len(ids) > max_len:      # drop over-length -> keep closing </think>+answer
            dropped += 1
            continue
        out.append({"input_ids": ids, "labels": labels})
        kept_tiers[tier_col[i]] += 1
    print(f"[cot] built {len(out)} examples (dropped {dropped} over-length); "
          f"tiers={dict(kept_tiers)}", flush=True)
    return out


def main():
    model_path = os.environ["MODEL_PATH"]
    out_dir = os.environ["OUT_DIR"]
    max_examples = int(os.environ.get("MAX_EXAMPLES", "90000"))
    max_len = int(os.environ.get("MAX_LEN", "2048"))
    target_eff = int(os.environ.get("TARGET_EFF", "48"))
    base_lr = float(os.environ.get("LR", "1e-5"))
    max_seconds = float(os.environ.get("MAX_SECONDS", "10800"))
    epochs = int(os.environ.get("EPOCHS", "1"))

    tok = setup_tokenizer(BASE_MODEL)
    examples = build_examples(tok, max_examples, max_len)

    print(f"[cot] loading {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to("cuda")
    model.config.use_cache = False

    t = time.time()
    train_lm(model, examples, tok=tok, out_dir=out_dir, epochs=epochs,
             base_lr=base_lr, max_seconds=max_seconds, target_eff=target_eff, tag="cot")
    print(f"[cot] done in {(time.time()-t)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
