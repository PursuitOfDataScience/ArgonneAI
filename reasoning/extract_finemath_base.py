"""
Extract a plain HF base-model dir from a pinned midtraining .pt checkpoint.

Midtraining writes only training checkpoints (model + optimizer + counters).
To start the reasoning pipeline on a midtrained base before (or instead of)
the run writing its final_model_complete, pin one checkpoint and materialize a
standalone HF dir (config.json + model.safetensors + tokenizer +
chat_template.jinja) -- exactly the layout midtraining.save_final_model_artifacts
would have produced, and exactly what sft.py / cot-sft.py expect as a base.

Everything is env-overridable so the same script pins ANY midtraining
checkpoint. ROPE_THETA MUST MATCH THE RUN THAT PRODUCED THE CHECKPOINT --
a mismatched theta writes a config that silently corrupts the model (§11):
  FineMath Phase-2 checkpoints                -> ROPE_THETA=10000 (default)
  INTERMIX checkpoints (§14, models/midtrain) -> ROPE_THETA=1000000
Example (intermix):
  CKPT=/project/rcc/youzhi/models/midtrain/checkpoint_step_345108.pt \
  ROPE_THETA=1000000 python reasoning/extract_finemath_base.py
"""

import glob
import os
import shutil
import sys

import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, "/home/youzhi/ArgonneAI")
from continue_pretrain import (
    ENABLE_INTERLEAVED_LOCAL_ATTENTION,
    ENABLE_QK_NORM,
    ENABLE_SANDWICH_NORM,
    ENABLE_V_NORM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LOCAL_ATTENTION_WINDOW,
    LOGIT_SOFTCAP,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Z_LOSS_WEIGHT,
)
from model import ArgonneConfig, ArgonneModel

CKPT = os.environ.get("CKPT", "/project/rcc/youzhi/models/midtrain_finemath_pinned/checkpoint_step_768847.pt")
_ckpt_stem = os.path.splitext(os.path.basename(CKPT))[0]
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(CKPT), f"extracted_{_ckpt_stem}"))
TOKENIZER_SRC = os.environ.get(  # tokenizer + chat_template source
    "TOKENIZER_SRC", "/project/rcc/youzhi/models/midtrain/final_model_complete_longmino"
)
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "13568"))
ROPE_THETA = float(os.environ.get("ROPE_THETA", "10000"))

# midtraining.sh treats a dir named final_model_complete beside live
# checkpoint_step_*.pt files as the phase-done marker and would stop the
# training chain -- never let an extract create one.
if os.path.basename(OUT.rstrip("/")) == "final_model_complete" and glob.glob(
    os.path.join(os.path.dirname(OUT.rstrip("/")), "checkpoint_step_*.pt")
):
    sys.exit(f"Refusing OUT={OUT}: it would create midtraining.sh's phase-done marker; choose another OUT.")

print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(
    f"  global_step={ckpt['global_step']} loss={ckpt['loss']:.4f} "
    f"tokens={ckpt['tokens_processed']:,} "
    f"midtrain_tokens={ckpt.get('midtraining_tokens_prior_phases',0)+ckpt.get('midtraining_tokens_processed',0):,}"
)
state = ckpt["model_state_dict"]

print(f"Loading tokenizer from: {TOKENIZER_SRC}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
vocab_size = len(tokenizer)
print(f"  vocab_size={vocab_size} eos={tokenizer.eos_token_id}")

config = ArgonneConfig(
    vocab_size=vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=BLOCK_SIZE,
    rope_theta=ROPE_THETA,
    # Explicit eos: a null eos in config.json makes downstream generation run
    # to max length instead of stopping (the §5 bug).
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_flash_attention=True,
    qk_norm=ENABLE_QK_NORM,
    v_norm=ENABLE_V_NORM,
    sandwich_norm=ENABLE_SANDWICH_NORM,
    z_loss_weight=Z_LOSS_WEIGHT,
    interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
    local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
    logit_softcap=LOGIT_SOFTCAP,
    tie_word_embeddings=True,
)
config.block_size = BLOCK_SIZE
config.rope_theta = ROPE_THETA
config._keep_in_fp32_modules = []

print("Building ArgonneModel and loading weights...")
model = ArgonneModel(config)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"  missing={list(missing)}  unexpected={list(unexpected)}")
assert not unexpected, f"Unexpected keys present: {unexpected}"
assert all("lm_head" in m for m in missing), f"Unexpected missing keys (not just tied lm_head): {missing}"
nparams = sum(p.numel() for p in model.parameters())
print(f"  params={nparams:,}")

# Trim embeddings to true vocab (mirrors save_final_model_artifacts).
embed = model.get_input_embeddings()
if embed.weight.shape[0] > vocab_size:
    print(f"  trimming embeddings {embed.weight.shape[0]} -> {vocab_size}")
    embed.weight = nn.Parameter(embed.weight[:vocab_size])
    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        lm_head.weight = nn.Parameter(lm_head.weight[:vocab_size])
    model.config.vocab_size = vocab_size

os.makedirs(OUT, exist_ok=True)
print(f"Saving HF model dir -> {OUT}")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
config.save_pretrained(OUT)

# Copy the Qwen3-style chat template (parses <think>...</think>) used downstream.
src_tpl = os.path.join(TOKENIZER_SRC, "chat_template.jinja")
if os.path.exists(src_tpl):
    shutil.copy2(src_tpl, os.path.join(OUT, "chat_template.jinja"))
    print("  copied chat_template.jinja")

print("DONE")
print("Contents:")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p):,} bytes)")
