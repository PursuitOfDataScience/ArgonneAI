"""
Extract a plain HF base-model dir from the LATEST FineMath midtraining .pt.

Test-run variant of extract_finemath_base.py: points at the latest available
FineMath checkpoint (step 833124, newer than the 768847 snapshot used in §11)
and writes a standalone HF base dir under midtrain_finemath_test/, for a full
SFT -> DPO -> CoT reasoning-pipeline repeat on the most-trained base.
"""

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

CKPT = "/project/rcc/youzhi/models/midtrain_finemath/checkpoint_step_833124.pt"
OUT = "/project/rcc/youzhi/models/midtrain_finemath_test/base/final_model_complete"
PHASE1_FINAL = "/project/rcc/youzhi/models/midtrain/final_model_complete_longmino"  # tokenizer + chat_template source
BLOCK_SIZE = 13568
ROPE_THETA = 10000.0

print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(
    f"  global_step={ckpt['global_step']} loss={ckpt['loss']:.4f} "
    f"tokens={ckpt['tokens_processed']:,} "
    f"midtrain_tokens={ckpt.get('midtraining_tokens_prior_phases',0)+ckpt.get('midtraining_tokens_processed',0):,}"
)
state = ckpt["model_state_dict"]

print(f"Loading tokenizer from: {PHASE1_FINAL}")
tokenizer = AutoTokenizer.from_pretrained(PHASE1_FINAL, trust_remote_code=True)
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

src_tpl = os.path.join(PHASE1_FINAL, "chat_template.jinja")
if os.path.exists(src_tpl):
    shutil.copy2(src_tpl, os.path.join(OUT, "chat_template.jinja"))
    print("  copied chat_template.jinja")

print("DONE")
print("Contents:")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p):,} bytes)")
