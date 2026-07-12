"""
Build a WEIGHT-SOUP base by linearly interpolating two same-lineage Argonne
checkpoints, and materialize it as a plain HF base dir (§17).

    souped = (1 - ALPHA) * SEED  +  ALPHA * INTERMIX

Why this exists (§17 "Is it worth continuing?"). The single-phase INTERMIX
midtraining broke the from-scratch numeracy ceiling (MATH 3->14/20) but its
LR=3e-4 over-wrote the seed's healthy world knowledge faster than the 60:40
replay protected it (GENERAL stuck 11/15, below the >=13 bar, and eroding). A
training-free average of the intermix checkpoint with its own seed recovers the
lost general for free: `0.35*seed + 0.65*intermix363908` scores MATH 15/20 AND
GEN 13/15 -- the first from-scratch Argonne base to clear the both-axes bar.
Both checkpoints share a lineage (intermix was SEEDED from the pretrain seed)
and the same RoPE regime (theta=1e6), so this is a clean linear interpolation in
weight space -- exactly the WiSE-FT / model-soup setup.

The output layout (config.json + model.safetensors + tokenizer +
chat_template.jinja) is identical to midtraining.save_final_model_artifacts and
extract_finemath_base.py, i.e. exactly what sft.py / cot-sft.py expect as a base.

ROPE_THETA MUST MATCH the regime BOTH checkpoints were trained under (theta=1e6
for the seed AND the intermix run -- see weekend.sh ROPE_THETA_OVERRIDE=1000000).
Using 1e4 here (the FineMath regime) would silently corrupt the model (the §11
trap, reversed). The default is 1e6; do not change it for a seed/intermix soup.

Everything is env-overridable:
  SEED_CKPT     pretrain seed .pt        (default: pretrain/checkpoint_step_329148.pt)
  INTERMIX_CKPT intermix .pt to soup     (default: midtrain/checkpoint_step_363908.pt -- PINNED per §17)
  ALPHA         weight on INTERMIX       (default: 0.65)
  OUT           output HF dir            (default: models/soup_seed_intermix_a065)
  ROPE_THETA    RoPE theta for config    (default: 1000000)
  TOKENIZER_SRC tokenizer + chat template source (default: argonne-3.0-base, theta=1e6 lineage)
  BLOCK_SIZE    max_position_embeddings  (default: 13568)
Example:
  ALPHA=0.65 python reasoning/build_soup_base.py
"""

import gc
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

SEED_CKPT = os.environ.get("SEED_CKPT", "/project/rcc/youzhi/models/pretrain/checkpoint_step_329148.pt")
INTERMIX_CKPT = os.environ.get("INTERMIX_CKPT", "/project/rcc/youzhi/models/midtrain/checkpoint_step_363908.pt")
ALPHA = float(os.environ.get("ALPHA", "0.65"))
OUT = os.environ.get("OUT", "/project/rcc/youzhi/models/soup_seed_intermix_a065")
ROPE_THETA = float(os.environ.get("ROPE_THETA", "1000000"))
TOKENIZER_SRC = os.environ.get("TOKENIZER_SRC", "/project/rcc/youzhi/models/pretrain/argonne-3.0-base")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "13568"))

assert 0.0 <= ALPHA <= 1.0, f"ALPHA must be in [0,1], got {ALPHA}"

# midtraining.sh treats a dir named final_model_complete beside live
# checkpoint_step_*.pt files as the phase-done marker and would stop the
# training chain -- never let a build create one next to live checkpoints.
if os.path.basename(OUT.rstrip("/")) == "final_model_complete" and glob.glob(
    os.path.join(os.path.dirname(OUT.rstrip("/")), "checkpoint_step_*.pt")
):
    sys.exit(f"Refusing OUT={OUT}: it would create midtraining.sh's phase-done marker; choose another OUT.")

# --- Idempotency: skip if the soup already exists (safe for auto-resubmits) ---
if os.path.isfile(os.path.join(OUT, "model.safetensors")) and os.path.isfile(os.path.join(OUT, "config.json")):
    print(f"Soup base already present at {OUT}/model.safetensors -- nothing to do.")
    sys.exit(0)

print(f"Loading tokenizer from: {TOKENIZER_SRC}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
vocab_size = len(tokenizer)
print(f"  vocab_size={vocab_size} eos={tokenizer.eos_token_id}")

# --- Weighted average of the two model_state_dicts, computed in fp32 ---------
# Memory-frugal: mmap the (~35 GB) checkpoints so tensors are paged in lazily,
# and accumulate one checkpoint at a time (peak RAM ~ one fp32 copy of the 2.88B
# params, ~11.5 GB) instead of holding both full checkpoints resident.
COMPONENTS = [
    ("SEED", SEED_CKPT, 1.0 - ALPHA),
    ("INTERMIX", INTERMIX_CKPT, ALPHA),
]
print(f"\nSouping with ALPHA={ALPHA}:  souped = {1.0 - ALPHA:.4g}*SEED + {ALPHA:.4g}*INTERMIX")

acc = {}          # key -> fp32 weighted-sum tensor (floating params only)
non_float = {}    # key -> tensor copied verbatim from SEED (buffers, ints)
key_sets = []
for tag, path, w in COMPONENTS:
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {tag} checkpoint not found: {path}")
    print(f"  [{tag}] load {path}  (weight {w:.4g})", flush=True)
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    step = ckpt.get("global_step", "?")
    state = ckpt["model_state_dict"]
    key_sets.append(set(state.keys()))
    for k, v in state.items():
        if torch.is_floating_point(v):
            contrib = v.detach().to(torch.float32) * w
            if k in acc:
                acc[k].add_(contrib)
            else:
                acc[k] = contrib
        elif tag == "SEED":
            # Non-float buffers (if any) are identical across the lineage; keep SEED's.
            non_float[k] = v.detach().clone()
    print(f"    global_step={step}  tensors={len(state)}", flush=True)
    del ckpt, state
    gc.collect()

if key_sets[0] != key_sets[1]:
    only_seed = sorted(key_sets[0] - key_sets[1])[:8]
    only_mix = sorted(key_sets[1] - key_sets[0])[:8]
    sys.exit(f"ERROR: checkpoints have mismatched keys (not a clean interpolation).\n"
             f"  only in SEED: {only_seed}\n  only in INTERMIX: {only_mix}")

souped_state = {**acc, **non_float}
print(f"  averaged {len(acc)} float tensors ({len(non_float)} non-float copied from SEED)")

# --- Build the model and load the souped weights ----------------------------
config = ArgonneConfig(
    vocab_size=vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=BLOCK_SIZE,
    rope_theta=ROPE_THETA,
    # Explicit eos/bos/pad: a null eos in config.json makes downstream generation
    # run to max length instead of stopping (the §5 bug).
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

print("\nBuilding ArgonneModel and loading souped weights...")
model = ArgonneModel(config)
missing, unexpected = model.load_state_dict(souped_state, strict=False)
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
print(f"\nSaving HF model dir -> {OUT}")
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
