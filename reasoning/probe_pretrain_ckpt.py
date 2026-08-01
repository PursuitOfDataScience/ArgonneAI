"""
Two-axis GATE probe for a raw argonne3.5 pretraining checkpoint (`checkpoint_step_*.pt`).

Why this exists (§20/§21 strategy, 2026-07-07): downstream RLVR/SFT on the 3.0 base is
exhausted; the lever is the argonne3.5 base now pretraining. This is the CHEAP steering
instrument for it -- run it read-only against the live pretrain's latest checkpoint every
~2-4B tokens to watch the two axes climb and extrapolate the gate-crossing token count,
WITHOUT perturbing the training job.

The GATE (from the §15 real-base control: Qwen1.5-0.5B, which the identical reasoning recipe
turned into a ~36/40 reasoner): a base is worth running the reasoning recipe on once ONE
checkpoint clears >=14/20 math AND >=14/15 general SIMULTANEOUSLY. No Argonne base ever had
both (3.0-base was 3/20 math, 13/15 general -- it fails ONLY the math axis, which is exactly
what FineMath-in-pretraining targets).

Reuses the EXACT probe sets + grading from base_probe_general.py / quick_base_probe.py so the
math/20 and general/15 numbers are directly comparable to every §11-§16 reading. The only new
piece is a loader that infers vocab from the checkpoint's own embedding -- argonne3.5 pads the
vocab 151669 -> 151680 (mult of 128, for fp8 lm_head), so base_probe_general.load_any's
`vocab = len(tok)` assumption would raise a shape mismatch.

Env:
  CKPT   checkpoint to probe (default: models/pretrain/checkpoint_last.pt -- always present)
  THETA  rope_theta (default 1e6 -- the argonne3.5 pretrain value; see its train log)
  TOK    tokenizer dir (default: the Qwen3-0.6B-Base tokenizer the 3.5 pretrain uses)
"""

import os
import sys

import torch
import torch.nn as nn

REASONING_DIR = "/home/youzhi/ArgonneAI/reasoning"
REPO_ROOT = "/home/youzhi/ArgonneAI"
for _p in (REASONING_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer

from model import ArgonneConfig, ArgonneModel
# Reuse the probe sets + graders verbatim (math/20 from quick_base_probe, general/15 here).
import base_probe_general as bpg
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

CKPT = os.environ.get("CKPT", "/project/rcc/youzhi/models/pretrain/checkpoint_last.pt").strip()
THETA = float(os.environ.get("THETA", "1000000"))
TOK = os.environ.get("TOK", "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base").strip()

MATH_GATE = 14   # /20
GEN_GATE = 14    # /15


def _strip_prefixes(state):
    """Undo torch.compile ('_orig_mod.') / DDP ('module.') key prefixes if present."""
    for pfx in ("_orig_mod.", "module."):
        if any(k.startswith(pfx) for k in state):
            state = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in state.items()}
    return state


def load_ckpt(path, theta, tok):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = _strip_prefixes(ckpt["model_state_dict"])
    # Infer vocab from the checkpoint's OWN embedding (3.5 pads 151669 -> 151680).
    ck_vocab = state["embed_tokens.weight"].shape[0]
    cfg = ArgonneConfig(
        vocab_size=ck_vocab, hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE_SIZE, max_position_embeddings=13568,
        rope_theta=theta, use_flash_attention=True, qk_norm=ENABLE_QK_NORM,
        v_norm=ENABLE_V_NORM, sandwich_norm=ENABLE_SANDWICH_NORM,
        z_loss_weight=Z_LOSS_WEIGHT,
        interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
        local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
        logit_softcap=LOGIT_SOFTCAP, tie_word_embeddings=True,
    )
    cfg.block_size = 13568
    cfg._keep_in_fp32_modules = []
    model = ArgonneModel(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Trim padded rows so greedy argmax can never land on an untrained pad token,
    # and keep the embed<->lm_head tie intact.
    vocab = len(tok)
    if ck_vocab > vocab:
        emb = model.get_input_embeddings()
        emb.weight = nn.Parameter(emb.weight[:vocab])
        lm = model.get_output_embeddings()
        if lm is not None:
            lm.weight = emb.weight
        model.config.vocab_size = vocab
    real_missing = [k for k in missing if k not in ("lm_head.weight",)]
    print(f"  loaded: ck_vocab={ck_vocab} tok_vocab={vocab} "
          f"missing={real_missing or 'none(+tied lm_head)'} unexpected={list(unexpected) or 'none'}",
          flush=True)
    return model.to(torch.bfloat16).to("cuda").eval(), ckpt


def main():
    print(f"{'=' * 78}\nARGONNE3.5 BASE-QUALITY GATE PROBE\n  ckpt : {CKPT}\n"
          f"  theta: {THETA:g}\n  tok  : {TOK}\n{'=' * 78}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    model, ckpt = load_ckpt(CKPT, THETA, tok)
    step = ckpt.get("global_step")
    toks = ckpt.get("tokens_processed")
    loss = ckpt.get("loss")
    print(f"  checkpoint: step={step} tokens_processed={toks:,} train_loss={loss}"
          if toks is not None else f"  checkpoint: step={step}", flush=True)

    mc, mrows = bpg.run_math(model, tok)
    print(f"\n  -- MATH ({len(bpg.MATH_PROBES)}) --", flush=True)
    for ok, q, gold, pred, snip in mrows:
        print(f"    [{'Y' if ok else 'n'}] {q[:44]:44s} gold={str(gold):<5} pred={str(pred):<6} | {snip}", flush=True)
    gc, grows = bpg.run_general(model, tok)
    print(f"\n  -- GENERAL ({len(bpg.GEN_PROBES)}) --", flush=True)
    for ok, q, gold, snip in grows:
        print(f"    [{'Y' if ok else 'n'}] {q[:44]:44s} want={gold:<12} | {snip}", flush=True)

    n_math, n_gen = len(bpg.MATH_PROBES), len(bpg.GEN_PROBES)
    passed = (mc >= MATH_GATE) and (gc >= GEN_GATE)
    print(f"\n{'#' * 78}", flush=True)
    print(f"  step={step}  tokens={toks:,}" if toks is not None else f"  step={step}", flush=True)
    print(f"  MATH {mc}/{n_math}  (gate >={MATH_GATE})     GENERAL {gc}/{n_gen}  (gate >={GEN_GATE})", flush=True)
    print(f"  GATE (>={MATH_GATE}/{n_math} math AND >={GEN_GATE}/{n_gen} general): "
          f"{'*** PASS -- run the reasoning recipe ***' if passed else 'not yet'}", flush=True)
    print(f"{'#' * 78}", flush=True)

    # Append a machine-readable trajectory row so the recurring probe builds a curve
    # of both axes vs tokens (extrapolate the gate-crossing). Header written once.
    traj = os.environ.get("TRAJ", "").strip()
    if traj:
        new = not os.path.exists(traj)
        with open(traj, "a") as f:
            if new:
                f.write("step\ttokens\tmath_/20\tgen_/15\tgate_pass\n")
            f.write(f"{step}\t{toks}\t{mc}\t{gc}\t{int(passed)}\n")
        print(f"  trajectory row appended -> {traj}", flush=True)


if __name__ == "__main__":
    main()
