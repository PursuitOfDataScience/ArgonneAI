"""Per-tier HELD-OUT cross-entropy probe for the argonne3.5 reasoning anneal.

WHY THIS EXISTS (the measurement gap it closes)
-----------------------------------------------
The whole 3.5 run trains with `Validation data: disabled` — there is no held-out loss
anywhere, on any stage. For the 1-epoch pretrain that is benign (never-repeated tokens =>
train loss ~= held-out loss), but it leaves the *reasoning anneal* blind in the way that
matters: the anneal mixes five tiers with wildly different intrinsic entropy (code is
cheap, reasoning traces are not), so its aggregate train loss cannot tell you whether the
anneal is actually buying math/reasoning or merely getting better at predicting boilerplate
Python. Benchmarks answer that eventually, but they are floor-bound and noisy on a 2.88B
base; a per-tier CE delta is ~100x more sensitive and is the right instrument for
EXTRAPOLATING the remaining ~81% of the anneal.

WHY THE PROBE TOKENS ARE GENUINELY UNSEEN
-----------------------------------------
build_reasoning_corpus.py `flatten --holdout_frac 0.25` carves each SOURCE into
main=[0, split) and holdout=[split, use), split = 0.75*use, where a source's token stream
is its shards concatenated in manifest order. main -> reasoning_anneal_flat.bin (the bin
the anneal is training on now); holdout -> reasoning_midtrain_flat.bin (reserved for the
later ctx-extension stage, NOT yet trained on). So the TAIL of a tier's LAST shard sits
strictly inside that tier's holdout region and has been seen by neither stage.

FORMAT / LOSS NOTES
-------------------
- .bin is llm.c-style: a 256*int32 (1024-byte) header then uint32 tokens. uint32, not
  uint16 -- the Qwen3 vocab (151,669) does not fit in uint16.
- ArgonneModel.forward applies `labels` with NO internal shift
  (F.cross_entropy(logits.view(-1,V), labels.view(-1))), so the caller must pass
  pre-shifted labels. We do.
- logit_softcap=15.0 is applied inside forward for both the 3.0 and 3.5 configs, so the
  reported CE is the model's real predictive distribution and the arms stay comparable.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

HEADER_BYTES = 256 * 4


def load_raw_ckpt(pt_path):
    """Build the model straight from a training .pt -- no extract-to-disk round trip.

    A trajectory probe needs many checkpoints, and extracting each one costs a 34.6 GB
    torch.load PLUS an 11.5 GB save_pretrained. This does the load once and keeps the model
    in memory at the checkpoint's OWN padded vocab (151680); the CE is computed over real
    token ids < 151669, so the 11 pad rows are never selected and the loss is identical to
    the trimmed export.
    """
    import gc
    from model import ArgonneConfig, ArgonneModel
    from continue_pretrain import (
        ENABLE_INTERLEAVED_LOCAL_ATTENTION, ENABLE_QK_NORM, ENABLE_SANDWICH_NORM, ENABLE_V_NORM,
        HIDDEN_SIZE, INTERMEDIATE_SIZE, LOCAL_ATTENTION_WINDOW, LOGIT_SOFTCAP,
        NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, Z_LOSS_WEIGHT,
    )
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    for pfx in ("_orig_mod.", "module."):
        if any(k.startswith(pfx) for k in state):
            state = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in state.items()}
    step = ck.get("global_step")
    ck = None
    gc.collect()  # drop the ~23 GB of Adam state before building the model
    cfg = ArgonneConfig(
        vocab_size=state["embed_tokens.weight"].shape[0], hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS, intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=13568, rope_theta=1000000.0, use_flash_attention=True,
        qk_norm=ENABLE_QK_NORM, v_norm=ENABLE_V_NORM, sandwich_norm=ENABLE_SANDWICH_NORM,
        z_loss_weight=Z_LOSS_WEIGHT,
        interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
        local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
        logit_softcap=LOGIT_SOFTCAP, tie_word_embeddings=True)
    cfg.block_size = 13568
    cfg._keep_in_fp32_modules = []
    m = ArgonneModel(cfg)
    missing, _ = m.load_state_dict(state, strict=False)
    assert not [k for k in missing if "lm_head" not in k], f"missing weights: {missing}"
    del state
    gc.collect()
    return m.to(torch.bfloat16), step


def load_tail(bin_path, n_tokens):
    """Return the last `n_tokens` tokens of a .bin (guaranteed inside the holdout tail)."""
    total = (os.path.getsize(bin_path) - HEADER_BYTES) // 4
    take = min(n_tokens, total)
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r", offset=HEADER_BYTES)
    return np.asarray(mm[total - take:total], dtype=np.int64), total


@torch.no_grad()
def tier_ce(model, tokens, block, batch, device):
    """Mean token-level CE over `tokens`, in (block+1)-token windows."""
    win = block + 1
    n_win = (len(tokens) - 1) // block
    if n_win == 0:
        return float("nan"), 0
    tot_loss, tot_tok = 0.0, 0
    for start in range(0, n_win, batch):
        rows = []
        for w in range(start, min(start + batch, n_win)):
            off = w * block
            rows.append(tokens[off:off + win])
        rows = [r for r in rows if len(r) == win]
        if not rows:
            break
        buf = torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)
        x, y = buf[:, :-1].contiguous(), buf[:, 1:].contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        ntok = y.numel()
        tot_loss += float(out.loss) * ntok
        tot_tok += ntok
    return tot_loss / max(1, tot_tok), tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated name=path pairs")
    ap.add_argument("--tiers", required=True,
                    help="comma-separated tier=binpath pairs")
    ap.add_argument("--tokens", type=int, default=2_000_000, help="held-out tokens per tier")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="report/a35_tier_ce.json")
    args = ap.parse_args()

    from model import ArgonneModel  # noqa: E402  (needs REPO on sys.path)

    models = [kv.split("=", 1) for kv in args.models.split(",") if kv.strip()]
    tiers = [kv.split("=", 1) for kv in args.tiers.split(",") if kv.strip()]

    # Load every tier's held-out tail ONCE and reuse across arms -> identical probe tokens
    # for every model, so the cross-arm deltas are exact (no sampling variance at all).
    probes = {}
    for name, path in tiers:
        if not os.path.exists(path):
            print(f"  [tier {name}] MISSING {path} -- skipped", flush=True)
            continue
        toks, total = load_tail(path, args.tokens)
        probes[name] = toks
        print(f"  [tier {name:12s}] tail {len(toks):,} of {total:,} tok  <- {os.path.basename(path)}",
              flush=True)

    device = "cuda"
    results = {}
    for name, path in models:
        print(f"\n===== {name}  ({path}) =====", flush=True)
        try:
            if path.endswith(".pt"):
                model, step = load_raw_ckpt(path)
                model = model.to(device).eval()
                print(f"  loaded raw ckpt, global_step={step}", flush=True)
            else:
                model = ArgonneModel.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            json.dump(results, open(args.out, "w"), indent=2)
            continue
        row = {}
        for tier, toks in probes.items():
            ce, n = tier_ce(model, toks, args.block, args.batch, device)
            row[tier] = {"ce": ce, "ppl": float(np.exp(min(ce, 20))), "tokens": n}
            print(f"  {tier:14s} CE {ce:7.4f}   PPL {np.exp(min(ce, 20)):9.3f}   ({n:,} tok)",
                  flush=True)
        results[name] = row
        del model
        torch.cuda.empty_cache()
        json.dump(results, open(args.out, "w"), indent=2)

    # Delta table: every arm vs the first arm listed (the reference).
    if len(models) > 1:
        ref = models[0][0]
        if isinstance(results.get(ref), dict) and "error" not in results.get(ref, {}):
            print(f"\n===== CE delta vs {ref} (negative = better) =====", flush=True)
            hdr = f"{'tier':14s}" + "".join(f"{m:>14s}" for m, _ in models[1:])
            print(hdr, flush=True)
            for tier in probes:
                line = f"{tier:14s}"
                for m, _ in models[1:]:
                    r = results.get(m, {})
                    if "error" in r or tier not in r:
                        line += f"{'--':>14s}"
                    else:
                        line += f"{r[tier]['ce'] - results[ref][tier]['ce']:>+14.4f}"
                print(line, flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
