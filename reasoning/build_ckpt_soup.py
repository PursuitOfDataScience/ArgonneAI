#!/usr/bin/env python3
"""Weight-soup two SAME-LINEAGE HF checkpoints: (1-a)*DPO + a*THINK.

Why (2026-07-05, §19): the soup-base downstream run produced `think_soup` =
`dpo_soup` + a CoT-SFT weight-delta (cot_soup.sh trains FROM dpo_soup, same
arch, same theta=1e6 -> same optimization basin). Post-CoT `think_soup` is a
strong-math specialist (10/10 both math modes) but its general chat REGRESSED
(gen 5/4) -- the diagnostic proved the *pre-CoT* `dpo_soup` was general-HEALTHY
(~7-8/10, concise, no loops); the CoT stage overwrote general. Anti-loop
DECODING was refuted (§18f): rep-penalty 1.3 + no-repeat-3 corrupts numbers
("80"->"8") and collapses math 10->4, only helping general-no-think loops.

Since `think_soup = dpo_soup + delta` in ONE basin, a LINEAR interpolation
    blend = dpo_soup + a*(think_soup - dpo_soup) = (1-a)*dpo_soup + a*think_soup
scales the CoT delta down: high `a` keeps math, low `a` recovers general. This
is training-FREE (minutes, no GPU) and directly targets the diagnosed
math<->general trade -- the same model-soup trick that BUILT the soup base.

Memory-frugal: streams tensors one key at a time via safetensors safe_open
(mmap, no full second copy on load). Preserves fp32. Copies config + tokenizer
(the chat template that carries think-mode support) from the THINK dir.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# tokenizer/config files to copy from the THINK dir. CRITICAL: these Argonne
# checkpoints keep the chat template in a SEPARATE `chat_template.jinja` file
# (NOT inside tokenizer_config.json) -- omitting it makes apply_chat_template
# raise "chat_template is not set" at eval time.
AUX = ("config.json", "chat_template.jinja", "tokenizer_config.json",
       "tokenizer.json", "special_tokens_map.json", "generation_config.json",
       "vocab.json", "merges.txt", "added_tokens.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpo", required=True, help="pre-CoT dir (weight 1-alpha)")
    p.add_argument("--think", required=True, help="post-CoT dir (weight alpha)")
    p.add_argument("--alpha", type=float, required=True,
                   help="weight on THINK; blend=(1-a)*dpo + a*think")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    dpo, think, out = Path(a.dpo), Path(a.think), Path(a.out)
    alpha = a.alpha
    print(f"soup: (1-{alpha})*{dpo.name} + {alpha}*{think.name} -> {out}")

    dpo_st = dpo / "model.safetensors"
    think_st = think / "model.safetensors"
    for f in (dpo_st, think_st):
        if not f.exists():
            raise SystemExit(f"missing weights: {f}")

    if (out / "model.safetensors").exists() and (out / "config.json").exists():
        print(f"  already built -> {out} (skip)")
        return

    out.mkdir(parents=True, exist_ok=True)

    with safe_open(str(dpo_st), framework="pt", device="cpu") as fd, \
         safe_open(str(think_st), framework="pt", device="cpu") as ft:
        kd, kt = set(fd.keys()), set(ft.keys())
        if kd != kt:
            only_d, only_t = kd - kt, kt - kd
            raise SystemExit(f"key mismatch: dpo-only={only_d} think-only={only_t}")
        blended = {}
        n = len(kt)
        for i, k in enumerate(sorted(kt)):
            td = fd.get_tensor(k)
            tt = ft.get_tensor(k)
            if td.shape != tt.shape:
                raise SystemExit(f"shape mismatch @ {k}: {td.shape} vs {tt.shape}")
            # blend in fp32 for precision, cast back to source dtype.
            b = (1.0 - alpha) * td.float() + alpha * tt.float()
            blended[k] = b.to(tt.dtype).contiguous()
            if i % 40 == 0 or i == n - 1:
                print(f"  [{i + 1}/{n}] {k}  {tuple(tt.shape)} {tt.dtype}", flush=True)

    save_file(blended, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in AUX:
        src = think / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"  saved -> {out}  ({len(blended)} tensors)")


if __name__ == "__main__":
    main()
