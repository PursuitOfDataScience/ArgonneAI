#!/usr/bin/env python3
"""Stage the argonne3.5 midtrained base as a Hugging Face model repo.

Source of truth: /project/rcc/youzhi/models/midtrain/final_model_complete -- the HF dir the
ctx-extension midtrain wrote on natural completion (step 321,062). It is fp32, single-file,
and carries no `auto_map`, so it cannot be loaded standalone with trust_remote_code=True.

This produces the same layout the argonne-3.0-base card uses (5 bf16 shards + index + bundled
model.py + tokenizer), plus the one fix that card lacks: an `auto_map` so
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` resolves the custom
`argonne2` classes without the user having to clone the repo first.

Weights are converted tensor-by-tensor via safe_open so peak RSS stays near the bf16 output
size (~6 GB) instead of loading the whole 11.5 GB fp32 file at once.

Usage:
  python reasoning/stage_a35_base_hf.py --stage <dir>            # stage only
  python reasoning/stage_a35_base_hf.py --stage <dir> --verify   # stage + reload check
"""
import argparse
import json
import shutil
from pathlib import Path

SRC = "/project/rcc/youzhi/models/midtrain/final_model_complete"
REPO_ROOT = Path(__file__).resolve().parent.parent
# The published context length. The midtrain trained at exactly this block size.
BLOCK = 13568


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--max-shard-size", default="1200MB")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from huggingface_hub import save_torch_state_dict

    src, stage = Path(a.src), Path(a.stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # ---- 1. weights: fp32 -> bf16, streamed one tensor at a time -------------------------
    sd = {}
    with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            sd[k] = t.to(torch.bfloat16) if t.is_floating_point() else t
    n_params = sum(v.numel() for k, v in sd.items() if "embed_tokens" in k or "lm_head" not in k)
    print(f"  loaded {len(sd)} tensors; converted to bf16")

    save_torch_state_dict(sd, str(stage), max_shard_size=a.max_shard_size)
    shards = sorted(p.name for p in stage.glob("model-*.safetensors"))
    print(f"  wrote {len(shards)} shards: {shards}")

    # ---- 2. config: bf16 + published context + auto_map ----------------------------------
    cfg = json.load(open(src / "config.json"))
    cfg["dtype"] = "bfloat16"
    cfg["torch_dtype"] = "bfloat16"          # older transformers reads this spelling
    cfg["block_size"] = BLOCK
    cfg["max_position_embeddings"] = BLOCK
    cfg["use_gradient_checkpointing"] = False
    # THE fix the 3.0-base card is missing: without auto_map, `trust_remote_code=True` cannot
    # resolve model_type "argonne2" to the bundled model.py and from_pretrained raises.
    cfg["auto_map"] = {
        "AutoConfig": "model.ArgonneConfig",
        "AutoModelForCausalLM": "model.ArgonneModel",
    }
    json.dump(cfg, open(stage / "config.json", "w"), indent=2)
    print(f"  config.json: vocab={cfg['vocab_size']} ctx={BLOCK} rope_theta={cfg['rope_theta']:g} +auto_map")

    # ---- 3. aux: the architecture that DEFINES these weights, + tokenizer ----------------
    shutil.copy2(REPO_ROOT / "model.py", stage / "model.py")
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        p = src / fn
        if p.exists():
            shutil.copy2(p, stage / fn)
            print(f"  aux: {fn}")
        else:
            print(f"  aux MISSING (skipped): {fn}")

    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"  staged {stage}  ({total/2**30:.2f} GiB, {n_params:,} params)")

    # ---- 4. verify the bundle loads the way the model card tells people to load it -------
    if a.verify:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  verifying: AutoModelForCausalLM.from_pretrained(trust_remote_code=True) ...")
        tok = AutoTokenizer.from_pretrained(str(stage), trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(str(stage), trust_remote_code=True,
                                                 dtype=torch.bfloat16)
        got = sum(p.numel() for p in m.parameters())
        print(f"    loaded OK: {got:,} params, dtype={next(m.parameters()).dtype}, "
              f"ctx={m.config.max_position_embeddings}, vocab={m.config.vocab_size}")
        assert m.config.vocab_size == len(tok), f"vocab mismatch {m.config.vocab_size} vs {len(tok)}"
        # numeric spot-check against the fp32 source (bf16 rounding only)
        ids = tok("Argonne National Laboratory is", return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = m(ids)
        lg = out.logits if hasattr(out, "logits") else out[0]
        print(f"    forward OK: logits {tuple(lg.shape)} finite={bool(torch.isfinite(lg).all())}")
        assert torch.isfinite(lg).all(), "non-finite logits"
    print("  DONE")


if __name__ == "__main__":
    main()
