#!/usr/bin/env python3
"""Deploy a winning Argonne-3.0-think checkpoint to the SAME HF model card.

Takes a local fp32 single-file checkpoint (e.g. a v6 soup blend), converts weights to bf16,
shards them, and bundles them with the EXACT aux files the live repo already uses (model.py,
tokenizer, chat_template) so loadability is identical to the shipped v2. Fixes the one latent
config bug: eos_token_id=None -> 151645 (<|im_end|>) so generation stops cleanly (§5/§16).

Stages to a local dir first and (with --verify) reloads via from_pretrained(trust_remote_code=True)
to prove the bundle is loadable BEFORE any push. --push uploads the staged folder to the repo.

Usage:
  # dry-run (stage + verify, NO push):
  python deploy_hf.py --src <winner_dir> --verify
  # real deploy:
  python deploy_hf.py --src <winner_dir> --verify --push --commit-msg "v3: ..."
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ID = "PursuitOfDataScience/Argonne-3.0-think"
AUX_FROM_HUB = ["model.py", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
EOS_ID = 151645  # <|im_end|>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="winner dir with fp32 model.safetensors + config.json")
    ap.add_argument("--stage", default="/project/rcc/youzhi/models/deploy_stage")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--verify", action="store_true", help="reload staged dir via from_pretrained")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--commit-msg", default="Update Argonne-3.0-think weights")
    ap.add_argument("--max-shard-size", default="1300MB")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, save_torch_state_dict

    src = Path(args.src)
    stage = Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 1) aux files from the LIVE repo (identical loadability to shipped v2)
    for fn in AUX_FROM_HUB:
        p = hf_hub_download(args.repo, fn)
        shutil.copy2(p, stage / fn)
        print(f"  aux <- hub: {fn}")

    # 2) config: start from the LIVE repo config (proven), fix eos, keep everything else
    cfg_p = hf_hub_download(args.repo, "config.json")
    cfg = json.load(open(cfg_p))
    cfg["eos_token_id"] = EOS_ID
    cfg["torch_dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    json.dump(cfg, open(stage / "config.json", "w"), indent=2)
    print(f"  config: eos_token_id -> {EOS_ID}, dtype bfloat16 (auto_map/model_type preserved)")

    # 3) weights: load fp32 winner, cast bf16, shard (preserves tied-embedding key set)
    st = src / "model.safetensors"
    if not st.exists():
        sys.exit(f"missing {st}")
    sd = load_file(str(st))
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    print(f"  weights: {len(sd)} tensors -> bf16, sharding (max {args.max_shard_size})")
    save_torch_state_dict(sd, str(stage), max_shard_size=args.max_shard_size)

    print("  staged files:", sorted(p.name for p in stage.iterdir()))

    # 4) verify loadable
    if args.verify:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  verifying from_pretrained(trust_remote_code=True) ...")
        tok = AutoTokenizer.from_pretrained(str(stage), trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(str(stage), trust_remote_code=True,
                                                 dtype=torch.bfloat16)
        np = sum(p.numel() for p in m.parameters())
        print(f"  VERIFY OK: loaded {np:,} params; eos={m.config.eos_token_id}")
        del m

    # 5) push
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"  PUSHING staged folder -> {args.repo}")
        api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model",
                          commit_message=args.commit_msg,
                          delete_patterns=["model-*.safetensors", "model.safetensors.index.json"])
        print("  PUSH complete.")
    else:
        print("  (dry-run: no push)")


if __name__ == "__main__":
    main()
