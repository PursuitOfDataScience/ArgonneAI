#!/usr/bin/env python3
"""Deploy the Qwen1.5-0.5B Argonne-recipe reasoner (`reason_Qwen1.5-0.5B/think`) to a NEW HF repo.

Native Qwen2 arch -> loads with plain transformers (no trust_remote_code). Converts fp32 weights to
bf16, bundles the tokenizer + ChatML chat_template + generation_config + README, reload-verifies, pushes.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ID = "PursuitOfDataScience/Argonne-Qwen1.5-0.5B-think"
# EXACT ChatML template the model was CoT-trained with (reason_control/common.CHATML_TEMPLATE).
# Embedded in tokenizer_config.json (robust across transformers versions; a bare chat_template.jinja
# is not auto-loaded by the training-env transformers).
CHATML = ("{% for m in messages %}"
          "{{ '<|im_start|>' + m['role'] + '\n' + (m['content'] | trim) + '<|im_end|>' + '\n' }}"
          "{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
IM_END = 151645   # <|im_end|> — ChatML end-of-turn; the model ends its response with this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/project/rcc/youzhi/models/reason_Qwen1.5-0.5B/think")
    ap.add_argument("--stage", default="/project/rcc/youzhi/models/deploy_stage_qwen05b")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--readme", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--commit-msg", default="Argonne-recipe Qwen1.5-0.5B reasoner (think)")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file
    from huggingface_hub import save_torch_state_dict

    src = Path(args.src)
    stage = Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
        if (src / fn).exists():
            shutil.copy2(src / fn, stage / fn)
            print(f"  aux: {fn}")

    # embed the ChatML template in tokenizer_config.json (robust) — the bare .jinja isn't auto-loaded
    tc = json.load(open(stage / "tokenizer_config.json"))
    tc["chat_template"] = CHATML
    json.dump(tc, open(stage / "tokenizer_config.json", "w"), indent=2)
    print("  tokenizer_config: embedded ChatML chat_template")

    cfg = json.load(open(src / "config.json"))
    cfg["torch_dtype"] = "bfloat16"
    cfg["eos_token_id"] = IM_END   # end-of-turn is <|im_end|>, not <|endoftext|>
    json.dump(cfg, open(stage / "config.json", "w"), indent=2)
    print(f"  config: torch_dtype bfloat16, eos_token_id -> {IM_END}")

    # generation_config: stop at <|im_end|> (and <|endoftext|> as fallback)
    gc_path = stage / "generation_config.json"
    gc = json.load(open(gc_path)) if gc_path.exists() else {}
    gc["eos_token_id"] = [IM_END, 151643]
    gc["pad_token_id"] = 151643
    json.dump(gc, open(gc_path, "w"), indent=2)
    print(f"  generation_config: eos_token_id -> {gc['eos_token_id']}")

    st = src / "model.safetensors"
    if not st.exists():
        sys.exit(f"missing {st}")
    sd = load_file(str(st))
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    print(f"  weights: {len(sd)} tensors -> bf16")
    save_torch_state_dict(sd, str(stage), max_shard_size="2GB")

    if args.readme and Path(args.readme).exists():
        shutil.copy2(args.readme, stage / "README.md")
        print("  README.md bundled")

    print("  staged:", sorted(p.name for p in stage.iterdir()))

    if args.verify:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  verifying from_pretrained ...")
        tok = AutoTokenizer.from_pretrained(str(stage))
        m = AutoModelForCausalLM.from_pretrained(str(stage), dtype=torch.bfloat16)
        np = sum(p.numel() for p in m.parameters())
        # quick chat-template sanity
        enc = tok.apply_chat_template([{"role": "user", "content": "Ana has 3 boxes with 12 pencils each. She gives away 8. How many are left?"}],
                                      tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        assert ids.shape[1] > 10, f"chat_template rendered only {ids.shape[1]} ids — template not applied!"
        gen = m.generate(ids, max_new_tokens=200, do_sample=False)
        txt = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=False)
        stopped = "<|im_end|>" in txt or gen.shape[1] < ids.shape[1] + 200
        print(f"  VERIFY OK: {np:,} params; prompt {ids.shape[1]} ids; has<think>={'<think>' in txt}; "
              f"has_boxed={'boxed' in txt}; stops_cleanly={stopped}")
        print("  --- sample generation ---\n" + txt[:400].replace(chr(10), " ") + "\n  ---")
        del m

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        print(f"  PUSHING -> {args.repo}")
        api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model",
                          commit_message=args.commit_msg,
                          delete_patterns=["model*.safetensors", "model.safetensors.index.json"])
        print("  PUSH complete.")
    else:
        print("  (dry-run: no push)")


if __name__ == "__main__":
    main()
