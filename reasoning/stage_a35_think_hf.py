#!/usr/bin/env python3
"""Stage the argonne3.5 reasoning model (blend_a085) as a Hugging Face model repo.

Source: /project/rcc/youzhi/models/a35_reason/blend_a085 = 0.85*think_v6 + 0.15*dpo.

FOUR CONFIG FIXES the source dir needs before it is publishable. These are not cosmetic --
the first one silently breaks generation for every `.generate()` user:

 1. eos_token_id 151643 -> 151645.  The source says 151643 (<|endoftext|>), inherited from the
    stage-C vLLM config patch where stop tokens are supplied separately by the engine so it never
    mattered. For a CHAT model loaded through transformers it matters completely: generation must
    halt at <|im_end|> (151645) at the end of the assistant turn, or the model runs on past its
    answer. deploy_hf.py flags this exact bug (§5/§16), and the published Argonne-3.0-think
    uses 151645.
 2. auto_map added, matching 3.0-think (AutoConfig + AutoModel + AutoModelForCausalLM), so
    trust_remote_code=True resolves the custom argonne2 classes standalone.
 3. use_cache false -> true. It is false because the dir was written by a TRAINING script;
    leaving it off disables the KV cache and makes generation quadratically slow.
 4. block_size 4096 -> 13568, to stop contradicting max_position_embeddings.

Verifies by reloading and running a real CHAT-templated generation, asserting the output
actually terminates on <|im_end|> -- i.e. it tests fix #1 rather than trusting it.
"""
import argparse
import json
import shutil
from pathlib import Path

SRC = "/project/rcc/youzhi/models/a35_reason/blend_a085"
REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCK = 13568
EOS_IM_END = 151645


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

    sd = {}
    with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            sd[k] = t.to(torch.bfloat16) if t.is_floating_point() else t
    print(f"  loaded {len(sd)} tensors -> bf16")
    save_torch_state_dict(sd, str(stage), max_shard_size=a.max_shard_size)
    print(f"  shards: {sorted(p.name for p in stage.glob('model-*.safetensors'))}")

    cfg = json.load(open(src / "config.json"))
    before = cfg.get("eos_token_id")
    cfg["eos_token_id"] = EOS_IM_END
    cfg["dtype"] = "bfloat16"
    cfg["torch_dtype"] = "bfloat16"
    cfg["use_cache"] = True
    cfg["block_size"] = BLOCK
    cfg["max_position_embeddings"] = BLOCK
    cfg["use_gradient_checkpointing"] = False
    cfg["auto_map"] = {
        "AutoConfig": "model.ArgonneConfig",
        "AutoModel": "model.ArgonneModel",
        "AutoModelForCausalLM": "model.ArgonneModel",
    }
    json.dump(cfg, open(stage / "config.json", "w"), indent=2)
    print(f"  config: eos {before} -> {EOS_IM_END} (<|im_end|>), use_cache=True, "
          f"block={BLOCK}, +auto_map")

    shutil.copy2(REPO_ROOT / "model.py", stage / "model.py")
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        p = src / fn
        if p.exists():
            shutil.copy2(p, stage / fn)
            print(f"  aux: {fn}")
        else:
            print(f"  aux MISSING: {fn}")

    if a.verify:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  verifying reload + CHAT generation termination ...")
        tok = AutoTokenizer.from_pretrained(str(stage), trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(str(stage), trust_remote_code=True,
                                                 dtype=torch.bfloat16).eval()
        assert m.config.eos_token_id == EOS_IM_END, m.config.eos_token_id
        msgs = [{"role": "user", "content": "What is 17 - 5?"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = m.generate(ids, max_length=ids.shape[1] + 200, do_sample=False)
        gen = out[0][ids.shape[1]:]
        # STRICTER than my first attempt, which false-passed: `eos in gen or len(gen) < 200`
        # was satisfied by a run that hit the full budget and degenerated into repetition.
        # The only thing that proves the eos path works is stopping EARLY.
        stopped = len(gen) < 200
        print(f"    generated {len(gen)} new tokens (budget 200); stopped_early={stopped}")
        print("    ---\n    " + tok.decode(gen, skip_special_tokens=True).strip()[:400])
        assert stopped, "generation ran to the full budget -- the eos stop did NOT take"
    print("  DONE")


if __name__ == "__main__":
    main()
