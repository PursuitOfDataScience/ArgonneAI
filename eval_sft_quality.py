#!/usr/bin/env python3
"""
Quick inference eval for the partially-trained SFT checkpoint.

Loads /project/rcc/youzhi/models/instruct/sft_ckpts/checkpoint-7033 (or the
path passed via --ckpt), applies the chat template, and prints generated
replies for a small fixed set of quality prompts.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

QUALITY_QUESTIONS = [
    "Hey! How's it going?",
    "I'm planning a weekend trip. Any tips for packing light?",
    "Explain what a black hole is in a way a 10-year-old would understand.",
    "I just failed an exam I studied really hard for. I feel terrible.",
    "What are three fun things to do on a rainy day, and why?",
    "Write a short poem about the ocean at night.",
    "What's the difference between machine learning and deep learning?",
    "Give me a quick recipe for garlic noodles.",
]


def detect_eos_from_template(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    for i in range(len(ids) - 1, -1, -1):
        token_str = tokenizer.decode([ids[i]]).strip()
        if token_str:
            return ids[i]
    return tokenizer.eos_token_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="/project/rcc/youzhi/models/instruct/sft_ckpts/checkpoint-7033")
    ap.add_argument("--tokenizer_path", type=str,
                    default="/project/rcc/youzhi/models/pretrain/argonne-3.0-base",
                    help="Path to a directory that contains tokenizer files (the ckpt subdir doesn't).")
    ap.add_argument("--argonne_root", type=str, default="/home/youzhi/ArgonneAI")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)  # greedy by default
    ap.add_argument("--max_seq_len", type=int, default=4096)
    args = ap.parse_args()

    sys.path.insert(0, args.argonne_root)
    from model import ArgonneConfig, ArgonneModel

    print("=" * 88)
    print(f"Loading checkpoint: {args.ckpt}")
    print("=" * 88)
    cfg_path = os.path.join(args.ckpt, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = ArgonneConfig(**{k: v for k, v in cfg_dict.items() if not k.startswith("_")})
    cfg.max_position_embeddings = args.max_seq_len
    cfg.block_size = args.max_seq_len
    cfg.use_flash_attention = True
    cfg._keep_in_fp32_modules = []

    print(f"  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads} "
          f"vocab={cfg.vocab_size} ctx={cfg.max_position_embeddings}")

    model = ArgonneModel(cfg)
    state = load_file(os.path.join(args.ckpt, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.tie_weights()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.to("cuda").eval()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = detect_eos_from_template(tok)
    if eos_id != tok.eos_token_id:
        old = repr(tok.eos_token)
        tok.eos_token_id = eos_id
        tok.eos_token = tok.convert_ids_to_tokens(eos_id)
        print(f"  EOS: {old} -> {repr(tok.eos_token)} (id={eos_id})")
    print(f"  pad_id={tok.pad_token_id} eos_id={tok.eos_token_id}")

    print()
    print("=" * 88)
    print("GENERATION OUTPUTS")
    print("=" * 88)
    do_sample = args.temperature > 0
    for i, q in enumerate(QUALITY_QUESTIONS, start=1):
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        if torch.is_tensor(prompt_ids):
            prompt_ids = prompt_ids.tolist()
        if isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(x) for x in prompt_ids]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        max_len = min(args.max_seq_len, input_ids.shape[1] + args.max_new_tokens)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_length=max_len,
                temperature=args.temperature if do_sample else 1.0,
                do_sample=do_sample,
            )
        gen = out[0, input_ids.shape[1]:].tolist()
        if eos_id in gen:
            gen = gen[: gen.index(eos_id)]
        reply = tok.decode(gen, skip_special_tokens=True).strip()

        print(f"\nQ{i}: {q}")
        print(f"A{i}: {reply}")

    print("\n" + "=" * 88)


if __name__ == "__main__":
    main()
