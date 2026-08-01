#!/usr/bin/env python3
"""Run text generation with an Argonne causal LM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with Argonne 3.0.")
    parser.add_argument(
        "--model-path",
        default="PursuitOfDataScience/Argonne3.0-base",
        help="Local checkpoint directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about scientific computing at Argonne National Laboratory.",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="How many new tokens to generate.",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff.")
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument("--sample", dest="do_sample", action="store_true", help="Enable sampling.")
    decode_group.add_argument("--greedy", dest="do_sample", action="store_false", help="Use greedy decoding.")
    parser.set_defaults(do_sample=True)
    parser.add_argument(
        "--device",
        default="auto",
        help='Target device ("auto", "cpu", "cuda", "cuda:0", ...).',
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Torch dtype to load the model with.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if device.startswith("cuda"):
        if dtype_arg == "fp32":
            return torch.float32
        if dtype_arg == "fp16":
            return torch.float16
        if dtype_arg == "bf16":
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    dtype = resolve_dtype(device, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    max_length = input_ids.shape[1] + args.max_new_tokens

    generate_kwargs = {
        "input_ids": input_ids,
        "max_length": max_length,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generate_kwargs["top_p"] = args.top_p
        generate_kwargs["top_k"] = args.top_k

    autocast_enabled = device.startswith("cuda") and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        output_ids = model.generate(**generate_kwargs)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
