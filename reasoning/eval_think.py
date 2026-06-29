#!/usr/bin/env python3
"""Sampling-based GPU eval for the Argonne 3.0 CoT ("think") checkpoint.

Built to avoid the four things that made the earlier CPU probes useless:
  1. runs on GPU in bf16 (not CPU);
  2. samples (temp ~0.7) instead of greedy argmax, which falls into the
     enumeration/repetition loops this 2.88B model is prone to;
  3. gives generous max_new_tokens (default 1024) so long trained think-spans
     actually reach </think> and the answer, instead of truncating mid-think;
  4. sets eos_token_id explicitly. config.json has eos_token_id=null, so
     generate() would otherwise never stop on <|im_end|>.

It deliberately does NOT use no_repeat_ngram_size: prompt-inclusive n-gram
bans were a previous source of gibberish in eval decoding. Repetition is
controlled with sampling + a mild repetition_penalty instead.

The prompt set is chosen to separate the failure modes:
  - EASY  : trivial arithmetic/algebra. These are absent from the training
            distribution (OpenR1-Math + codeforces only), so failures here are
            data-coverage, not scale.
  - MEDIUM: standard but slightly involved.
  - HARD  : competition-math, matching the training distribution. Failures
            here point at the 2.88B capability ceiling.
  - TRAIN : a problem taken verbatim from the training set (dataset[0],
            answer 1870). If the model can't do a trained example, that is a
            different signal than either coverage or scale.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Registers ArgonneConfig/ArgonneModel so AutoModel can build the architecture.
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass


PROMPTS = [
    ("EASY",  "If 2x + 5 = 17, what is x?"),
    ("EASY",  "What is the sum of the first 10 positive integers?"),
    ("EASY",  "A rectangle has length 8 and width 3. What is its perimeter?"),
    ("EASY",  "What is 15% of 80?"),
    ("MEDIUM", "How many positive divisors does 360 have?"),
    ("MEDIUM", "Solve for x: x^2 - 5x + 6 = 0."),
    ("HARD",  "Find the sum of all positive integers n such that "
              "n^2 + 12n - 2007 is a perfect square."),
    ("TRAIN", "Let M = {1, 2, 3, ..., 1995}. Let A be a subset of M satisfying: "
              "if x is in A, then 15x is not in A. What is the maximum number of "
              "elements in A?"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Sampling GPU eval for the think checkpoint.")
    p.add_argument("--model-path",
                   default="/project/rcc/youzhi/models/instruct/think_ckpts")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--num-samples", type=int, default=1,
                   help="Samples per prompt (>1 gives a sense of variance / pass@k).")
    p.add_argument("--greedy", action="store_true",
                   help="Greedy decoding instead of sampling (for A/B comparison).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default=None,
                   help="Append a transcript here (in addition to stdout).")
    return p.parse_args()


@torch.inference_mode()
def sample_generate(model, input_ids, *, max_new_tokens, temperature, top_k, top_p,
                    do_sample, eos_id, repetition_penalty=1.0):
    """Token-by-token decode that actually stops on eos.

    The model's own generate() ignores eos entirely (runs to max_length) and
    applies repetition_penalty / n-gram bans over the *whole* input_ids
    including the prompt -- the prompt-inclusive banning that has produced
    gibberish before. This loop stops on eos and penalizes only generated
    tokens.
    """
    device = model.embed_tokens.weight.device
    ctx = model.config.max_position_embeddings
    cur = input_ids.to(device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        chunk = cur[:, -ctx:]
        logits = model.forward(chunk).logits[:, -1, :].float()
        logits = logits / max(temperature, 1e-6)

        if repetition_penalty != 1.0 and generated:
            gen_tok = torch.tensor(sorted(set(generated)), device=logits.device)
            sl = logits[0, gen_tok]
            logits[0, gen_tok] = torch.where(sl < 0, sl * repetition_penalty,
                                             sl / repetition_penalty)

        if do_sample:
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p:
                s_logits, s_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(s_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits = logits.masked_fill(
                    remove.scatter(1, s_idx, remove), float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        tok = int(next_token.item())
        generated.append(tok)
        cur = torch.cat([cur, next_token.to(cur.device)], dim=-1)
        if tok == eos_id:
            break
    return generated


class Tee:
    """Mirror prints to stdout and an optional log file."""

    def __init__(self, log_path):
        self.fh = open(log_path, "a") if log_path else None

    def __call__(self, *parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def main():
    args = parse_args()
    out = Tee(args.log)

    if not torch.cuda.is_available():
        out("ERROR: CUDA not available; this eval is meant to run on a GPU node.")
        sys.exit(1)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.manual_seed(args.seed)

    out("=" * 70)
    out(f"Argonne 3.0 think eval  {_dt.datetime.now().isoformat(timespec='seconds')}")
    out(f"model        : {args.model_path}")
    mode = "greedy" if args.greedy else (
        f"sample temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    out(f"decoding     : {mode} rep_penalty={args.repetition_penalty}")
    out(f"max_new_tok  : {args.max_new_tokens} | samples/prompt: {args.num_samples}")
    out("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # config.json leaves eos/pad null; pull them from the tokenizer so generate()
    # actually stops on <|im_end|>.
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None \
        else tokenizer.convert_tokens_to_ids("<|im_end|>")

    # AutoModelForCausalLM.from_pretrained is the self-healing override in
    # model.py: it re-inits the rotary buffers and re-ties lm_head after load.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    n_eos = 0
    n_closed = 0
    n_total = 0

    for tag, prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_tensors="pt",
        )
        # This chat template has {% generation %} blocks, so apply_chat_template
        # returns a BatchEncoding (input_ids + assistant_masks), not a bare
        # tensor. Pull out input_ids.
        input_ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)

        for s in range(args.num_samples):
            n_total += 1
            with torch.autocast("cuda", dtype=dtype):
                gen_ids = sample_generate(
                    model, input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    do_sample=not args.greedy,
                    eos_id=eos_id,
                    repetition_penalty=args.repetition_penalty,
                )
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)

            eos_reached = eos_id in gen_ids
            think_closed = "</think>" in text
            # Answer = whatever follows </think> (the model trained on
            # "<think>...</think>\n\n**Answer:** ...").
            answer = text.split("</think>", 1)[1].strip() if think_closed else ""
            answer = answer.replace("<|im_end|>", "").strip()

            n_eos += int(eos_reached)
            n_closed += int(think_closed)

            head = f"[{tag}] sample {s+1}/{args.num_samples}" if args.num_samples > 1 else f"[{tag}]"
            out("\n" + "-" * 70)
            out(f"{head}  PROMPT: {prompt}")
            out(f"  gen_tokens={len(gen_ids)}  think_closed={think_closed}  eos={eos_reached}")
            out("  --- generation ---")
            out(text.strip())
            if think_closed:
                out("  --- parsed answer ---")
                out(answer if answer else "(empty after </think>)")

    out("\n" + "=" * 70)
    out(f"SUMMARY: {n_total} generations | "
        f"reached </think>: {n_closed}/{n_total} | "
        f"reached eos: {n_eos}/{n_total}")
    out("=" * 70)
    out.close()


if __name__ == "__main__":
    main()
