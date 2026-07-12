#!/usr/bin/env python3
"""Basic-numeracy probe across the Argonne 3.0 checkpoint chain.

Question this answers: can these models do grade-school arithmetic at all,
*without* think mode? A 2.88B model should. If the base/SFT/DPO checkpoints
already fail, the reasoning failure in the CoT ("think") model is inherited
from upstream training, not caused by CoT SFT. If they pass but the think
model fails, CoT SFT degraded it.

Plain chat prompting, enable_thinking=False, greedy decode, short answers.
Uses a manual decode loop because ArgonneModel.generate() ignores eos and
applies prompt-inclusive repetition bans.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
# model.py lives at the repo ROOT (parent of reasoning/) and self-registers the
# `argonne2` arch with AutoConfig/AutoModelForCausalLM at import time. Without
# importing it, AutoModelForCausalLM.from_pretrained raises KeyError('argonne2').
# Put BOTH reasoning/ and the repo root on the path so the import resolves
# regardless of where the script is launched from.
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401  (registers argonne2)
except ModuleNotFoundError:
    pass

# (prompt, expected answer) — expected is for eyeballing only, not auto-graded.
MATH_PROBES = [
    ("What is 17 - 5?", "12"),
    ("What is 8 + 3?", "11"),
    ("What is 7 times 6?", "42"),
    ("What is 100 divided by 4?", "25"),
    ("Solve for x: 2x + 5 = 17.", "6"),
    ("What is half of 80?", "40"),
    ("What is 15% of 80?", "12"),
    ("What is the sum 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10?", "55"),
    ("A rectangle has length 8 and width 3. What is its perimeter?", "22"),
    ("How many positive divisors does 12 have?", "6 (1,2,3,4,6,12)"),
]

# Non-math: factual recall, commonsense, language, non-numeric logic, instruction.
GENERAL_PROBES = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What gas do humans breathe in to stay alive?", "oxygen"),
    ("In one sentence, what does a refrigerator do?", "keeps food cold"),
    ("Correct the grammar: 'She don't like apples.'", "She doesn't like apples."),
    ("Tom is taller than Sam. Sam is taller than Bob. Who is the shortest?", "Bob"),
    ("List three primary colors.", "red, yellow, blue"),
    ("In one sentence, what is photosynthesis?", "plants make food from sunlight"),
    ("Is the sun a star or a planet?", "star"),
]

PROBE_SETS = {"math": MATH_PROBES, "general": GENERAL_PROBES}


@torch.inference_mode()
def decode_loop(model, input_ids, *, max_new_tokens, eos_id,
                do_sample=False, temperature=0.7, top_k=50, top_p=0.95,
                repetition_penalty=1.0, no_repeat_ngram=0):
    device = model.embed_tokens.weight.device
    ctx = model.config.max_position_embeddings
    cur = input_ids.to(device)
    out = []
    for _ in range(max_new_tokens):
        logits = model.forward(cur[:, -ctx:]).logits[:, -1, :].float()
        # Repetition penalty + no-repeat-ngram over GENERATED tokens only (never
        # the prompt -- §5's prompt-inclusive-ban bug). Defaults (1.0 / 0) are no-ops.
        if repetition_penalty != 1.0 and out:
            for t in set(out):
                v = logits[0, t]
                logits[0, t] = v / repetition_penalty if v > 0 else v * repetition_penalty
        if no_repeat_ngram > 0 and len(out) >= no_repeat_ngram:
            n = no_repeat_ngram
            prefix = tuple(out[-(n - 1):]) if n > 1 else ()
            for i in range(len(out) - n + 1):
                if tuple(out[i:i + n - 1]) == prefix:
                    logits[0, out[i + n - 1]] = float("-inf")
        if do_sample:
            logits = logits / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p:
                s_logits, s_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(s_logits, dim=-1), dim=-1)
                rm = cum > top_p
                rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = False
                logits = logits.masked_fill(rm.scatter(1, s_idx, rm), float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        tok = int(nxt.item())
        out.append(tok)
        cur = torch.cat([cur, nxt.to(cur.device)], dim=-1)
        if tok == eos_id:
            break
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-paths", nargs="+", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--probe-set", choices=list(PROBE_SETS), default="math",
                   help="Which question set to run.")
    p.add_argument("--enable-think", action="store_true",
                   help="Prompt with think mode on (default off).")
    p.add_argument("--sample", action="store_true",
                   help="Sample (temp 0.7) instead of greedy.")
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="Penalize already-generated tokens (1.0 = off). 1.3 anti-loop.")
    p.add_argument("--no-repeat-ngram", type=int, default=0,
                   help="Ban repeated generated n-grams of this size (0 = off).")
    p.add_argument("--log", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    fh = open(args.log, "a") if args.log else None

    def out(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    if not torch.cuda.is_available():
        out("ERROR: CUDA not available."); sys.exit(1)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    out("=" * 70)
    out(f"Argonne 3.0 probe [{args.probe_set}]  {_dt.datetime.now().isoformat(timespec='seconds')}")
    out(f"decoding: {'sample' if args.sample else 'greedy'}, "
        f"think={'ON' if args.enable_think else 'off'}")
    out("=" * 70)

    for mp in args.model_paths:
        out("\n" + "#" * 70)
        out(f"# MODEL: {mp}")
        out("#" * 70)
        tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        eos_id = tok.eos_token_id if tok.eos_token_id is not None \
            else tok.convert_tokens_to_ids("<|im_end|>")
        model = AutoModelForCausalLM.from_pretrained(
            mp, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True)
        model.to(device); model.eval()

        for prompt, expected in PROBE_SETS[args.probe_set]:
            messages = [{"role": "user", "content": prompt}]
            try:
                enc = tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=args.enable_think, return_tensors="pt")
            except Exception:
                enc = tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt")
            input_ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
            with torch.autocast("cuda", dtype=dtype):
                gen = decode_loop(model, input_ids,
                                  max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                                  do_sample=args.sample,
                                  repetition_penalty=args.repetition_penalty,
                                  no_repeat_ngram=args.no_repeat_ngram)
            text = tok.decode(gen, skip_special_tokens=False).strip()
            # When thinking, show the final answer after </think>.
            final = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
            final = final.replace("<|im_end|>", "").strip()
            out("\n  Q:", prompt, f"   [expect {expected}]")
            out("  A:", text.replace("<|im_end|>", "").replace("\n", "\n     "))
            if args.enable_think:
                closed = "</think>" in text
                out("  FINAL-AFTER-COT:", (final if final else "(none / think not closed)"),
                    f"  [closed={closed}]")

        del model
        torch.cuda.empty_cache()

    if fh:
        fh.close()


if __name__ == "__main__":
    main()
