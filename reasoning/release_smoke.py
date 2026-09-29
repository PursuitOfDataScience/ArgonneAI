#!/usr/bin/env python3
"""Smoke-test a STAGED Hub release bundle the way a user will load it, and collect sample replies.

What a release has to survive: load the staged folder with
`trust_remote_code=True` and NO manual `register_argonne()`, render prompts with the bundled chat
template, and generate WITHOUT passing `eos_token_id` -- the default is the one thing no training or
evaluation probe exercises, and a wrong `config.eos_token_id` makes `.generate()` run to max_length on
every turn. A reply that hits the length cap is reported as NOT TERMINATED and the run exits non-zero.

  python reasoning/release_smoke.py --model-dir <staged dir> --kind think|instruct [--no-think] --out samples.json
CPU is fine (fp32; a 2B model generates a few tokens/s); pass --device cuda when a GPU is free.
"""
import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "think": [
        "A shop sells pencils 3 for $2. How much do 12 pencils cost?",
        "Tom has 5 boxes with 12 apples each. He gives away 17 apples. How many apples does he have left?",
        "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
        "What is the sum of the interior angles of a hexagon, in degrees?",
    ],
    "instruct": [
        "Explain in three sentences why the sky is blue.",
        "Write a short haiku about autumn leaves.",
        "Give me three tips for writing a clear email to a busy colleague.",
        "What is the difference between a list and a tuple in Python?",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--kind", choices=["think", "instruct"], required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-think", action="store_true",
                    help="render with enable_thinking=False, which puts the empty think block in the prompt")
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, torch_dtype=dtype)
    model.to(a.device).eval()
    cfg = model.config
    print(f"[smoke] loaded {a.model_dir} in {time.time() - t0:.0f}s  eos_token_id={cfg.eos_token_id} "
          f"max_position_embeddings={getattr(cfg, 'max_position_embeddings', None)} "
          f"tokenizer.eos={tok.eos_token!r}", flush=True)
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    print(f"[smoke] lm_head tied to embed_tokens: {tied}", flush=True)

    results, failed = [], 0
    for q in PROMPTS[a.kind]:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True,
                                       **({"enable_thinking": False} if a.no_think else {}))
        ids = tok(text, return_tensors="pt")["input_ids"].to(a.device)
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_length=ids.shape[1] + a.max_new, do_sample=False)
        gen = out[0][ids.shape[1]:]
        n = int(gen.shape[0])
        ended = n < a.max_new or (n > 0 and int(gen[-1]) == cfg.eos_token_id)
        reply = tok.decode(gen, skip_special_tokens=True)
        failed += 0 if ended else 1
        print(f"\n[smoke] Q: {q}\n[smoke] {n} tokens in {time.time() - t1:.0f}s, "
              f"{'terminated' if ended else 'NOT TERMINATED (hit the length cap)'}\n{reply}", flush=True)
        results.append({"prompt": q, "reply": reply, "tokens": n, "terminated": bool(ended)})
    if a.out:
        json.dump({"model_dir": a.model_dir, "kind": a.kind, "eos_token_id": cfg.eos_token_id,
                   "tied": bool(tied), "no_think": a.no_think, "results": results}, open(a.out, "w"), indent=1)
    print(f"\n[smoke] {len(results) - failed}/{len(results)} replies terminated without an eos_token_id argument")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
