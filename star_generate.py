#!/usr/bin/env python3
"""STaR / rejection-sampling generation for RLVR-style self-improvement.

Sample K traces per gsm8k problem from the current think model, keep only the
ones whose final \\boxed{} answer matches the gold, and save the correct
(question, model-trace) pairs as an SFT dataset. Fine-tuning on these
reinforces the model's OWN correct reasoning — the offline, stable form of
RLVR (reward = verified-correct).

Batching note: ArgonneModel.forward forces attention_mask=None (pure causal,
no padding support), so we batch the K identical copies of ONE prompt (same
length, no padding) and iterate problems sequentially.
"""

import argparse, json, re, sys, datetime as _dt
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass

GSM8K = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"


def norm(s):
    """Normalize a numeric answer string for comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "").replace(" ", "")
    s = s.rstrip(".")
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    v = m.group(0)
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def extract_boxed(text):
    bx = re.findall(r"\\boxed\{([^}]*)\}", text)
    if bx:
        return norm(bx[-1])
    m = re.search(r"answer is[:\s$]*([-\d.,]+)", text, re.I)
    return norm(m.group(1)) if m else None


@torch.inference_mode()
def batched_sample(model, input_ids, *, max_new_tokens, eos_id, temperature, top_k, top_p):
    """input_ids: [B, L] (B identical copies). Returns list of B token-lists.

    Uses the model's KV cache: prefill the prompt once, then feed one token per
    step. ~100x+ faster than recomputing the full sequence each step.
    """
    device = model.embed_tokens.weight.device
    cur = input_ids.to(device)
    B = cur.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    gen = [[] for _ in range(B)]
    past = None
    step_in = cur
    for _ in range(max_new_tokens):
        out = model(step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float() / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            rm = cum > top_p
            rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = False
            logits = logits.masked_fill(rm.scatter(1, si, rm), float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)  # [B,1]
        for b in range(B):
            if not done[b]:
                t = int(nxt[b, 0])
                gen[b].append(t)
                if t == eos_id:
                    done[b] = True
        step_in = nxt  # cache holds the prefix; only feed the new token
        if bool(done.all()):
            break
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/project/rcc/youzhi/models/instruct/think_mix2_ckpts")
    ap.add_argument("--out", default="/project/rcc/youzhi/data/star_correct_v1")
    ap.add_argument("--n-problems", type=int, default=1500)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True)
    model.to("cuda"); model.eval()

    problems = []
    for ln in open(GSM8K):
        o = json.loads(ln)
        gold = extract_boxed(o["answer"])
        if gold is not None:
            problems.append((o["question"], gold))
        if len(problems) >= args.n_problems:
            break
    print(f"Loaded {len(problems)} gsm8k problems with gold. "
          f"Sampling K={args.k} @ temp {args.temperature}.", flush=True)

    kept, solved, total_correct = [], 0, 0
    # failure-mode tally across all samples (diagnostic: truncation vs capability)
    fm = {"correct": 0, "wrong": 0, "unclosed": 0, "no_answer": 0}
    t0 = _dt.datetime.now()
    for pi, (q, gold) in enumerate(problems):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        batch = ids.repeat(args.k, 1)
        gens = batched_sample(model, batch, max_new_tokens=args.max_new_tokens,
                              eos_id=eos_id, temperature=args.temperature,
                              top_k=args.top_k, top_p=args.top_p)
        good = []
        for g in gens:
            text = tok.decode(g, skip_special_tokens=True)
            pred = extract_boxed(text)
            if "</think>" not in text:
                fm["unclosed"] += 1
            elif pred is None:
                fm["no_answer"] += 1
            elif pred == gold:
                fm["correct"] += 1
            else:
                fm["wrong"] += 1
            if pred == gold and "</think>" in text:
                good.append(text.strip())
        if good:
            solved += 1
            for tr in good[:args.max_keep_per_problem]:
                kept.append({"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": tr}], "tier": "star_gsm8k"})
            total_correct += len(good)
        if (pi + 1) % 50 == 0:
            el = (_dt.datetime.now() - t0).total_seconds()
            print(f"  [{pi+1}/{len(problems)}] solved={solved} "
                  f"({100*solved/(pi+1):.0f}% pass@{args.k}) kept={len(kept)} "
                  f"| samples {fm} | {el/(pi+1):.2f}s/prob", flush=True)
        # Incremental save so a wall-time kill doesn't lose everything.
        if (pi + 1) % 200 == 0 and kept:
            _save(kept, args.out)

    print(f"\nDONE: {solved}/{len(problems)} solved (pass@{args.k}={100*solved/len(problems):.1f}%), "
          f"{len(kept)} correct traces kept", flush=True)
    _save(kept, args.out)
    print(f"saved -> {args.out}", flush=True)


def _save(rows, out):
    import shutil, os
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    Dataset.from_list(rows).save_to_disk(tmp)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.rename(tmp, out)


if __name__ == "__main__":
    main()
