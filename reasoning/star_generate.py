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
# model.py lives at the repo ROOT (parent of reasoning/) and self-registers the
# `argonne2` arch. Put BOTH reasoning/ and root on the path so AutoModel can load
# checkpoints that don't bundle model.py (e.g. soup_blend_a085; no auto_map).
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass

GSM8K = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
MATH = "/project/rcc/youzhi/data/nlile_hendrycks-MATH-benchmark"
MATH_MAX_LEVEL = 3  # lvl 1-3 only: where the model has a non-trivial pass rate


def load_problems(source, n_problems, seed=0):
    """Return list of (question, gold, tier). source in {gsm8k, math, both}."""
    import random as _r
    probs = []
    if source in ("gsm8k", "both"):
        g = []
        for ln in open(GSM8K):
            o = json.loads(ln)
            gold = extract_boxed(o["answer"])
            if gold is not None:
                g.append((o["question"], gold, "star_gsm8k"))
        probs += g
    if source in ("math", "both"):
        from datasets import load_from_disk
        ds = load_from_disk(MATH)["train"]
        m = []
        for r in ds:
            if int(r.get("level", 9)) > MATH_MAX_LEVEL:
                continue
            # Keep ONLY purely-numeric answers. norm() extracts the *first*
            # number, so symbolic answers ("-4x^4-7x+14", "3/2", "\frac12")
            # would yield a bogus gold and poison verification. Require the
            # whole cleaned answer to be a plain int/decimal.
            cleaned = str(r["answer"]).strip().replace("$", "").replace(",", "").replace(" ", "")
            if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
                continue
            gold = norm(cleaned)
            if gold is not None:
                m.append((r["problem"], gold, "star_math"))
        probs += m
    _r.Random(seed).shuffle(probs)
    if n_problems and n_problems > 0:
        probs = probs[:n_problems]
    return probs


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
        # A model can emit an astronomically large number (runaway loop / huge product);
        # float(v) then overflows to inf and int(inf) raises OverflowError. Guard both so a
        # single pathological trace never crashes a grading/generation job (this norm is the
        # shared primitive behind eval_math/clean_eval/vllm_bon/vllm_rollouts/grpo).
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
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
    device = next(model.parameters()).device   # arch-agnostic (argonne + Qwen2/Llama)
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


def autofit_k(model, prompt_ids, *, eos_id, target_frac=0.85, max_k=512,
              probe_tokens=64, temperature=0.8, top_k=50, top_p=0.95, verbose=True):
    """Pick the largest generation group K (identical copies of ONE prompt) whose peak
    HBM stays <= target_frac of the card. The model has no cross-prompt batching, so K is
    the only knob to fill HBM during generation. Probes with a SHORT generation on the
    given (ideally longest) prompt so the prefill logit spike is captured; OOM-safe.

    Returns K>=1. Note: generation on a small model is HBM-light AND compute-scales with K,
    so a high target may pick a K that is slow per problem — cap via max_k for time budgets.
    """
    total = torch.cuda.get_device_properties(0).total_memory
    best = 1
    for k in (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        if k > max_k:
            break
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = prompt_ids.repeat(k, 1)
            batched_sample(model, batch, max_new_tokens=probe_tokens, eos_id=eos_id,
                           temperature=temperature, top_k=top_k, top_p=top_p)
            frac = torch.cuda.max_memory_allocated() / total
            if verbose:
                print(f"  [autofit] K={k:<4} probe peak {frac*100:.0f}% HBM", flush=True)
            if frac <= target_frac:
                best = k
            if frac >= target_frac:
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if verbose:
                print(f"  [autofit] K={k:<4} OOM -> stop; using K={best}", flush=True)
            break
    torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/project/rcc/youzhi/models/instruct/think_mix2_ckpts")
    ap.add_argument("--out", default="/project/rcc/youzhi/data/star_correct_v1")
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=1500)
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N problems (chunking / resume after a kill)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    ap.add_argument("--all-out", default=None,
                    help="ALSO persist EVERY rollout with its label {correct,wrong,unclosed,"
                         "no_answer} + gold here (the corpus for RLVR-DPO / a verifier). Off if unset.")
    ap.add_argument("--target-hbm", type=float, default=0.0,
                    help="auto-fit K to this HBM fraction (0=off, use fixed --k); OOM-safe.")
    ap.add_argument("--max-k", type=int, default=256, help="cap for --target-hbm auto-fit.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True)
    model.to("cuda"); model.eval()

    problems = load_problems(args.source, args.n_problems)
    if args.start:
        problems = problems[args.start:]

    eff_k = args.k
    if args.target_hbm > 0:
        longest_q = max(problems, key=lambda p: len(p[0]))[0]
        penc = tok.apply_chat_template(
            [{"role": "user", "content": longest_q}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
        pid = penc["input_ids"] if hasattr(penc, "keys") else penc
        eff_k = autofit_k(model, pid.to("cuda"), eos_id=eos_id, target_frac=args.target_hbm,
                          max_k=args.max_k, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p)

    from collections import Counter as _C
    print(f"Loaded {len(problems)} problems (source={args.source}, start={args.start}) "
          f"{dict(_C(t for _, _, t in problems))}. "
          f"Sampling K={eff_k} @ temp {args.temperature}, max_new={args.max_new_tokens}."
          f"{' [+all-out]' if args.all_out else ''}", flush=True)

    kept, all_rows, solved, total_correct = [], [], 0, 0
    # failure-mode tally across all samples (diagnostic: truncation vs capability)
    fm = {"correct": 0, "wrong": 0, "unclosed": 0, "no_answer": 0}
    t0 = _dt.datetime.now()
    for pi, (q, gold, tier) in enumerate(problems):
        torch.cuda.reset_peak_memory_stats()
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        while True:
            try:
                batch = ids.repeat(eff_k, 1)
                gens = batched_sample(model, batch, max_new_tokens=args.max_new_tokens,
                                      eos_id=eos_id, temperature=args.temperature,
                                      top_k=args.top_k, top_p=args.top_p)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if eff_k <= 1:
                    raise
                eff_k = max(1, eff_k // 2)
                print(f"  [oom] halved K -> {eff_k} at problem {pi}", flush=True)
        good = []
        for g in gens:
            text = tok.decode(g, skip_special_tokens=True)
            pred = extract_boxed(text)
            if "</think>" not in text:
                label = "unclosed"
            elif pred is None:
                label = "no_answer"
            elif pred == gold:
                label = "correct"
            else:
                label = "wrong"
            fm[label] += 1
            if label == "correct":
                good.append(text.strip())
            if args.all_out:
                all_rows.append({"question": q, "trace": text.strip(), "label": label,
                                 "pred": pred if pred is not None else "", "gold": gold, "tier": tier})
        if good:
            solved += 1
            for tr in good[:args.max_keep_per_problem]:
                kept.append({"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": tr}], "tier": tier})
            total_correct += len(good)
        if pi == 0 or (pi + 1) % 25 == 0:
            el = (_dt.datetime.now() - t0).total_seconds()
            _hbm = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            print(f"  [{pi+1}/{len(problems)}] solved={solved} "
                  f"({100*solved/(pi+1):.0f}% pass@{eff_k}) kept={len(kept)} "
                  f"| samples {fm} | {el/(pi+1):.2f}s/prob "
                  f"hbm={_hbm*100:.0f}% ({torch.cuda.max_memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB)",
                  flush=True)
        # Incremental save so a wall-time kill doesn't lose everything.
        if (pi + 1) % 200 == 0:
            if kept:
                _save(kept, args.out)
            if args.all_out and all_rows:
                _save(all_rows, args.all_out)

    print(f"\nDONE: {solved}/{len(problems)} solved (pass@{eff_k}={100*solved/len(problems):.1f}%), "
          f"{len(kept)} correct traces kept", flush=True)
    if kept:
        _save(kept, args.out)
        print(f"saved -> {args.out}", flush=True)
    if args.all_out and all_rows:
        _save(all_rows, args.all_out)
        print(f"saved ALL {len(all_rows)} labeled rollouts -> {args.all_out}", flush=True)


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
