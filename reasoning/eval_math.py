#!/usr/bin/env python3
"""Programmatic GSM8K / MATH grader for the Argonne 3.0 reasoning models (§22).

Why this exists: `eval_numeracy.py` is a 10-item *eyeballed* probe (no auto-grading);
the only honest quantitative numbers so far came from lm-eval (§20d). This is the
missing programmatic judge — it reuses the verified `extract_boxed`/`norm`/`load_problems`/
`batched_sample` primitives from `star_generate.py` and reports:

  * single-sample accuracy  (matches §21's ~2.6% GSM8K)
  * pass@k                  (any of K correct  -> the latent-capability ceiling, §21 ~48%)
  * filtered majority vote  (self-consistency diagnostic: drop unclosed/no-box, then vote)
  * failure-mode tally      (correct / wrong / unclosed / no_answer)

and implements **budget-forced termination** (`--think-budget N`, §22c Phase 0): at N
generated tokens, force-inject `\n</think>\n\nThe answer is \boxed{` into every sequence
that has NOT yet closed its `<think>` span, then let it finish a short answer tail. This
is s1-style force-STOP (the pathology here is over-thinking / non-termination: ~42% of
traces never close `</think>`). It is DISTINCT from the refuted §18f rep-penalty fix: it
forces a stop and bans nothing, so it cannot corrupt digits.

Batching obeys the model's no-padding constraint: K identical copies of ONE prompt (same
length) per problem, iterating problems sequentially (same pattern as star_generate.py).
Per-sequence forced-token injection is safe because each batch element keeps its own KV.
"""

import argparse
import datetime as _dt
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
# model.py lives at the repo ROOT and self-registers the `argonne2` arch; put both
# reasoning/ and root on the path so AutoModel can load checkpoints without auto_map.
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401  (registers argonne2)
except ModuleNotFoundError:
    pass

# Reuse the VERIFIED verifier + loader + HBM auto-fit (single source of truth).
from star_generate import extract_boxed, norm, load_problems, autofit_k  # noqa: E402

CLOSE_STR = "\n</think>\n\nThe answer is \\boxed{"


@torch.inference_mode()
def sample_batch(model, tok, input_ids, *, max_new_tokens, eos_id, do_sample,
                 temperature, top_k, top_p, think_budget=0, close_ids=None):
    """input_ids: [B, L] (B identical copies of ONE prompt). Returns list of B token-lists.

    KV-cached (prefill once, one token/step). If think_budget>0, at step==think_budget any
    sequence that has not emitted `</think>` yet is switched onto a forced closure queue
    (close_ids), after which it resumes normal sampling to fill the boxed answer.
    """
    device = model.embed_tokens.weight.device
    cur = input_ids.to(device)
    B = cur.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    gen = [[] for _ in range(B)]
    forced = [[] for _ in range(B)]          # per-seq queue of forced token ids
    forced_triggered = False
    past = None
    step_in = cur
    close_ids = close_ids or []

    for step in range(max_new_tokens):
        out = model(step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        if do_sample:
            logits = logits / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
                rm = cum > top_p
                rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = False
                logits = logits.masked_fill(rm.scatter(1, si, rm), float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)

        # Trigger force-close for still-open sequences exactly once, at the budget.
        if think_budget and not forced_triggered and step == think_budget and close_ids:
            forced_triggered = True
            for b in range(B):
                if not done[b] and "</think>" not in tok.decode(gen[b], skip_special_tokens=True):
                    forced[b] = list(close_ids)

        chosen = nxt.clone()
        for b in range(B):
            if done[b]:
                continue
            if forced[b]:
                t = forced[b].pop(0)
                chosen[b, 0] = t
            else:
                t = int(chosen[b, 0])
            gen[b].append(t)
            if t == eos_id and not forced[b]:
                done[b] = True
        step_in = chosen
        if bool(done.all()):
            break
    return gen


def classify(text, gold, think_mode):
    """Return (label, pred). Labels: correct / wrong / unclosed / no_answer."""
    pred = extract_boxed(text)
    if think_mode and "</think>" not in text:
        return "unclosed", pred
    if pred is None:
        return "no_answer", pred
    if pred == gold:
        return "correct", pred
    return "wrong", pred


def grade_model(mp, problems, args, out):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None \
        else tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(
        mp, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True)
    model.to("cuda"); model.eval()

    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False) if args.think_budget else []
    think_mode = args.enable_think

    # HBM auto-fit: pick K (copies of one prompt) to fill the card. Only meaningful when
    # sampling; greedy pass@1 (k==1, no --sample) stays K=1. Probe the LONGEST prompt so the
    # prefill logit spike is captured.
    eff_k = args.k
    if args.target_hbm > 0 and (args.sample or args.k > 1):
        longest_q = max(problems, key=lambda p: len(p[0]))[0]
        penc = tok.apply_chat_template(
            [{"role": "user", "content": longest_q}], tokenize=True,
            add_generation_prompt=True, enable_thinking=think_mode, return_tensors="pt")
        pid = penc["input_ids"] if hasattr(penc, "keys") else penc
        eff_k = autofit_k(model, pid.to("cuda"), eos_id=eos_id,
                          target_frac=args.target_hbm, max_k=args.max_k,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        out(f"  [autofit] K={eff_k} for target {args.target_hbm:.0%} HBM (max_k={args.max_k})")
    do_sample = (args.sample or eff_k > 1)

    fm = {"correct": 0, "wrong": 0, "unclosed": 0, "no_answer": 0}
    n_first_correct = 0            # single-sample accuracy (sample[0])
    n_pass_k = 0                   # >=1 of K correct
    n_majority_correct = 0         # filtered-majority vote correct
    n_majority_scorable = 0        # problems with >=1 closed+boxed sample
    t0 = _dt.datetime.now()

    for pi, (q, gold, tier) in enumerate(problems):
        torch.cuda.reset_peak_memory_stats()
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True,
            enable_thinking=think_mode, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        # OOM-safe: if a rare long prompt overflows, halve K for the rest of the run.
        while True:
            try:
                batch = ids.repeat(eff_k, 1)
                with torch.autocast("cuda", dtype=dtype):
                    gens = sample_batch(
                        model, tok, batch, max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                        do_sample=do_sample, temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        think_budget=args.think_budget, close_ids=close_ids)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if eff_k <= 1:
                    raise
                eff_k = max(1, eff_k // 2)
                do_sample = (args.sample or eff_k > 1)
                out(f"  [oom] halved K -> {eff_k} at problem {pi}")

        preds, any_correct, votes = [], False, Counter()
        for j, g in enumerate(gens):
            text = tok.decode(g, skip_special_tokens=True)
            label, pred = classify(text, gold, think_mode)
            fm[label] += 1
            if label == "correct":
                any_correct = True
                if j == 0:
                    n_first_correct += 1
            # filtered majority: only closed+boxed samples vote
            if pred is not None and (not think_mode or "</think>" in text):
                votes[pred] += 1
        if any_correct:
            n_pass_k += 1
        if votes:
            n_majority_scorable += 1
            if votes.most_common(1)[0][0] == gold:
                n_majority_correct += 1

        if pi == 0 or (pi + 1) % 25 == 0:
            el = (_dt.datetime.now() - t0).total_seconds()
            hbm = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            out(f"  [{pi+1}/{len(problems)}] pass@1(first)={100*n_first_correct/(pi+1):.1f}% "
                f"pass@{eff_k}={100*n_pass_k/(pi+1):.1f}% "
                f"maj={100*n_majority_correct/(pi+1):.1f}% | {fm} "
                f"| {el/(pi+1):.2f}s/prob hbm={hbm*100:.0f}%")

    n = len(problems)
    total_samples = sum(fm.values())
    peak_hbm = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
    out("\n  " + "-" * 66)
    out(f"  MODEL: {mp}")
    out(f"  config: source={args.source} n={n} K={eff_k} think={'ON' if think_mode else 'off'} "
        f"decode={'sample' if do_sample else 'greedy'} "
        f"think_budget={args.think_budget or 'off'} max_new={args.max_new_tokens} "
        f"peak_hbm={peak_hbm*100:.0f}%")
    out(f"  single-sample accuracy : {100*fm['correct']/max(total_samples,1):.2f}%  "
        f"({fm['correct']}/{total_samples} samples)")
    out(f"  pass@1 (sample[0])     : {100*n_first_correct/n:.2f}%  ({n_first_correct}/{n})")
    out(f"  pass@{eff_k:<15}: {100*n_pass_k/n:.2f}%  ({n_pass_k}/{n})   <- latent-capability ceiling")
    out(f"  filtered majority vote : {100*n_majority_correct/n:.2f}%  ({n_majority_correct}/{n} all) "
        f"| {100*n_majority_correct/max(n_majority_scorable,1):.2f}% of {n_majority_scorable} scorable")
    unclosed_pct = 100 * fm['unclosed'] / max(total_samples, 1)
    out(f"  failure modes          : {fm}  (unclosed={unclosed_pct:.1f}%)")
    out("  " + "-" * 66)

    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-paths", nargs="+", required=True)
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--k", type=int, default=1, help="samples per problem (K=1 -> pass@1)")
    ap.add_argument("--target-hbm", type=float, default=0.0,
                    help="auto-fit K (copies of one prompt) to fill this HBM fraction, e.g. 0.85 "
                         "(0=off, use fixed --k). Only applies when sampling; OOM-safe.")
    ap.add_argument("--max-k", type=int, default=256,
                    help="cap for --target-hbm auto-fit (bounds per-problem compute/wall-time).")
    ap.add_argument("--sample", action="store_true", help="sample even when K==1 (else greedy)")
    ap.add_argument("--enable-think", action="store_true", default=True,
                    help="think mode on (default). Use --no-enable-think for the direct channel.")
    ap.add_argument("--no-enable-think", dest="enable_think", action="store_false")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="force-close </think> at N generated tokens (0=off). Requires think mode.")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="1024 (not eval_numeracy's 200 — CoT spans truncate at 200).")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    if args.think_budget and not args.enable_think:
        print("WARN: --think-budget is a no-op without think mode; ignoring.", flush=True)
        args.think_budget = 0

    fh = open(args.log, "a") if args.log else None

    def out(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    torch.manual_seed(args.seed)
    problems = load_problems(args.source, args.n_problems, seed=args.seed)
    out("=" * 70)
    out(f"eval_math [{args.source}]  {_dt.datetime.now().isoformat(timespec='seconds')}")
    out(f"n={len(problems)} K={args.k} think={'ON' if args.enable_think else 'off'} "
        f"think_budget={args.think_budget or 'off'} max_new={args.max_new_tokens}")
    out("=" * 70)
    for mp in args.model_paths:
        grade_model(mp, problems, args, out)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
