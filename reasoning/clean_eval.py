#!/usr/bin/env python3
"""HONEST held-out math eval for the Argonne 3.0 reasoning models (contamination-free).

WHY THIS EXISTS (2026-07-09 finding):
`build_sft_mix.py:gen_gsm8k()` reads `gsm8k_main_curated/shards/shard_00000.jsonl` with NO
split filter. That shard holds train(7473)+test(1319) pooled, and the CoT-SFT mix draws 8,500
of 8,792 rows -> **~94% of GSM8K TEST, with worked <think>/\\boxed solutions, is in the
training data of every Argonne think model**. On top of that, `star_generate.py` calls
`load_problems(..)` with the DEFAULT seed=0 and START=0, so the STaR corpus covers
problems[0:440] of the same seed-0 shuffle that `eval_math.py --n-problems 200` evaluates on
-> 200/200 overlap. Every GSM8K number in thinking_training.md (greedy 2.5%, self-cons 13%,
pass@32 42.5%, pass@256 82%, the STaR 3->6% "win") is therefore measured on CONTAMINATED data.

THE CONTROLLED TEST this script runs: SVAMP and ASDiv are elementary word problems (1-2 steps)
that appear in NO training mix -- and they are EASIER than GSM8K (2-8 steps). So:
  * if the GSM8K numbers reflect real capability -> score should be HIGHER on the easier clean sets
  * if they reflect memorization                 -> score will be LOWER on the clean sets
Also reports pass@K, which is where memorization inflates most (sampling stumbles onto a
memorized solution). The "82% latent ceiling" that motivated the whole §22 verifier/best-of-N
program is exactly a pass@K number.

Sources: svamp (1000, clean) | asdiv (2305, clean) | math500 (MATH test, numeric-only; never
directly trained but OpenMathReasoning/Mixture-of-Thoughts carry indirect-leak risk) |
gsm8k (the CONTAMINATED reference set -- same seed-0 shuffle as eval_math.py, for direct A/B).

Uses the VALIDATED vLLM port (reasoning/vllm_argonne.py). Budget-forcing (s1-style force-close
of `</think>`) is implemented here on the FAST path as a 2-pass generate (it existed only in the
slow HF sampler in eval_math.py).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star_generate import extract_boxed, norm  # verified primitives  # noqa: E402

DATA = "/project/rcc/youzhi/data"
CLOSE_STR = "\n</think>\n\nThe answer is \\boxed{"


# --------------------------- clean problem loaders ---------------------------
def load_clean(source, n, seed=0):
    """Return list of (question, gold). All golds normalized with the verified norm()."""
    from datasets import load_from_disk
    probs = []
    if source == "svamp":
        d = load_from_disk(f"{DATA}/svamp_clean")
        rows = list(d["train"]) + list(d["test"])
        for r in rows:
            q = (r["Body"].strip() + " " + r["Question"].strip()).strip()
            g = norm(str(r["Answer"]))
            if g is not None:
                probs.append((q, g))
    elif source == "asdiv":
        d = load_from_disk(f"{DATA}/asdiv_clean")
        for r in d["validation"]:
            q = (r["body"].strip() + " " + r["question"].strip()).strip()
            # answers look like "9 (apples)" -> take the leading number
            g = norm(str(r["answer"]))
            if g is not None:
                probs.append((q, g))
    elif source == "math500":
        d = load_from_disk(f"{DATA}/nlile_hendrycks-MATH-benchmark")["test"]
        for r in d:
            cleaned = str(r["answer"]).strip().replace("$", "").replace(",", "").replace(" ", "")
            if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
                continue  # numeric-only, same rule as star_generate.load_problems
            g = norm(cleaned)
            if g is not None:
                probs.append((r["problem"], g))
    elif source == "mawps":
        # MU-NLPC/Calc-mawps test (520) — independent classic word problems, CLEAN (no training mix).
        d = load_from_disk(f"{DATA}/mawps_clean")
        for r in d:
            g = norm(str(r["gold"]))
            if g is not None:
                probs.append((r["question"].strip(), g))
    elif source in ("gsmplus", "gsm_plus"):
        # qintongli/GSM-Plus test (9233) — ADVERSARIAL perturbations of GSM8K test. Semi-clean:
        # gsm8k-DERIVED (the model was contaminated on gsm8k test, §23), but perturbations change
        # the answer so memorization doesn't transfer. Use as a robustness check, NOT a primary judge.
        d = load_from_disk(f"{DATA}/gsmplus_test")
        for r in d:
            g = norm(str(r["gold"]))
            if g is not None:
                probs.append((r["question"].strip(), g))
    elif source in ("gsm8k", "gsm8k_test"):
        # The CONTAMINATED reference set. Replicates star_generate.load_problems exactly
        # (same file, same seed-0 shuffle) so the A/B is apples-to-apples.
        path = f"{DATA}/gsm8k_main_curated/shards/shard_00000.jsonl"
        pool = []
        for ln in open(path):
            o = json.loads(ln)
            g = extract_boxed(o["answer"])
            if g is None:
                continue
            if source == "gsm8k_test" and o.get("split") != "test":
                continue
            pool.append((o["question"], g))
        probs = pool
    else:
        raise ValueError(source)
    random.Random(seed).shuffle(probs)
    return probs[:n] if n and n > 0 else probs


# --------------------------- vLLM decode helpers -----------------------------
def build_ids(tok, q, think=True):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=think)
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    if len(enc) > 0 and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(x) for x in enc]


def budget_forced_generate(llm, tok, prompt_ids, budget, tail, temperature=0.0):
    """s1-style force-close on the vLLM path (2 passes).

    Pass 1: generate <= budget tokens. Pass 2: for every sequence that has NOT emitted
    `</think>`, append CLOSE_STR and generate a short answer tail; for sequences that closed
    but produced no \\boxed answer, just continue. Bans nothing (distinct from the refuted
    §18f rep-penalty), so it cannot corrupt digits.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp1 = SamplingParams(n=1, temperature=temperature, max_tokens=budget)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp1)
    texts = [o.outputs[0].text for o in outs]
    gen_ids = [list(o.outputs[0].token_ids) for o in outs]

    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False)
    need, meta = [], []
    for i, t in enumerate(texts):
        closed = "</think>" in t
        has_ans = extract_boxed(t) is not None
        if closed and has_ans:
            continue
        cont = prompt_ids[i] + gen_ids[i] + (close_ids if not closed else [])
        need.append(TokensPrompt(prompt_token_ids=cont))
        meta.append((i, closed))
    if need:
        sp2 = SamplingParams(n=1, temperature=temperature, max_tokens=tail)
        outs2 = llm.generate(need, sp2)
        for (i, closed), o in zip(meta, outs2):
            texts[i] = texts[i] + ("" if closed else CLOSE_STR) + o.outputs[0].text
    return texts


def grade(texts_per_problem, golds, think=True):
    """Return metrics dict over per-problem lists of generations."""
    n = len(golds)
    fm = Counter()
    n_first = n_pass = n_maj = 0
    tot = corr = 0
    for texts, gold in zip(texts_per_problem, golds):
        votes = Counter()
        any_c = False
        for j, t in enumerate(texts):
            pred = extract_boxed(t)
            if think and "</think>" not in t:
                fm["unclosed"] += 1
            elif pred is None:
                fm["no_answer"] += 1
            elif pred == gold:
                fm["correct"] += 1
            else:
                fm["wrong"] += 1
            tot += 1
            if pred is not None and pred == gold:
                corr += 1
                any_c = True
                if j == 0:
                    n_first += 1
            if pred is not None and (not think or "</think>" in t):
                votes[pred] += 1
        if any_c:
            n_pass += 1
        if votes and votes.most_common(1)[0][0] == gold:
            n_maj += 1
    k = len(texts_per_problem[0]) if texts_per_problem else 0
    return {"n": n, "k": k, "single_acc": 100 * corr / max(tot, 1),
            "pass1": 100 * n_first / n, "passk": 100 * n_pass / n,
            "majority": 100 * n_maj / n, "fm": dict(fm),
            "n_first": n_first, "n_pass": n_pass, "n_maj": n_maj}


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval (percent) for k/n; returns '[lo-hi]' string."""
    import math
    if n == 0:
        return "[--]"
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"[{100*(center-half):.1f}-{100*(center+half):.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", nargs="+", default=["svamp", "asdiv", "gsm8k"])
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--budget-tail", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    fh = open(args.log, "a") if args.log else None

    def out(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)

    out("=" * 78)
    out(f"CLEAN EVAL  model={args.model}")
    out(f"  n={args.n_problems}/source  K={args.k}  think_budget={args.think_budget}  seed={args.seed}")
    out("=" * 78)

    summary = []
    for src in args.sources:
        probs = load_clean(src, args.n_problems, seed=args.seed)
        if not probs:
            out(f"  [{src}] NO PROBLEMS -- skipped"); continue
        golds = [g for _, g in probs]
        pids = [build_ids(tok, q, think=True) for q, _ in probs]
        tag = ("CONTAMINATED" if src.startswith("gsm8k")
               else "semi-clean" if src in ("gsmplus", "gsm_plus") else "clean")

        # 1) plain greedy
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)
        g_texts = [[o.outputs[0].text] for o in llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)]
        m_greedy = grade(g_texts, golds)

        # 2) greedy + budget-forcing
        bf = budget_forced_generate(llm, tok, pids, args.think_budget, args.budget_tail, 0.0)
        m_bf = grade([[t] for t in bf], golds)

        # 3) sampled K -> self-consistency + pass@K
        sps = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                             top_k=args.top_k, max_tokens=args.max_new_tokens)
        s_texts = [[c.text for c in o.outputs] for o in llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sps)]
        m_s = grade(s_texts, golds)

        nn = len(probs)
        out(f"\n  ---- [{src}]  ({tag})  n={nn} ----")
        out(f"    greedy pass@1            : {m_greedy['pass1']:.2f}% {wilson_ci(m_greedy['n_first'], nn)}   fm={m_greedy['fm']}")
        out(f"    greedy + budget-forcing  : {m_bf['pass1']:.2f}% {wilson_ci(m_bf['n_first'], nn)}   fm={m_bf['fm']}")
        out(f"    single-sample acc (T={args.temperature}) : {m_s['single_acc']:.2f}%")
        out(f"    self-consistency (K={args.k}) : {m_s['majority']:.2f}% {wilson_ci(m_s['n_maj'], nn)}")
        out(f"    pass@{args.k:<3}(latent ceiling) : {m_s['passk']:.2f}% {wilson_ci(m_s['n_pass'], nn)}")
        summary.append((src, tag, m_greedy['pass1'], m_bf['pass1'], m_s['majority'], m_s['passk']))

    out("\n" + "=" * 78)
    out(f"  {'source':<12} {'kind':<14} {'greedy':>8} {'+budget':>9} {'self-cons':>10} {'pass@K':>8}")
    for s, t, a, b, c, d in summary:
        out(f"  {s:<12} {t:<14} {a:>7.2f}% {b:>8.2f}% {c:>9.2f}% {d:>7.2f}%")
    out("=" * 78)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
