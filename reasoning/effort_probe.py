#!/usr/bin/env python3
"""Measure the REASONING-EFFORT PROFILE of an Argonne think model (vLLM fast path).

WHY THIS EXISTS (2026-08-02). §32 shipped `blend_a085` (= argonne-3.5-think): clean SVAMP/ASDiv
greedy 65.0/73.0, self-cons(K=8) 74.0/82.7, pass@8 90.7/92.3. Two facts about *effort* fall out
of those numbers and neither is measured anywhere:

  1. **Budget-forcing now adds exactly 0.00.** v6 was trained short-only ON PURPOSE (§23e) to
     force native `</think>` termination, and it worked (`unclosed` 53.7% -> 1.3%). So the model
     has no "think harder" knob left: it closes early and commits. Whether MORE sequential
     compute would help is unknown -- nobody has tried *suppressing* the close and extending.
  2. **The floor-to-ceiling gap is ~20pt** (greedy 65 vs pass@8 91). On the 3.0 base, every
     lever that tried to close such a gap died of SIGNAL STARVATION (§9/§20 GRPO: a binary
     reward left most groups with zero gradient at ~2% accuracy). At 59-65% single-sample
     accuracy that failure mode is *structurally absent* -- but "structurally absent" is a
     prediction, so this script MEASURES the reward density before any RL is paid for.

Modes (each writes a machine-readable JSON next to the log so later stages can diff):
  budget   accuracy vs generation budget + think-length distribution + acc binned by length.
           Answers: is the model length-limited at all, or does it saturate far below the cap?
  extend   s1-style FORCED CONTINUATION. Suppress the first `</think>`, inject a continuation
           cue, generate again, repeat N times, then force-close. Answers: can this model spend
           more sequential effort productively? Reports the wrong->correct / correct->wrong flip
           matrix per extension, which a scalar accuracy hides.
  density  RLVR reward-density audit on a TRAIN pool: per-problem correct-count histogram at
           K samples, `signal_groups` = fraction with 0<c<K (the groups that carry a
           group-relative gradient), and the variance-weighted mean. This is the go/no-go for RL.
  passk    pass@k and majority@k vs k (subsampled from one K-sample draw) + the never-solved
           set. Answers: where does the ceiling saturate, and how much is left to convert?

All problem pools reuse `clean_eval.load_clean` (contamination-audited) plus TRAIN-only pools
added here (`gsm8k_train`, `math_train*`) for generation/RL fuel -- never an eval set.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = "/project/rcc/youzhi/data"
CLOSE_STR = "\n</think>\n\nThe answer is $\\boxed{"


_GSM_GOLD = None


def gsm8k_gold_map():
    """question -> GSM8K's OWN gold (`#### N`), for verifying curated-shard solutions.

    The curated shard's `answer` is a MODEL-WRITTEN solution, so any gold read out of it is the
    generator's answer, not the dataset's: measured 6.62% wrong (see load_pool). Anything that
    treats those solutions as SFT targets needs to check them against this map -- the mix
    builders' `canonicalize_gsm` verified `extract_boxed(content) == gold` where BOTH sides came
    from the same generated text, which is a self-consistency check and cannot catch a wrong
    answer. Returns {} if the materialised pool is absent, so callers degrade to the old
    behaviour rather than crashing; check for the empty dict if you want to hard-fail.
    """
    global _GSM_GOLD
    if _GSM_GOLD is None:
        path = f"{DATA}/gsm8k_train_authoritative/train.jsonl"
        _GSM_GOLD = {}
        if os.path.exists(path):
            for ln in open(path):
                o = json.loads(ln)
                _GSM_GOLD[o["question"].strip()] = o["gold"]
    return _GSM_GOLD


# ------------------------------------------------------------------ pools ----
def load_pool(source, n, seed=0):
    """(question, gold) list. Delegates eval sets to clean_eval; adds TRAIN-only pools."""
    from star_generate import extract_boxed, norm

    if source in ("gsm8k_train", "math_train", "math_train_hard", "math_train_easy"):
        probs = []
        if source == "gsm8k_train":
            # ⚠️GOLD COMES FROM THE DATASET, NOT FROM A GENERATED SOLUTION. The curated shard's
            # `answer` field is a MODEL-WRITTEN solution, so taking its last \boxed{} as gold
            # inherits both the generator's arithmetic errors and extract_boxed's brace bug.
            # Measured 2026-08-12 against GSM8K's own `#### N`: 280 of 4,229 golds WRONG (6.62%)
            # -- 197 generator errors, 83 nested-brace parses (`\boxed{\$14{,}000}` -> "14").
            # Those golds label every rollout, every never-solved selection and every DPO pair on
            # this line, so a wrong one makes a correct trace score incorrect and can park a
            # solvable problem in the coverage hole permanently. The extractor ALSO drops any row
            # whose generated solution has no \boxed at all, which silently cost 3,152 of GSM8K
            # train's 7,473 problems -- 42% of the largest training pool, invisible because
            # load_pool returned a plausible 4,229.
            auth = f"{DATA}/gsm8k_train_authoritative/train.jsonl"
            if os.path.exists(auth):
                for ln in open(auth):
                    o = json.loads(ln)
                    probs.append((o["question"], o["gold"]))
            else:
                # fallback only -- an offline node without the materialised pool. Same behaviour as
                # before the fix, including its 6.62% wrong-gold rate; the warning is the point.
                print(f"!! WARNING {auth} missing; falling back to curated-shard golds, which are "
                      f"6.62% WRONG and cover only 4,229 of 7,473 train problems", file=sys.stderr)
                path = f"{DATA}/gsm8k_main_curated/shards/shard_00000.jsonl"
                for ln in open(path):
                    o = json.loads(ln)
                    if o.get("split") != "train":  # the §23 contamination rule: TRAIN only
                        continue
                    g = extract_boxed(o["answer"])
                    if g is not None:
                        probs.append((o["question"], g))
        else:
            from datasets import load_from_disk
            d = load_from_disk(f"{DATA}/nlile_hendrycks-MATH-benchmark")["train"]
            lo, hi = {"math_train": (1, 5), "math_train_easy": (1, 3),
                      "math_train_hard": (4, 5)}[source]
            for r in d:
                lvl = str(r.get("level", ""))
                m = re.search(r"(\d)", lvl)
                if not m or not (lo <= int(m.group(1)) <= hi):
                    continue
                cleaned = str(r["answer"]).strip().replace("$", "").replace(",", "").replace(" ", "")
                if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
                    continue                        # numeric-only, so norm() can verify
                g = norm(cleaned)
                if g is not None:
                    probs.append((r["problem"], g))
        random.Random(seed).shuffle(probs)
        return probs[:n] if n and n > 0 else probs

    if source.startswith("jsonl:"):
        # any (question, gold) JSONL as a pool -- used for the distractor-perturbed gsm8k-train
        # robustness pool (build_perturb_pool.py). Never point this at an eval set.
        from star_generate import norm as _norm
        probs = []
        for ln in open(source[len("jsonl:"):]):
            o = json.loads(ln)
            g = o.get("gold")
            g = _norm(str(g)) if g is not None else None
            if g is not None:
                probs.append((o["question"], g))
        random.Random(seed).shuffle(probs)
        return probs[:n] if n and n > 0 else probs

    from clean_eval import load_clean
    return load_clean(source, n, seed=seed)


# ------------------------------------------------------------------ engine ---
def make_llm(model, max_model_len, gpu_util, seed=0, enforce_eager=True):
    """NOTE `enforce_eager` defaults to True to MATCH `clean_eval.py`, the published judge.

    2026-08-02: greedy pass@1 on this model moved 65.0 -> 71.0 on the same 300 SVAMP problems
    purely by changing the engine config (compiled+cudagraph and/or a different max_model_len),
    so the engine config is part of the measurement and is recorded in every JSON this writes.
    """
    import vllm_argonne
    vllm_argonne.register()
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=gpu_util, max_model_len=max_model_len,
              enforce_eager=bool(enforce_eager), seed=seed)
    return llm, tok


def prompt_ids(tok, q, think=True):
    from clean_eval import build_ids
    return build_ids(tok, q, think=think)


def n_think_tokens(tok, text):
    """Tokens strictly inside <think>...</think> (or the whole gen if never closed)."""
    i = text.find("</think>")
    seg = text if i < 0 else text[:i]
    return len(tok.encode(seg, add_special_tokens=False))


def grade_one(text, gold, extract_boxed):
    pred = extract_boxed(text)
    closed = "</think>" in text
    if not closed:
        return "unclosed", pred
    if pred is None:
        return "no_answer", pred
    return ("correct" if pred == gold else "wrong"), pred


# ------------------------------------------------------------------- modes ---
def mode_budget(args, llm, tok, probs, out):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from star_generate import extract_boxed

    ids = [prompt_ids(tok, q) for q, _ in probs]
    golds = [g for _, g in probs]
    rows = []
    per_budget_lens = {}
    for b in args.budgets:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=b)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)
        texts = [o.outputs[0].text for o in outs]
        fm = Counter()
        lens = []
        correct_by_len = []
        for t, g in zip(texts, golds):
            lab, _ = grade_one(t, g, extract_boxed)
            fm[lab] += 1
            L = n_think_tokens(tok, t)
            lens.append(L)
            correct_by_len.append((L, lab == "correct"))
        acc = 100.0 * fm["correct"] / len(golds)
        per_budget_lens[b] = correct_by_len
        rows.append({"budget": b, "acc": acc, "fm": dict(fm),
                     "len_mean": sum(lens) / len(lens),
                     "len_p50": sorted(lens)[len(lens) // 2],
                     "len_p90": sorted(lens)[int(0.9 * len(lens))],
                     "len_max": max(lens)})
        print(f"  budget {b:>5}  acc {acc:6.2f}%  think-len mean {rows[-1]['len_mean']:6.1f} "
              f"p50 {rows[-1]['len_p50']:>4} p90 {rows[-1]['len_p90']:>4} max {rows[-1]['len_max']:>4}  {dict(fm)}",
              flush=True)

    # accuracy binned by trace length at the LARGEST budget (is long thinking a symptom of hard?)
    big = max(args.budgets)
    bins = [(0, 60), (60, 100), (100, 150), (150, 250), (250, 400), (400, 10 ** 9)]
    binned = []
    for lo, hi in bins:
        sel = [c for L, c in per_budget_lens[big] if lo <= L < hi]
        if sel:
            binned.append({"lo": lo, "hi": hi, "n": len(sel),
                           "acc": 100.0 * sum(sel) / len(sel)})
    print(f"\n  accuracy vs think-length (budget={big}):")
    for b in binned:
        print(f"    len [{b['lo']:>4},{b['hi'] if b['hi'] < 10**9 else 'inf'!s:>4})  n={b['n']:>4}  acc {b['acc']:6.2f}%")
    out["budget"] = {"rows": rows, "binned": binned}


def mode_extend(args, llm, tok, probs, out):
    """s1-style forced continuation: suppress `</think>`, inject a cue, regenerate, repeat.

    Implementation note: vLLM cannot 'unclose' text, so each round re-generates from the
    *truncated* prefix (everything before the first `</think>`) + the cue. That keeps every
    round on the model's own distribution and costs one generate per round.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from star_generate import extract_boxed

    ids = [prompt_ids(tok, q) for q, _ in probs]
    golds = [g for _, g in probs]
    cue_ids = tok.encode(args.cue, add_special_tokens=False)
    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False)

    # round 0 = plain greedy
    sp = SamplingParams(n=1, temperature=args.extend_temp, max_tokens=args.think_budget)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)
    gen_ids = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    gen_budget = [len(g) for g in gen_ids]        # running decoded-token count per problem

    def close_and_grade(cur_ids, cur_texts, tag):
        """Force-close whatever thinking exists, read the answer, grade."""
        need, meta = [], []
        finals = list(cur_texts)
        for i, t in enumerate(cur_texts):
            if "</think>" in t and extract_boxed(t) is not None:
                continue
            pre = cur_ids[i]
            if "</think>" not in t:
                pre = pre + close_ids
            need.append(TokensPrompt(prompt_token_ids=ids[i] + pre))
            meta.append((i, "</think>" in t))
        if need:
            sp2 = SamplingParams(n=1, temperature=0.0, max_tokens=args.tail)
            o2 = llm.generate(need, sp2)
            for (i, closed), o in zip(meta, o2):
                finals[i] = cur_texts[i] + ("" if closed else CLOSE_STR) + o.outputs[0].text
        labs = [grade_one(t, g, extract_boxed)[0] for t, g in zip(finals, golds)]
        fm = Counter(labs)
        acc = 100.0 * fm["correct"] / len(golds)
        thinklen = sum(n_think_tokens(tok, t) for t in finals) / len(finals)
        # cumulative DECODED tokens, i.e. the real serving cost of this many extensions -- the
        # final think length understates it, because each round's text is regenerated from a
        # truncated prefix rather than appended to.
        gen = sum(gen_budget) / len(golds) if gen_budget else 0.0
        print(f"  {tag:<14} acc {acc:6.2f}%  mean think-len {thinklen:7.1f}  "
              f"mean decoded {gen:7.1f}  {dict(fm)}", flush=True)
        return acc, labs, dict(fm), thinklen, gen

    acc0, labs0, fm0, tl0, gen0 = close_and_grade(gen_ids, texts, "extend x0")
    rows = [{"round": 0, "acc": acc0, "fm": fm0, "think_len": tl0, "decoded": gen0}]

    cur_ids, cur_texts = gen_ids, texts
    for r in range(1, args.n_extend + 1):
        nxt_ids, nxt_texts = [], []
        reqs = []
        for i, t in enumerate(cur_texts):
            j = t.find("</think>")
            if j < 0:                      # never closed -> just keep thinking
                pre_ids = cur_ids[i]
            else:                          # truncate the close, splice the cue
                pre_txt = t[:j]
                pre_ids = tok.encode(pre_txt, add_special_tokens=False)
            pre_ids = pre_ids + cue_ids
            reqs.append(TokensPrompt(prompt_token_ids=ids[i] + pre_ids))
            nxt_ids.append(pre_ids)
        spx = SamplingParams(n=1, temperature=args.extend_temp, max_tokens=args.extend_tokens)
        ox = llm.generate(reqs, spx)
        for i, o in enumerate(ox):
            add = list(o.outputs[0].token_ids)
            nxt_ids[i] = nxt_ids[i] + add
            gen_budget[i] += len(cue_ids) + len(add)
            nxt_texts.append(tok.decode(nxt_ids[i], skip_special_tokens=True))
        cur_ids, cur_texts = nxt_ids, nxt_texts
        acc, labs, fm, tl, gen = close_and_grade(cur_ids, cur_texts, f"extend x{r}")
        flips = Counter()
        for a, b in zip(labs0, labs):
            flips[f"{'C' if a == 'correct' else 'X'}->{'C' if b == 'correct' else 'X'}"] += 1
        print(f"                 flips vs x0: {dict(flips)}", flush=True)
        rows.append({"round": r, "acc": acc, "fm": fm, "think_len": tl, "decoded": gen,
                     "flips_vs_x0": dict(flips),
                     "net_flip": flips.get("X->C", 0) - flips.get("C->X", 0)})
    out["extend"] = {"cue": args.cue, "extend_tokens": args.extend_tokens, "rows": rows}


def mode_density(args, llm, tok, probs, out):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from star_generate import extract_boxed

    ids = [prompt_ids(tok, q) for q, _ in probs]
    golds = [g for _, g in probs]
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_new_tokens, seed=args.seed)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)

    hist = Counter()
    per = []
    fm = Counter()
    for o, g in zip(outs, golds):
        c = 0
        for cand in o.outputs:
            lab, _ = grade_one(cand.text, g, extract_boxed)
            fm[lab] += 1
            if lab == "correct":
                c += 1
        hist[c] += 1
        per.append(c)
    n = len(per)
    K = args.k
    signal = sum(1 for c in per if 0 < c < K)
    allc = sum(1 for c in per if c == K)
    none = sum(1 for c in per if c == 0)
    pbar = sum(per) / (n * K)
    # group-relative-advantage strength proxy: mean std of a binary group
    import math
    varw = sum(math.sqrt((c / K) * (1 - c / K)) for c in per) / n
    print(f"\n  n={n}  K={K}  T={args.temperature}")
    print(f"  single-sample acc      : {100*pbar:6.2f}%")
    print(f"  SIGNAL groups (0<c<K)  : {100*signal/n:6.2f}%   ({signal}/{n})")
    print(f"  saturated  (c==K)      : {100*allc/n:6.2f}%   ({allc}/{n})")
    print(f"  dead       (c==0)      : {100*none/n:6.2f}%   ({none}/{n})")
    print(f"  mean sqrt(p(1-p))      : {varw:6.4f}   (max 0.5; GRPO advantage scale)")
    print(f"  pass@{K:<3}               : {100*(n-none)/n:6.2f}%")
    print(f"  format mix             : {dict(fm)}")
    print("  correct-count histogram:")
    for c in range(K + 1):
        if hist[c]:
            print(f"    c={c:<3} {hist[c]:>4}  {'#' * min(60, hist[c])}")
    out["density"] = {"n": n, "k": K, "temperature": args.temperature,
                      "single_acc": 100 * pbar, "signal_frac": 100 * signal / n,
                      "saturated_frac": 100 * allc / n, "dead_frac": 100 * none / n,
                      "mean_sqrt_pq": varw, "passk": 100 * (n - none) / n,
                      "hist": {str(k): v for k, v in sorted(hist.items())},
                      "fm": dict(fm), "per_problem": per}


def mode_greedy(args, llm, tok, probs, out):
    """One greedy pass; dumps the exact generations so two engine configs can be text-diffed.

    Exists because a scalar accuracy cannot distinguish "the model changed" from "the kernel
    changed": if two configs disagree on 20/300 problems, the texts tell you immediately whether
    that is a handful of near-tie argmax flips or a systematically different decode.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from star_generate import extract_boxed

    ids = [prompt_ids(tok, q) for q, _ in probs]
    golds = [g for _, g in probs]
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)
    texts = [o.outputs[0].text for o in outs]
    fm = Counter()
    labs, preds = [], []
    for t, g in zip(texts, golds):
        lab, p = grade_one(t, g, extract_boxed)
        fm[lab] += 1
        labs.append(lab)
        preds.append(p)
    acc = 100.0 * fm["correct"] / len(golds)
    tl = sum(n_think_tokens(tok, t) for t in texts) / len(texts)
    print(f"  greedy acc {acc:6.2f}%   mean think-len {tl:7.1f}   {dict(fm)}")
    out["greedy"] = {"acc": acc, "fm": dict(fm), "think_len": tl,
                     "labels": labs, "preds": preds,
                     "texts": texts if args.dump_texts else None}


def mode_passk(args, llm, tok, probs, out):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from star_generate import extract_boxed

    ids = [prompt_ids(tok, q) for q, _ in probs]
    golds = [g for _, g in probs]
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_new_tokens, seed=args.seed)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)
    preds = []
    for o, g in zip(outs, golds):
        row = []
        for cand in o.outputs:
            lab, p = grade_one(cand.text, g, extract_boxed)
            row.append((lab, p))
        preds.append(row)

    rng = random.Random(args.seed)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= args.k]
    rows = []
    for k in ks:
        pk = mj = 0
        for row, g in zip(preds, golds):
            best_pk = best_mj = 0
            for _ in range(args.reps):
                sub = rng.sample(row, k) if k < len(row) else row
                if any(lab == "correct" for lab, _ in sub):
                    best_pk += 1
                votes = Counter(p for lab, p in sub if p is not None and lab in ("correct", "wrong"))
                if votes and votes.most_common(1)[0][0] == g:
                    best_mj += 1
            pk += best_pk / args.reps
            mj += best_mj / args.reps
        n = len(golds)
        rows.append({"k": k, "passk": 100 * pk / n, "majority": 100 * mj / n})
        print(f"  k={k:<4} pass@k {100*pk/n:6.2f}%   majority@k {100*mj/n:6.2f}%", flush=True)
    never = [i for i, row in enumerate(preds) if not any(lab == "correct" for lab, _ in row)]
    print(f"\n  never-solved at K={args.k}: {len(never)}/{len(golds)} = {100*len(never)/len(golds):.2f}%")
    out["passk"] = {"rows": rows, "never_solved": never, "k": args.k,
                    "questions_never": [probs[i][0][:160] for i in never[:40]]}


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["budget", "extend", "density", "passk", "greedy"])
    ap.add_argument("--enforce-eager", type=int, default=1,
                    help="1 = match clean_eval.py (the published judge). 0 = compiled+cudagraph.")
    ap.add_argument("--dump-texts", action="store_true", help="store generations in --json-out")
    ap.add_argument("--pool", default="svamp")
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--reps", type=int, default=8, help="subsample reps for pass@k/majority@k")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--tail", type=int, default=48)
    ap.add_argument("--n-extend", type=int, default=3)
    ap.add_argument("--extend-tokens", type=int, default=192)
    ap.add_argument("--extend-temp", type=float, default=0.0)
    ap.add_argument("--cue", default="\nWait, let me double-check that.\n")
    ap.add_argument("--log", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    probs = load_pool(args.pool, args.n_problems, seed=args.seed)
    header = (f"{'='*78}\nEFFORT PROBE  mode={args.mode}  model={args.model}\n"
              f"  pool={args.pool}  n={len(probs)}  seed={args.seed}\n{'='*78}")
    print(header, flush=True)

    llm, tok = make_llm(args.model, args.max_model_len, args.gpu_util, seed=args.seed,
                        enforce_eager=args.enforce_eager)
    out = {"model": args.model, "mode": args.mode, "pool": args.pool, "n": len(probs),
           "engine": {"enforce_eager": bool(args.enforce_eager),
                      "max_model_len": args.max_model_len, "gpu_util": args.gpu_util,
                      "max_new_tokens": args.max_new_tokens, "seed": args.seed}}
    {"budget": mode_budget, "extend": mode_extend, "density": mode_density,
     "passk": mode_passk, "greedy": mode_greedy}[args.mode](args, llm, tok, probs, out)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    main()
