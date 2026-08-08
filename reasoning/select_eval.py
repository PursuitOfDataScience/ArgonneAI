#!/usr/bin/env python3
"""Selection-strategy sweep for the Argonne-3.0-think models (2026-07-10).

WHY: on CLEAN held-out data (SVAMP/ASDiv) the shipped models show a huge SELECTION gap:
self-consistency captures ~36-51% while pass@32 is ~70-74% (clean_eval.py, 2026-07-09).
~50-60% of sampled traces never close `</think>`, so plain self-consistency THROWS THEM AWAY
(only closed+boxed samples vote). The memory-flagged, never-measured cheap lever is
**budget-forced self-consistency**: force-close EVERY still-open sample, then vote over all K --
recruiting the ~57% unclosed non-voters. This script measures it, plus confidence-weighted
voting, all derived from ONE sampled set (cheap), against the pass@K ceiling.

Strategies compared (same K samples per problem):
  1. plain self-consistency        : majority vote over closed+boxed samples (clean_eval baseline)
  2. conf-weighted self-consistency : vote weighted by exp(mean-token-logprob) over closed+boxed
  3. budget-forced self-consistency : force-close every UNclosed sample (append CLOSE_STR + short
                                      greedy tail), then majority vote over ALL K  <-- the lever
  4. conf-weighted budget-forced SC : (3) but weighted by the sample's mean-token-logprob
  5. pass@K (full) / pass@K (forced): the oracle ceilings for context

Uses the VALIDATED vLLM port and reuses clean_eval's loaders/force-close + star_generate's verifier.
No weights changed. Deployable = strategies 1-4 (no gold needed at inference).
"""
import argparse
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star_generate import extract_boxed, norm  # noqa: E402  verified primitives
from clean_eval import load_clean, build_ids, CLOSE_STR  # noqa: E402  clean loaders + force-close str


def mean_logprob(comp):
    """Geometric-mean token probability of a vLLM CompletionOutput (weight in (0,1]).
    Falls back to 1.0 (uniform) if cumulative_logprob is unavailable."""
    n = len(comp.token_ids) if comp.token_ids is not None else 0
    clp = getattr(comp, "cumulative_logprob", None)
    if clp is None or n == 0:
        return 1.0
    return math.exp(clp / n)


def vote(preds_weights):
    """preds_weights: list of (pred, weight). Return (unweighted_winner, weighted_winner)."""
    unw = Counter()
    w = defaultdict(float)
    for pred, weight in preds_weights:
        unw[pred] += 1
        w[pred] += weight
    unw_win = unw.most_common(1)[0][0] if unw else None
    w_win = max(w.items(), key=lambda kv: kv[1])[0] if w else None
    return unw_win, w_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", nargs="+", default=["svamp", "asdiv", "gsm8k"])
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--budget-tail", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--dump-preds", default=None,
                    help="write per-problem {golds, preds_full, preds_forced} JSON for offline null control")
    args = ap.parse_args()
    dump = {} if args.dump_preds else None

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
    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)

    out("=" * 84)
    out(f"SELECT EVAL  model={args.model}")
    out(f"  n={args.n_problems}/source  K={args.k}  temp={args.temperature}  seed={args.seed}")
    out("=" * 84)

    summary = []
    for src in args.sources:
        probs = load_clean(src, args.n_problems, seed=args.seed)
        if not probs:
            out(f"  [{src}] NO PROBLEMS -- skipped"); continue
        golds = [g for _, g in probs]
        pids = [build_ids(tok, q, think=True) for q, _ in probs]
        tag = "CONTAMINATED" if src.startswith("gsm8k") else "clean"

        # ---- ONE sampled set: K per problem, full length, with logprobs ----
        sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_tokens=args.max_new_tokens, logprobs=1)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)

        # per problem: list of dicts {text, closed, pred, weight, tokens}
        samples = []
        for o in outs:
            row = []
            for c in o.outputs:
                t = c.text
                closed = "</think>" in t
                row.append({"text": t, "closed": closed, "pred": extract_boxed(t),
                            "weight": mean_logprob(c), "tokens": list(c.token_ids)})
            samples.append(row)

        # ---- force-close the UNclosed subset (append CLOSE_STR + short greedy tail) ----
        need, meta = [], []
        for pi, row in enumerate(samples):
            for j, s in enumerate(row):
                good = s["closed"] and s["pred"] is not None
                if good:
                    continue
                cont = pids[pi] + s["tokens"] + (close_ids if not s["closed"] else [])
                need.append(TokensPrompt(prompt_token_ids=cont))
                meta.append((pi, j))
        if need:
            sp2 = SamplingParams(n=1, temperature=0.0, max_tokens=args.budget_tail)
            outs2 = llm.generate(need, sp2)
            for (pi, j), o in zip(meta, outs2):
                s = samples[pi][j]
                forced_text = s["text"] + ("" if s["closed"] else CLOSE_STR) + o.outputs[0].text
                s["forced_pred"] = extract_boxed(forced_text)
        for row in samples:
            for s in row:
                if "forced_pred" not in s:      # was already closed+boxed
                    s["forced_pred"] = s["pred"]

        # ---- score every strategy on the same samples ----
        n = len(probs)
        single_ok = single_tot = 0
        c_passk = c_passk_f = 0
        c_sc = c_wsc = c_bfsc = c_wbfsc = 0
        voters_full = voters_bf = 0
        for row, gold in zip(samples, golds):
            # single-sample acc + full pass@K (closed+boxed only, as clean_eval)
            closed_correct = False
            for s in row:
                single_tot += 1
                if s["closed"] and s["pred"] == gold:
                    single_ok += 1; closed_correct = True
            if closed_correct:
                c_passk += 1
            # forced pass@K (any forced sample correct)
            if any(s["forced_pred"] == gold for s in row):
                c_passk_f += 1
            # plain / weighted self-consistency over closed+boxed
            pw = [(s["pred"], s["weight"]) for s in row if s["closed"] and s["pred"] is not None]
            voters_full += len(pw)
            sc, wsc = vote(pw)
            c_sc += (sc == gold); c_wsc += (wsc == gold)
            # budget-forced self-consistency over ALL K (forced_pred)
            pwf = [(s["forced_pred"], s["weight"]) for s in row if s["forced_pred"] is not None]
            voters_bf += len(pwf)
            bfsc, wbfsc = vote(pwf)
            c_bfsc += (bfsc == gold); c_wbfsc += (wbfsc == gold)

        out(f"\n  ---- [{src}]  ({tag})  n={n}  K={args.k} ----")
        out(f"    single-sample acc (T={args.temperature})   : {100*single_ok/max(single_tot,1):.2f}%")
        out(f"    voters/problem  full={voters_full/n:.1f}  forced={voters_bf/n:.1f}  (of K={args.k})")
        out(f"    [1] self-consistency (plain)     : {100*c_sc/n:.2f}%")
        out(f"    [2] self-consistency (conf-wt)   : {100*c_wsc/n:.2f}%")
        out(f"    [3] BUDGET-FORCED self-cons      : {100*c_bfsc/n:.2f}%   <-- lever")
        out(f"    [4] budget-forced (conf-wt)      : {100*c_wbfsc/n:.2f}%")
        out(f"    pass@{args.k} full / forced        : {100*c_passk/n:.2f}% / {100*c_passk_f/n:.2f}%")
        summary.append((src, tag, 100*c_sc/n, 100*c_wsc/n, 100*c_bfsc/n, 100*c_wbfsc/n,
                        100*c_passk/n))
        if dump is not None:
            sp_ = lambda x: None if x is None else str(x)
            dump[src] = {
                "golds": [sp_(g) for g in golds],
                "preds_full": [[sp_(s["pred"]) for s in row] for row in samples],
                "preds_forced": [[sp_(s["forced_pred"]) for s in row] for row in samples],
            }

    out("\n" + "=" * 84)
    out(f"  {'source':<12} {'kind':<13} {'SC':>7} {'wSC':>7} {'BF-SC':>7} {'wBF-SC':>8} {'pass@K':>8}")
    for s, t, a, b, c, d, e in summary:
        out(f"  {s:<12} {t:<13} {a:>6.2f}% {b:>6.2f}% {c:>6.2f}% {d:>7.2f}% {e:>7.2f}%")
    out("=" * 84)
    if dump is not None:
        import json
        json.dump(dump, open(args.dump_preds, "w"))
        out(f"  [dumped per-problem preds -> {args.dump_preds}]")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
