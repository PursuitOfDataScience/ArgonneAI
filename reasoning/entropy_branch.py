#!/usr/bin/env python3
"""Spend the sampling budget WHERE the trace goes wrong, not on eight more whole traces.

WHY. argonne4-think's headroom is selection, not knowledge: greedy 43.4 under pass@8 68.9, and
`fail_taxonomy.py` on 93,912 on-policy rollouts says 67.9% of its wrong traces state no false
equation and never touch gold -- a coherent derivation of the wrong thing. Self-consistency@8
recovers only 7.5 of those 25.5 points because eight independent rollouts re-make the same early
decision eight times: gold is the plurality answer in just 65.2% of the problems the model can
solve at all.

The 2026 failure-dynamics literature says where to spend the budget instead: >85% of failure onsets
land in the first 30% of a trajectory, 43.5% of wrong traces contain exactly ONE invalid segment,
invalid segments carry a local ENTROPY SPIKE (p<0.001), and >20% of failed trajectories reach the
right answer from an alternative continuation of the SAME PREFIX. That last number is the whole
argument: the prefix is fine, one transition is not.

So: decode greedily once, find the first step boundary whose token entropy exceeds the 90th
percentile of the entropy seen so far, and re-run only the tail from there -- three short branches
(continue / "Wait," / "Let me reconsider:") -- then keep the most confident continuation.

WHAT MAKES THIS A REAL EXPERIMENT AND NOT A FLATTERING ONE. Three branches cost more than one
greedy pass, so beating greedy proves nothing on its own. Every run therefore also scores a
COMPUTE-MATCHED control: the same number of independent full samples under the same selector. If
branching is not better than resampling, the entropy targeting bought nothing and the honest
report is that it is just extra compute. Both are printed side by side, with per-item `ok` arrays
so the comparison is paired.

  python reasoning/entropy_branch.py --model <dir> --pools asdiv svamp --n 500 \
      --json-out report/a4_entbranch.json
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CUES = ["", "Wait, ", "Let me reconsider: "]     # branch 0 is a plain continuation
CLOSE = "\n</think>\n\nThe answer is $\\boxed{"


def entropy_from_logprobs(d):
    """Shannon entropy of the top-k slice, renormalised. vLLM only returns the top-k, so this is
    a lower bound on the true entropy -- fine for a RELATIVE trigger (percentile of its own
    history), which is what the trigger is."""
    if not d:
        return 0.0
    lps = [v.logprob for v in d.values()]
    m = max(lps)
    ws = [math.exp(x - m) for x in lps]
    z = sum(ws)
    ps = [w / z for w in ws]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def is_boundary(text):
    """A reasoning-step boundary: a newline, or the end of a sentence/clause."""
    t = text.strip()
    return ("\n" in text) or t.endswith((".", ":", "?", "!", ";"))


def find_trigger(ent, toks, tok, quantile, min_hist, stride):
    """First step boundary whose entropy exceeds the `quantile` of everything before it.

    Adaptive and instance-specific on purpose: a fixed entropy threshold means something
    different on a 12-token arithmetic step and a 200-token algebra derivation.
    """
    if len(ent) < min_hist + 2:
        return None
    for i in range(min_hist, len(ent) - 1):
        txt = tok.decode([toks[i]])
        if not (is_boundary(txt) or (stride and i % stride == 0)):
            continue
        hist = sorted(ent[:i])
        thr = hist[min(len(hist) - 1, int(quantile * len(hist)))]
        if ent[i] > thr:
            return i
    return None


def mean_ent(lps):
    es = [entropy_from_logprobs(d) for d in (lps or []) if d]
    return sum(es) / len(es) if es else 9e9


def mean_lp(lps, ids):
    if not lps:
        return -9e9
    tot, n = 0.0, 0
    for d, t in zip(lps, ids):
        if d and t in d:
            tot += d[t].logprob
            n += 1
    return tot / n if n else -9e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", nargs="+", default=["asdiv", "svamp"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--branch-tokens", type=int, default=320)
    ap.add_argument("--n-logprobs", type=int, default=20)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--min-hist", type=int, default=16)
    ap.add_argument("--stride", type=int, default=0,
                    help="also allow a trigger every Nth token (0 = boundaries only)")
    ap.add_argument("--control-temp", type=float, default=0.8)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from effort_probe import load_pool, make_llm, prompt_ids, grade_one
    from star_generate import extract_boxed, norm

    llm, tok = make_llm(a.model, gpu_util=a.gpu_util, max_model_len=a.max_model_len, seed=a.seed)
    res = {}

    for pool in a.pools:
        probs = load_pool(pool, a.n, seed=a.seed)
        golds = [norm(str(g)) for _, g in probs]
        pids = [prompt_ids(tok, q) for q, _ in probs]
        t0 = time.time()

        # ---- pass A: one greedy trace, with per-token top-k logprobs ----------------------
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens,
                            logprobs=a.n_logprobs)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
        g_text = [o.outputs[0].text for o in outs]
        g_ids = [list(o.outputs[0].token_ids) for o in outs]
        g_lps = [o.outputs[0].logprobs for o in outs]
        g_ent = [[entropy_from_logprobs(d) for d in (lp or [])] for lp in g_lps]

        greedy_ok = [grade_one(t, g, extract_boxed)[0] == "correct" for t, g in zip(g_text, golds)]
        greedy_fm = Counter(grade_one(t, g, extract_boxed)[0] for t, g in zip(g_text, golds))

        # ---- find the trigger and build the branch prompts --------------------------------
        trig, reqs, meta = [], [], []
        for i in range(len(probs)):
            b = find_trigger(g_ent[i], g_ids[i], tok, a.quantile, a.min_hist, a.stride)
            trig.append(b)
            if b is None:
                continue
            pre = g_ids[i][:b]
            for ci, cue in enumerate(CUES):
                cue_ids = tok.encode(cue, add_special_tokens=False) if cue else []
                reqs.append(TokensPrompt(prompt_token_ids=pids[i] + pre + cue_ids))
                meta.append((i, ci, len(pre), len(cue_ids)))
        n_trig = sum(1 for b in trig if b is not None)
        print(f"[eb/{pool}] triggered on {n_trig}/{len(probs)} "
              f"({n_trig / len(probs) * 100:.1f}%); median trigger at "
              f"{sorted(b / max(1, len(g_ids[i])) for i, b in enumerate(trig) if b is not None)[max(0, n_trig // 2)] * 100 if n_trig else 0:.0f}% "
              f"of the trace", flush=True)

        spb = SamplingParams(n=1, temperature=0.0, max_tokens=a.branch_tokens,
                             logprobs=a.n_logprobs)
        bouts = llm.generate(reqs, spb) if reqs else []

        # candidate 0 for every problem is the original greedy trace
        cands = defaultdict(list)     # i -> [(text, mean_ent, mean_lp, tag)]
        for i in range(len(probs)):
            cands[i].append((g_text[i], mean_ent(g_lps[i]), mean_lp(g_lps[i], g_ids[i]), "greedy"))
        for (i, ci, npre, ncue), o in zip(meta, bouts):
            pre_txt = tok.decode(g_ids[i][:npre])
            full = pre_txt + CUES[ci] + o.outputs[0].text
            lp = o.outputs[0].logprobs
            cands[i].append((full, mean_ent(lp), mean_lp(lp, list(o.outputs[0].token_ids)),
                             f"branch{ci}"))

        # ---- COMPUTE-MATCHED control: len(CUES) independent full samples ------------------
        spc = SamplingParams(n=len(CUES), temperature=a.control_temp, top_p=0.95,
                             max_tokens=a.max_tokens, logprobs=a.n_logprobs)
        couts = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], spc)
        ctrl = defaultdict(list)
        for i, o in enumerate(couts):
            ctrl[i].append((g_text[i], mean_ent(g_lps[i]), mean_lp(g_lps[i], g_ids[i]), "greedy"))
            for j, c in enumerate(o.outputs):
                ctrl[i].append((c.text, mean_ent(c.logprobs),
                                mean_lp(c.logprobs, list(c.token_ids)), f"samp{j}"))

        # ---- selectors -------------------------------------------------------------------
        def score(pool_cands, key, force_close):
            ok, fm, picks = [], Counter(), Counter()
            for i in range(len(probs)):
                cs = pool_cands[i]
                if force_close:
                    cs = [(t if "</think>" in t else t + CLOSE + "}$.", e, l, g) for t, e, l, g in cs]
                if key == "ent":
                    best = min(cs, key=lambda c: c[1])
                elif key == "lp":
                    best = max(cs, key=lambda c: c[2])
                else:                                  # plurality of the candidates' answers
                    votes = Counter()
                    for t, _, _, _ in cs:
                        p = extract_boxed(t)
                        if p is not None:
                            votes[p] += 1
                    if votes:
                        top = votes.most_common(1)[0][0]
                        best = next(c for c in cs if extract_boxed(c[0]) == top)
                    else:
                        best = cs[0]
                lab, _ = grade_one(best[0], golds[i], extract_boxed)
                ok.append(lab == "correct")
                fm[lab] += 1
                picks[best[3]] += 1
            return ok, fm, picks

        cfgs = {}
        for name, src in (("branch", cands), ("control", ctrl)):
            for key in ("ent", "lp", "vote"):
                for fc, sfx in ((False, ""), (True, "+bud")):
                    ok, fm, picks = score(src, key, fc)
                    cfgs[f"{name}_{key}{sfx}"] = {"ok": ok, "fm": dict(fm), "picks": dict(picks)}
        cfgs["greedy"] = {"ok": greedy_ok, "fm": dict(greedy_fm), "picks": {}}
        # the pass@ceiling of the branch set: does ANY branch reach gold?
        for name, src in (("branch", cands), ("control", ctrl)):
            anyok = []
            for i in range(len(probs)):
                anyok.append(any(grade_one(t, golds[i], extract_boxed)[0] == "correct"
                                 for t, _, _, _ in src[i]))
            cfgs[f"{name}_oracle"] = {"ok": anyok, "fm": {}, "picks": {}}

        res[pool] = {"n": len(probs), "n_trigger": n_trig, "cfgs": cfgs,
                     "secs": time.time() - t0}
        print(f"\n=== {pool}  n={len(probs)}  triggered={n_trig}  "
              f"{time.time() - t0:.0f}s ===")
        base = sum(greedy_ok) / len(greedy_ok) * 100
        for k in sorted(cfgs):
            acc = sum(cfgs[k]["ok"]) / len(probs) * 100
            print(f"  {k:20s} {acc:6.2f}  ({acc - base:+5.2f} vs greedy)  "
                  f"{cfgs[k].get('picks', {})}")

    if a.json_out:
        json.dump({"model": a.model, "args": vars(a), "res": res},
                  open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")

    print("\n" + "=" * 70)
    print("POOL-MEAN")
    keys = sorted(set(k for p in res for k in res[p]["cfgs"]))
    for k in keys:
        vals = [sum(res[p]["cfgs"][k]["ok"]) / res[p]["n"] * 100 for p in res if k in res[p]["cfgs"]]
        gvals = [sum(res[p]["cfgs"]["greedy"]["ok"]) / res[p]["n"] * 100 for p in res]
        print(f"  {k:20s} {sum(vals) / len(vals):6.2f}  "
              f"({sum(vals) / len(vals) - sum(gvals) / len(gvals):+5.2f} vs greedy)")


if __name__ == "__main__":
    main()
