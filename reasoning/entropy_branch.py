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

# THE SELECTORS. `vote` is what self-consistency does today and what caps this model: on a4's own
# rollouts gold is the plurality answer in only 65.2% of the problems it can solve at all. The other
# three are reward-model-free alternatives from the self-certainty line (arXiv 2502.18581), which
# reports self-certainty + Borda beating self-consistency on GSM8K/MATH and, unlike perplexity,
# continuing to improve as N grows. Self-certainty is KL(uniform || p) averaged over tokens, i.e.
# log V - H(p) up to a constant, so ranking by MINIMUM mean entropy is ranking by maximum
# self-certainty -- `ent` is that argmax pick, `borda` and `wvote` combine it with the vote.
SELECTORS = ("vote", "ent", "lp", "borda", "wvote", "vtb")

# ⚠️`vtb` (vote-then-tie-break) is the selector with the strongest prior, and the reason is a
# measurement, not taste. Over the 6,565 train problems where gold appears among the answered
# candidates, the vote already picks gold in 66.5% and loses in 33.5% -- and among those losses
# **78.2% are near-ties: 38.6% are EXACT ties (margin 0) and 39.6% are margin 1.** Gold carries just
# 1 of 8 votes in 70.6% of the losses, but so does the wrong answer that beats it: a4's answer
# distribution over 8 samples is FRAGMENTED, so plurality is close to arbitrary there -- an exact tie
# is currently broken by Counter.most_common insertion order.
# So a selector does not need to be a good verifier. It needs to be slightly better than a coin flip
# on near-ties. `vtb` restricts the confidence signal to exactly that decision and never lets it
# override a clear plurality, which makes it strictly lower-variance than `ent`/`borda`/`wvote`.

# A sixth selector, behind --selfverify, and the last FREE one available. §41h killed every text
# feature; self-certainty is a property of the generating distribution. This asks the model a
# different question instead: shown a problem and a candidate's answer, how much mass does it put on
# "Yes" vs "No"? No training, no second model. The prior is mixed -- §22i measured a LEARNED verifier
# failing on this line, and the 2026 process-verification result reports meta-cognition "amplifying
# confusion without sufficient model capacity" at small scale -- but zero-shot p(Yes) as a RERANKER
# has never been measured here, and it is the difference between "no free selector exists" and "we
# did not look".
VERIFY_TMPL = ("Problem: {q}\n\nProposed answer: {a}\n\n"
               "Is the proposed answer correct? Reply with exactly one word, Yes or No.")


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
    ap.add_argument("--tie-slack", type=int, default=1,
                    help="vtb considers any answer within this many votes of the plurality. 1 covers "
                         "78.2% of the vote's measured losses on a4's own rollouts.")
    ap.add_argument("--selfverify", type=int, default=0,
                    help="also score every candidate by the model's own zero-shot p(Yes) that the "
                         "answer is correct, and add the selfvfy / vfyvote selectors")
    ap.add_argument("--control-n", type=int, default=0,
                    help="0 = compute-match the branch arm; >0 for the selector study (e.g. 8)")
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
        fracs = sorted(b / max(1, len(g_ids[i])) for i, b in enumerate(trig) if b is not None)
        med = fracs[len(fracs) // 2] * 100 if fracs else 0.0
        print(f"[eb/{pool}] triggered on {n_trig}/{len(probs)} "
              f"({n_trig / len(probs) * 100:.1f}%); median trigger at {med:.0f}% of the trace",
              flush=True)

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

        # ---- control: independent full samples ------------------------------------------
        # At --control-n = len(CUES) this is COMPUTE-MATCHED to the branch arm, which is what makes
        # "branching beat greedy" mean anything. Raising it turns the same run into the selector
        # study: self-certainty's reported advantage over plurality voting GROWS with N, and a4's
        # vote is the thing that caps at 65.2% of its recoverable headroom.
        n_ctrl = a.control_n or len(CUES)
        spc = SamplingParams(n=n_ctrl, temperature=a.control_temp, top_p=0.95,
                             max_tokens=a.max_tokens, logprobs=a.n_logprobs)
        couts = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], spc)
        ctrl = defaultdict(list)
        for i, o in enumerate(couts):
            ctrl[i].append((g_text[i], mean_ent(g_lps[i]), mean_lp(g_lps[i], g_ids[i]), "greedy"))
            for j, c in enumerate(o.outputs):
                ctrl[i].append((c.text, mean_ent(c.logprobs),
                                mean_lp(c.logprobs, list(c.token_ids)), f"samp{j}"))

        # ---- optional: zero-shot self-verification score per candidate --------------------
        # Exact, not top-k: build the verification prompt twice, once ending in "Yes" and once in
        # "No", and read the logprob vLLM reports for that final PROMPT token. Reading a generated
        # top-20 instead would silently score nothing on candidates where neither word makes the
        # cut, which is exactly the case a weak verifier produces.
        # ⚠️Scored for BOTH candidate sets and keyed by (set, item, candidate). Keying by
        # (item, candidate) alone silently reuses the branch arm's scores for the control arm, whose
        # candidate j is a different trace entirely -- which would have made the control's selfvfy
        # column meaningless while looking perfectly plausible.
        vscore = {}
        if a.selfverify:
            def vprompt(q, ans, word):
                msg = [{"role": "user", "content": VERIFY_TMPL.format(q=q, a=ans)}]
                ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if len(ids) and isinstance(ids[0], (list, tuple)):
                    ids = ids[0]
                return [int(x) for x in ids] + tok.encode(word, add_special_tokens=False)

            vreqs, vmeta = [], []
            for name, src in (("branch", cands), ("control", ctrl)):
                for i in range(len(probs)):
                    for ci, (t, _, _, _) in enumerate(src[i]):
                        ans = extract_boxed(t)
                        if ans is None:
                            continue
                        for w in ("Yes", "No"):
                            vreqs.append(TokensPrompt(
                                prompt_token_ids=vprompt(probs[i][0], ans, w)))
                            vmeta.append((name, i, ci, w))
            if vreqs:
                vouts = llm.generate(vreqs, SamplingParams(n=1, temperature=0.0, max_tokens=1,
                                                           prompt_logprobs=0))
                raw = {}
                for (name, i, ci, w), o in zip(vmeta, vouts):
                    pl = o.prompt_logprobs
                    lp = -20.0
                    if pl and pl[-1]:
                        e = pl[-1].get(o.prompt_token_ids[-1])
                        if e is not None:
                            lp = e.logprob
                    raw[(name, i, ci, w)] = lp
                for (name, i, ci, w) in list(raw):
                    if w != "Yes":
                        continue
                    y, n_ = raw[(name, i, ci, "Yes")], raw.get((name, i, ci, "No"))
                    if n_ is None:
                        continue
                    vscore[(name, i, ci)] = math.exp(y) / (math.exp(y) + math.exp(n_) + 1e-12)
                got = len(vscore)
                print(f"[eb/{pool}] self-verification scored {got} candidates over both arms; "
                      f"mean p(Yes) {sum(vscore.values()) / max(1, got):.3f}", flush=True)

        # ---- selectors -------------------------------------------------------------------
        def score(pool_cands, key, force_close, setname="branch"):
            ok, fm, picks = [], Counter(), Counter()
            for i in range(len(probs)):
                cs = pool_cands[i]
                if force_close:
                    cs = [(t if "</think>" in t else t + CLOSE + "}$.", e, l, g) for t, e, l, g in cs]
                answers = [extract_boxed(t) for t, _, _, _ in cs]
                if key == "ent":
                    best = min(cs, key=lambda c: c[1])
                elif key == "lp":
                    best = max(cs, key=lambda c: c[2])
                elif key in ("vtb", "vtb_vfy"):
                    votes = Counter(aa for aa in answers if aa is not None)
                    if not votes:
                        best = cs[0]
                    else:
                        tc = max(votes.values())
                        near = [x for x, c in votes.items() if tc - c <= a.tie_slack]
                        def strength(x):
                            js = [j for j, aa in enumerate(answers) if aa == x]
                            if key == "vtb_vfy":
                                return max(vscore.get((setname, i, j), -1.0) for j in js)
                            return -min(cs[j][1] for j in js)     # -min entropy = max certainty
                        top = max(near, key=strength)
                        best = min((c for c, aa in zip(cs, answers) if aa == top),
                                   key=lambda c: c[1])
                elif key == "selfvfy":
                    # highest p(Yes) wins; candidates the verifier could not score fall back to
                    # their self-certainty so the arm is never decided by a missing score
                    best = cs[max(range(len(cs)),
                                  key=lambda j: (vscore.get((setname, i, j), -1.0), -cs[j][1]))]
                elif key == "vfyvote":
                    tally = Counter()
                    for j, aa in enumerate(answers):
                        if aa is not None:
                            tally[aa] += vscore.get((setname, i, j), 0.0)
                    if tally and max(tally.values()) > 0:
                        top = tally.most_common(1)[0][0]
                        best = min((c for c, bb in zip(cs, answers) if bb == top),
                                   key=lambda c: c[1])
                    else:
                        best = cs[0]
                elif key in ("borda", "wvote"):
                    # aggregate over ANSWERS, weighting each candidate by its self-certainty:
                    # Borda by rank (robust to the scale of the confidence metric), wvote by the
                    # raw certainty value. Both fall back to the plurality when no candidate
                    # produced an answer at all.
                    order = sorted(range(len(cs)), key=lambda j: cs[j][1])   # most certain first
                    tally = Counter()
                    for rank, j in enumerate(order):
                        if answers[j] is None:
                            continue
                        w = (len(cs) - rank) if key == "borda" else max(0.0, 10.0 - cs[j][1])
                        tally[answers[j]] += w
                    if tally:
                        top = tally.most_common(1)[0][0]
                        best = min((c for c, aa in zip(cs, answers) if aa == top),
                                   key=lambda c: c[1])
                    else:
                        best = cs[0]
                else:                                  # plurality of the candidates' answers
                    votes = Counter(aa for aa in answers if aa is not None)
                    if votes:
                        top = votes.most_common(1)[0][0]
                        best = next(c for c, aa in zip(cs, answers) if aa == top)
                    else:
                        best = cs[0]
                lab, _ = grade_one(best[0], golds[i], extract_boxed)
                ok.append(lab == "correct")
                fm[lab] += 1
                picks[best[3]] += 1
            return ok, fm, picks

        # §41l's premise, re-measured on THESE pools rather than assumed from the train pools:
        # among the problems where the vote loses but gold is present, how far behind is gold? If the
        # losses are not tie-heavy here, `vtb` has little to work with and that must be visible in
        # the same report as its score.
        marg = Counter()
        n_recov = n_votewin = 0
        for i in range(len(probs)):
            answers = [extract_boxed(t) for t, _, _, _ in ctrl[i]]
            v = Counter(x for x in answers if x is not None)
            if not v or golds[i] not in v:
                continue
            n_recov += 1
            tc = max(v.values())
            if v.most_common(1)[0][0] == golds[i]:
                n_votewin += 1
            else:
                marg[min(tc - v[golds[i]], 4)] += 1
        nl = sum(marg.values())
        if nl:
            near = marg[0] + marg[1]
            print(f"[eb/{pool}] vote-loss margins over {n_recov} recoverable items: "
                  f"vote wins {n_votewin} ({n_votewin / n_recov * 100:.1f}%), loses {nl}; "
                  f"of the losses margin0 {marg[0] / nl * 100:.1f}% margin1 {marg[1] / nl * 100:.1f}% "
                  f"-> near-ties {near / nl * 100:.1f}% (train pools measured 78.2%)", flush=True)

        cfgs = {}
        keys = list(SELECTORS) + (["selfvfy", "vfyvote", "vtb_vfy"]
                                  if a.selfverify and vscore else [])
        for name, src in (("branch", cands), ("control", ctrl)):
            for key in keys:
                for fc, sfx in ((False, ""), (True, "+bud")):
                    ok, fm, picks = score(src, key, fc, name)
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
