#!/usr/bin/env python3
"""Build a SELF-VERIFICATION / SELF-CORRECTION tier from a labeled-rollout corpus.

THE PROBLEM THIS ATTACKS (§33 Phase-0, 2026-08-02). The shipped argonne-3.5-think has no usable
"think harder" knob, and it is not merely absent -- it is NEGATIVE:

  s1-style forced continuation (suppress `</think>`, inject a cue, regenerate), n=300 clean:
    asdiv greedy   x0 73.3% -> x1 69.7% -> x3 69.0%      net flips at x3: C->X 22 vs X->C 9
    svamp greedy   x0 65.7% -> x1 65.7% -> x3 64.0%      net flips at x3: C->X 12 vs X->C  7
    asdiv T=0.7    x0 67.7% -> x1 58.3% -> x3 57.3%      net flips at x3: C->X 47 vs X->C 16
  and accuracy is INVERSELY related to trace length: svamp [60-100) tok = 88% correct,
  [100-150) = 57%, [250-400) = 25%, [400+) = 0%.

So more sequential compute makes this model worse, and long thinking is a symptom of failure
rather than a route out of it. That is a direct consequence of how it was built: §23e/v6 trained
it on SHORT closed correct traces ONLY, which is exactly what fixed non-termination (`unclosed`
53.7%->1.3%) and is also exactly why continuing a trace is off-distribution.

THE FIX ATTEMPTED HERE. Give the model a *trained* continuation mode, keyed on the SAME cue
string the probe injects, in two flavours built from its own rollouts:

  verify_confirm   correct trace -> cue -> a mechanically-generated recheck of every `a op b = c`
                   step (each one re-evaluated in python, so the check text cannot lie) -> answer.
                   Teaches: extra tokens are for re-checking what you already computed.
  verify_rederive  correct trace A -> cue -> a *step-signature-distinct* correct trace B for the
                   same problem -> answer. Teaches: on the cue, derive it a second independent way
                   and commit when they agree -- i.e. self-consistency folded into ONE pass, which
                   matters because majority@2 already beats greedy on this model (svamp pass@2
                   72.2% vs greedy 65.0%).
  verify_fix       a WRONG rollout -> cue -> honest transition -> the model's own CORRECT rollout
                   -> answer. Teaches: when the check fails, re-derive instead of committing.

HONESTY RULE (this cost a redesign). The first build clipped each wrong trace to its leading 60%
and appended "That line of reasoning does not hold up" -- for 3,054 of 3,500 rows, i.e. wherever
no arithmetic error could be *located*. But a wrong trace's prefix is frequently correct (the error
comes later), so those rows were telling the model to abandon sound reasoning -- plausibly the very
behaviour that makes its forced continuations lose 2-3 correct answers for each one they gain. Now
a row may only assert what was verified in python: `verify_fix` either NAMES a specific
arithmetically-wrong step it located, or keeps the whole wrong attempt (which demonstrably did
reach a wrong answer) behind a neutral "let me recompute this independently".

Both end in the deployed close, so termination pressure is preserved. Nothing in the generated
verification text is asserted unless it was verified in python -- the tier is correct-by
construction, which is the property the §10/§23e targeted tiers had and the STaR tiers did not.

The falsifiable prediction, which `effort_probe.py --mode extend` tests directly: after training
on this tier, extension should stop being net-negative. If x1 still loses to x0, the tier failed
and no amount of accuracy improvement elsewhere should be attributed to "reasoning effort".
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rft_generate import EQ, _val, has_bad_arith, is_degenerate, step_signature  # noqa: E402

CUE = "\nWait, let me double-check that.\n"
CLOSE_FMT = "\n</think>\n\nThe answer is $\\boxed{%s}$."
SENT_END = re.compile(r"(?<=[.!?\n])\s+")


def fmt_num(x):
    """Render a float without a trailing .0 so check lines read like the model's own arithmetic."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def recheck_text(think, gold, max_steps=6):
    """Mechanically re-verify every explicit equation. Returns None if there is nothing to check.

    Every line emitted is computed here in python, so this text is true by construction. Steps
    that do NOT verify are skipped rather than "corrected", because a confirm-flavour row must
    only ever contain confirmations (the trace it came from is already gold-correct).
    """
    lines = []
    seen = set()
    for m in EQ.finditer(think):
        try:
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        v = _val(a, op, b)
        if v is None or abs(v - c) > 1e-6:
            continue
        key = (a, op.lower(), b, c)
        if key in seen:
            continue
        seen.add(key)
        o = {"x": "*", "X": "*", "×": "*"}.get(op, op)
        lines.append(f"{fmt_num(a)} {o} {fmt_num(b)} = {fmt_num(v)} ✓")
        if len(lines) >= max_steps:
            break
    if not lines:
        return None
    return ("Let me recheck each step: " + "; ".join(lines) +
            f". Every step checks out, so the answer is {gold}.")


def truncate_at_first_bad(think):
    """Cut a wrong think block just after the sentence holding its first bad equation.

    Keeping the WHOLE wrong attempt would teach the model to produce a full wrong derivation
    before every answer -- doubling trace length and re-importing the over-thinking v6 removed.
    Returns (prefix, bad_desc) or None when the trace has no locatable arithmetic error.
    """
    for m in EQ.finditer(think):
        try:
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        v = _val(a, op, b)
        if v is None or abs(v - c) <= 1e-6:
            continue
        end = m.end()
        nxt = SENT_END.search(think, end)
        cut = nxt.end() if nxt else min(len(think), end + 1)
        o = {"x": "*", "X": "*", "×": "*"}.get(op, op)
        desc = (f"I wrote {fmt_num(a)} {o} {fmt_num(b)} = {fmt_num(c)}, but "
                f"{fmt_num(a)} {o} {fmt_num(b)} = {fmt_num(v)}. Let me redo this properly.")
        return think[:cut].rstrip(), desc
    return None


def clip_words(text, frac):
    """Keep the leading `frac` of a think block, rounded to a sentence boundary."""
    target = int(len(text) * frac)
    cuts = [m.end() for m in SENT_END.finditer(text) if m.end() <= max(target, 40)]
    return text[:cuts[-1]].rstrip() if cuts else text[:target].rstrip()


def strip_think(trace):
    """Return the inner text of <think>...</think>, or None."""
    t = trace.strip()
    i = t.find("</think>")
    if i < 0:
        return None
    s = t[:i]
    j = s.find("<think>")
    return s[j + len("<think>"):].strip("\n") if j >= 0 else s.strip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-jsonl", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="/project/rcc/youzhi/models/a35_reason/blend_a085")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--n-confirm", type=int, default=3000)
    ap.add_argument("--n-rederive", type=int, default=3000)
    ap.add_argument("--n-fix", type=int, default=3000)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--max-tok", type=int, default=1024,
                    help="higher than the v6 cap of 768 ON PURPOSE: a verify row is a short trace "
                         "plus its recheck, and clipping it would truncate the thing being taught")
    ap.add_argument("--min-think-tok", type=int, default=48)
    ap.add_argument("--wrong-clip-frac", type=float, default=0.6,
                    help="when a wrong trace has no locatable bad equation, keep this leading share")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    by_q = defaultdict(lambda: {"pool": None, "gold": None, "correct": [], "wrong": []})
    for p in args.all_jsonl:
        for ln in open(p):
            o = json.loads(ln)
            e = by_q[o["question"]]
            e["pool"] = o.get("pool", "?")
            e["gold"] = o["gold"]
            if o["label"] == "correct":
                e["correct"].append(o["trace"])
            elif o["label"] == "wrong":
                e["wrong"].append(o["trace"])
    print(f"problems with >=1 correct: {sum(1 for e in by_q.values() if e['correct'])}/{len(by_q)}")
    print(f"problems with correct AND wrong: "
          f"{sum(1 for e in by_q.values() if e['correct'] and e['wrong'])}")

    qs = list(by_q.keys())
    rng.shuffle(qs)
    rows = []
    stat = Counter()

    # ---- verify_confirm: correct trace -> cue -> mechanical recheck -> answer -------------
    for q in qs:
        if stat["confirm"] >= args.n_confirm:
            break
        e = by_q[q]
        if not e["correct"]:
            continue
        gold = e["gold"]
        made = 0
        for tr in sorted(e["correct"], key=len):
            if made >= args.max_per_problem:
                break
            if has_bad_arith(tr):
                stat["confirm_drop_bad_arith"] += 1
                continue
            th = strip_think(tr)
            if th is None:
                continue
            if is_degenerate(th, gold, args.min_think_tok, tok, True, 0):
                stat["confirm_drop_degenerate"] += 1
                continue
            chk = recheck_text(th, gold)
            if chk is None:
                stat["confirm_drop_no_checkable_step"] += 1
                continue
            content = "<think>\n" + th + CUE + chk + CLOSE_FMT % gold
            n = len(tok.encode(content, add_special_tokens=False))
            if n > args.max_tok:
                stat["confirm_drop_too_long"] += 1
                continue
            rows.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": content}],
                         "tier": "verify_confirm", "num_tokens": n, "pool": e["pool"]})
            stat["confirm"] += 1
            made += 1

    def clean_positives(e):
        """Correct traces that pass step-verification and the degeneracy test, shortest first."""
        outs = []
        for tr in sorted(e["correct"], key=len):
            if has_bad_arith(tr):
                continue
            th = strip_think(tr)
            if th is None or is_degenerate(th, e["gold"], args.min_think_tok, tok, True, 0):
                continue
            outs.append(th)
        return outs

    # ---- verify_rederive: correct A -> cue -> DISTINCT correct B -> answer ----------------
    for q in qs:
        if stat["rederive"] >= args.n_rederive:
            break
        e = by_q[q]
        pos = clean_positives(e)
        if len(pos) < 2:
            stat["rederive_drop_need_2_positives"] += 1
            continue
        gold = e["gold"]
        a = pos[0]
        b = next((x for x in pos[1:] if step_signature(x) != step_signature(a)), None)
        if b is None:
            stat["rederive_drop_no_distinct_second"] += 1
            continue
        content = ("<think>\n" + a + CUE +
                   "Let me derive it a second way to be sure.\n" + b +
                   f"\nBoth ways give {gold}." + CLOSE_FMT % gold)
        n = len(tok.encode(content, add_special_tokens=False))
        if n > args.max_tok:
            stat["rederive_drop_too_long"] += 1
            continue
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": content}],
                     "tier": "verify_rederive", "num_tokens": n, "pool": e["pool"]})
        stat["rederive"] += 1

    # ---- verify_fix: wrong attempt -> cue -> HONEST transition -> correct re-derivation ----
    for q in qs:
        if stat["fix"] >= args.n_fix:
            break
        e = by_q[q]
        if not (e["correct"] and e["wrong"]):
            continue
        gold = e["gold"]
        pos = clean_positives(e)
        if not pos:
            stat["fix_drop_no_clean_positive"] += 1
            continue
        good = pos[0]
        # prefer a wrong trace whose error can actually be LOCATED -- that row can name the step
        wrongs = sorted(e["wrong"], key=len)
        located = [w for w in wrongs if truncate_at_first_bad(strip_think(w) or "") is not None]
        made = 0
        for wtr in (located + [w for w in wrongs if w not in located]):
            if made >= args.max_per_problem:
                break
            wth = strip_think(wtr)
            if wth is None or len(tok.encode(wth, add_special_tokens=False)) < args.min_think_tok:
                continue
            hit = truncate_at_first_bad(wth)
            if hit is not None:
                prefix, desc = hit                  # names a step verified wrong in python
                stat["fix_kind_bad_eq"] += 1
            else:
                # keep the WHOLE wrong attempt: it demonstrably reached a wrong answer, so a
                # neutral "recompute" is true, whereas dismissing a prefix would not be.
                prefix, desc = wth, "Let me recompute this independently."
                stat["fix_kind_full_wrong"] += 1
            content = ("<think>\n" + prefix + CUE + desc + "\n" + good + CLOSE_FMT % gold)
            n = len(tok.encode(content, add_special_tokens=False))
            if n > args.max_tok:
                stat["fix_drop_too_long"] += 1
                continue
            rows.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": content}],
                         "tier": "verify_fix", "num_tokens": n, "pool": e["pool"]})
            stat["fix"] += 1
            made += 1

    rng.shuffle(rows)
    with open(args.jsonl_out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(rows)} rows -> {args.jsonl_out}")
    print(f"  tiers: {Counter(r['tier'] for r in rows).most_common()}")
    print(f"  pools: {Counter(r['pool'] for r in rows).most_common()}")
    toks = [r["num_tokens"] for r in rows]
    if toks:
        print(f"  tokens: mean {sum(toks)/len(toks):.0f}  p50 {sorted(toks)[len(toks)//2]}  "
              f"p95 {sorted(toks)[int(.95*len(toks))]}  max {max(toks)}")
    print(f"  stats: {dict(stat)}")
    if args.stats_out:
        json.dump({"stats": dict(stat), "n": len(rows), "args": vars(args)},
                  open(args.stats_out, "w"), indent=1)
    if rows:
        print("\n---- example verify_confirm ----")
        ex = next((r for r in rows if r["tier"] == "verify_confirm"), None)
        if ex:
            print(ex["messages"][1]["content"][:1400])
        print("\n---- example verify_fix ----")
        ex = next((r for r in rows if r["tier"] == "verify_fix"), None)
        if ex:
            print(ex["messages"][1]["content"][:1400])


if __name__ == "__main__":
    main()
