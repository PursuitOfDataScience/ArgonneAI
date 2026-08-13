#!/usr/bin/env python3
"""Read an ON-POLICY rollout dump and say WHY the model is wrong, in categories that map to levers.

WHY THIS EXISTS. Twelve post-training arms on argonne4-think moved pass@8 62.4 -> 69.0 and left
greedy at ~43. `gate_report.py` localises that to one number (accuracy among traces that ANSWERED,
~50% vs 3.5-think's 70%) but stops there: "wrong" is a single bucket, and the fixes for its
contents are different and mutually exclusive. Training a4 on more traces cannot be the answer to
a readout bug, and a decode change cannot be the answer to an arithmetic bug.

This splits `wrong` into buckets that each name a lever:

  reached_not_selected   gold appears inside the trace but the boxed answer is something else
                         -> READOUT. Fixable at decode/format level, no capability needed.
  bad_arith              some explicit `a op b = c` in the trace is false
                         -> EXECUTION. Fixable with drills / tool use / step-level preference.
  no_gold_no_arith       gold never appears and every stated equation checks out
                         -> PLAN. The model solved a different problem correctly. Needs capability.

and reports, per problem, how much of the pass@k headroom a *selector* could reach (is gold the
plurality answer?) versus how much needs the policy to change at all.

Input is the `*_all.jsonl` written by rft_generate / build_rlvr_pairs: one row per rollout with
question / pool / trace / label / pred / gold.

  python reasoning/fail_taxonomy.py --jsonl /project/rcc/youzhi/data/a4_dpo/a4_dpo_all.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rft_generate import EQ, has_bad_arith, _val  # noqa: E402
from star_generate import norm  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    """Every number the trace states, normalised the way the grader normalises answers."""
    out = set()
    for m in NUM.finditer(text):
        s = m.group(0).replace(",", "")
        out.add(norm(s))
        if s.endswith(".0"):
            out.add(norm(s[:-2]))
    return out


def eq_list(text):
    return [(m.group(1), m.group(2).lower(), m.group(3), m.group(4)) for m in EQ.finditer(text)]


def first_bad_eq(text):
    """Index of the first false equation, or None. Localises where a trace goes off the rails."""
    for i, m in enumerate(EQ.finditer(text)):
        try:
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        v = _val(a, op, b)
        if v is not None and abs(v - c) > 1e-6:
            return i, m.start()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    by_q = defaultdict(list)
    n = 0
    with open(a.jsonl) as f:
        for line in f:
            r = json.loads(line)
            by_q[(r["pool"], r["question"])].append(r)
            n += 1
            if a.max_rows and n >= a.max_rows:
                break
    print(f"[taxonomy] {n:,} rollouts over {len(by_q):,} problems "
          f"(K={n / max(1, len(by_q)):.1f})\n")

    lab = Counter()
    wrong_bucket = Counter()
    arith_by_label = Counter()
    tot_by_label = Counter()
    eqs_by_label = defaultdict(list)
    chars_by_label = defaultdict(list)
    badeq_pos = []                       # relative position of the first false equation
    # per-problem view
    pk = Counter()                       # #correct out of K -> #problems
    plurality_ok = Counter()             # among solvable-but-unreliable, is gold the plurality?
    gold_reached_anywhere = Counter()
    per_pool = defaultdict(Counter)

    for (pool, q), rows in by_q.items():
        gold = norm(str(rows[0]["gold"]))
        ncorr = 0
        preds = []
        any_gold_in_trace = False
        for r in rows:
            L = r["label"]
            lab[L] += 1
            tot_by_label[L] += 1
            per_pool[pool][L] += 1
            tr = r["trace"]
            chars_by_label[L].append(len(tr))
            eqs_by_label[L].append(len(eq_list(tr)))
            bad = has_bad_arith(tr)
            if bad:
                arith_by_label[L] += 1
            nums = numbers_in(tr)
            if gold in nums:
                any_gold_in_trace = True
            if L == "correct":
                ncorr += 1
            if L == "wrong":
                if gold in nums:
                    wrong_bucket["reached_not_selected"] += 1
                elif bad:
                    wrong_bucket["bad_arith"] += 1
                else:
                    wrong_bucket["no_gold_no_arith"] += 1
                fb = first_bad_eq(tr)
                if fb is not None and len(tr) > 0:
                    badeq_pos.append(fb[1] / len(tr))
            if L in ("correct", "wrong"):
                preds.append(norm(str(r.get("pred"))))
        pk[ncorr] += 1
        if any_gold_in_trace:
            gold_reached_anywhere["yes"] += 1
        else:
            gold_reached_anywhere["no"] += 1
        if preds:
            top, cnt = Counter(preds).most_common(1)[0]
            if 0 < ncorr < len(rows):
                plurality_ok["gold_is_plurality" if top == gold else "wrong_is_plurality"] += 1

    K = round(n / max(1, len(by_q)))
    print("=== rollout labels ===")
    for L, c in lab.most_common():
        pct = c / n * 100
        arate = arith_by_label[L] / max(1, tot_by_label[L]) * 100
        mc = sum(chars_by_label[L]) / max(1, len(chars_by_label[L]))
        me = sum(eqs_by_label[L]) / max(1, len(eqs_by_label[L]))
        print(f"  {L:11s} {c:7,d} {pct:5.1f}%   bad_arith {arate:5.1f}%   "
              f"chars {mc:6.0f}   equations {me:4.1f}")

    W = sum(wrong_bucket.values())
    print(f"\n=== the WRONG bucket, split by lever ({W:,} traces) ===")
    for k in ("reached_not_selected", "bad_arith", "no_gold_no_arith"):
        c = wrong_bucket[k]
        print(f"  {k:22s} {c:7,d} {c / max(1, W) * 100:5.1f}%   ({c / n * 100:4.1f}% of all rollouts)")
    if badeq_pos:
        badeq_pos.sort()
        med = badeq_pos[len(badeq_pos) // 2]
        print(f"  first false equation sits at {med * 100:.0f}% through the trace (median of "
              f"{len(badeq_pos):,}) -- a prefix that long is shared with the fixed version")

    print(f"\n=== per problem (K={K}) ===")
    npro = len(by_q)
    solved_any = sum(c for k, c in pk.items() if k > 0)
    print(f"  never solved in K   {pk[0]:6,d} {pk[0] / npro * 100:5.1f}%")
    print(f"  solved at least 1x  {solved_any:6,d} {solved_any / npro * 100:5.1f}%")
    print(f"  solved ALL {K}x      {pk[K]:6,d} {pk[K] / npro * 100:5.1f}%")
    unrel = solved_any - pk[K]
    print(f"  solvable-unreliable {unrel:6,d} {unrel / npro * 100:5.1f}%  <- the headroom")
    tot_pl = sum(plurality_ok.values())
    if tot_pl:
        g = plurality_ok["gold_is_plurality"]
        print(f"    of those, gold is the PLURALITY answer in {g:,}/{tot_pl:,} = {g / tot_pl * 100:.1f}%"
              f"  -> a majority vote reaches it; the other {100 - g / tot_pl * 100:.1f}% need a "
              f"selector better than voting (verifier) or a different policy")
    gy = gold_reached_anywhere["yes"]
    print(f"  gold appears somewhere in SOME trace: {gy:,}/{npro:,} = {gy / npro * 100:.1f}%")

    print("\n=== by pool ===")
    for pool, c in per_pool.items():
        t = sum(c.values())
        print(f"  {pool:16s} n={t:6,d}  " + "  ".join(f"{k} {v / t * 100:.1f}%" for k, v in c.most_common()))

    if a.json_out:
        json.dump({"n": n, "problems": len(by_q), "labels": lab, "wrong_bucket": wrong_bucket,
                   "arith_by_label": arith_by_label, "pk": {str(k): v for k, v in pk.items()},
                   "plurality": plurality_ok,
                   "gold_reached_anywhere": gold_reached_anywhere,
                   "per_pool": {p: dict(c) for p, c in per_pool.items()}},
                  open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
