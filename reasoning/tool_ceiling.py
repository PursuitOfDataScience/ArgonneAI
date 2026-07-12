#!/usr/bin/env python3
"""Estimate the tool-use CEILING from already-generated candidate traces (CPU, free).

Tool-use (calculator offload) only helps traces whose PROCEDURE is right but ARITHMETIC is
wrong. This bounds that: over the banked vLLM candidates, of the wrong-but-closed traces, how
many are 'interceptable' (contain a detectably-wrong inline `a op b = c` step, OR the correct
gold equals the true value of some inline `a op b` the model miscomputed) vs 'structural'
(all inline steps correct but wrong answer = wrong operand/procedure — tool-use can't fix).
High interceptable fraction => tool-use has real headroom => worth the full build.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_generate import extract_boxed, norm  # noqa: E402

EQ = re.compile(r"(-?\d+(?:\.\d+)?)\s*([-+*/×xX])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def _val(a, op, b):
    op = op.lower()
    if op == "+": return a + b
    if op == "-": return a - b
    if op in ("*", "×", "x"): return a * b
    if op == "/": return a / b if b else None
    return None


def analyze(text, gold):
    """Return dict flags for a closed trace."""
    n_step = n_bad = 0
    gold_recoverable = False
    gnum = None
    try:
        gnum = float(gold)
    except (TypeError, ValueError):
        pass
    for m in EQ.finditer(text):
        a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        v = _val(a, op, b)
        if v is None:
            continue
        n_step += 1
        if abs(v - c) > 1e-6:
            n_bad += 1
            if gnum is not None and abs(v - gnum) < 1e-6:
                gold_recoverable = True  # model did the right op, wrote wrong result, true=gold
    return n_step, n_bad, gold_recoverable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="/project/rcc/youzhi/data/vllm_bon_candidates.json")
    args = ap.parse_args()
    rows = json.load(open(args.candidates))

    closed = wrong = correct = 0
    wrong_interceptable = wrong_gold_recoverable = wrong_structural = 0
    closed_with_eq = 0
    # per-PROBLEM: does the problem have >=1 wrong trace that a calculator would flip to gold?
    prob_recoverable = 0
    n_prob_wrong_only = 0  # problems with no correct trace but >=1 gold-recoverable wrong trace

    for r in rows:
        gold = r["gold"]
        any_correct = False
        any_gold_recover = False
        for t in r["candidates"]:
            if "</think>" not in t:
                continue
            closed += 1
            pred = extract_boxed(t)
            n_step, n_bad, gold_rec = analyze(t, gold)
            if n_step > 0:
                closed_with_eq += 1
            if pred is not None and pred == gold:
                correct += 1
                any_correct = True
            else:
                wrong += 1
                if n_bad > 0:
                    wrong_interceptable += 1
                    if gold_rec:
                        wrong_gold_recoverable += 1
                        any_gold_recover = True
                else:
                    wrong_structural += 1
        if not any_correct and any_gold_recover:
            n_prob_wrong_only += 1

    print("=" * 66)
    print(f"tool-use ceiling from {len(rows)} problems, {closed} closed traces")
    print("=" * 66)
    print(f"  closed w/ any inline `a op b=`  : {closed_with_eq} ({100*closed_with_eq/max(closed,1):.0f}%)  <- tool surface")
    print(f"  correct / wrong (closed)        : {correct} / {wrong}")
    if wrong:
        print(f"  wrong INTERCEPTABLE (bad step)  : {wrong_interceptable} ({100*wrong_interceptable/wrong:.0f}% of wrong)")
        print(f"    of which gold-recoverable     : {wrong_gold_recoverable} ({100*wrong_gold_recoverable/wrong:.0f}% of wrong)  <- calc would flip to GOLD")
        print(f"  wrong STRUCTURAL (steps ok)     : {wrong_structural} ({100*wrong_structural/wrong:.0f}% of wrong)  <- tool-use CAN'T fix")
    print(f"  PROBLEMS solvable ONLY via calc  : {n_prob_wrong_only}/{len(rows)} "
          f"({100*n_prob_wrong_only/len(rows):.1f}%)  <- headroom tool-use ADDS beyond current")
    print("=" * 66)


if __name__ == "__main__":
    main()
