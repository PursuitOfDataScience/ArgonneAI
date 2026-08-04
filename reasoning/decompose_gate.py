#!/usr/bin/env python
"""decompose_gate.py -- split an effort_gate result by the per-item annotations the eval sets already ship.

§34's headline mechanisms were all invisible at the set level and obvious one level down:

  GSM-Plus  by perturbation_type : nt0 gained +26.0 on integer-decimal-fraction conversion and +1.1 on
                                   `adding operation` -- i.e. the gain is arithmetic EXECUTION, not
                                   comprehension. The set-level number is +7.5 and says none of this.
  ASDiv     by solution_type     : 54% of items are single-operator, and nt0 regressed on all four of
                                   those types while gaining +8 to +10 on Algebra-2 / TVQ-Final. The
                                   set-level number is -3.5, i.e. two opposite effects cancelling.
  SVAMP     by operator count    : 76% single-operator. Same story.

That also corrected §33s's claim that the suite contained no single-step arithmetic (§34am): it is mostly
single-step arithmetic in prose. The annotations were in the datasets all along.

`effort_gate --json-out` stores per-item `ok` vectors and `clean_eval.load_clean` is deterministic for a
given (source, n, seed), so any gate result can be re-cut offline at zero GPU cost.

Usage:
  python decompose_gate.py --glob '.../report/g_both_*.json' --pool asdiv --n 1000 --ref base
  python decompose_gate.py --glob '.../report/g_*.json' --pool gsmplus --n 500 --config budget
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import sys
from collections import Counter, defaultdict


def annotations(pool: str, probs):
    """-> (list of per-item labels aligned to probs, label-kind string). '?' where unmatched."""
    from datasets import load_from_disk
    DATA = "/project/rcc/youzhi/data"

    if pool == "asdiv":
        d = load_from_disk(f"{DATA}/asdiv_clean")["validation"]
        lab = {(r["body"].strip() + " " + r["question"].strip()).strip(): r["solution_type"] for r in d}
        return [lab.get(q.strip(), "?") for q, _ in probs], "solution_type"

    if pool == "svamp":
        d = load_from_disk(f"{DATA}/svamp_clean")
        rows = list(d["train"]) + list(d["test"])
        lab = {(r["Body"].strip() + " " + r["Question"].strip()).strip():
               f"{sum(r['Equation'].count(o) for o in '+-*/')}-op ({r['Type']})" for r in rows}
        return [lab.get(q.strip(), "?") for q, _ in probs], "operator count (Type)"

    if pool in ("gsmplus", "gsm_plus"):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from datasets import load_dataset
        d = load_dataset("qintongli/GSM-Plus", split="test")
        lab = dict(zip((q.strip() for q in d["question"]), d["perturbation_type"]))
        return [lab.get(q.strip(), "?") for q, _ in probs], "perturbation_type"

    if pool == "math500":
        d = load_from_disk(f"{DATA}/nlile_hendrycks-MATH-benchmark")["test"]
        lab = {}
        for r in d:
            lv = re.search(r"(\d)", str(r.get("level", "")))
            lab[str(r["problem"]).strip()] = f"L{lv.group(1)}" if lv else "L?"
        return [lab.get(q.strip(), "?") for q, _ in probs], "MATH level"

    return ["(no annotation available)"] * len(probs), "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="gate JSONs (per-model files are fine)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, required=True, help="MUST match the gate's --n for this pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="greedy")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--family", default=None, help="average arms whose name starts with this")
    ap.add_argument("--min-n", type=int, default=20, help="hide slices smaller than this")
    args = ap.parse_args()

    sys.path.insert(0, "/project/rcc/youzhi/a35_effort/tools")
    from clean_eval import load_clean
    probs = load_clean(args.pool, args.n, seed=args.seed)
    labels, kind = annotations(args.pool, probs)

    ok = {}
    for f in sorted(globmod.glob(args.glob)):
        try:
            o = json.load(open(f))
        except Exception:
            continue
        for nm, pools in o.get("res", {}).items():
            cfg = pools.get(args.pool, {}).get(args.config)
            if cfg and len(cfg["ok"]) == len(probs):
                ok[nm] = [bool(v) for v in cfg["ok"]]
    if args.ref not in ok:
        print(f"reference '{args.ref}' not found for pool={args.pool} config={args.config}; "
              f"present: {sorted(ok)}")
        return
    arms = sorted(n for n in ok if n != args.ref
                  and (args.family is None or n.startswith(args.family)))
    if not arms:
        print(f"no arms found; present: {sorted(ok)}")
        return

    matched = sum(1 for x in labels if x != "?")
    print(f"pool={args.pool}  n={len(probs)}  config={args.config}  ref={args.ref}")
    print(f"arms={arms} (averaged)   annotation={kind}   matched {matched}/{len(probs)}\n")
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    print(f"{kind:<38}{'n':>6}{args.ref:>9}{'arms':>9}{'delta':>8}")
    print("-" * 70)
    rows = []
    for lab, idx in by.items():
        if len(idx) < args.min_n:
            continue
        r = 100 * sum(ok[args.ref][i] for i in idx) / len(idx)
        a = 100 * sum(sum(ok[m][i] for i in idx) / len(idx) for m in arms) / len(arms)
        rows.append((a - r, lab, len(idx), r, a))
    for d, lab, n, r, a in sorted(rows, reverse=True):
        print(f"{lab[:38]:<38}{n:>6}{r:>8.1f}%{a:>8.1f}%{d:>+8.1f}")
    hidden = sum(len(i) for l, i in by.items() if len(i) < args.min_n)
    if hidden:
        print(f"\n({hidden} items in slices smaller than --min-n {args.min_n} not shown)")


if __name__ == "__main__":
    main()
