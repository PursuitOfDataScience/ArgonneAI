#!/usr/bin/env python
"""Summarise lm-eval result JSONs, printing acc AND acc_norm side by side.

WHY THIS EXISTS. The summary loop copy-pasted into the campaign launchers (`323_general.sh`,
`342_general_genfix.sh`, ...) does this:

    for k, x in v.items():
        if k.startswith("acc"): vals.append(x); break     # <-- takes whichever comes FIRST

`acc,none` and `acc_norm,none` both start with "acc", so which one it reports depends on dict order.
On 2026-08-04 that produced a released-model mean of **51.47** against §33i's recorded **55.21** for
the same checkpoint — a 3.74pt phantom regression that is entirely arc_easy/hellaswag/openbookqa
being length-normalised or not:

    task            acc     acc_norm
    arc_easy      65.03        59.60
    hellaswag     45.35        59.91
    openbookqa    25.00        37.00

Both numbers are correct; mixing them across runs is not. Printing both makes the comparison
impossible to get wrong, and makes it obvious when a historical reference used the other one.
(winogrande reports only `acc`, so acc_norm falls back to acc for that task.)

Usage:
  python reasoning/lmeval_summary.py --glob '<report>/lmeval_*.json' [--ref base]
  python reasoning/lmeval_summary.py --files a.json b.json --tasks arc_easy hellaswag
"""
import argparse
import glob as globmod
import json
import os

DEFAULT_TASKS = ("arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa", "winogrande")


def read(path, tasks):
    d = json.load(open(path))
    acc, norm, per = [], [], {}
    for t in tasks:
        v = d.get(t) or {}
        a = v.get("acc,none")
        if a is None:
            continue
        n = v.get("acc_norm,none", a)       # winogrande has no acc_norm
        acc.append(100 * a)
        norm.append(100 * n)
        per[t] = (100 * a, 100 * n)
    if not acc:
        return None
    return {"acc": sum(acc) / len(acc), "acc_norm": sum(norm) / len(norm),
            "n_tasks": len(acc), "per_task": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument("--ref", default="base", help="model name to show deltas against")
    ap.add_argument("--per-task", action="store_true")
    a = ap.parse_args()

    files = list(a.files) + (sorted(globmod.glob(a.glob)) if a.glob else [])
    if not files:
        raise SystemExit("no files matched")

    res = {}
    for f in files:
        name = os.path.basename(f)
        for pre, suf in (("lmeval_", ".json"),):
            if name.startswith(pre):
                name = name[len(pre):]
            if name.endswith(suf):
                name = name[: -len(suf)]
        r = read(f, a.tasks)
        if r:
            res[name] = r

    order = ([a.ref] if a.ref in res else []) + sorted(k for k in res if k != a.ref)
    w = max(len(k) for k in order) + 2
    print(f"{'model':<{w}}{'acc':>9}{'acc_norm':>11}{'tasks':>7}"
          + (f"{'d acc':>9}{'d norm':>9}" if a.ref in res else ""))
    print("-" * (w + 27 + (18 if a.ref in res else 0)))
    ref = res.get(a.ref)
    for k in order:
        r = res[k]
        line = f"{k:<{w}}{r['acc']:>9.2f}{r['acc_norm']:>11.2f}{r['n_tasks']:>7}"
        if ref and k != a.ref:
            line += f"{r['acc']-ref['acc']:>+9.2f}{r['acc_norm']-ref['acc_norm']:>+9.2f}"
        print(line)

    if a.per_task:
        for k in order:
            print(f"\n---- {k}")
            for t, (x, y) in res[k]["per_task"].items():
                print(f"  {t:<16} acc {x:6.2f}   acc_norm {y:6.2f}")

    print("\nQuote WHICH metric you are using. Historical numbers in thinking_training.md "
          "(e.g. base = 55.21) are acc_norm.")


if __name__ == "__main__":
    main()
