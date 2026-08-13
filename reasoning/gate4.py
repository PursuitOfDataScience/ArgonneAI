#!/usr/bin/env python
"""The four-pool pooled PAIRED readout — the only trustworthy number on this line.

WHY THIS EXISTS RATHER THAN gate_report.py. `gate_report` prints per-pool rows and a delta for ONE
metric. That is the shape that produced 2026-08-12's worst mistake: gate call 1 (asdiv+svamp) read
`pass@8 +2.05, p=0.003` and was reported as "coverage moved"; the four-pool number is +0.83, p=0.156,
because svamp is +2.70 (p=0.013) while gsmplus is −3.80 (p=0.040) — nominally significant in OPPOSITE
directions. Pool-to-pool spread on this line is ±3pt and the effects being chased are ~1pt, so any
readout narrower than four pools says whatever the chosen pools prefer.

So this tool refuses to be that shape. It:
  * POOLS the four clean pools into one 3,000-item paired comparison and prints that FIRST;
  * prints DISCORDANCE next to every delta, because a null score is not a null intervention -- r8 vs
    repairlo differ on 17.0% of items for a net +0.57, and `b≈c` is exactly what makes a paired test
    read null (§41bw);
  * prints a per-pool sign check underneath, so heterogeneity is visible rather than averaged away;
  * WARNS if math500 is included -- it is contaminated for this line (17/319 near-dups) and is not
    part of the four-pool standard.

Usage:
  python reasoning/gate4.py --json "report/a4divrep_gate_*.json" --baseline a4r8repair_a100
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLEAN4 = ["asdiv", "svamp", "gsmplus", "mawps"]
METRICS = ["greedy", "budget", "extend1", "selfcons8", "pass8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--baseline", required=True, help="model key to compare everything against")
    ap.add_argument("--pools", nargs="+", default=CLEAN4)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    a = ap.parse_args()

    from gate_report import mcnemar

    res = {}
    for pat in a.json:
        for f in sorted(glob.glob(pat)):
            for m, pools in json.load(open(f))["res"].items():
                res.setdefault(m, {}).update(pools)
    if not res:
        print("no gate JSONs matched"); return 2

    have = [p for p in a.pools if all(p in v for v in res.values())]
    missing = [p for p in a.pools if p not in have]
    if missing:
        print(f"⚠️  pools missing from some models, EXCLUDED: {missing}")
    if "math500" in have:
        print("⚠️  math500 is CONTAMINATED for this line (17/319 near-dups) and is not part of the "
              "four-pool standard — quote it separately, never pooled.")
    if len(have) < 4:
        print(f"⚠️⚠️  ONLY {len(have)} POOL(S) ({have}). This is the readout that misled on 2026-08-12; "
              f"the pool spread is ±3pt. Treat anything below as PROVISIONAL.")

    def cat(m, metric):
        out = []
        for p in have:
            out += res[m][p][metric]["ok"]
        return out

    n = len(cat(a.baseline, a.metrics[0]))
    print(f"\nPOOLED {have}  n={n} identical items")
    hdr = f"{'model':22s}" + "".join(f"{m:>11s}" for m in a.metrics)
    print(hdr); print("-" * len(hdr))
    for m in sorted(res):
        row = f"{m.replace('_a100',''):22s}"
        for metric in a.metrics:
            row += f"{100*sum(cat(m, metric))/n:11.2f}"
        print(row + ("   <- baseline" if m == a.baseline else ""))

    print(f"\nPAIRED vs {a.baseline.replace('_a100','')}  (delta, p, and how many items DISAGREE)")
    for m in sorted(res):
        if m == a.baseline:
            continue
        print(f"  {m.replace('_a100',''):22s}")
        for metric in a.metrics:
            x, y = cat(m, metric), cat(a.baseline, metric)
            b, c, p = mcnemar(x, y)
            d = 100 * sum(x) / n - 100 * sum(y) / n
            sig = "  SIG" if p < 0.05 else ""
            print(f"      {metric:10s} {d:+6.2f}  p={p:8.3g}  disagree {b+c:4d} "
                  f"({100*(b+c)/n:4.1f}%){sig}")

    print(f"\nPER-POOL SIGN CHECK on greedy (heterogeneity is invisible in the pooled number)")
    for m in sorted(res):
        if m == a.baseline:
            continue
        cells = []
        for p in have:
            x = res[m][p]["greedy"]["ok"]; y = res[a.baseline][p]["greedy"]["ok"]
            cells.append(f"{p}:{100*sum(x)/len(x)-100*sum(y)/len(y):+.2f}")
        signs = [1 if float(c.split(":")[1]) > 0 else -1 for c in cells]
        flag = "" if abs(sum(signs)) == len(signs) else "   ⚠️MIXED SIGNS"
        print(f"  {m.replace('_a100',''):22s} " + "  ".join(cells) + flag)


if __name__ == "__main__":
    sys.exit(main())
