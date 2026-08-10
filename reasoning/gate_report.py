#!/usr/bin/env python3
"""Turn effort_gate JSONs into a paired comparison table WITH a failure decomposition.

WHY THIS EXISTS. A gate score alone cannot tell you what to do next. On this line the same
-16pt headline has meant three different things at different times: a termination defect
(traces never closing `</think>`), a selection defect (pass@k >> greedy), or a plain capability
gap. Those need opposite fixes, and the distinguishing numbers -- accuracy among traces that
ANSWERED, the unclosed rate, and trace length -- are already in the gate JSON but are not printed
by anything. Hand-computing them per arm is how you end up quoting a stale prior instead of
reading your own run.

Reads one or more `--json` files (they may cover different pools), prints per-pool and pool-mean
greedy/self-cons/pass@k for every model, deltas vs a `--baseline` model, exact McNemar on the
greedy column (the pairing is what makes small deltas interpretable -- items are identical across
models within one gate call), and the decomposition.

  python reasoning/gate_report.py --json report/a4rft_gate_*.json --baseline a4e1_a085
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def mcnemar(a_ok, b_ok):
    """Exact two-sided binomial test on discordant pairs. Returns (b, c, p)."""
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import comb
    # two-sided: sum of probabilities of outcomes at least as extreme
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--baseline", default="", help="model key to compute deltas against")
    ap.add_argument("--metric", default="greedy", choices=["greedy", "selfcons8", "pass8"])
    a = ap.parse_args()

    files = [f for pat in a.json for f in sorted(glob.glob(pat)) if os.path.exists(f)]
    if not files:
        raise SystemExit("no gate JSONs matched")

    # pool -> model -> record
    data = defaultdict(dict)
    for f in files:
        res = json.load(open(f))["res"]
        for model, pools in res.items():
            for pool, cfgs in pools.items():
                data[pool][model] = cfgs

    pools = list(data)
    models = sorted({m for p in pools for m in data[p]})
    base = a.baseline if a.baseline in models else ""

    print(f"pools: {pools}")
    print(f"{'pool':9s} {'model':16s} {'greedy':>7s} {'sc@8':>7s} {'pass@8':>7s} | "
          f"{'corr':>5s} {'wrong':>5s} {'uncl':>5s} {'no_ans':>6s} | {'acc|ANS':>8s} {'t_len':>6s} | {'Δ':>6s} {'p':>9s}")
    print("-" * 118)
    means = defaultdict(lambda: defaultdict(list))
    for pool in pools:
        for m in models:
            if m not in data[pool]:
                continue
            r = data[pool][m]
            g = r[a.metric]
            n = len(g["ok"])
            acc = sum(g["ok"]) / n * 100
            sc = sum(r["selfcons8"]["ok"]) / n * 100
            p8 = sum(r["pass8"]["ok"]) / n * 100
            fm = r["greedy"]["fm"]
            c, w = fm.get("correct", 0), fm.get("wrong", 0)
            u, na = fm.get("unclosed", 0), fm.get("no_answer", 0)
            ans = c + w
            accans = c / ans * 100 if ans else float("nan")
            d, ptxt = "", ""
            if base and m != base and base in data[pool]:
                d = f"{acc - sum(data[pool][base][a.metric]['ok'])/n*100:+6.2f}"
                _, _, p = mcnemar(g["ok"], data[pool][base][a.metric]["ok"])
                ptxt = f"{p:9.2e}"
            print(f"{pool:9s} {m:16s} {acc:7.2f} {sc:7.2f} {p8:7.2f} | {c:5d} {w:5d} {u:5d} {na:6d} | "
                  f"{accans:7.1f}% {r['greedy']['think_len']:6.1f} | {d:>6s} {ptxt:>9s}")
            means[m]["acc"].append(acc); means[m]["sc"].append(sc); means[m]["p8"].append(p8)
            means[m]["accans"].append(accans); means[m]["uncl"].append(u / n * 100)
            means[m]["tl"].append(r["greedy"]["think_len"])
        print()

    print("=" * 118)
    print(f"POOL-MEAN over {len(pools)} pools")
    bm = sum(means[base]["acc"]) / len(pools) if base else None
    for m in models:
        v = means[m]
        mean = lambda k: sum(v[k]) / len(v[k])
        d = f"  Δ {mean('acc') - bm:+6.2f}" if bm is not None and m != base else ""
        print(f"  {m:16s} greedy {mean('acc'):6.2f}  sc@8 {mean('sc'):6.2f}  pass@8 {mean('p8'):6.2f}  |  "
              f"acc|ANS {mean('accans'):5.1f}%  unclosed {mean('uncl'):5.2f}%  t_len {mean('tl'):6.1f}{d}")


if __name__ == "__main__":
    main()
