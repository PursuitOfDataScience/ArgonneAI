#!/usr/bin/env python3
"""Every arm ever gated on one line, ranked, with the decode-time configs and the diagnosis columns.

WHY, separately from `gate_report.py`. gate_report answers "how does THIS arm compare to its baseline
on the pools THIS gate call covered", which is the right question inside one experiment and the wrong
one after fifteen. Arms accumulate across dozens of JSONs, each covering a different subset of models
and pools, and the question becomes "what is the best thing we have, and what did each lever actually
move". Answering that by hand is how a stale number gets quoted -- it is why the standing note on this
line said the stronger-teacher lever was untested after it had already been run and refuted.

Reads every `report/<prefix>*_gate_*.json`, keeps the largest-n record per (pool, model), and prints
the pool-mean of every decode config plus the failure decomposition. Arms missing a pool are dropped
from the ranking rather than averaged over fewer pools, because a 4-pool mean is not comparable to a
5-pool mean.

⚠️Cross-arm deltas here are NOT paired: two arms may come from different gate calls on the same items
but different engine instances, and this line has measured ±0.87 pool-mean seed noise and ±0.30 from
pure GPU non-determinism. Use this to rank and to spot what moved; use `gate_report.py --baseline`
on a single JSON for a p-value.

  python reasoning/arms_table.py --prefix a4 --pools asdiv svamp mawps gsmplus math500
"""
import argparse
import glob
import json
import os
from collections import defaultdict

CFGS = ["greedy", "budget", "extend1", "extend2", "extend3", "selfcons8", "pass8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="a4", help="report/<prefix>*_gate_*.json")
    ap.add_argument("--pools", nargs="+",
                    default=["asdiv", "svamp", "mawps", "gsmplus", "math500"])
    ap.add_argument("--sort", default="greedy", choices=CFGS)
    ap.add_argument("--allow-partial", action="store_true",
                    help="also show arms missing some pools (flagged, and never mixed into a rank)")
    a = ap.parse_args()

    best = defaultdict(dict)            # pool -> model -> (n, rec, source)
    for f in sorted(glob.glob(f"report/{a.prefix}*_gate_*.json")):
        try:
            res = json.load(open(f))["res"]
        except Exception:
            continue
        for model, pools in res.items():
            for pool, rec in pools.items():
                n = len(rec["greedy"]["ok"])
                if model not in best[pool] or n > best[pool][model][0]:
                    best[pool][model] = (n, rec, os.path.basename(f).split("_gate")[0])

    models = sorted({m for p in best for m in best[p]})
    rows, partial = [], []
    for m in models:
        have = [p for p in a.pools if m in best[p]]
        vals = defaultdict(list)
        acc_ans, uncl, tl, ns, src = [], [], [], [], set()
        for p in have:
            n, r, s = best[p][m]
            src.add(s)
            ns.append(n)
            for c in CFGS:
                if c in r:
                    vals[c].append(sum(r[c]["ok"]) / n * 100)
            fm = r["greedy"]["fm"]
            c_, w_ = fm.get("correct", 0), fm.get("wrong", 0)
            acc_ans.append(c_ / (c_ + w_) * 100 if c_ + w_ else float("nan"))
            uncl.append(fm.get("unclosed", 0) / n * 100)
            tl.append(r["greedy"]["think_len"])
        if not have:
            continue
        mean = lambda x: sum(x) / len(x) if x else float("nan")
        row = (mean(vals[a.sort]), m, [mean(vals[c]) for c in CFGS],
               mean(acc_ans), mean(uncl), mean(tl), len(have), sorted(src)[:2])
        (rows if len(have) == len(a.pools) else partial).append(row)

    rows.sort(reverse=True)
    partial.sort(reverse=True)
    hdr = f"{'arm':20s}" + "".join(f"{c:>9s}" for c in CFGS) + \
          f"{'acc|ANS':>9s}{'uncl%':>7s}{'t_len':>7s}"
    print(f"pools: {a.pools}   sorted by {a.sort}")
    print(hdr)
    print("-" * len(hdr))
    for _, m, v, aa, un, tl, npool, src in rows:
        print(f"{m:20s}" + "".join(f"{x:9.2f}" for x in v) +
              f"{aa:8.1f}%{un:7.2f}{tl:7.1f}")
    if partial and a.allow_partial:
        print(f"\n-- INCOMPLETE (fewer than {len(a.pools)} pools; NOT comparable to the above) --")
        for _, m, v, aa, un, tl, npool, src in partial:
            print(f"{m:20s}" + "".join(f"{x:9.2f}" for x in v) +
                  f"{aa:8.1f}%{un:7.2f}{tl:7.1f}   [{npool}/{len(a.pools)} pools, {','.join(src)}]")
    elif partial:
        print(f"\n({len(partial)} arm(s) hidden for missing pools: "
              f"{', '.join(m for _, m, *_ in partial[:8])}; --allow-partial to show)")


if __name__ == "__main__":
    main()
