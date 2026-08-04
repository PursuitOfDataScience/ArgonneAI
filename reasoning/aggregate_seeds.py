#!/usr/bin/env python
"""aggregate_seeds.py -- seed-average a family of effort_gate JSONs against a shared baseline.

`effort_gate.py` reports each arm vs the reference with exact McNemar, but not the SEED MEAN across
arms of the same recipe -- and §33p/§33u established that on this recipe a one-epoch CoT-SFT run
carries ~1.7pt of run-to-run variation on a 5-set mean, so the seed mean is the only number worth
quoting. This reads the per-model JSONs written by a gate task and prints, per pool:

    base acc | each seed's acc and its paired McNemar p vs base | seed mean | mean delta | spread

Usage:
  python aggregate_seeds.py --glob '/path/report/g_nt0_*.json' [--ref base] [--config greedy]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import re
from collections import defaultdict
from math import comb


def mcnemar(a, b):
    """Exact two-sided McNemar on paired boolean vectors -> (n01, n10, p)."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 0, 0, 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for the gate's per-model JSONs")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--config", default="greedy",
                    help="which decode config to aggregate (greedy | budget | extend1 ...)")
    ap.add_argument("--family", default=None,
                    help="only aggregate arms whose name starts with this (default: all non-ref)")
    args = ap.parse_args()

    # model -> pool -> ok-vector, merged across the n-group JSONs
    res: dict[str, dict[str, list]] = defaultdict(dict)
    for f in sorted(globmod.glob(args.glob)):
        try:
            o = json.load(open(f))
        except Exception as e:                                  # noqa: BLE001
            print(f"  (skip unreadable {f}: {e})")
            continue
        for name, pools in o.get("res", {}).items():
            for pool, cfgs in pools.items():
                cfg = cfgs.get(args.config)
                if cfg and "ok" in cfg:
                    res[name][pool] = cfg["ok"]

    if args.ref not in res:
        print(f"NO REFERENCE '{args.ref}' found. models present: {sorted(res)}")
        return
    arms = sorted(n for n in res if n != args.ref
                  and (args.family is None or n.startswith(args.family)))
    if not arms:
        print(f"no arms found (models present: {sorted(res)})")
        return

    pools = [p for p in ("svamp", "asdiv", "mawps", "gsmplus", "math500") if p in res[args.ref]]
    print(f"config={args.config}   ref={args.ref}   arms={arms}\n")
    hdr = f"{'pool':<10}{'n':>6}{'base':>8}" + "".join(f"{a[-5:]:>9}" for a in arms) + \
          f"{'mean':>9}{'delta':>8}{'spread':>8}"
    print(hdr)
    print("-" * len(hdr))
    means, base_means = [], []
    for pool in pools:
        b = res[args.ref][pool]
        n = len(b)
        bacc = 100 * sum(b) / n
        accs, ps = [], []
        for a in arms:
            v = res[a].get(pool)
            if not v or len(v) != n:
                accs.append(None)
                ps.append(None)
                continue
            accs.append(100 * sum(v) / n)
            ps.append(mcnemar(b, v)[2])
        got = [x for x in accs if x is not None]
        m = sum(got) / len(got) if got else float("nan")
        spread = (max(got) - min(got)) if len(got) > 1 else 0.0
        cells = "".join(f"{x:>9.2f}" if x is not None else f"{'--':>9}" for x in accs)
        print(f"{pool:<10}{n:>6}{bacc:>8.2f}{cells}{m:>9.2f}{m - bacc:>+8.2f}{spread:>8.2f}")
        for a, p in zip(arms, ps):
            if p is not None and p < 0.05:
                print(f"           ^ {a}: paired McNemar vs {args.ref} p={p:.4g}")
        means.append(m)
        base_means.append(bacc)
    if means:
        mm = sum(means) / len(means)
        bm = sum(base_means) / len(base_means)
        print("-" * len(hdr))
        print(f"{'MEAN':<10}{'':>6}{bm:>8.2f}{'':>{9*len(arms)}}{mm:>9.2f}{mm - bm:>+8.2f}")
        print(f"\n({len(pools)}-set unweighted mean; §33p's noise scale on this recipe is ~1.7pt, "
              f"so a delta inside that is not a result.)")


if __name__ == "__main__":
    main()
