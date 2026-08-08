#!/usr/bin/env python3
"""Chance-collision null control for pass@K / self-consistency / single-sample accuracy.

WHY (2026-07-10 eval-integrity finding): SVAMP/ASDiv golds are tiny integers (SVAMP ~54% of
golds in [0,20], ASDiv ~47%; modal answers 1-5). A model that merely emits small integers
scores a large pass@K BY CHANCE. Nothing in clean_eval.py/select_eval.py corrects for this, so
the headline "pass@32 = 70-76% latent ceiling" is inflated by lucky numeric collisions. This
grader measures the EXCESS-over-chance using the model's OWN predictions (no new GPU):

  for B random permutations pi:  re-score problem i's K predictions against gold[pi(i)]
  (a DIFFERENT problem's gold) and recompute the metric -> a null distribution driven purely by
  the answer-marginal collision structure. excess = observed - null_mean (with the null 95% band).

Two nulls: (1) global permutation; (2) magnitude-bucketed permutation (shuffle golds only within
the same answer-magnitude bucket) which additionally removes difficulty/magnitude correlation.

Input: the --dump-preds JSON from select_eval.py  {source: {golds, preds_full, preds_forced}}.
"""
import argparse
import json
import random
from collections import Counter


def mode_correct(preds, gold):
    v = Counter(p for p in preds if p is not None)
    return bool(v) and v.most_common(1)[0][0] == gold


def metrics(preds_by_prob, golds):
    """Return (single_acc, passk, self_cons) as fractions."""
    n = len(golds)
    tot = sc = pk = mc = 0
    for preds, g in zip(preds_by_prob, golds):
        for p in preds:
            tot += 1
            if p is not None and p == g:
                sc += 1
        if any(p is not None and p == g for p in preds):
            pk += 1
        if mode_correct(preds, g):
            mc += 1
    k = len(preds_by_prob[0]) if preds_by_prob else 0
    return sc / max(tot, 1), pk / n, mc / n, k


def magnitude_bucket(g):
    try:
        v = abs(float(g))
    except (TypeError, ValueError):
        return "nonnum"
    for hi, name in [(5, "0-5"), (20, "6-20"), (100, "21-100"), (1000, "101-1k")]:
        if v <= hi:
            return name
    return ">1k"


def permuted_golds(golds, rng, bucketed):
    idx = list(range(len(golds)))
    if not bucketed:
        rng.shuffle(idx)
        return [golds[j] for j in idx]
    # shuffle within magnitude buckets
    buckets = {}
    for i, g in enumerate(golds):
        buckets.setdefault(magnitude_bucket(g), []).append(i)
    out = [None] * len(golds)
    for b, members in buckets.items():
        shuf = members[:]
        rng.shuffle(shuf)
        for src_i, dst_i in zip(members, shuf):
            out[dst_i] = golds[src_i]
    return out


def null_band(preds_by_prob, golds, B, bucketed, seed=0):
    rng = random.Random(seed)
    sc, pk, mc = [], [], []
    for _ in range(B):
        pg = permuted_golds(golds, rng, bucketed)
        s, p, m, _ = metrics(preds_by_prob, pg)
        sc.append(s); pk.append(p); mc.append(m)
    def band(xs):
        xs = sorted(xs)
        return (sum(xs) / len(xs), xs[int(0.975 * len(xs))])  # (mean, p97.5)
    return band(sc), band(pk), band(mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="select_eval.py --dump-preds JSON")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    data = json.load(open(args.preds))
    print("=" * 92)
    print(f"CHANCE-COLLISION NULL CONTROL  {args.label}  (B={args.B} permutations)")
    print("  excess = observed - null_mean ; a metric is 'real capability' only far above its null p97.5")
    print("=" * 92)
    for src, d in data.items():
        golds = d["golds"]
        for fam in ("preds_full", "preds_forced"):
            preds = d[fam]
            s_o, p_o, m_o, k = metrics(preds, golds)
            (s_gm, s_g95), (p_gm, p_g95), (m_gm, m_g95) = null_band(preds, golds, args.B, False)
            (s_bm, s_b95), (p_bm, p_b95), (m_bm, m_b95) = null_band(preds, golds, args.B, True)
            ftag = "full" if fam == "preds_full" else "forced"
            print(f"\n  [{src} / {ftag}]  n={len(golds)}  K={k}")
            print(f"    {'metric':<14}{'observed':>10}{'null(glob)':>12}{'excess':>9}{'null(mag)':>12}{'excess':>9}")
            for name, o, gm, g95, bm, b95 in [
                ("single-acc", s_o, s_gm, s_g95, s_bm, s_b95),
                ("self-cons", m_o, m_gm, m_g95, m_bm, m_b95),
                (f"pass@{k}", p_o, p_gm, p_g95, p_bm, p_b95),
            ]:
                print(f"    {name:<14}{100*o:>9.1f}%{100*gm:>11.1f}%{100*(o-gm):>8.1f}{100*bm:>11.1f}%{100*(o-bm):>8.1f}")
    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
