"""Paired comparison of two evaluated decision models on the SAME test items (matched by example id).

    python decision/compare.py --a eval_run1.npz --b eval_run2.npz --split test [--temps_a ... --temps_b ...]

Accuracy is compared pair by pair on items both models saw: per question type, per source, and overall, with a
percentile-bootstrap 95% CI on the accuracy difference (items resampled, 10,000 draws) and an exact McNemar test
(two-sided binomial on the discordant pairs). n and the discordant counts are printed, so a difference that rests
on a handful of items is visible as such.
"""
import argparse
import math

import numpy as np

TYPES = ["choice", "score", "noul"]


def load(path, split):
    z = np.load(path, allow_pickle=False)
    if f"{split}_ids" not in z:
        raise SystemExit(f"{path} has no {split}_ids (evaluated before ids were saved; re-run evaluate.py)")
    names = list(z[f"{split}_source_names"])
    L = z[f"{split}_logits"]
    correct = L.argmax(1) == z[f"{split}_label"]
    return {str(i): (bool(c), int(t), names[int(s)]) for i, c, t, s in
            zip(z[f"{split}_ids"], correct, z[f"{split}_type"], z[f"{split}_source"])}


def mcnemar_p(b, c):
    """Exact two-sided McNemar: P(X <= min(b, c)) * 2 under Binomial(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(pairs, rng):
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    d = b - a
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    bc = int(((a == 1) & (b == 0)).sum())
    cb = int(((a == 0) & (b == 1)).sum())
    return {"n": n, "acc_a": a.mean(), "acc_b": b.mean(), "delta": d.mean(), "ci": (lo, hi),
            "a_only": bc, "b_only": cb, "p": mcnemar_p(bc, cb)}


def fmt(name, r):
    return (f"{name:28s} n={r['n']:6d}  A {r['acc_a']:.4f}  B {r['acc_b']:.4f}  B-A {r['delta']:+.4f} "
            f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  discordant A-only {r['a_only']} / B-only {r['b_only']}  "
            f"McNemar p {r['p']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    A, B = load(args.a, args.split), load(args.b, args.split)
    common = sorted(set(A) & set(B))
    print(f"{len(common)} common {args.split} items ({len(A)} in A, {len(B)} in B)")
    rng = np.random.default_rng(0)
    print(fmt("ALL", compare([(A[i][0], B[i][0]) for i in common], rng)))
    for t, name in enumerate(TYPES):
        sel = [i for i in common if A[i][1] == t]
        if sel:
            print(fmt(name, compare([(A[i][0], B[i][0]) for i in sel], rng)))
    for src in sorted({A[i][2] for i in common}):
        sel = [i for i in common if A[i][2] == src]
        if len(sel) >= 50:
            print(fmt("  " + src, compare([(A[i][0], B[i][0]) for i in sel], rng)))


if __name__ == "__main__":
    main()
