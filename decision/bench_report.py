"""Score argonne-4.5-decision and open baselines on the same never-trained decision sets with one piece of code.

    python decision/bench_report.py --ours eval.npz --temps decision_config.json --baseline name=path.npz ... --out report.json

Accuracy per set and calibration error (10-bin ECE on the top answer) where a model gives real probabilities
(`ood_scored`; a baseline that writes its answer out has accuracy only). Every model is scored on exactly the items
all models answered, matched by id.
"""
import argparse
import json

import numpy as np

SETS = [("banking77", "Banking77 intents, 77 options"), ("newsgroups", "20 Newsgroups, 20 options"),
        ("fin_topic", "Finance news topics, 20 options"), ("fin_sentiment", "Finance news sentiment, 3 options"),
        ("mmlu", "MMLU, 4 options"), ("truthfulqa", "TruthfulQA MC1"),
        ("banking77_rubric", "Banking77 intents, as a written rule"), ("newsgroups_rubric", "20 Newsgroups, as a written rule"),
        ("fin_topic_rubric", "Finance news topics, as a written rule"),
        ("enron_spam", "Spam email (yes / no)"), ("aegis", "Unsafe chat request (yes / no)")]


def load(path, temps=None):
    z = np.load(path)
    L = z["ood_logits"].astype(np.float64)
    names = np.array(z["ood_source_names"])
    src = names[z["ood_source"]]
    typ = z["ood_type"] if "ood_type" in z else np.zeros(len(L))
    scored = z["ood_scored"] if "ood_scored" in z else np.ones(len(L), dtype=bool)
    if temps:                                          # the model's own fitted temperatures, per question type
        tt = np.array([temps.get({0: "choice", 1: "score", 2: "noul"}[int(t)], 1.0) for t in typ])[:, None]
        L = L / tt
    L = np.where(np.isfinite(L), L, -np.inf)
    z_ = L - L.max(1, keepdims=True)
    P = np.exp(z_)
    P /= P.sum(1, keepdims=True)
    return {i: (p, int(y), s, bool(sc)) for i, p, y, s, sc in zip(z["ood_ids"], P, z["ood_label"], src, scored)}


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok, dtype=float)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins) if b else (conf <= 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--temps", default="")
    ap.add_argument("--baseline", action="append", default=[], help="name=path.npz")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    temps = json.load(open(args.temps)).get("temperatures") if args.temps else None
    models = {"argonne-4.5-decision": load(args.ours, temps)}
    for b in args.baseline:
        name, path = b.split("=", 1)
        models[name] = load(path)
    common = set.intersection(*(set(m) for m in models.values()))
    rep = {"n_common": len(common), "sets": {}}
    for key, label in SETS:
        ids = sorted(i for i in common if models["argonne-4.5-decision"][i][2] == key)
        if not ids:
            continue
        row = {"label": label, "n": len(ids)}
        for name, m in models.items():
            ok = [int(m[i][0].argmax() == m[i][1]) for i in ids]
            sc = [i for i in ids if m[i][3]]
            row[name] = {"acc": round(100 * float(np.mean(ok)), 1),
                         "ece": round(ece([m[i][0].max() for i in sc], [m[i][0].argmax() == m[i][1] for i in sc]), 3)
                         if len(sc) == len(ids) else None}
        rep["sets"][key] = row
    for name in models:                                 # unweighted mean over the sets
        accs = [r[name]["acc"] for r in rep["sets"].values()]
        eces = [r[name]["ece"] for r in rep["sets"].values() if r[name]["ece"] is not None]
        rep.setdefault("mean", {})[name] = {"acc": round(float(np.mean(accs)), 1),
                                            "ece": round(float(np.mean(eces)), 3) if len(eces) == len(accs) else None}
    # paired, on identical items: ours minus each baseline, pooled over every set; bootstrap over items, exact McNemar
    from math import comb
    ids = sorted(i for i in common if any(models["argonne-4.5-decision"][i][2] == k for k, _ in SETS))
    a = np.array([models["argonne-4.5-decision"][i][0].argmax() == models["argonne-4.5-decision"][i][1] for i in ids], float)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    for name, m in models.items():
        if name == "argonne-4.5-decision":
            continue
        b = np.array([m[i][0].argmax() == m[i][1] for i in ids], float)
        d = a - b
        ci = np.percentile([d[s].mean() for s in boots], [2.5, 97.5])
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        k, n = min(n10, n01), n10 + n01
        p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
        rep.setdefault("paired", {})[name] = {"items": len(ids), "ours_minus_baseline_pt": round(100 * d.mean(), 1),
                                              "ci95_pt": [round(100 * ci[0], 1), round(100 * ci[1], 1)],
                                              "ours_only_right": n10, "baseline_only_right": n01, "mcnemar_p": p}
    txt = json.dumps(rep, indent=1)
    if args.out:
        open(args.out, "w").write(txt)
    names = list(models)
    print(f"{'set':42s}" + "".join(f"{n[:22]:>24s}" for n in names))
    for r in rep["sets"].values():
        print(f"{r['label'][:42]:42s}" + "".join(
            f"{r[n]['acc']:>14.1f}" + (f" ({r[n]['ece']:.3f})" if r[n]['ece'] is not None else " " * 8) + "  " for n in names))
    print(f"{'mean':42s}" + "".join(f"{rep['mean'][n]['acc']:>14.1f}" + (f" ({rep['mean'][n]['ece']:.3f})" if rep['mean'][n]['ece'] is not None else " " * 8) + "  " for n in names))


if __name__ == "__main__":
    main()
