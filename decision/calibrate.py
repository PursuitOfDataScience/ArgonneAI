"""Fit one temperature per question type on the VAL logits and write it into the model's decision_config.json.

    python decision/calibrate.py --model DIR --eval eval.json      (reads eval.npz next to eval.json)

Temperature scaling (divide the option logits by T before the softmax) keeps every argmax, so accuracy is
unchanged; it only fixes over- or under-confidence. T is fitted by minimising log loss on val (a proper scoring
rule, the same objective Jev's RLCD rewards), then test and ood are re-scored with it so the gain is measured on
data the fit never saw. The per-type result lands in eval.json under "calibrated".
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.train import ece  # noqa: E402

TYPES = ["choice", "score", "noul"]


def nll(L, T, temp):
    z = L / temp
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-(T * np.where(T > 0, logp, 0)).sum(1).mean())


def fit_temperature(L, T):
    """Golden-section search on log T in [0.2, 5] (log loss is unimodal in T for fixed logits)."""
    a, b = np.log(0.2), np.log(5.0)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(L, T, np.exp(c)), nll(L, T, np.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(L, T, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(L, T, np.exp(d))
    return float(np.exp((a + b) / 2))


def scored(L, T, label, temp):
    z = L / temp
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    ok = pred == label
    return {"acc": float(ok.mean()), "nll": nll(L, T, temp), "ece": ece(conf, ok),
            "brier": float(((p - T) ** 2).sum(1).mean()), "n": int(len(L))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()
    raw = np.load(os.path.splitext(args.eval)[0] + ".npz")
    temps, report = {}, {}
    for t, name in enumerate(TYPES):
        if "val_logits" not in raw:
            break
        sel = raw["val_type"] == t
        if not sel.any():
            continue
        L, T = raw["val_logits"][sel].astype(np.float64), raw["val_target"][sel].astype(np.float64)
        L = np.where(np.isfinite(L), L, -1e9)
        temps[name] = round(fit_temperature(L, T), 4)
        report[name] = {"temperature": temps[name]}
        for split in ("val", "test", "ood"):
            if f"{split}_logits" not in raw:
                continue
            s2 = raw[f"{split}_type"] == t
            if not s2.any():
                continue
            L2 = np.where(np.isfinite(raw[f"{split}_logits"][s2]), raw[f"{split}_logits"][s2], -1e9).astype(np.float64)
            T2 = raw[f"{split}_target"][s2].astype(np.float64)
            y2 = raw[f"{split}_label"][s2]
            report[name][split] = {"before": scored(L2, T2, y2, 1.0), "after": scored(L2, T2, y2, temps[name])}
    cfg_path = os.path.join(args.model, "decision_config.json")
    cfg = json.load(open(cfg_path))
    cfg["temperatures"] = temps
    cfg["calibration"] = {"method": "temperature scaling, one per type, fitted on val by log loss", "eval": args.eval}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    ev = json.load(open(args.eval))
    ev["calibrated"] = report
    with open(args.eval, "w") as f:
        json.dump(ev, f, indent=1)
    for name, r in report.items():
        line = f"[calibrate] {name}: T={r['temperature']}"
        for split in ("test", "ood"):
            if split in r:
                line += (f" | {split} ece {r[split]['before']['ece']:.3f} -> {r[split]['after']['ece']:.3f}, "
                         f"nll {r[split]['before']['nll']:.3f} -> {r[split]['after']['nll']:.3f}, acc {r[split]['after']['acc']:.3f}")
        print(line, flush=True)
    print(f"[calibrate] wrote temperatures {temps} into {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
