#!/usr/bin/env python3
"""Build the argonne-4.5 training-loss figure for the README and HF cards, and print per-stage stats.

The a4.5 sibling of `plot_a4_loss.py`. Three stages, attributed by GLOBAL step (the three launchers all
log `Step <global> | Loss | PPL | Tokens <cumulative> | LR`):
  pretrain            steps       1..203,451   block 1,024     -> argonne-4.5 pretrain
  phase A anneal      steps 203,452..236,866   block 1,024     -> argonne-4.5-base
  phase B ctx 13,568  steps 236,867..268,461   block 13,568    -> argonne-4.5-base-ctx13568

The run moved between machines, so its logs live in several directories: the first chain's
`<n>-train.out` (steps 50..64,850), then `slice-001.out` and the `a45_*` slice logs. Files are replayed in
order and rows are keyed by step, so a step that was trained again after a rollback (a slice that resumed
from an earlier checkpoint, or a segment retrained after a corrupted compile in August) keeps only its LAST
occurrence. The first chain goes first, in its slice-counter order (three of its logs were rewritten on
2026-09-17, so their mtimes are not when they were written); every later log goes by mtime, the one clock
they share (their names mix UTC and local stamps). A slice named `_step<N>_` resumed at global step N, so
a row below N is a local counter from a restart that never loaded its step, and is dropped.

  python reasoning/plot_a45_loss.py <log dir> [<log dir> ...] --out plots/argonne4_5_loss_plot.png
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"Step (\d+) \| Loss: ([\d.]+) \| PPL: ([\d.eE+]+) \| Tokens: ([\d,]+) \| LR: ([\d.eE+-]+)")
KEEP_RE = re.compile(r"^(\d+-train|slice-\d+|a45_.+)\.out$")   # production logs, not probes or measurements
CHAIN_RE = re.compile(r"^(\d+)-train\.out$")
RESUME_RE = re.compile(r"_step(\d+)_")
STAGES = ("pretrain", "anneal", "midtrain_b")
BOUNDS = {"pretrain": 203451, "anneal": 236866}          # last global step of each stage
LABELS = {"pretrain": "pretrain (block 1,024)", "anneal": "phase A anneal (block 1,024)",
          "midtrain_b": "phase B ctx 13,568"}
COLORS = {"pretrain": "#2b6cb0", "anneal": "#b7791f", "midtrain_b": "#2f855a"}


def stage_of(step):
    if step <= BOUNDS["pretrain"]:
        return "pretrain"
    return "anneal" if step <= BOUNDS["anneal"] else "midtrain_b"


def rolling_median(xs, window=21):
    if len(xs) < 3:
        return list(xs)
    half = max(1, window // 2)
    return [sorted(xs[max(0, i - half):i + half + 1])[len(xs[max(0, i - half):i + half + 1]) // 2]
            for i in range(len(xs))]


def collect(dirs):
    files = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.out")):
            name = os.path.basename(p)
            if KEEP_RE.match(name):
                m = CHAIN_RE.match(name)
                files.append(((0, int(m.group(1))) if m else (1, os.path.getmtime(p)), p))
    rows, dropped = {}, 0
    for _, path in sorted(files):
        m = RESUME_RE.search(os.path.basename(path))
        floor = int(m.group(1)) if m else 0
        text = open(path, errors="ignore").read().replace("\r", "\n")
        for m in STEP_RE.finditer(text):
            step = int(m.group(1))
            if step < floor:
                dropped += 1
                continue
            rows[step] = (int(m.group(4).replace(",", "")), float(m.group(2)), float(m.group(3)),
                          float(m.group(5)), stage_of(step))
    return rows, len(files), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="plots/argonne4_5_loss_plot.png")
    a = ap.parse_args()
    rows, n_files, dropped = collect(a.dirs)
    steps = sorted(rows)
    assert steps, "no Step lines found"
    tok = [rows[s][0] / 1e9 for s in steps]
    loss = [rows[s][1] for s in steps]
    ppl = [rows[s][2] for s in steps]
    lr = [rows[s][3] for s in steps]
    stage = [rows[s][4] for s in steps]
    print(f"parsed {len(steps)} logged points from {n_files} logs, steps {steps[0]}..{steps[-1]}, "
          f"{tok[-1]:.2f}B cumulative tokens; dropped {dropped} local-counter rows")
    stats = {}
    for name in STAGES:
        idx = [i for i, s in enumerate(stage) if s == name]
        if not idx:
            continue
        lo, hi = idx[0], idx[-1]
        last50 = sorted(loss[i] for i in idx[-50:])
        stats[name] = dict(first_step=steps[lo], last_step=steps[hi], tokens_start_B=round(tok[lo], 3),
                           tokens_end_B=round(tok[hi], 3), tokens_B=round(tok[hi] - tok[lo], 3),
                           loss_first=loss[lo], loss_last=loss[hi], loss_median_last50=round(last50[len(last50) // 2], 4),
                           lr_first=lr[lo], lr_last=lr[hi], n_points=len(idx))
        s = stats[name]
        rate = (rows[steps[hi]][0] - rows[steps[lo]][0]) / max(1, steps[hi] - steps[lo])
        s["tokens_per_step"] = round(rate)
        s["rows_off_token_line"] = sum(abs(rows[steps[i]][0] - rows[steps[lo]][0] - rate * (steps[i] - steps[lo]))
                                       > 0.01 * rate * max(1, steps[i] - steps[lo]) for i in idx)
        print(f"  {name:11s} steps {s['first_step']:>7}..{s['last_step']:<7} tokens {s['tokens_start_B']:>7.2f}B -> "
              f"{s['tokens_end_B']:>7.2f}B ({s['tokens_B']:>6.2f}B)  LR {s['lr_first']:.2e} -> {s['lr_last']:.2e}  "
              f"loss_med(last50) {s['loss_median_last50']}  points {s['n_points']}  "
              f"tok/step {s['tokens_per_step']:,} (rows off that line: {s['rows_off_token_line']})")
    # the biggest step gaps in the logged record (a gap is a slice whose log is missing, not lost training)
    gaps = sorted(((steps[i] - steps[i - 1], steps[i - 1], steps[i]) for i in range(1, len(steps))), reverse=True)[:3]
    print("largest logged-step gaps:", ", ".join(f"{g} ({p}->{q})" for g, p, q in gaps))

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name in STAGES:
        idx = [i for i, s in enumerate(stage) if s == name]
        if not idx:
            continue
        xs = [tok[i] for i in idx]
        ax[0].plot(xs, [loss[i] for i in idx], lw=0.6, color=COLORS[name], alpha=0.30)
        ax[0].plot(xs, rolling_median([loss[i] for i in idx]), lw=1.1, color=COLORS[name], label=LABELS[name])
        ax[1].plot(xs, [ppl[i] for i in idx], lw=0.6, color=COLORS[name], alpha=0.30)
        ax[1].plot(xs, rolling_median([ppl[i] for i in idx]), lw=1.1, color=COLORS[name])
        ax[2].plot(xs, [lr[i] for i in idx], lw=1.0, color=COLORS[name])
    for name in ("anneal", "midtrain_b"):
        if name in stats:
            for axis in ax:
                axis.axvline(stats[name]["tokens_start_B"], color="#718096", ls="--", lw=0.8)
    ax[0].set_ylabel("train loss"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title("Argonne 4.5 training loss, perplexity, and LR vs cumulative tokens\n"
                    "(faint = raw logged step; solid = rolling median)", fontsize=11)
    ax[0].set_ylim(top=min(max(loss), 6.0))
    ax[1].set_ylabel("perplexity"); ax[1].set_yscale("log")
    ax[2].set_ylabel("learning rate"); ax[2].set_yscale("log")
    ax[2].set_xlabel("cumulative tokens (billions)")
    for axis in ax:
        axis.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=140)
    json.dump(stats, open(os.path.splitext(a.out)[0] + "_stages.json", "w"), indent=2)
    print(f"wrote {a.out} and its _stages.json")


if __name__ == "__main__":
    main()
