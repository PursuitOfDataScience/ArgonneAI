#!/usr/bin/env python3
"""Build the argonne3.5 training-loss figure for the HF model card, and print per-stage stats.

Parses every `Step N | Loss: L | PPL: P | Tokens: T | LR: R` line the three stages wrote into
report/*-train.out (slices 1..105 cover the whole run: pretrain -> reasoning anneal ->
ctx-extension midtrain). Slices overlap by a few steps where a wall-time save was re-done on
resume, so rows are de-duplicated by step, keeping the last occurrence.

Stage boundaries are detected from the logs themselves (the stage banner each slice prints),
not hardcoded, so the figure stays correct if the run is extended.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_RE = re.compile(r"^Step (\d+) \| Loss: ([\d.]+) \| PPL: ([\d.eE+]+) \| Tokens: ([\d,]+) \| LR: ([\d.eE+-]+)")


def slice_index(p):
    m = re.search(r"/(\d+)-train\.out$", p)
    return int(m.group(1)) if m else 0


def main():
    rows = {}           # step -> (tokens, loss, ppl, lr)
    stage_of_slice = {}
    for p in sorted(glob.glob(os.path.join(REPO, "report", "*-train.out")), key=slice_index):
        stage = "pretrain"
        with open(p, errors="ignore") as fh:
            for line in fh:
                if line.startswith("Stage: continue_pretrain.py (MIDTRAIN"):
                    stage = "midtrain"
                elif line.startswith("Stage: continue_pretrain.py"):
                    stage = "anneal"
                m = STEP_RE.match(line)
                if m:
                    step = int(m.group(1))
                    rows[step] = (int(m.group(4).replace(",", "")), float(m.group(2)),
                                  float(m.group(3)), float(m.group(5)), stage)
        stage_of_slice[slice_index(p)] = stage

    steps = sorted(rows)
    if not steps:
        sys.exit("no Step lines found in report/*-train.out")
    tok = [rows[s][0] / 1e9 for s in steps]
    loss = [rows[s][1] for s in steps]
    ppl = [rows[s][2] for s in steps]
    lr = [rows[s][3] for s in steps]
    stage = [rows[s][4] for s in steps]

    # stage boundaries (first step of each stage after the first)
    bounds = []
    for i in range(1, len(steps)):
        if stage[i] != stage[i - 1]:
            bounds.append((steps[i], tok[i], stage[i]))

    print(f"parsed {len(steps)} logged points, steps {steps[0]}..{steps[-1]}, "
          f"{tok[-1]:.2f}B cumulative tokens")
    stats = {}
    for name in ("pretrain", "anneal", "midtrain"):
        idx = [i for i, s in enumerate(stage) if s == name]
        if not idx:
            continue
        lo, hi = idx[0], idx[-1]
        stats[name] = dict(first_step=steps[lo], last_step=steps[hi],
                           tokens_start_B=round(tok[lo], 3), tokens_end_B=round(tok[hi], 3),
                           tokens_B=round(tok[hi] - tok[lo], 3),
                           loss_first=loss[lo], loss_last=loss[hi],
                           lr_first=lr[lo], lr_last=lr[hi], n_points=len(idx))
        s = stats[name]
        print(f"  {name:9s} steps {s['first_step']:>7}..{s['last_step']:<7} "
              f"tokens {s['tokens_start_B']:>7.2f}B -> {s['tokens_end_B']:>7.2f}B "
              f"({s['tokens_B']:>6.2f}B)  LR {s['lr_first']:.2e} -> {s['lr_last']:.2e}")

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    colors = {"pretrain": "#2b6cb0", "anneal": "#b7791f", "midtrain": "#2f855a"}
    for name in ("pretrain", "anneal", "midtrain"):
        idx = [i for i, s in enumerate(stage) if s == name]
        if not idx:
            continue
        ax[0].plot([tok[i] for i in idx], [loss[i] for i in idx], lw=0.7,
                   color=colors[name], label=name)
        ax[1].plot([tok[i] for i in idx], [ppl[i] for i in idx], lw=0.7, color=colors[name])
        ax[2].plot([tok[i] for i in idx], [lr[i] for i in idx], lw=0.9, color=colors[name])
    # Stage labels go at the BOTTOM of the loss panel: the top-right corner is where the
    # legend sits and where the pretrain curve peaks, so a top annotation collides with both.
    for _, tb, name in bounds:
        for a in ax:
            a.axvline(tb, color="#718096", ls="--", lw=0.8)
        ax[0].annotate(name, xy=(tb, ax[0].get_ylim()[0]), xytext=(4, 6),
                       textcoords="offset points", fontsize=8, color="#4a5568")
    ax[0].set_ylabel("train loss"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title("Argonne 3.5-base — training loss, perplexity, and LR vs cumulative tokens")
    ax[1].set_ylabel("perplexity"); ax[1].set_yscale("log")
    ax[2].set_ylabel("learning rate"); ax[2].set_yscale("log")
    ax[2].set_xlabel("cumulative tokens (billions)")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()

    out_png = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "report", "a35_loss_plot.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")
    out_json = os.path.splitext(out_png)[0] + "_stages.json"
    json.dump(stats, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
