#!/usr/bin/env python3
"""Build the argonne4.0 training-loss figure for the HF model card, and print per-stage stats.

The a4 sibling of `plot_a35_loss.py`, with two differences forced by how a4 actually ran:

1. **Four stages, not three** — pretrain -> reasoning anneal (phase A) -> ctx-extension to 13,568
   (phase B) -> ctx-extension to 65,536 (phase C).
2. **Phase C came from a DIFFERENT launcher** (`midtrain_c_a4.sh`, not the marker-gated
   `run_full_training.sh` chain), so it writes `report/argonne4.0/a4midc-<jobid>.out` and prints
   no `Stage:` banner. Stage attribution is therefore filename-based for phase C and
   banner-based for the other three. Parsing only the banners silently drops phase C entirely.

Rows are de-duplicated by step keeping the last occurrence, because a wall-time slice that saves
and resumes re-logs a few steps.

Phase C's per-step loss swings ~0.5 to ~2.5 between adjacent logged steps: at block 65,536 one
step is 15 sequences and the corpus interleaves long arXiv with short replay, so a single step's
loss is a sample of the mixture, not a trend. The raw trace is drawn faintly and a rolling median
solid on top; reading a trend off the raw phase-C line is a mistake.
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
LOGDIR = os.path.join(REPO, "report", "argonne4.0")
STEP_RE = re.compile(r"^Step (\d+) \| Loss: ([\d.]+) \| PPL: ([\d.eE+]+) \| Tokens: ([\d,]+) \| LR: ([\d.eE+-]+)")

STAGES = ("pretrain", "anneal", "midtrain_b", "midtrain_c")
LABELS = {
    "pretrain": "pretrain (block 1,024)",
    "anneal": "phase A anneal (block 1,024)",
    "midtrain_b": "phase B ctx 13,568",
    "midtrain_c": "phase C ctx 65,536",
}
COLORS = {"pretrain": "#2b6cb0", "anneal": "#b7791f", "midtrain_b": "#2f855a", "midtrain_c": "#9b2c2c"}


def _num(path, pattern):
    m = re.search(pattern, path)
    return int(m.group(1)) if m else 0


def rolling_median(xs, window=21):
    if len(xs) < 3:
        return list(xs)
    half = max(1, window // 2)
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        chunk = sorted(xs[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def collect():
    """step -> (tokens, loss, ppl, lr, stage). Later files overwrite earlier ones."""
    rows = {}
    # the marker-gated chain: stage comes from the banner each slice prints
    for path in sorted(glob.glob(os.path.join(LOGDIR, "*-train.out")),
                       key=lambda p: _num(p, r"/(\d+)-train\.out$")):
        stage = "pretrain"
        with open(path, errors="ignore") as fh:
            for line in fh:
                if line.startswith("Stage: continue_pretrain.py (MIDTRAIN phase B"):
                    stage = "midtrain_b"
                elif line.startswith("Stage: continue_pretrain.py (MIDTRAIN phase A"):
                    stage = "anneal"
                elif line.startswith("Stage: continue_pretrain.py"):
                    stage = "anneal"
                m = STEP_RE.match(line)
                if m:
                    rows[int(m.group(1))] = (int(m.group(4).replace(",", "")), float(m.group(2)),
                                             float(m.group(3)), float(m.group(5)), stage)
    # phase C: separate launcher, no banner -> attribute by filename
    for path in sorted(glob.glob(os.path.join(LOGDIR, "a4midc-*.out")),
                       key=lambda p: _num(p, r"a4midc-(\d+)\.out$")):
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = STEP_RE.match(line)
                if m:
                    rows[int(m.group(1))] = (int(m.group(4).replace(",", "")), float(m.group(2)),
                                             float(m.group(3)), float(m.group(5)), "midtrain_c")
    return rows


def main():
    rows = collect()
    steps = sorted(rows)
    if not steps:
        sys.exit(f"no Step lines found under {LOGDIR}")
    tok = [rows[s][0] / 1e9 for s in steps]
    loss = [rows[s][1] for s in steps]
    ppl = [rows[s][2] for s in steps]
    lr = [rows[s][3] for s in steps]
    stage = [rows[s][4] for s in steps]

    print(f"parsed {len(steps)} logged points, steps {steps[0]}..{steps[-1]}, "
          f"{tok[-1]:.2f}B cumulative tokens")
    stats = {}
    for name in STAGES:
        idx = [i for i, s in enumerate(stage) if s == name]
        if not idx:
            continue
        lo, hi = idx[0], idx[-1]
        stats[name] = dict(first_step=steps[lo], last_step=steps[hi],
                           tokens_start_B=round(tok[lo], 3), tokens_end_B=round(tok[hi], 3),
                           tokens_B=round(tok[hi] - tok[lo], 3),
                           loss_first=loss[lo], loss_last=loss[hi],
                           loss_median_last50=round(
                               sorted([loss[i] for i in idx[-50:]])[len(idx[-50:]) // 2], 4),
                           lr_first=lr[lo], lr_last=lr[hi], n_points=len(idx))
        s = stats[name]
        print(f"  {name:11s} steps {s['first_step']:>7}..{s['last_step']:<7} "
              f"tokens {s['tokens_start_B']:>7.2f}B -> {s['tokens_end_B']:>7.2f}B "
              f"({s['tokens_B']:>6.2f}B)  LR {s['lr_first']:.2e} -> {s['lr_last']:.2e}  "
              f"loss_med(last50) {s['loss_median_last50']}")

    bounds = [(steps[i], tok[i], stage[i]) for i in range(1, len(steps)) if stage[i] != stage[i - 1]]

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name in STAGES:
        idx = [i for i, s in enumerate(stage) if s == name]
        if not idx:
            continue
        xs = [tok[i] for i in idx]
        ax[0].plot(xs, [loss[i] for i in idx], lw=0.6, color=COLORS[name], alpha=0.30)
        ax[0].plot(xs, rolling_median([loss[i] for i in idx]), lw=1.1, color=COLORS[name],
                   label=LABELS[name])
        ax[1].plot(xs, [ppl[i] for i in idx], lw=0.6, color=COLORS[name], alpha=0.30)
        ax[1].plot(xs, rolling_median([ppl[i] for i in idx]), lw=1.1, color=COLORS[name])
        ax[2].plot(xs, [lr[i] for i in idx], lw=1.0, color=COLORS[name])
    for _, tb, name in bounds:
        for a in ax:
            a.axvline(tb, color="#718096", ls="--", lw=0.8)
        ax[0].annotate(name.replace("midtrain_", "phase "), xy=(tb, ax[0].get_ylim()[0]),
                       xytext=(4, 6), textcoords="offset points", fontsize=8, color="#4a5568")
    ax[0].set_ylabel("train loss"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title("Argonne 4.0-base — training loss, perplexity, and LR vs cumulative tokens\n"
                    "(faint = raw logged step; solid = rolling median)", fontsize=11)
    ax[1].set_ylabel("perplexity"); ax[1].set_yscale("log")
    ax[2].set_ylabel("learning rate"); ax[2].set_yscale("log")
    ax[2].set_xlabel("cumulative tokens (billions)")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()

    out_png = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "report", "a4_loss_plot.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")
    out_json = os.path.splitext(out_png)[0] + "_stages.json"
    json.dump(stats, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
