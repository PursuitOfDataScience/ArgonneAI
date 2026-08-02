#!/usr/bin/env python3
"""Model-card figures for Argonne 3.5-think.

Palette is the validated categorical default, slots 1-3 (blue / orange / aqua), which clears
all-pairs CVD and normal-vision floors in light mode. Aqua sits below 3:1 on the light surface,
so the relief rule applies -- every bar carries a visible direct value label, which is also why
these read fine in grayscale/print.

Design choices worth stating:
 - Grouped bars, not a dual axis: every panel plots ONE measure family (percent), so a single
   y-scale is always correct.
 - Direct labels on every bar AND a legend, so identity never depends on color alone.
 - Recessive grid and axes; text in ink colors, never in a series color.
 - Fig 1's 3.0 numbers come from the controlled head-to-head run (same judge, same job) when
   report/h2h_a30.log exists; otherwise it falls back to the §23a recorded values and SAYS SO in
   the subtitle, so a reader is never misled about provenance.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (light)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "report/card_figs"
os.makedirs(OUT, exist_ok=True)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK2)


def labelled_bars(ax, groups, series, colors, ymax=100, fmt="{:.1f}"):
    """groups: list[str]; series: list[(name, [values])]. 2px visual gap between bars."""
    n = len(series)
    width = 0.80 / n
    xs = range(len(groups))
    for i, ((name, vals), c) in enumerate(zip(series, colors)):
        off = (i - (n - 1) / 2) * width
        pos = [x + off for x in xs]
        ax.bar(pos, vals, width=width * 0.94, color=c, label=name, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + ymax * 0.018, fmt.format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="medium")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, ymax)
    style(ax)


def parse_clean_eval(path):
    """-> {'svamp': (greedy, budget, selfcons, passk), 'asdiv': (...)} from a clean_eval log."""
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    out = {}
    for src in ("svamp", "asdiv"):
        m = re.search(rf"^\s*{src}\s+clean\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%",
                      txt, re.M)
        if m:
            out[src] = tuple(float(x) for x in m.groups())
    return out or None


# ---------------------------------------------------------------- fig 1: vs 3.0-think
h2h30, h2h35 = parse_clean_eval("report/h2h_a30.log"), parse_clean_eval("report/h2h_a35.log")
if h2h30 and h2h35:
    a30 = [h2h30["svamp"][0], h2h30["svamp"][2], h2h30["asdiv"][0], h2h30["asdiv"][2]]
    a35 = [h2h35["svamp"][0], h2h35["svamp"][2], h2h35["asdiv"][0], h2h35["asdiv"][2]]
    prov = "controlled head-to-head: both models, one job, identical judge / n=300 / K=8 / seed"
else:
    a30 = [18.0, 36.3, 22.7, 51.0]        # §23a, measured on v2 blend_star_a06
    a35 = [65.00, 74.00, 73.00, 82.67]
    prov = "3.0 column = §23a recorded values for v2 (NOT re-measured here); 3.5 measured 2026-08-02"

fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURFACE)
labelled_bars(ax, ["SVAMP\ngreedy", "SVAMP\nself-cons", "ASDiv\ngreedy", "ASDiv\nself-cons"],
              [("Argonne 3.0-think", a30), ("Argonne 3.5-think", a35)], [S2, S1])
ax.set_ylabel("accuracy (%)", color=INK2, fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.set_title("Argonne 3.5-think vs 3.0-think — uncontaminated SVAMP / ASDiv",
             color=INK, fontsize=12.5, fontweight="semibold", pad=14, loc="left")
ax.text(0, 1.015, prov, transform=ax.transAxes, fontsize=8, color=MUTED)
ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left", ncol=2,
          bbox_to_anchor=(0, -0.13))
fig.tight_layout()
fig.savefig(f"{OUT}/vs_3p0.png", dpi=160, facecolor=SURFACE)
print(f"wrote {OUT}/vs_3p0.png  ({'head-to-head' if h2h30 else 'fallback'} provenance)")

# ---------------------------------------------------------------- fig 2: termination
fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
labelled_bars(ax, ["SVAMP", "ASDiv"],
              [("same base, generic CoT mix", [53.7, 59.7]),
               ("Argonne 3.5-think (short-trace mix)", [1.3, 2.0])],
              [S2, S1], ymax=70)
ax.set_ylabel("answers never emitted (%)", color=INK2, fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.set_title("Non-termination solved at the weights", color=INK, fontsize=12.5,
             fontweight="semibold", pad=14, loc="left")
ax.text(0, 1.015, "greedy `no_answer` rate — the trace never closes </think>, so no answer is produced",
        transform=ax.transAxes, fontsize=8, color=MUTED)
ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left",
          bbox_to_anchor=(0, -0.13))
fig.tight_layout()
fig.savefig(f"{OUT}/termination.png", dpi=160, facecolor=SURFACE)
print(f"wrote {OUT}/termination.png")

# ---------------------------------------------------------------- fig 3: attribution
stages = ["prev base\n+ generic CoT", "new base\n+ generic CoT", "new base\n+ v6 mix",
          "+ SFT/DPO\n+ soup (shipped)"]
fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=SURFACE)
labelled_bars(ax, stages,
              [("greedy pass@1", [24.00, 25.67, 62.33, 65.00]),
               ("self-consistency K=8", [43.00, 62.33, 73.67, 74.00]),
               ("pass@8 (ceiling)", [58.67, 74.00, 91.33, 90.67])],
              [S1, S2, S3])
ax.set_ylabel("SVAMP accuracy (%)", color=INK2, fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.set_title("What moved the number: the base raised the ceiling, the mix raised the floor",
             color=INK, fontsize=12.5, fontweight="semibold", pad=14, loc="left")
ax.text(0, 1.015,
        "step 2 lifts the ceiling (+15 pass@8) with greedy FLAT; step 3 converts it (+36.7 greedy)",
        transform=ax.transAxes, fontsize=8, color=MUTED)
ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left", ncol=3,
          bbox_to_anchor=(0, -0.13))
fig.tight_layout()
fig.savefig(f"{OUT}/attribution.png", dpi=160, facecolor=SURFACE)
print(f"wrote {OUT}/attribution.png")
