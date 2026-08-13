#!/usr/bin/env python3
"""Emit the model-card benchmark table from lm-eval result JSONs, as markdown.

WHY A SCRIPT AND NOT HAND-TYPING. A release table is ~40 numbers copied out of JSON into markdown,
and `lmeval_summary.py`'s docstring already records what that costs: a 3.74pt phantom regression
caused purely by reading `acc` from one run and `acc_norm` from another. This applies ONE metric
rule to every arm and prints the source file for each column, so the card cannot silently mix them.

THE METRIC RULE (identical to thinking_training.md §39d/§39f, so the rows merge with the banked
tables): `acc_norm` for the multiple-choice tasks, `acc` for winogrande and mmlu, which do not
report a length-normalised variant. gsm8k is reported separately in both strict-match and
flexible-extract because it is the only GENERATIVE task in the suite and the two disagree in ways
that matter (strict-match rewards emitting the `#### N` format, not just getting the answer right).

Usage:
  python reasoning/release_table.py \
      "Argonne 4.0-base=report/a4_release_lmeval_a4_pc_112674.json" \
      "stage 3 (ctx 13,568)=report/a4_pcgate_lmeval_a4_pb.json" \
      "Llama-3.2-1B=report/a4_pcanchor_lmeval_llama32_1b.json" \
      "Qwen3-0.6B-Base=report/a4_pcanchor_lmeval_qwen3_06b.json"
"""
import json
import sys

# (task, metric key) -- ordered as the card prints them
MC = [
    ("arc_challenge", "acc_norm,none"),
    ("arc_easy", "acc_norm,none"),
    ("hellaswag", "acc_norm,none"),
    ("piqa", "acc_norm,none"),
    ("sciq", "acc_norm,none"),
    ("openbookqa", "acc_norm,none"),
    ("winogrande", "acc,none"),          # no acc_norm reported
    ("mmlu", "acc,none"),                # no acc_norm reported
]
GSM8K = [("exact_match,strict-match", "gsm8k strict-match"),
         ("exact_match,flexible-extract", "gsm8k flexible-extract")]
PRETTY = {"arc_challenge": "arc_challenge", "arc_easy": "arc_easy", "hellaswag": "hellaswag",
          "piqa": "piqa", "sciq": "sciq", "openbookqa": "openbookqa",
          "winogrande": "winogrande *(acc)*", "mmlu": "mmlu *(acc)*"}


def load(path):
    return json.load(open(path))


def cell(v):
    return "—" if v is None else f"{v:.2f}"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    arms = []
    for spec in sys.argv[1:]:
        name, _, path = spec.partition("=")
        arms.append((name, path, load(path)))

    rows = {}
    for task, key in MC:
        rows[task] = [(100 * d[task][key] if task in d and key in d.get(task, {}) else None)
                      for _, _, d in arms]
    means = []
    for i in range(len(arms)):
        vals = [rows[t][i] for t, _ in MC if rows[t][i] is not None]
        means.append(sum(vals) / len(vals) if len(vals) == len(MC) else None)

    hdr = " | ".join(n for n, _, _ in arms)
    print(f"| task | {hdr} |")
    print("|---|" + "---:|" * len(arms))
    for task, _ in MC:
        print(f"| {PRETTY[task]} | " + " | ".join(cell(v) for v in rows[task]) + " |")
    print(f"| **8-task mean** | " + " | ".join(f"**{cell(v)}**" for v in means) + " |")
    for key, label in GSM8K:
        vals = [(100 * d["gsm8k"][key] if "gsm8k" in d and key in d.get("gsm8k", {}) else None)
                for _, _, d in arms]
        bold = "**" if "strict" in key else ""
        print(f"| {bold}{label}{bold} | "
              + " | ".join(f"{bold}{cell(v)}{bold}" for v in vals) + " |")

    print("\nsources:")
    for name, path, _ in arms:
        print(f"  {name:<28} {path}")
    missing = {t for t, _ in MC for i, (_, _, d) in enumerate(arms) if rows[t][i] is None}
    if missing:
        print("\nWARNING incomplete tasks (means suppressed for those arms):", sorted(missing))


if __name__ == "__main__":
    main()
