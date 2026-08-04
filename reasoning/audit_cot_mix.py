#!/usr/bin/env python
"""audit_cot_mix.py -- report what cot-sft.py will ACTUALLY train on, per tier, before you train.

Both bugs found in §34 were invisible from the launcher and from the training log:

  1. `--max_think_tokens` defaults to 128, so 31.3% of v6's reasoning rows were severed
     mid-derivation (80-89% of the four hard math tiers) and nothing said so.
  2. `--preserve_raw_reasoning` defaults to 0, which routes rows through
     canonicalize_reasoning_turn(); when that returns None the row is silently REPLACED by a
     random other row, so the row count and step count are unchanged. 3,029 of 26,428 rows
     (11.5%) never trained, including 80.7% of `synth_arith`, and nothing said so either.

Both would have been caught in seconds by printing this table. Run it on any mix before training:

  python audit_cot_mix.py --data /project/rcc/youzhi/data/cot_sft_mix_v6 \
      --model /project/rcc/youzhi/models/a35_reason/dpo \
      --max-seq-length 1024 --max-think-tokens 128 --preserve-raw-reasoning 0

Compare two flag settings directly with --compare, which is the form that makes a bug obvious.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def load_cotsft():
    """Import cot-sft.py (hyphen in the name blocks a normal import)."""
    p = Path(__file__).resolve().parent / "cot-sft.py"
    spec = importlib.util.spec_from_file_location("cotsft", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cotsft"] = m
    spec.loader.exec_module(m)
    return m


SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def audit(m, ds, tok, max_seq, max_think, preserve_raw, allow_non_reasoning):
    """-> per-tier (total, dropped, truncated, conclusion-deleted) counts.

    `concl` counts rows where canonicalization deleted a sentence that STATES THE RESULT (contains
    "answer" and a digit). That was the most damaging of the three §34 defects and the least
    visible: on `gsm8k_train_short` it hit 94.5% of kept rows, so the model was trained on
    derivations that compute a value and never conclude -- the direct cause of §33v's
    double-application bug. It is not a row count, so nothing else surfaces it.
    """
    tot, dropped, trunc, concl = Counter(), Counter(), Counter(), Counter()
    for r in ds:
        tier = r.get("tier", "?")
        tot[tier] += 1
        conv = m.clean_messages(r["messages"])
        if not conv:
            dropped[tier] += 1
            continue
        built = m.build_masked_example(
            conv, tok, max_seq, max_think_tokens=max_think,
            preserve_raw_reasoning=bool(preserve_raw),
            allow_non_reasoning=bool(allow_non_reasoning))
        if built is None:
            dropped[tier] += 1
            continue
        asst = [x for x in conv if x["role"] == "assistant"]
        if not asst:
            continue
        sp = THINK.search(asst[-1]["content"])
        if sp and max_think > 0:
            if len(tok.encode(sp.group(1).strip(), add_special_tokens=False)) > max_think:
                trunc[tier] += 1
        if sp and not preserve_raw:
            orig = sp.group(1).strip()
            cleaned = m.clean_training_think_span(orig)
            if cleaned:
                lost = [s.strip() for s in SENT.split(orig)
                        if s.strip() and re.sub(r"\s+", " ", s).strip() not in cleaned]
                if any(re.search(r"\banswer\b", s, re.I) and re.search(r"\d", s) for s in lost):
                    concl[tier] += 1
    return tot, dropped, trunc, concl


def report(label, tot, dropped, trunc, concl):
    T = sum(tot.values())
    S = T - sum(dropped.values())
    print(f"\n===== {label} =====")
    print(f"{'tier':<20}{'rows':>7}{'dropped':>13}{'truncated':>13}"
          f"{'concl-del':>13}{'eff%':>7}")
    print("-" * 78)
    for tier, n in tot.most_common():
        eff = 100 * (n - dropped[tier]) / S if S else 0
        flags = []
        if n and dropped[tier] / n > 0.25:
            flags.append("DROPPED")
        if n and trunc[tier] / n > 0.5:
            flags.append("TRUNCATED")
        if n and concl[tier] / n > 0.5:
            flags.append("CONCLUSION DELETED")
        f = ("  <-- " + " + ".join(flags)) if flags else ""
        print(f"{tier:<20}{n:>7}{dropped[tier]:>7} ({100*dropped[tier]/n:4.1f}%)"
              f"{trunc[tier]:>7} ({100*trunc[tier]/n:4.1f}%)"
              f"{concl[tier]:>7} ({100*concl[tier]/n:4.1f}%){eff:>6.1f}%{f}")
    print("-" * 78)
    D, TR, C = sum(dropped.values()), sum(trunc.values()), sum(concl.values())
    print(f"{'TOTAL':<20}{T:>7}{D:>7} ({100*D/T:4.1f}%){TR:>7} ({100*TR/T:4.1f}%)"
          f"{C:>7} ({100*C/T:4.1f}%)")
    if D:
        print("  NOTE: dropped rows are REPLACED by a random surviving row (ReasoningDataset."
              "__getitem__),\n        so row count, step count and the loss curve are all unchanged "
              "-- see 'eff%'.")
    if C:
        print("  NOTE: 'concl-del' = canonicalization deleted a sentence stating the RESULT from "
              "inside\n        <think>. The model is then trained to compute and not conclude "
              "(§34ac).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="for the tokenizer")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--max-think-tokens", type=int, default=128)
    ap.add_argument("--preserve-raw-reasoning", type=int, default=0)
    ap.add_argument("--allow-non-reasoning", type=int, default=1)
    ap.add_argument("--compare", action="store_true",
                    help="also audit max_think_tokens=0 + preserve_raw_reasoning=1 (the 3.0-line "
                         "settings) and print both, which is how §34's bugs became obvious")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    m = load_cotsft()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_from_disk(args.data)
    ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    print(f"data={args.data}  rows={len(ds)}  max_seq={args.max_seq_length}")

    report(f"as configured (max_think={args.max_think_tokens}, "
           f"preserve_raw={args.preserve_raw_reasoning})",
           *audit(m, ds, tok, args.max_seq_length, args.max_think_tokens,
                  args.preserve_raw_reasoning, args.allow_non_reasoning))
    if args.compare:
        report("3.0-line settings (max_think=0, preserve_raw=1)",
               *audit(m, ds, tok, args.max_seq_length, 0, 1, args.allow_non_reasoning))


if __name__ == "__main__":
    main()
