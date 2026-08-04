#!/usr/bin/env python3
"""Single-step arithmetic tier: SHORT trace, NO verification cue, correct by construction.

WHY (§33s). The verify tier cost **−23.8pt on one-step arithmetic** (shipped 40/80 vs candidate
21/80): it taught the model that "Wait, let me double-check that" is part of the default trace, so the
cue fires on `2 + 2`, the second derivation has nowhere to go, and it invents an error — then the
verification re-derives the same wrong way and *confirms* it. Two causes, both addressed:
  1. the cue fired UNCONDITIONALLY  -> `build_verify_tier.py --min-eqs N` now restricts verify rows to
     genuinely multi-step derivations, so the cue is never associated with one-step work;
  2. v6's single-step `synth_arith` tier (2,500 rows, 9.4%) was diluted to 7.0% by a tier taking 26%
     of the mix -> this file rebuilds that capability explicitly, at a controllable share.

Every row here is a one- or two-step computation with a SHORT think block and NO cue, so it is the
direct counter-example to "always double-check": a finished one-step answer gets committed.

CONTAMINATION. The gate (`simple_arith_probe.py`) generates its items from a fixed seed; every probe
question string is loaded and EXCLUDED here, and this generator uses different operand ranges and
phrasings. Same distribution, disjoint items -- ordinary methodology. Generalisation beyond the
trained phrasings is checked by running the probe at a fresh `--probe-seed`.
"""
import argparse
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
if RDIR not in sys.path:
    sys.path.insert(0, RDIR)

CLOSE = "\n</think>\n\nThe answer is $\\boxed{%s}$."
NUM_TMPL = ["What is {a} {s} {b}?", "Compute {a} {s} {b}.", "Calculate {a} {s} {b}.",
            "{a} {s} {b} = ?", "Work out {a} {s} {b}.", "Find the value of {a} {s} {b}."]
WORD = [
    ("{n} has {a} {u} and gets {b} more. How many {u} does {n} have now?", "+"),
    ("{n} had {a} {u} and gave away {b}. How many {u} are left?", "-"),
    ("Each box holds {b} {u}. How many {u} are in {a} boxes?", "*"),
    ("{n} shares {a} {u} equally among {b} friends. How many {u} does each friend get?", "/"),
    ("{n} buys {a} {u} at ${b} each. What is the total cost in dollars?", "*"),
]
NAMES = ["Tom", "Mia", "Raj", "Ana", "Leo", "Zoe", "Ivan", "Nia", "Omar", "Elsa", "Hugo", "Kira"]
UNITS = ["apples", "pencils", "stickers", "marbles", "books", "coins", "cards", "sweets",
         "balloons", "tickets", "shells", "buttons"]
SYM = {"+": "+", "-": "-", "*": "×", "/": "÷"}


def fmt(x):
    if isinstance(x, Fraction):
        x = float(x)
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.4f}".rstrip("0").rstrip(".")


def compute(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return Fraction(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-numeric", type=int, default=3000)
    ap.add_argument("--n-word", type=int, default=1200)
    ap.add_argument("--n-twostep", type=int, default=800,
                    help="two-step rows, still cue-free, so the tier does not teach 'one step only'")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--exclude-probe", action="store_true", default=True,
                    help="drop any question string that appears in simple_arith_probe.py's gate")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    banned = set()
    if args.exclude_probe:
        try:
            from simple_arith_probe import PROBES
            banned = {q.strip() for q, _ in PROBES}
            print(f"excluding {len(banned)} gate questions")
        except Exception as e:
            print(f"WARN could not load the gate for exclusion: {e!r}")

    rows, seen = [], set()

    def add(q, think, gold, tier):
        q = q.strip()
        if q in banned or q in seen:
            return False
        seen.add(q)
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant",
                                   "content": "<think>\n" + think + CLOSE % gold}],
                     "tier": tier, "num_tokens": 0})
        return True

    # ---- pure numeric, ONE step -------------------------------------------
    tries = 0
    while sum(1 for r in rows if r["tier"] == "arith_1step") < args.n_numeric and tries < 60000:
        tries += 1
        op = rng.choice("+-*/")
        if op == "+":
            a, b = rng.randint(2, 999), rng.randint(2, 999)
        elif op == "-":
            a, b = rng.randint(10, 999), rng.randint(2, 99)
            if b > a:
                a, b = b, a
        elif op == "*":
            a, b = rng.randint(2, 40), rng.randint(2, 20)
        else:
            b, q = rng.randint(2, 20), rng.randint(2, 60)
            a = b * q
        g = compute(a, op, b)
        s = SYM[op]
        add(rng.choice(NUM_TMPL).format(a=a, b=b, s=s),
            f"{a} {s} {b} = {fmt(g)}.", fmt(g), "arith_1step")

    # ---- one-step word problems ------------------------------------------
    tries = 0
    while sum(1 for r in rows if r["tier"] == "arith_word") < args.n_word and tries < 60000:
        tries += 1
        tmpl, op = rng.choice(WORD)
        if op == "/":
            b, q = rng.randint(2, 9), rng.randint(2, 20)
            a = b * q
        elif op == "*":
            a, b = rng.randint(2, 25), rng.randint(2, 12)
        elif op == "-":
            a, b = rng.randint(12, 200), rng.randint(2, 11)
        else:
            a, b = rng.randint(3, 200), rng.randint(2, 60)
        g = compute(a, op, b)
        s = SYM[op]
        add(tmpl.format(n=rng.choice(NAMES), u=rng.choice(UNITS), a=a, b=b),
            f"{a} {s} {b} = {fmt(g)}.", fmt(g), "arith_word")

    # ---- two-step, still cue-free ----------------------------------------
    tries = 0
    while sum(1 for r in rows if r["tier"] == "arith_2step") < args.n_twostep and tries < 60000:
        tries += 1
        a, b, c = rng.randint(2, 60), rng.randint(2, 20), rng.randint(2, 40)
        op1, op2 = rng.choice(["*+", "*-", "+*"])
        if op1 == "*":
            m = a * b
            g = m + c if op2 == "+" else m - c
            think = f"{a} × {b} = {m}. Then {m} {'+' if op2 == '+' else '−'} {c} = {fmt(g)}."
            q = f"What is {a} × {b} {'+' if op2 == '+' else '−'} {c}?"
        else:
            m = a + b
            g = m * c
            think = f"{a} + {b} = {m}. Then {m} × {c} = {fmt(g)}."
            q = f"What is ({a} + {b}) × {c}?"
        add(q, think, fmt(g), "arith_2step")

    # ---- token counts -----------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/project/rcc/youzhi/models/a35_reason/blend_a085",
                                        trust_remote_code=True)
    for r in rows:
        r["num_tokens"] = len(tok.encode(r["messages"][1]["content"], add_special_tokens=False))

    rng.shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    t = [r["num_tokens"] for r in rows]
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"  tiers: {Counter(r['tier'] for r in rows).most_common()}")
    print(f"  tokens: mean {sum(t)/len(t):.0f}  p50 {sorted(t)[len(t)//2]}  max {max(t)}")
    print("\n---- examples ----")
    for tier in ("arith_1step", "arith_word", "arith_2step"):
        ex = next((r for r in rows if r["tier"] == tier), None)
        if ex:
            print(f"[{tier}] Q: {ex['messages'][0]['content']}")
            print(f"          A: {ex['messages'][1]['content']!r}")


if __name__ == "__main__":
    main()
