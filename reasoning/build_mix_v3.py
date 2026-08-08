#!/usr/bin/env python3
"""Build cot_sft_mix_v3 = cot_sft_mix_v2 + a targeted MULTI-STEP tier.

Why: data-calibration (mix v2), STaR, and a properly-configured GRPO all hit the
SAME ceiling — the model handles single facts (mix v2's synth_arith fixed those)
but fails on multi-STEP chains. The four families the 4-quadrant eval keeps
failing, across every checkpoint:
  1. two-operation linear algebra   (2x + 5 = 17  -> x = 6)
  2. sequential / series sums       (1 + 2 + ... + 10 = 55)   <- also the loop trap
  3. formula + substitution geometry(perimeter = 2(l+w))
  4. divisor counting               (# positive divisors of 12 = 6)

This tier supplies SHORT, CORRECT, multi-step <think> traces for exactly those
families — correct BY CONSTRUCTION (every number is computed in Python; we then
re-verify each trace's \\boxed answer with the same extractor used for RLVR).
We keep all of mix v2 as the anchor (the zero-sum-diet lesson: don't let a
narrow tier erode general/no-think). Output: cot_sft_mix_v3, save_to_disk.
"""

import random
from collections import Counter
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

V2 = "/project/rcc/youzhi/data/cot_sft_mix_v2"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v3"
TOK = "/project/rcc/youzhi/models/instruct/dpo_ckpts"
PER_FAMILY = 5000          # 4 families -> ~20k multi-step traces (~17% of the mix)
MAX_TOKENS = 4000
SEED = 20260616
rng = random.Random(SEED)

# Cross-check golds with the exact RLVR extractor.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_generate import extract_boxed, norm  # noqa


def algebra():
    """ax + b = c  /  ax - b = c, integer solution x."""
    x = rng.randint(1, 15)
    a = rng.randint(2, 12)
    b = rng.randint(1, 50)
    if rng.random() < 0.5:
        c = a * x + b
        q = rng.choice([f"Solve for x: {a}x + {b} = {c}.",
                        f"If {a}x + {b} = {c}, what is x?",
                        f"Find x: {a}x + {b} = {c}."])
        step = (f"We have {a}x + {b} = {c}. Subtract {b} from both sides: "
                f"{a}x = {c} - {b} = {c - b}. Divide both sides by {a}: "
                f"x = {c - b} / {a} = {x}")
    else:
        c = a * x - b
        q = rng.choice([f"Solve for x: {a}x - {b} = {c}.",
                        f"If {a}x - {b} = {c}, what is x?",
                        f"Find x: {a}x - {b} = {c}."])
        step = (f"We have {a}x - {b} = {c}. Add {b} to both sides: "
                f"{a}x = {c} + {b} = {c + b}. Divide both sides by {a}: "
                f"x = {c + b} / {a} = {x}")
    return q, step, x


def series_sum():
    """sum 1..n, first n even / odd, or a general consecutive range m..n."""
    kind = rng.choice(["nat", "nat", "even", "odd", "range", "range"])
    if kind == "nat":
        n = rng.randint(5, 200); s = n * (n + 1) // 2
        q = rng.choice([f"What is the sum 1 + 2 + 3 + ... + {n}?",
                        f"Find the sum of the first {n} positive integers.",
                        f"Add up all the integers from 1 to {n}."])
        step = (f"The sum of the first n positive integers is n(n+1)/2. "
                f"Here n = {n}, so the sum is {n} \\times {n + 1} / 2 = "
                f"{n * (n + 1)} / 2 = {s}")
    elif kind == "even":
        n = rng.randint(5, 150); s = n * (n + 1)
        q = rng.choice([f"What is the sum of the first {n} positive even numbers?",
                        f"Find 2 + 4 + 6 + ... up to the {n}th even number."])
        step = (f"The sum of the first n even numbers is n(n+1). "
                f"Here n = {n}, so the sum is {n} \\times {n + 1} = {s}")
    elif kind == "odd":
        n = rng.randint(5, 150); s = n * n
        q = rng.choice([f"What is the sum of the first {n} positive odd numbers?",
                        f"Find 1 + 3 + 5 + ... up to the {n}th odd number."])
        step = (f"The sum of the first n odd numbers is n^2. "
                f"Here n = {n}, so the sum is {n}^2 = {s}")
    else:
        m = rng.randint(2, 90); n = m + rng.randint(3, 90)
        cnt = n - m + 1; s = (m + n) * cnt // 2
        q = rng.choice([f"What is the sum {m} + {m + 1} + ... + {n}?",
                        f"Find the sum of all integers from {m} to {n}."])
        step = (f"The sum of consecutive integers from a to b is (a+b)(b-a+1)/2. "
                f"Here a = {m} and b = {n}, with {cnt} terms, so the sum is "
                f"({m} + {n}) \\times {cnt} / 2 = {m + n} \\times {cnt} / 2 = "
                f"{(m + n) * cnt} / 2 = {s}")
    return q, step, s


def geometry():
    kind = rng.choice(["rect_p", "rect_a", "sq_p", "sq_a", "tri_p"])
    if kind == "rect_p":
        l, w = rng.randint(2, 30), rng.randint(2, 30); r = 2 * (l + w)
        q = f"A rectangle has length {l} and width {w}. What is its perimeter?"
        step = (f"The perimeter of a rectangle is 2 \\times (length + width). "
                f"That is 2 \\times ({l} + {w}) = 2 \\times {l + w} = {r}")
    elif kind == "rect_a":
        l, w = rng.randint(2, 30), rng.randint(2, 30); r = l * w
        q = f"A rectangle has length {l} and width {w}. What is its area?"
        step = (f"The area of a rectangle is length \\times width = "
                f"{l} \\times {w} = {r}")
    elif kind == "sq_p":
        s = rng.randint(2, 40); r = 4 * s
        q = f"A square has side length {s}. What is its perimeter?"
        step = f"The perimeter of a square is 4 \\times side = 4 \\times {s} = {r}"
    elif kind == "sq_a":
        s = rng.randint(2, 40); r = s * s
        q = f"A square has side length {s}. What is its area?"
        step = f"The area of a square is side^2 = {s}^2 = {r}"
    else:
        a, b, c = (rng.randint(2, 30) for _ in range(3)); r = a + b + c
        q = f"A triangle has sides {a}, {b}, and {c}. What is its perimeter?"
        step = (f"The perimeter is the sum of the side lengths: "
                f"{a} + {b} + {c} = {r}")
    return q, step, r


def divisors():
    """# positive divisors of N via prime-factorization exponents."""
    primes = rng.sample([2, 3, 5, 7, 11, 13], k=rng.randint(1, 3))
    primes.sort()
    exps = [rng.randint(1, 4) for _ in primes]
    n = 1
    for p, e in zip(primes, exps):
        n *= p ** e
    if n > 9999 or n < 4:
        return None
    fac = " \\times ".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in zip(primes, exps))
    prodparts = " \\times ".join(f"({e}+1)" for e in exps)
    nums = " \\times ".join(str(e + 1) for e in exps)
    d = 1
    for e in exps:
        d *= (e + 1)
    q = rng.choice([f"How many positive divisors does {n} have?",
                    f"How many positive divisors does {n} have in total?",
                    f"Find the number of positive divisors of {n}."])
    step = (f"First factor {n}: {n} = {fac}. The number of positive divisors is "
            f"the product of (exponent + 1) over the prime factors: "
            f"{prodparts} = {nums} = {d}")
    return q, step, d


FAMILIES = {"ms_algebra": algebra, "ms_series": series_sum,
            "ms_geometry": geometry, "ms_divisors": divisors}


def build_family(name, gen, target, tok):
    kept, seen_q = [], set()
    tries = 0
    while len(kept) < target and tries < target * 40:
        tries += 1
        out = gen()
        if out is None:
            continue
        q, step, gold = out
        if q in seen_q:
            continue
        content = f"<think>\n{step}.\n</think>\n\nThe answer is $\\boxed{{{gold}}}$."
        # VERIFY correct-by-construction with the RLVR extractor.
        if extract_boxed(content) != norm(str(gold)):
            continue
        msgs = [{"role": "user", "content": q},
                {"role": "assistant", "content": content}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                      enable_thinking=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        ntok = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
        if ntok > MAX_TOKENS:
            continue
        seen_q.add(q)
        kept.append({"messages": msgs, "tier": name, "num_tokens": ntok})
    print(f"  {name:12s}: kept {len(kept):5d}/{target} (tries {tries})")
    return kept


def main():
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    print(f"Loading anchor mix v2 <- {V2}")
    v2 = load_from_disk(V2)
    v2_rows = [{"messages": r["messages"], "tier": r["tier"],
                "num_tokens": r["num_tokens"]} for r in v2]
    print(f"  v2: {len(v2_rows)} rows {dict(Counter(r['tier'] for r in v2_rows))}")

    print("Generating multi-step tier:")
    ms_rows = []
    for name, gen in FAMILIES.items():
        ms_rows += build_family(name, gen, PER_FAMILY, tok)

    all_rows = v2_rows + ms_rows
    rng.shuffle(all_rows)
    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(OUT)
    print(f"\nTOTAL: {len(ds)} -> {OUT}")
    print("composition:", dict(Counter(ds["tier"])))


if __name__ == "__main__":
    main()
