#!/usr/bin/env python3
"""Distractor-augmented gsm8k-TRAIN problems — a robustness pool for §33's follow-up round.

WHY (2026-08-02, after §33n). Across five held-out sets the verify tier's two significant gains were
ASDiv (+4.80, p=0.0005) and **GSM-Plus (+5.20, p=0.017)** — and GSM-Plus is where the model is
weakest in absolute terms (shipped greedy 27.8%, pass@8 ~59%), so it holds the most headroom. But the
tier's fuel was rollouts on *clean* phrasings (gsm8k-train, MATH L1-3); nothing in it trains the
model to survive a perturbed problem. This builds the missing pool.

WHAT IT DOES, and the ONE axis it covers. GSM-Plus perturbs along ~8 axes (numeric substitution,
digit expansion, int->decimal, added/reversed operation, rephrasing, distractor insertion, critical
thinking). Only **distractor insertion** can be done programmatically with the gold answer provably
unchanged, so that is the only one done here — an irrelevant sentence about a DIFFERENT subject,
carrying its own numbers, spliced in before the final question. Everything else would need a teacher
model to re-derive the gold, and a wrong gold poisons the tier silently.

CONTAMINATION. Perturbations are built from gsm8k **split=="train"** only. GSM-Plus derives from
gsm8k **test**. Disjoint problems, similar perturbation style = ordinary train/test methodology
(§23's rule), not leakage.

HONEST LIMITATION, recorded so it is not forgotten: an inserted sentence can in principle make a
problem ambiguous rather than merely harder. Two guards: (1) the distractor names a different
subject and asks nothing; (2) `--require-solvable` keeps only problems the model already answers
correctly with the majority of its samples on the CLEAN version, so a later failure is attributable
to the perturbation rather than to the problem being beyond it. Even so, treat the resulting tier as
distractor-robustness fuel, not as a general GSM-Plus surrogate.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GSM = "/project/rcc/youzhi/data/gsm8k_main_curated/shards/shard_00000.jsonl"
NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
SENT = re.compile(r"(?<=[.!?])\s+")
NAMES = ["Dana", "Marco", "Priya", "Kwame", "Lena", "Ibrahim", "Sofia", "Hiroshi",
         "Aisha", "Tomas", "Ingrid", "Rafael"]
TEMPLATES = [
    "{name} separately owns {n} {unit}, which are not part of this order.",
    "Earlier that week {name} counted {n} {unit} in a different shop.",
    "{name}, who is unrelated to this, keeps {n} {unit} at home.",
    "A nearby store lists {n} {unit}, though {name} never buys any.",
    "For comparison, {name} once had {n} {unit} in another town.",
]
UNITS = ["boxes", "tickets", "apples", "pencils", "coins", "bottles", "chairs",
         "notebooks", "bags", "candles", "stamps", "marbles"]


def load_train():
    from star_generate import extract_boxed
    out = []
    for ln in open(GSM):
        o = json.loads(ln)
        if o.get("split") != "train":
            return_gold = None
        else:
            return_gold = extract_boxed(o["answer"])
        if return_gold is not None:
            out.append((o["question"].strip(), return_gold))
    return out


def perturb(q, rng):
    """Splice one irrelevant sentence in before the final question sentence."""
    parts = [s for s in SENT.split(q.strip()) if s.strip()]
    if len(parts) < 2:
        return None
    # a distractor number that does not already appear, so it cannot be mistaken for a given
    present = set(NUM.findall(q))
    cand = [str(x) for x in range(3, 97) if str(x) not in present]
    if not cand:
        return None
    d = TEMPLATES[rng.randrange(len(TEMPLATES))].format(
        name=NAMES[rng.randrange(len(NAMES))], n=rng.choice(cand),
        unit=UNITS[rng.randrange(len(UNITS))])
    return " ".join(parts[:-1] + [d, parts[-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSONL of {question, gold}")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--solvable-jsonl", nargs="*", default=None,
                    help="rollout corpora; keep only problems whose MAJORITY sample was correct")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    probs = load_train()
    print(f"gsm8k train with parsable gold: {len(probs)}")

    keep_q = None
    if args.solvable_jsonl:
        votes, gold_of = {}, {}
        for p in args.solvable_jsonl:
            for ln in open(p):
                o = json.loads(ln)
                if o["label"] in ("correct", "wrong") and o.get("pred"):
                    votes.setdefault(o["question"], Counter())[o["pred"]] += 1
                    gold_of[o["question"]] = o["gold"]
        keep_q = {q for q, v in votes.items()
                  if v and v.most_common(1)[0][0] == gold_of.get(q)}
        print(f"solvable filter: {len(keep_q)} problems whose clean majority is correct")

    rng.shuffle(probs)
    n_written = n_skip = 0
    with open(args.out, "w") as fh:
        for q, g in probs:
            if n_written >= args.n:
                break
            if keep_q is not None and q not in keep_q:
                n_skip += 1
                continue
            pq = perturb(q, rng)
            if pq is None:
                n_skip += 1
                continue
            fh.write(json.dumps({"question": pq, "gold": g, "orig": q}) + "\n")
            n_written += 1
    print(f"wrote {n_written} perturbed problems -> {args.out}  (skipped {n_skip})")
    if n_written:
        first = json.loads(open(args.out).readline())
        print("\n---- example ----\nORIG: " + first["orig"][:300] +
              "\nPERT: " + first["question"][:400] + f"\nGOLD: {first['gold']}")


if __name__ == "__main__":
    main()
