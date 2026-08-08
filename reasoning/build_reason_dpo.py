#!/usr/bin/env python3
"""Build RLVR-DPO preference pairs from the STaR labeled-rollout corpus (§22 lever #3).

RLVR-DPO sidesteps GRPO's group-advantage collapse: DPO needs only ONE positive + ONE
negative per prompt (a logsigmoid contrast), and pass@32≈42% means ~half of problems have
BOTH a verified-correct-closed trace AND a wrong/unclosed one. star_generate.py --all-out
saved every rollout with its label; this groups by question and emits {chosen, rejected}
message-list pairs in the format reason_control/dpo.py's build_pairs() consumes
(chosen = [user, assistant<correct>], rejected = [user, assistant<negative>]).

Two pair TYPES, both useful:
  * correct  vs  WRONG (closed, wrong answer)   -> pressures final-answer arithmetic
  * correct  vs  UNCLOSED (never closed </think>) -> pressures termination (the ~50% failure)
Chosen = the SHORTEST correct-closed trace per question (compress pass@1; avoid rewarding length).
"""
import argparse
import random
from collections import defaultdict, Counter
from datasets import load_from_disk, Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="/project/rcc/youzhi/data/star_all_soup_r1",
                    help="labeled-rollout corpus from star_generate.py --all-out")
    ap.add_argument("--out", default="/project/rcc/youzhi/data/star_dpo_soup_r1")
    ap.add_argument("--max-pairs-per-q", type=int, default=2,
                    help="at most this many pairs per question (1 wrong + 1 unclosed by default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ds = load_from_disk(args.all)
    by_q = defaultdict(lambda: {"correct": [], "wrong": [], "unclosed": [], "no_answer": []})
    for r in ds:
        by_q[r["question"]][r["label"]].append(r["trace"])

    pairs, n_wrong_pairs, n_unclosed_pairs = [], 0, 0
    for q, g in by_q.items():
        if not g["correct"]:
            continue
        chosen_trace = min(g["correct"], key=len)  # shortest correct-closed
        chosen = [{"role": "user", "content": q},
                  {"role": "assistant", "content": chosen_trace}]
        negs = []
        if g["wrong"]:
            negs.append(("wrong", rng.choice(g["wrong"])))
        if g["unclosed"]:
            negs.append(("unclosed", rng.choice(g["unclosed"])))
        if not negs and g["no_answer"]:
            negs.append(("no_answer", rng.choice(g["no_answer"])))
        rng.shuffle(negs)
        for kind, neg_trace in negs[:args.max_pairs_per_q]:
            if neg_trace.strip() == chosen_trace.strip():
                continue
            pairs.append({
                "chosen": chosen,
                "rejected": [{"role": "user", "content": q},
                             {"role": "assistant", "content": neg_trace}],
                "neg_kind": kind,
            })
            if kind == "wrong":
                n_wrong_pairs += 1
            elif kind == "unclosed":
                n_unclosed_pairs += 1

    rng.shuffle(pairs)
    print(f"questions with >=1 correct: {sum(1 for g in by_q.values() if g['correct'])}")
    print(f"DPO pairs: {len(pairs)}  (correct-vs-wrong={n_wrong_pairs}, "
          f"correct-vs-unclosed={n_unclosed_pairs}, other={len(pairs)-n_wrong_pairs-n_unclosed_pairs})")
    print(f"neg_kind dist: {dict(Counter(p['neg_kind'] for p in pairs))}")
    Dataset.from_list(pairs).save_to_disk(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
