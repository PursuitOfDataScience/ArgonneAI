#!/usr/bin/env python3
"""Build the generative-verifier SFT dataset from the STaR labeled-rollout corpus (§22 lever #1).

pass@256 ≈ 82% but single-sample ≈ 4%: the right answer is usually IN the sample set, the model
just can't pick it. A VERIFIER picks it. Since the base is a tied-embedding causal LM (no value
head), we train a GENERATIVE verifier: given (problem, candidate solution), emit 'Yes'/'No'. Then
best-of-N = sample N, verifier-score each, keep the highest — this is what reaches toward the 82%.

Training data is nearly free: star_generate.py --all-out already labeled every rollout
{correct, wrong, unclosed, no_answer}. We frame each as a chat:
  user: <problem> + <candidate solution> + "Is the final answer correct? Reply Yes or No."
  assistant: "Yes" (label==correct) / "No" (otherwise)
Rerank-time candidates are always closed+boxed, so the key discrimination is correct-vs-wrong;
we include a smaller share of unclosed/no_answer as 'No' so the verifier also rejects junk.
Classes are balanced (correct traces are the scarce positive)."""
import argparse
import random
from collections import Counter
from datasets import load_from_disk, Dataset

PROMPT_TMPL = (
    "You are grading a math solution.\n\n"
    "Problem:\n{q}\n\n"
    "Proposed solution:\n{sol}\n\n"
    "Is the final boxed answer correct? Reply with only 'Yes' or 'No'."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="/project/rcc/youzhi/data/star_all_soup_r1")
    ap.add_argument("--out", default="/project/rcc/youzhi/data/verifier_sft_soup_r1")
    ap.add_argument("--neg-per-pos", type=float, default=1.5,
                    help="No:Yes ratio (verifier sees more wrong than right at rerank time)")
    ap.add_argument("--junk-frac", type=float, default=0.25,
                    help="fraction of the 'No' set drawn from unclosed/no_answer (else 'wrong')")
    ap.add_argument("--pos-upsample", type=int, default=1, help="repeat the scarce positives")
    ap.add_argument("--max-sol-chars", type=int, default=6000, help="drop pathologically long traces")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ds = load_from_disk(args.all)
    pos, wrong, junk = [], [], []
    for r in ds:
        sol = r["trace"].strip()
        if not sol or len(sol) > args.max_sol_chars:
            continue
        ex = {"messages": [
            {"role": "user", "content": PROMPT_TMPL.format(q=r["question"], sol=sol)},
            {"role": "assistant", "content": "Yes" if r["label"] == "correct" else "No"}],
            "tier": "verifier"}
        if r["label"] == "correct":
            pos.append(ex)
        elif r["label"] == "wrong":
            wrong.append(ex)
        else:  # unclosed / no_answer
            junk.append(ex)

    pos_up = pos * args.pos_upsample
    n_no = int(len(pos_up) * args.neg_per_pos)
    n_junk = min(len(junk), int(n_no * args.junk_frac))
    n_wrong = min(len(wrong), n_no - n_junk)
    rng.shuffle(wrong); rng.shuffle(junk)
    neg = wrong[:n_wrong] + junk[:n_junk]
    rows = pos_up + neg
    rng.shuffle(rows)

    lbls = Counter(m["messages"][-1]["content"] for m in rows)
    print(f"positives(correct)={len(pos)} upsampled={len(pos_up)} | wrong={len(wrong)} junk={len(junk)}")
    print(f"TOTAL={len(rows)}  label dist (Yes/No): {dict(lbls)}")
    Dataset.from_list(rows).save_to_disk(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
