#!/usr/bin/env python3
"""Build RLVR-DPO pairs from a labeled-rollout corpus, with MODE-AWARE negatives.

WHY DPO AND NOT MORE SFT (2026-08-02, after Arm A). Adding 6,607 on-policy gold+step-verified
traces to the v6 diet moved nothing (svamp greedy 65.0->63.0, self-cons 74.0->76.0; asdiv
73.0->73.7, 82.7->80.3 -- all inside the n=300 noise floor, no consistent sign). That is the
expected outcome of a likelihood objective here: the model ALREADY samples the correct trace often
(clean pass@32 97-98%), so raising its likelihood a little does not change which trace is the
ARGMAX -- and greedy decoding returns the argmax. Moving an argmax needs a CONTRASTIVE objective
that pushes the wrong mode DOWN while pushing the right trace UP.

WHY THIS IS NOT RE-PAYING A DEAD LEVER. §23d lists "weight-space RLVR-DPO" as killed on
2026-07-10. That verdict was reached with **321 pairs** built from a model solving ~2.6% of
problems -- the contrast signal genuinely was too weak. The same corpus construction on
argonne-3.5-think yields ~10^4 pairs from measured 75-82% signal groups. Also distinct: §32's DPO
stage (a measured no-op, loss pinned at ln 2) trained on `argilla_dpo-mix-7k` GENERAL preference
data, not on verifiable-reward pairs, so it is no evidence about this.

THE MODE-AWARE PART (new). `vllm_rollouts.py --dpo-out` samples negatives uniformly from the wrong
traces. But the trace that needs its probability lowered is specifically the one carrying the
MODAL wrong answer -- that is what greedy is going to emit. So negatives are drawn
majority-answer-first, and `--only-mode-wrong` restricts the corpus to problems where the mode is
wrong at all (a problem whose mode is already right needs no repair, and pairing on it just risks
degrading a working behaviour).

Chosen traces inherit the Arm-A filters: gold-verified, no arithmetically-wrong step, and
non-degenerate (>= min think tokens AND the gold value must appear inside <think>) -- the last one
matters more here than in SFT, because a DPO chosen sample that asserts the answer without
deriving it teaches the policy to raise the likelihood of unjustified assertions.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rft_generate import has_bad_arith, is_degenerate, step_signature  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-jsonl", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="/project/rcc/youzhi/models/a35_reason/blend_a085")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pos", type=int, default=2)
    ap.add_argument("--max-neg", type=int, default=2)
    ap.add_argument("--only-mode-wrong", action="store_true")
    ap.add_argument("--min-think-tok", type=int, default=48)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    by_q = defaultdict(lambda: {"gold": None, "pool": None, "correct": [], "wrong": [],
                                "votes": Counter(), "wrong_by_pred": defaultdict(list)})
    fm = Counter()
    for p in args.all_jsonl:
        for ln in open(p):
            o = json.loads(ln)
            e = by_q[o["question"]]
            e["gold"] = o["gold"]
            e["pool"] = o.get("pool", "?")
            fm[o["label"]] += 1
            if o["label"] in ("correct", "wrong") and o.get("pred"):
                e["votes"][o["pred"]] += 1
            if o["label"] == "correct":
                e["correct"].append(o["trace"])
            elif o["label"] == "wrong":
                e["wrong"].append(o["trace"])
                if o.get("pred"):
                    e["wrong_by_pred"][o["pred"]].append(o["trace"])
    print(f"rollouts {sum(fm.values())} labels={dict(fm)} problems={len(by_q)}")

    pairs = []
    stat = Counter()
    for q, e in by_q.items():
        gold, votes = e["gold"], e["votes"]
        if not e["correct"] or not e["wrong"]:
            stat["skip_need_both"] += 1
            continue
        mode = votes.most_common(1)[0][0] if votes else None
        mode_wrong = (mode is not None and mode != gold)
        if args.only_mode_wrong and not mode_wrong:
            stat["skip_mode_already_right"] += 1
            continue
        stat["problems_mode_wrong" if mode_wrong else "problems_mode_right"] += 1

        pos, sigs = [], set()
        for t in sorted(e["correct"], key=len):
            if len(pos) >= args.max_pos:
                break
            if has_bad_arith(t):
                stat["pos_drop_bad_arith"] += 1
                continue
            th = t.split("</think>")[0]
            th = th.split("<think>")[-1]
            if is_degenerate(th, gold, args.min_think_tok, tok, True, 0):
                stat["pos_drop_degenerate"] += 1
                continue
            sg = step_signature(t)
            if sg in sigs:
                continue
            sigs.add(sg)
            pos.append(t)
        if not pos:
            stat["skip_no_clean_positive"] += 1
            continue

        # negatives: the MODAL wrong answer's traces first -- that is the one greedy will emit
        negs = []
        if mode_wrong and e["wrong_by_pred"].get(mode):
            negs += sorted(e["wrong_by_pred"][mode], key=len)[:args.max_neg]
            stat["neg_from_mode"] += len(negs)
        if len(negs) < args.max_neg:
            rest = [t for t in e["wrong"] if t not in negs]
            extra = rng.sample(rest, min(args.max_neg - len(negs), len(rest)))
            negs += extra
            stat["neg_random"] += len(extra)

        for c in pos:
            for r in negs:
                if c.strip() == r.strip():
                    continue
                pairs.append({
                    "chosen": [{"role": "user", "content": q},
                               {"role": "assistant", "content": c.strip()}],
                    "rejected": [{"role": "user", "content": q},
                                 {"role": "assistant", "content": r.strip()}],
                    "neg_kind": "wrong",
                    "mode_wrong": bool(mode_wrong),
                    "pool": e["pool"]})

    rng.shuffle(pairs)
    if args.max_pairs and len(pairs) > args.max_pairs:
        print(f"  capping {len(pairs)} -> {args.max_pairs}")
        pairs = pairs[:args.max_pairs]
    Dataset.from_list(pairs).save_to_disk(args.out)
    print(f"\nwrote {len(pairs)} pairs -> {args.out}")
    print(f"  mode_wrong share: {100*sum(1 for p in pairs if p['mode_wrong'])/max(len(pairs),1):.1f}%")
    print(f"  pools: {Counter(p['pool'] for p in pairs).most_common()}")
    print(f"  stats: {dict(stat)}")
    if args.stats_out:
        json.dump({"n_pairs": len(pairs), "stats": dict(stat), "fm": dict(fm),
                   "args": vars(args)}, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
