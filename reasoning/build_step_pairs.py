#!/usr/bin/env python3
"""Preference pairs over the FIRST REASONING STEP, valued by how often that opening reaches gold.

WHY THE FIRST STEP AND NOT THE WHOLE TRACE. Two measurements on argonne4-think, both from this
session:

  * `fail_taxonomy.py` (93,912 on-policy rollouts): 67.9% of wrong traces state no false equation
    and never touch gold -- a coherent derivation of the wrong thing -- and **79.0% of them already
    differ from a correct derivation at equation index 0**, with a median shared-equation prefix of
    0%. The trace is lost at its first computation, not at some later slip.
  * Whole-trace RLVR-DPO produced the best `acc|ANSWERED` of any a4 arm (52.6% vs the 50.1%
    baseline) and STILL lost greedy, because unclosed went 13.7% -> 22.4% and mean think length
    230 -> 312. That is textbook unregularised-DPO length drift, and on this line every arm that
    lengthened traces has lost.

Pairs that differ only in a ~40-token opening cannot express a length preference: chosen and
rejected share the prompt, sit within a few tokens of each other, and neither carries a
terminator. So this keeps the mechanism that moved `acc|ANSWERED` and removes the failure mode that
cancelled it.

WHY VALUED, NOT OUTCOME-LABELLED. Taking "the first step of a trace that happened to end correct"
as the chosen sample is a weak proxy -- a wrong trace often opens correctly and fails later. With
K=8 rollouts per problem the openings can be GROUPED (by their equation, or by wording when there
is no equation) and each group scored by the fraction of its rollouts that reached gold. That is a
Monte-Carlo value estimate of the opening itself, so the pair contrasts a demonstrably good first
move against a demonstrably dead one rather than two arbitrary prefixes.

The chosen opening's own equation is arithmetic-checked. Note this deliberately does NOT run
`has_bad_arith` over the whole trace: that matcher fires on 18.7% of CORRECT traces because it
reads `1/3 = 5` out of `1/2 + 1/3 = 5/6`, so applying it broadly would drop good data. Checking the
single equation inside the ~40-token opening is precise.

Output is the schema `rlvr_dpo.py` already consumes (chosen / rejected as message lists +
neg_kind), so training is `rlvr_dpo.py --no-append-eos` with no new trainer.

  python reasoning/build_step_pairs.py \
      --all-jsonl /project/rcc/youzhi/data/a4_dpo/a4_dpo_all.jsonl \
      --tokenizer /project/rcc/youzhi/models/a4_think_final/think_combo \
      --out /project/rcc/youzhi/data/a4_step_pairs --stats-out report/a4_step_pairs.json
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rft_generate import EQ, _val  # noqa: E402
from star_generate import norm     # noqa: E402

WS = re.compile(r"\s+")
# an opening that already states the final answer is not a first reasoning step (see the filter
# at the pair-construction site for the reward-hacking case this exists to remove)
ANSWER_MARK = re.compile(r"\\boxed\{|####")


def strip_think_open(trace):
    t = trace.lstrip()
    if t.startswith("<think>"):
        t = t[len("<think>"):]
    return t.lstrip("\n")


def first_step(body, tok, max_tokens, min_chars):
    """The opening move: up to the end of the first computed line / paragraph / sentence.

    Capped in TOKENS because the cap is what keeps the pair length-neutral -- an opening allowed to
    run to 200 tokens would reintroduce exactly the length preference this file exists to avoid.
    """
    end = None
    m = EQ.search(body)
    if m:                                   # first line that completes an equation
        nl = body.find("\n", m.end())
        end = len(body) if nl < 0 else nl
    if end is None:
        p = body.find("\n\n")
        if p >= 0:
            end = p
    if end is None:
        for i, ch in enumerate(body):
            if ch in ".?!;" and i + 1 >= min_chars:
                end = i + 1
                break
    if end is None:
        end = len(body)
    step = body[:end].strip()
    if len(step) < min_chars:
        return None
    ids = tok.encode(step, add_special_tokens=False)
    if len(ids) > max_tokens:
        step = tok.decode(ids[:max_tokens]).rstrip()
    return step


def step_key(step):
    """Group openings that are the SAME MOVE in different words: its equation if it has one."""
    m = EQ.search(step)
    if m:
        return "eq:%s%s%s=%s" % (m.group(1), m.group(2).lower(), m.group(3), m.group(4))
    return "tx:" + WS.sub(" ", step.lower())[:60]


def opening_arith_ok(step):
    """Only the equation inside the OPENING is checked (see the module docstring)."""
    m = EQ.search(step)
    if not m:
        return True
    try:
        a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
    except ValueError:
        return True
    v = _val(a, op, b)
    return v is None or abs(v - c) <= 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-jsonl", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-step-tokens", type=int, default=64)
    ap.add_argument("--min-step-chars", type=int, default=24)
    ap.add_argument("--min-gap", type=float, default=0.5,
                    help="required difference in gold-rate between the chosen and rejected opening")
    ap.add_argument("--min-group", type=int, default=1, help="min rollouts backing an opening")
    ap.add_argument("--pairs-per-problem", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--stats-out", default="")
    a = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    rng = random.Random(a.seed)

    by_q = defaultdict(list)
    for path in a.all_jsonl:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_q[(r["pool"], r["question"])].append(r)
    print(f"[steps] {sum(len(v) for v in by_q.values()):,} rollouts over {len(by_q):,} problems")

    stat = Counter()
    rows = []
    gaps = []
    for (pool, q), rs in sorted(by_q.items()):
        groups = defaultdict(lambda: {"n": 0, "ok": 0, "steps": []})
        for r in rs:
            body = strip_think_open(r["trace"])
            step = first_step(body, tok, a.max_step_tokens, a.min_step_chars)
            if step is None:
                stat["no_step"] += 1
                continue
            g = groups[step_key(step)]
            g["n"] += 1
            g["ok"] += 1 if r["label"] == "correct" else 0
            g["steps"].append(step)
        cands = [(v["ok"] / v["n"], k, v) for k, v in groups.items() if v["n"] >= a.min_group]
        if len(cands) < 2:
            stat["fewer_than_two_openings"] += 1
            continue
        cands.sort(reverse=True)
        best_rate, _, best = cands[0]
        # the rejected opening must be a demonstrably DEAD one: zero of its rollouts reached gold
        dead = [c for c in cands if c[0] == 0.0]
        if best_rate <= 0.0 or not dead:
            stat["no_contrast" if best_rate <= 0.0 else "no_dead_opening"] += 1
            continue
        # prefer the dead opening with the MOST rollouts behind it -- that is the one greedy is
        # most likely to emit, which is the mode that needs pushing down
        worst = max(dead, key=lambda c: c[2]["n"])
        if best_rate - worst[0] < a.min_gap:
            stat["gap_too_small"] += 1
            continue
        made = 0
        for _ in range(a.pairs_per_problem):
            ch = rng.choice(best["steps"])
            rj = rng.choice(worst[2]["steps"])
            if WS.sub(" ", ch.lower()) == WS.sub(" ", rj.lower()):
                stat["identical_text"] += 1
                continue
            if not opening_arith_ok(ch):
                stat["chosen_bad_arith"] += 1
                continue
            # ⚠️REWARD-HACKED CHOSEN OPENINGS. A trace is graded `correct` from the LAST \boxed{} anywhere in
            # it -- `clean_eval.grade` reads `pred` independently of closure -- so a rollout that blurts
            # `\boxed{<gold>}` in its first sentence is labelled correct no matter how incoherent the rest
            # is. With --min-group 1 one such rollout scores that opening 1.00 and it becomes the CHOSEN
            # sample. Observed verbatim on the first build: chosen = "...the x-dependence of the expression
            # is \boxed{2}, and this is the only term..." against a rejected opening that was a perfectly
            # sane restatement of the problem. Training DPO on that pair teaches the model to emit the
            # answer inside the think block -- degenerate, and the opposite of a first REASONING step.
            # A first step states a computation, never the final answer, so this is a definitional filter,
            # not a heuristic.
            if ANSWER_MARK.search(ch):
                stat["chosen_states_answer"] += 1
                continue
            rows.append({
                "chosen": [{"role": "user", "content": q},
                           {"role": "assistant", "content": "<think>\n" + ch}],
                "rejected": [{"role": "user", "content": q},
                             {"role": "assistant", "content": "<think>\n" + rj}],
                "neg_kind": "wrong", "pool": pool,
                "chosen_rate": best_rate, "rejected_rate": worst[0],
                "chosen_n": best["n"], "rejected_n": worst[2]["n"],
            })
            gaps.append(best_rate - worst[0])
            made += 1
        stat[f"kept_{pool}"] += made

    rng.shuffle(rows)
    if a.max_pairs and len(rows) > a.max_pairs:
        rows = rows[:a.max_pairs]
    print(f"[steps] pairs={len(rows):,}")
    for k, v in sorted(stat.items()):
        print(f"    {k:26s} {v:7,d}")
    if rows:
        ctok = [len(tok.encode(r["chosen"][-1]["content"], add_special_tokens=False)) for r in rows]
        rtok = [len(tok.encode(r["rejected"][-1]["content"], add_special_tokens=False)) for r in rows]
        mean = lambda x: sum(x) / len(x)
        # THE check that this is length-neutral. If chosen is systematically longer than rejected,
        # DPO will learn "longer" and reproduce the whole-trace arm's failure.
        print(f"[steps] chosen {mean(ctok):.1f} tok vs rejected {mean(rtok):.1f} tok "
              f"(delta {mean(ctok) - mean(rtok):+.1f}); mean value gap {mean(gaps):.2f}")
        print("--- example ---")
        print("  Q:        ", rows[0]["chosen"][0]["content"][:160])
        print(f"  CHOSEN  ({rows[0]['chosen_rate']:.2f}, n={rows[0]['chosen_n']}): "
              f"{rows[0]['chosen'][-1]['content'][:200]!r}")
        print(f"  REJECTED({rows[0]['rejected_rate']:.2f}, n={rows[0]['rejected_n']}): "
              f"{rows[0]['rejected'][-1]['content'][:200]!r}")
        Dataset.from_list(rows).save_to_disk(a.out)
        print(f"[steps] wrote {a.out}")
    if a.stats_out:
        json.dump({"pairs": len(rows), "stat": stat, "args": vars(a)},
                  open(a.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
