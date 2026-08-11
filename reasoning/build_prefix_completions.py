#!/usr/bin/env python3
"""Let a STRONGER same-tokenizer model finish a4's OWN partial traces, keep the splices that reach gold.

WHY THIS AND NOT THE THINGS THAT ALREADY FAILED. §41bf decomposed the remaining five-pool gap to
3.5-think additively:

    COVERAGE  (pass@8)              +6.36   <- 79% of the gap
    SELECTION (sc@8 given pass@8)   +1.11
    FLOOR     (greedy vs sc@8)      +0.62

Six rounds of per-token on-policy KD took the SELECTION gap from 18.00 to 14.65 against the target's
13.54, i.e. a4 now picks among what it can reach nearly as well as the model it is chasing. That is
why the deployed metric saturated at round 3 and why every sharpening arm since has returned ~1pt:
**the lever is nearly exhausted because the quantity it moves is nearly closed.** What is left is
that a4 never solves 31.24% of problems in 8 samples against 3.5-think's 24.88%, and nothing built
so far adds a solution the model cannot already produce.

THE THREE FAILURES THIS IS SHAPED AROUND, each of which rules out an easier version of it:
  * §41c  off-policy imitation of Llama-3.1-8B: a measured NULL that pushed `acc|ANSWERED` DOWN to
          46.1%. Imitating a stronger model's traces WHOLESALE does not transfer -- the target
          distribution is too far from anything a4 visits.
  * §41m/§41n  gold-anchored self-distillation: REFUTED at -21.6 (p<1e-14), because a4 is *damaged*
          by information in its context (61.25 plain -> 56.25 with the correct answer -> 30.00 with a
          wrong one, §41h). Anything that puts the answer in the prompt is capped by that.
  * §41bb/§41bg  per-token KD from any stronger same-tokenizer teacher: 0-for-5 on termination,
          because per-token KD transfers the teacher's trace-length policy and that policy IS the
          token distribution.
Prefix completion evades all three by construction. The prefix is **a4's own**, so the target starts
at a state a4 actually visits and inherits a4's style and length regime rather than the completer's.
Nothing privileged enters the *student's* context -- the completer sees the prefix, the student only
ever sees the question. And the objective is plain CE on a verified trace, so there is no divergence
term to blow up on terminator columns and no hazard ratio to satisfy.

WHAT IS DELIBERATELY MEASURED RATHER THAN ASSUMED:
  * `--include-empty-prefix` emits a prefix-free variant of every problem, which is exactly what
    `gen_teacher.py` already does. That is the CONTROL: if empty-prefix splices train as well as
    a4-prefixed ones, prefix conditioning is doing nothing and this is just §41c again with a
    different teacher. The stats JSON reports yield and length separately for the two.
  * `--max-trace-tokens` hard-caps the FINAL spliced trace (default 330 = round 6's own p90 think
    length). Every arm on this line that lengthened traces has lost (§38j/§41f/§41bb), so a method
    whose targets come from a longer-form model must bound length at the DATA level, not hope.
  * Only completions whose `\\boxed` answer equals gold survive. A base model has no notion of
    correctness (§41bg) -- this filter is what a verified completer buys over one.

Output rows use the rollout-dump schema, so training needs NO new trainer:
    python reasoning/opd_train.py --kd-weight 0 --ce-weight 1 --labels correct \\
        --rollouts <this output> --student <ckpt> ...

  python reasoning/build_prefix_completions.py \\
      --rollouts /project/rcc/youzhi/data/a4_opd_opd_r3_r4/all.jsonl \\
      --completer Qwen/Qwen3-4B \\
      --out /project/rcc/youzhi/data/a4_prefix_comp/all.jsonl \\
      --stats-out report/a4_prefix_comp.json
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed, norm    # noqa: E402  verified primitives

CLOSE_STR = "\n</think>\n\nThe answer is \\boxed{"
THINK_OPEN = "<think>"


def strip_think_open(trace):
    t = trace.lstrip()
    if t.startswith(THINK_OPEN):
        t = t[len(THINK_OPEN):]
    return t.lstrip("\n")


def cut_prefix(body, tok, max_tokens, min_tokens):
    """Truncate a trace body to <= max_tokens, ending at a LINE boundary.

    Cutting mid-sentence would hand the completer a fragment no well-formed trace contains, so the
    splice would be teaching a style that never occurs. Falls back to a sentence boundary, then to a
    hard token cut, and returns None when there is not enough material to be worth completing.
    """
    ids = tok.encode(body, add_special_tokens=False)
    if len(ids) < min_tokens:
        return None
    if len(ids) > max_tokens:
        body = tok.decode(ids[:max_tokens])
    for sep in ("\n\n", "\n"):
        p = body.rfind(sep)
        if p > 0 and len(tok.encode(body[:p], add_special_tokens=False)) >= min_tokens:
            return body[:p].rstrip()
    for i in range(len(body) - 1, 0, -1):
        if body[i] in ".?!":
            cand = body[:i + 1]
            if len(tok.encode(cand, add_special_tokens=False)) >= min_tokens:
                return cand
    return body.rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--completer", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default="")
    ap.add_argument("--select", choices=["never", "rare", "all"], default="never",
                    help="never = 0/K rollouts correct (the 31.24% coverage hole, the point of this "
                         "file); rare = 1..floor(K/4); all = every problem")
    ap.add_argument("--pools", nargs="*", default=None,
                    help="restrict to these TRAIN pools. ⚠️MATTERS FOR MEASUREMENT INTEGRITY, not just "
                         "transfer: the coverage hole is 57.1%% math_train_hard (2,586 of 4,531 "
                         "never-solved problems, a 67.5%% never-solved rate there vs 24.8%% on "
                         "gsm8k_train), and MATH-train near-dups MATH-500 in every self-generated mix "
                         "on this line. Training on math_train_hard completions therefore makes the "
                         "math500 column uninterpretable while leaving the four CLEAN grade-school "
                         "pools (asdiv/svamp/mawps/gsmplus) unaffected -- so it is allowed, but the "
                         "headline must be the four-pool number and math500 must be flagged. Pass "
                         "`--pools gsm8k_train math_train_easy` to avoid the issue entirely at the "
                         "cost of 57%% of the fuel.")
    ap.add_argument("--prefixes-per-problem", type=int, default=2)
    ap.add_argument("--prefix-tokens", type=int, default=96,
                    help="tokens of a4's own trace to keep. §41b: 79% of wrong traces already differ "
                         "from a correct derivation at equation index 0, so a LONG prefix mostly "
                         "carries the mistake forward -- this is short on purpose.")
    ap.add_argument("--min-prefix-tokens", type=int, default=16)
    ap.add_argument("--include-empty-prefix", type=int, default=1,
                    help="also emit a prefix-free variant per problem: the CONTROL that decides "
                         "whether prefix conditioning matters at all")
    ap.add_argument("--completer-instruction",
                    default="\n\nKeep the remaining reasoning brief: at most 5 short steps, then give the "
                            "final answer as \\boxed{}.",
                    help="appended to the question in the COMPLETER's prompt only. Needed because "
                         "--max-trace-tokens drops any splice longer than the student's own p90, so an "
                         "unprompted long-form completer would have most of its output discarded rather "
                         "than merely truncated -- the cap silently becomes a yield problem. "
                         "⚠️This does NOT contradict §41bg, where brevity conditioning degraded Qwen3-4B: "
                         "that measured per-token AGREEMENT with a4's tokens under a divergence loss. This "
                         "is generation quality, a different quantity, and `gen_teacher.py` already "
                         "conditions for concision the same way. The student never sees this text -- only "
                         "the resulting trace is kept -- so it cannot repeat §41m's context-poisoning.")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-completion-tokens", type=int, default=256)
    ap.add_argument("--tail", type=int, default=48)
    ap.add_argument("--max-trace-tokens", type=int, default=330,
                    help="hard cap on the FINAL spliced trace (round 6's own p90 think length). "
                         "Every arm that lengthened traces on this line has lost.")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=46)
    a = ap.parse_args()

    import vllm_argonne
    vllm_argonne.register()          # transformers-5.x tokenizer shim; needed even for native Qwen3
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.completer, trust_remote_code=True)

    # ---- group a4's rollouts by problem and pick the coverage hole ------------------------------
    by_q = defaultdict(list)
    for path in a.rollouts:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_q[(r["pool"], r["question"])].append(r)
    stat = Counter()
    chosen = []
    for (pool, q), rs in sorted(by_q.items()):
        if a.pools and pool not in a.pools:
            stat["skip_pool_filtered"] += 1
            continue
        n_ok = sum(1 for r in rs if r["label"] == "correct")
        K = len(rs)
        if a.select == "never" and n_ok != 0:
            stat["skip_already_solved"] += 1
            continue
        if a.select == "rare" and not (0 < n_ok <= max(1, K // 4)):
            stat["skip_not_rare"] += 1
            continue
        chosen.append((pool, q, rs))
    rng.shuffle(chosen)
    if a.max_problems:
        chosen = chosen[:a.max_problems]
    print(f"[pfx] {len(by_q):,} problems -> {len(chosen):,} selected (--select {a.select}); "
          f"{dict(stat)}", flush=True)
    if not chosen:
        raise SystemExit("nothing selected; check --select against the dump's labels")

    # ---- build prefix-conditioned prompts -------------------------------------------------------
    # The completer's OWN chat template is used, because the completer is the model generating. The
    # tokenizers are id-identical (verified: len(tok)=151,669, <think>=151667), so a4's prefix tokens
    # are valid tokens here -- that identity is the whole reason a cross-model splice is even defined.
    def prompt_for(q, prefix):
        enc = tok.apply_chat_template([{"role": "user", "content": q + a.completer_instruction}],
                                      tokenize=True,
                                      add_generation_prompt=True, enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if len(enc) and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        ids = [int(x) for x in enc]
        # Qwen3's template may already open the think block; adding a second <think> would produce a
        # prompt no trace ever contains.
        opened = THINK_OPEN in tok.decode(ids[-8:])
        add = ("" if opened else THINK_OPEN + "\n") + (prefix + "\n" if prefix else "")
        return ids + (tok.encode(add, add_special_tokens=False) if add else [])

    jobs = []     # (prompt_ids, pool, question, gold, prefix, kind)
    for pool, q, rs in chosen:
        gold = norm(str(rs[0].get("gold", "")))
        seen = set()
        cands = [r for r in rs if r.get("trace")]
        rng.shuffle(cands)
        made = 0
        for r in cands:
            if made >= a.prefixes_per_problem:
                break
            pfx = cut_prefix(strip_think_open(r["trace"]), tok, a.prefix_tokens, a.min_prefix_tokens)
            if not pfx or pfx in seen:
                continue
            seen.add(pfx)
            made += 1
            jobs.append((prompt_for(q, pfx), pool, q, gold, pfx, "pfx"))
        stat["problems_with_prefix"] += 1 if made else 0
        stat["no_usable_prefix"] += 0 if made else 1
        if a.include_empty_prefix:
            jobs.append((prompt_for(q, ""), pool, q, gold, "", "empty"))
    print(f"[pfx] {len(jobs):,} prompts ({sum(1 for j in jobs if j[5]=='pfx'):,} prefixed, "
          f"{sum(1 for j in jobs if j[5]=='empty'):,} empty-prefix control), "
          f"{a.samples} samples each", flush=True)

    # ---- generate, then force-close anything still open ------------------------------------------
    llm = LLM(model=a.completer, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len, seed=a.seed)
    # ⚠️NO per-request `seed=` here. A request-level seed with n>1 risks collapsing the n candidates
    # onto each other, which would silently halve the sample budget while looking like it worked.
    # Reproducibility comes from the engine seed passed to LLM(...) above.
    sp = SamplingParams(n=a.samples, temperature=a.temperature, top_p=0.95,
                        max_tokens=a.max_completion_tokens)
    outs = llm.generate([TokensPrompt(prompt_token_ids=j[0]) for j in jobs], sp)

    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False)
    need, meta, texts = [], [], []
    for ji, o in enumerate(outs):
        for ci, cand in enumerate(o.outputs):
            t = cand.text
            texts.append(t)
            if "</think>" not in t:
                need.append(TokensPrompt(prompt_token_ids=jobs[ji][0] + list(cand.token_ids)
                                         + close_ids))
                meta.append(len(texts) - 1)
    stat["forced_closed"] = len(need)
    if need:
        # identical to clean_eval's budget-forcing, which is worth +2.25 five-pool at eval time --
        # here it converts a truncated completion into a usable target instead of discarding it
        outs2 = llm.generate(need, SamplingParams(n=1, temperature=0.0, max_tokens=a.tail))
        for idx, o in zip(meta, outs2):
            texts[idx] = texts[idx] + CLOSE_STR + o.outputs[0].text

    # ---- grade, cap length, emit -----------------------------------------------------------------
    rows, lens, kept_by_kind = [], [], Counter()
    ti = 0
    solved_problems = set()
    for ji, o in enumerate(outs):
        _, pool, q, gold, pfx, kind = jobs[ji]
        for _ci in range(len(o.outputs)):
            t = texts[ti]; ti += 1
            stat[f"gen_{kind}"] += 1
            pred = extract_boxed(t)
            if pred is None:
                stat[f"drop_no_answer_{kind}"] += 1
                continue
            if pred != gold:
                stat[f"drop_wrong_{kind}"] += 1
                continue
            body = (pfx + "\n" if pfx else "") + t.lstrip("\n")
            trace = THINK_OPEN + "\n" + body
            n_tok = len(tok.encode(trace, add_special_tokens=False))
            if n_tok > a.max_trace_tokens:
                stat[f"drop_too_long_{kind}"] += 1
                continue
            rows.append({"pool": pool, "question": q, "trace": trace, "gold": gold,
                         "pred": pred, "label": "correct",
                         "src": "prefix_completion", "kind": kind, "prefix_tokens":
                         len(tok.encode(pfx, add_special_tokens=False)) if pfx else 0})
            lens.append(n_tok)
            kept_by_kind[kind] += 1
            solved_problems.add((pool, q))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lens.sort()
    med = lens[len(lens) // 2] if lens else 0
    p90 = lens[int(0.9 * len(lens))] if lens else 0
    print(f"\n[pfx] kept {len(rows):,} traces over {len(solved_problems):,} of {len(chosen):,} "
          f"never-solved problems ({100*len(solved_problems)/max(1,len(chosen)):.1f}% newly solvable)")
    print(f"[pfx] by kind: {dict(kept_by_kind)}   trace tokens median {med} p90 {p90}")
    for k in ("pfx", "empty"):
        g = stat.get(f"gen_{k}", 0)
        if g:
            print(f"[pfx] {k:6s} yield {kept_by_kind[k]:6d}/{g:6d} = {100*kept_by_kind[k]/g:5.2f}%  "
                  f"(wrong {stat.get(f'drop_wrong_{k}',0)}, no_answer "
                  f"{stat.get(f'drop_no_answer_{k}',0)}, too_long {stat.get(f'drop_too_long_{k}',0)})")
    print(f"[pfx] wrote {a.out}", flush=True)

    if a.stats_out:
        json.dump({"rows": len(rows), "problems_selected": len(chosen),
                   "problems_newly_solvable": len(solved_problems),
                   "kept_by_kind": dict(kept_by_kind), "stat": dict(stat),
                   "trace_tokens": {"median": med, "p90": p90,
                                    "mean": sum(lens) / len(lens) if lens else 0},
                   "args": vars(a)}, open(a.stats_out, "w"), indent=1)
        print(f"[pfx] wrote {a.stats_out}", flush=True)


if __name__ == "__main__":
    main()
