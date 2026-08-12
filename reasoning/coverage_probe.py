#!/usr/bin/env python3
"""Did CE on verified solutions install the behaviour ON THE TRAINED PROBLEMS THEMSELVES?

WHY THIS EXISTS. §41bn measured the coverage arm as a clean null: 4,303 gold-verified Qwen3-14B
solutions to problems a4 had NEVER solved in 8 samples, trained as plain CE, moved eval pass@8 by
-0.03 (p=1.00). §41bq then showed coverage is 76-91% of the entire remaining gap to 3.5-think. So
whether that null is fundamental or fixable is the most consequential open question on this line,
and the two candidate explanations make OPPOSITE predictions about one cheap measurement:

  (a) SCALE / OPTIMISATION. 4,303 traces x ~275 tokens is ~1.2M tokens, one epoch at lr 5e-6. That
      may simply be too little to install new behaviour at all. PREDICTION: the model still fails
      the TRAINED problems -- it did not even memorise them -- and scaling the corpus is worth doing.
  (b) FUNDAMENTAL. The traces were absorbed as surface form; the reasoning they encode is out of
      reach for a 1.04B model whose base is -26.5 mmlu / -41.2 gsm8k behind Qwen3-0.6B-Base (§39).
      PREDICTION: the model now solves the TRAINED problems (memorisation worked) but nothing
      transfers, so coverage is problem-specific and post-training cannot buy it.

Distinguishing them decides whether §41bn's withdrawal of the coverage-scaling branch was right.
That withdrawal rested on "the first 4,303 traces moved the metric by nothing, so more will not
either" -- which is only sound under (b). Under (a) it is exactly backwards.

WHAT IT MEASURES. K samples at the eval temperature on the specific problems that appear in the
coverage training file, reporting pass@K / per-sample solve rate / solved-all-K. The problems are
TRAIN problems, so this is deliberately a MEMORISATION probe and its numbers must never be quoted as
a capability result -- that is what the held-out pools in `effort_gate.py` are for.

⚠️ONE MODEL PER INVOCATION (vLLM keeps its KV cache after the LLM object is dropped). Call it once
per checkpoint and compare the JSONs; `reasoning/a4_covprobe.sh` does exactly that.

  python reasoning/coverage_probe.py \
      --model /project/.../think_pfxcomp --label pfxcomp \
      --coverage /project/rcc/youzhi/data/a4_pfxcomp/all.jsonl \
      --rollouts /project/rcc/youzhi/data/a4_opd_opd_r3_r4/all.jsonl \
      --k 8 --json-out report/a4_covprobe_pfxcomp.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed, norm    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # ⚠️ONE MODEL PER PROCESS, deliberately. vLLM does not release its KV cache when an LLM object is
    # deleted, so a second LLM() in the same process dies with "Free memory on device ..." -- which is
    # exactly why effort_gate.py runs every model in an isolated subprocess. A --models loop here would
    # work for the first model and fail for every one after it.
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--coverage", required=True,
                    help="the coverage training file; its questions ARE the trained problems")
    ap.add_argument("--rollouts", required=True,
                    help="the pre-training rollout dump, used to confirm each probed problem really was "
                         "never-solved (0/K) by the starting policy -- otherwise 'newly solved' is vacuous")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    # ---- the trained problems, and gold ---------------------------------------------------------
    trained = {}
    with open(a.coverage) as f:
        for line in f:
            r = json.loads(line)
            trained[(r["pool"], r["question"])] = norm(str(r["gold"]))
    # ---- confirm they were 0/K for the starting policy ------------------------------------------
    solved_before = defaultdict(bool)
    with open(a.rollouts) as f:
        for line in f:
            r = json.loads(line)
            k = (r["pool"], r["question"])
            if k in trained and r["label"] == "correct":
                solved_before[k] = True
    probe = [(p, q, g) for (p, q), g in sorted(trained.items()) if not solved_before[(p, q)]]
    dropped = len(trained) - len(probe)
    if a.max_problems:
        probe = probe[:a.max_problems]
    print(f"[probe] {len(trained):,} problems in the coverage file; {dropped:,} were already solved by the "
          f"starting policy and are excluded; probing {len(probe):,} genuinely never-solved ones", flush=True)
    if not probe:
        raise SystemExit("nothing to probe")

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from clean_eval import build_ids

    out = {}
    for spec in [f"{a.label or os.path.basename(a.model.rstrip('/'))}={a.model}"]:
        label, path = spec.split("=", 1)
        print(f"\n######## {label} = {path} ########", flush=True)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        llm = LLM(model=path, dtype="bfloat16", trust_remote_code=True,
                  gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len, seed=a.seed)
        prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, q)) for _p, q, _g in probe]
        outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                                                    max_tokens=a.max_new_tokens))
        n_any = n_all = tot_correct = tot_samples = 0
        per_pool = defaultdict(lambda: [0, 0])
        for (pool, _q, gold), o in zip(probe, outs):
            hits = sum(1 for c in o.outputs if extract_boxed(c.text) == gold)
            tot_correct += hits
            tot_samples += len(o.outputs)
            n_any += hits > 0
            n_all += hits == len(o.outputs)
            per_pool[pool][0] += hits > 0
            per_pool[pool][1] += 1
        res = {"n_problems": len(probe), "pass_at_k": 100 * n_any / len(probe),
               "solve_rate": 100 * tot_correct / max(1, tot_samples),
               "all_k": 100 * n_all / len(probe),
               "per_pool": {p: 100 * v[0] / v[1] for p, v in per_pool.items()}}
        out[label] = res
        print(f"[probe] {label}: pass@{a.k} {res['pass_at_k']:.2f}%  per-sample solve rate "
              f"{res['solve_rate']:.2f}%  solved all {a.k}: {res['all_k']:.2f}%", flush=True)
        print(f"[probe] {label}: per pool " +
              "  ".join(f"{p} {v:.1f}%" for p, v in sorted(res["per_pool"].items())), flush=True)
        del llm

    print("\n######## SUMMARY: pass@%d on the TRAINED never-solved problems ########" % a.k)
    for label, r in out.items():
        print(f"  {label:12s} pass@{a.k} {r['pass_at_k']:6.2f}%   solve-rate {r['solve_rate']:5.2f}%")
    print("""
INTERPRETATION, fixed in advance:
  a LARGE rise -> CE installed the behaviour on these problems but it did not transfer to the eval
                  pools. Coverage is problem-specific; scaling the corpus buys memorisation, not
                  reach. §41bn's withdrawal of the scaling branch stands.
  ~NO rise     -> CE did not even install the trained behaviour, so the eval null is about scale or
                  optimisation rather than reachability, and the scaling branch must be REOPENED.
⚠️These are TRAIN problems. The numbers are a memorisation probe and are not a capability result.""")

    if a.json_out:
        json.dump({"args": vars(a), "res": out}, open(a.json_out, "w"), indent=1)
        print(f"[probe] wrote {a.json_out}")


if __name__ == "__main__":
    main()
