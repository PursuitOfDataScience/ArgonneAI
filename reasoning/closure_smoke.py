#!/usr/bin/env python3
"""90-second check that a checkpoint still TERMINATES, before spending an hour gating it.

WHY THIS EXISTS (2026-08-10). The first on-policy-distillation arm trained cleanly for 33 minutes --
reverse KL 0.85 -> 0.32, teacher/student argmax agreement 76% -> 85%, every logged number healthy --
and came out with a **96.95% unclosed rate and greedy 1.75** against a 56.70 baseline. It never
emits `</think>`; it rambles to the token cap on essentially every item. The full gate then spent
another 40 GPU-minutes measuring self-consistency and pass@8 on a model that cannot finish a
sentence, and had to be cancelled.

One greedy pass over a couple of hundred items would have caught it in 90 seconds. So: run this
between training and the gate, and skip the gate for any arm that fails it.

Exit code 0 = healthy, 3 = closure collapse (above --max-unclosed), 2 = could not run. The
launcher branches on that.

  python reasoning/closure_smoke.py --model <dir> --pools asdiv --n 200 --max-unclosed 0.45
"""
import argparse
import json
import os
import sys

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", nargs="+", default=["asdiv"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-unclosed", type=float, default=0.45,
                    help="fail above this. The healthy checkpoints on this line sit at 0.07-0.12 "
                         "and the worst gated arm at 0.22, so 0.45 fails only a real collapse.")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    from collections import Counter
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from effort_probe import load_pool, make_llm, prompt_ids, grade_one
    from star_generate import extract_boxed, norm

    llm, tok = make_llm(a.model, gpu_util=a.gpu_util, max_model_len=a.max_model_len, seed=0)
    out, worst = {}, 0.0
    for pool in a.pools:
        probs = load_pool(pool, a.n, seed=0)
        ids = [prompt_ids(tok, q) for q, _ in probs]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids],
                            SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens))
        fm = Counter()
        dec = 0
        for o, (_, g) in zip(outs, probs):
            lab, _ = grade_one(o.outputs[0].text, norm(str(g)), extract_boxed)
            fm[lab] += 1
            dec += len(o.outputs[0].token_ids)
        n = len(probs)
        unc = fm["unclosed"] / n
        worst = max(worst, unc)
        out[pool] = {"n": n, "fm": dict(fm), "unclosed": unc,
                     "greedy": fm["correct"] / n, "mean_decoded": dec / n}
        print(f"[smoke/{pool}] greedy {fm['correct'] / n * 100:5.2f}%  unclosed "
              f"{unc * 100:5.2f}%  no_answer {fm['no_answer'] / n * 100:5.2f}%  "
              f"mean decoded {dec / n:.0f} tok", flush=True)

    if a.json_out:
        json.dump({"model": a.model, "res": out, "max_unclosed": a.max_unclosed},
                  open(a.json_out, "w"), indent=1)
    if worst > a.max_unclosed:
        print(f"FAIL closure collapse: unclosed {worst * 100:.1f}% > "
              f"{a.max_unclosed * 100:.0f}% -- this checkpoint does not terminate, skip the gate",
              flush=True)
        raise SystemExit(3)
    print(f"[smoke] PASS  worst unclosed {worst * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
