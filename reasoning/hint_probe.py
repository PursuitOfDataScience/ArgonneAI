#!/usr/bin/env python3
"""Can the model USE information placed in its context? The cheapest question nobody asked.

WHY (2026-08-10, §41n). Gold-anchored self-distillation put the verified answer -- and, where one
existed, a correct derivation -- into a FROZEN COPY of argonne4-think's own context, then trained the
unhinted model to match the hinted model's next-token distribution on its own traces. The teacher's
distribution barely moved: **JSD 0.0113 with 95.0% argmax agreement**, and 2,811 training steps
reduced it only to 0.0098. Telling this model the answer changes 5% of what it says next.

There are two very different explanations and they imply opposite next steps:
  (a) the divergence is IRREDUCIBLE -- the teacher conditions on information the student's input does
      not contain, so matching it is impossible in principle and the method is mis-specified here;
  (b) the model cannot EXPLOIT in-context information at all, in which case the hindsight objective
      never had a signal to give, and the same deficit would cap every retrieval-, hint-, or
      few-shot-based method on this base -- a far more fundamental finding than any reasoning gap.

(b) is directly testable in about two minutes, and it has never been tested on this line. Give the
model the problem with the answer written in the prompt and see whether it says the answer.

Four conditions, same items, greedy, one engine:
  plain      the deployed prompt (the baseline)
  answer     + "(Reference: the correct final answer is X.)"          <- the GASD hint, verbatim
  derivation + the answer AND a correct derivation
  wrong      + a DELIBERATELY WRONG answer                            <- the control that matters

`wrong` is what makes this a measurement rather than a demonstration. If accuracy rises under `answer`
but is unchanged under `wrong`, the model is reading the hint and reasoning with it. If accuracy
follows the hint in BOTH directions, it is copying, which is still in-context use but says nothing
about reasoning. If NEITHER moves, the model is ignoring its context, and that is finding (b).

  python reasoning/hint_probe.py --model <dir> --pools asdiv svamp --n 200
"""
import argparse
import json
import os
import sys
from collections import Counter

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HINT_ANS = "\n\nReference (already verified, for guidance): the correct final answer is {gold}."
HINT_DER = ("\n\nReference (already verified, for guidance): the correct final answer is {gold}."
            " A correct derivation is: {sol}")


def perturb(gold):
    """A plainly wrong but same-shaped answer, so the control differs only in correctness."""
    try:
        v = float(gold)
        w = v + 1 if abs(v) < 1e6 else v / 2
        return str(int(w)) if float(w).is_integer() else str(w)
    except ValueError:
        return gold + "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", nargs="+", default=["asdiv", "svamp"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from effort_probe import load_pool, make_llm, prompt_ids, grade_one
    from star_generate import extract_boxed, norm

    llm, tok = make_llm(a.model, gpu_util=a.gpu_util, max_model_len=a.max_model_len, seed=0)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens)
    res = {}

    for pool in a.pools:
        probs = load_pool(pool, a.n, seed=0)
        golds = [norm(str(g)) for _, g in probs]
        # a reference derivation from the model itself is not available offline here, so the
        # `derivation` condition reuses the answer plus a one-line skeleton; the informative
        # comparison is plain vs answer vs wrong.
        conds = {
            "plain":  [q for q, _ in probs],
            "answer": [q + HINT_ANS.format(gold=g) for (q, _), g in zip(probs, golds)],
            "wrong":  [q + HINT_ANS.format(gold=perturb(g)) for (q, _), g in zip(probs, golds)],
        }
        out = {}
        for name, qs in conds.items():
            ids = [prompt_ids(tok, q) for q in qs]
            outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)
            fm, ok, said_hint = Counter(), [], 0
            for o, g, q in zip(outs, golds, qs):
                lab, pred = grade_one(o.outputs[0].text, g, extract_boxed)
                fm[lab] += 1
                ok.append(lab == "correct")
                # did it emit the number that was IN the prompt, whatever that number was?
                hinted = norm(str(perturb(g))) if name == "wrong" else g
                if name != "plain" and pred is not None and pred == hinted:
                    said_hint += 1
            n = len(qs)
            out[name] = {"acc": sum(ok) / n * 100, "fm": dict(fm), "ok": ok,
                         "echoed_hint": said_hint / n * 100 if name != "plain" else None}
            extra = "" if out[name]["echoed_hint"] is None else \
                f"  echoed the hinted number {out[name]['echoed_hint']:5.1f}%"
            print(f"[hint/{pool}] {name:10s} acc {out[name]['acc']:6.2f}%{extra}", flush=True)
        base = out["plain"]["acc"]
        print(f"[hint/{pool}] => answer {out['answer']['acc'] - base:+6.2f}pt vs plain,  "
              f"wrong {out['wrong']['acc'] - base:+6.2f}pt vs plain", flush=True)
        res[pool] = out

    if a.json_out:
        json.dump({"model": a.model, "res": res}, open(a.json_out, "w"), indent=1)
        print(f"wrote {a.json_out}")

    print("\n" + "=" * 70)
    for k in ("plain", "answer", "wrong"):
        vals = [res[p][k]["acc"] for p in res]
        eh = [res[p][k]["echoed_hint"] for p in res if res[p][k]["echoed_hint"] is not None]
        e = f"   echoed {sum(eh) / len(eh):5.1f}%" if eh else ""
        print(f"  {k:10s} {sum(vals) / len(vals):6.2f}%{e}")
    print("\nread: answer>>plain and wrong~=plain -> it reasons WITH the hint;")
    print("      answer>>plain and wrong<<plain  -> it COPIES the hint (in-context use, not reasoning);")
    print("      neither moves                   -> it ignores its context, which caps every")
    print("                                         hint/retrieval/few-shot method on this base.")


if __name__ == "__main__":
    main()
