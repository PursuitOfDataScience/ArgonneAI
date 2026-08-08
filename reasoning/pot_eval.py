#!/usr/bin/env python3
"""Program-of-Thought (PoT) evaluation for the Argonne 3.0 reasoning models.

THE HYPOTHESIS (§22 lever #2, never tried on soup_blend_a085): the residual failure
is *arithmetic execution* inside *correct procedures* (`8+3=7`), and pass@256=82% vs
greedy=2.5% means the capability is latent but un-selectable. Both are fixed by moving
computation OUT of the token stream and INTO a Python interpreter:

  * GENERATION: the model writes a short Python program (the *procedure* — its strength);
    Python does the *arithmetic* (its weakness). It never has to compute `8+3` itself.
  * SELECTION: executing K programs and voting over the *executed* answers is an
    EXTERNAL, non-base-limited verifier (the one §22i said was needed to capture the 82%
    ceiling — a same-base learned verifier failed: best-of-N 13.5% ~= vote 13.0%).

This is a TRAINING-FREE probe: few-shot the existing model to emit code, execute, grade.
It reuses the VALIDATED vLLM port (reasoning/vllm_argonne.py) for fast, HBM-filling
sampling, and the verified `load_problems`/`norm` primitives from star_generate.py so the
GSM8K problem set + numeric normalization match every prior number in the doc (seed 0).

Reports, on the SAME 200 GSM8K problems used throughout §22:
  * greedy pass@1                 (vs NL greedy 2.5% / budget-forced 7.5%)
  * execution-verified self-consistency (majority vote over EXECUTED answers, K samples)
                                  (vs NL self-consistency 13.0%)
  * pass@K                        (vs NL pass@32 42.5% / pass@256 82%)
  * code-health diagnostics       (has-code / ran-ok / produced-number / exec-error rates)

Two phases in one job: (1) vLLM generates all programs; (2) CPU executes+grades them in a
timeout-guarded subprocess pool (model-generated code -> sandboxed by process + timeout).
"""
import argparse
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star_generate import load_problems, norm  # noqa: E402

SYSTEM = ("You are an expert at solving math word problems by writing a short Python "
          "program. Read the problem, then write a Python program that computes the "
          "answer step by step using variables, and prints ONLY the final numeric answer. "
          "Do not do the arithmetic yourself — let Python compute it.")

# Few-shot PoT exemplars: variables + expressions (NOT pre-computed literals) so the model
# imitates OFFLOADING arithmetic to Python rather than doing it in its head.
SHOTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?",
     "```python\napril = 48\nmay = april / 2\ntotal = april + may\nprint(int(total))\n```"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
     "babysitting. How much did she earn?",
     "```python\nrate_per_hour = 12\nminutes = 50\nearned = rate_per_hour * (minutes / 60)\n"
     "print(earned)\n```"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the "
     "money she needs. Her parents decided to give her $15 for that purpose, and her "
     "grandparents twice as much as her parents. How much more money does Betty need to "
     "buy the wallet?",
     "```python\ncost = 100\nbetty = cost / 2\nparents = 15\ngrandparents = 2 * parents\n"
     "have = betty + parents + grandparents\nneed = cost - have\nprint(int(need))\n```"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts "
     "in total does it take?",
     "```python\nblue = 2\nwhite = blue / 2\ntotal = blue + white\nprint(int(total))\n```"),
]

CODE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text):
    blocks = CODE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    # Fallback: the tail looks like a program (has print( or assignments).
    if "print(" in text:
        # take from the first line that mentions an assignment or print
        lines = text.splitlines()
        start = 0
        for i, ln in enumerate(lines):
            if re.match(r"\s*\w+\s*=", ln) or ln.strip().startswith("print("):
                start = i
                break
        cand = "\n".join(lines[start:]).strip()
        return cand or None
    return None


def run_code(code, timeout=6, cwd=None):
    """Execute model-written code in an isolated subprocess. Return (norm_answer, status)."""
    if not code:
        return None, "no_code"
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception:
        return None, "spawn_err"
    if p.returncode != 0:
        return None, "exec_err"
    out = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    if not out:
        return None, "no_output"
    tail = out[-1].strip()
    # A real GSM8K answer is a small number; a giant digit run is a runaway program.
    # norm() does int(float(x)) which raises OverflowError on inf, so guard it.
    if len(tail) > 40:
        return None, "bignum"
    try:
        val = norm(tail)
    except (OverflowError, ValueError):
        return None, "bignum"
    return (val, "ok") if val is not None else (None, "no_number")


def build_ids(tok, question, think):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in SHOTS:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=think)
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    if len(enc) > 0 and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(x) for x in enc]


def grade_texts(texts_per_problem, golds, cwd):
    """texts_per_problem: list (per problem) of list-of-str generations.
    Returns per-problem list of (norm_answer_or_None, status)."""
    flat, idx = [], []
    for pi, texts in enumerate(texts_per_problem):
        for ci, t in enumerate(texts):
            flat.append(extract_code(t))
            idx.append((pi, ci))
    results = [None] * len(flat)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for j, r in enumerate(ex.map(lambda c: run_code(c, cwd=cwd), flat)):
            results[j] = r
    per = [[None] * len(t) for t in texts_per_problem]
    for (pi, ci), r in zip(idx, results):
        per[pi][ci] = r
    return per


def report(tag, per_problem, golds, out):
    n = len(per_problem)
    status_tally = Counter()
    n_first_correct = n_pass_k = n_maj = n_maj_scorable = 0
    tot = corr = 0
    for texts, gold in zip(per_problem, golds):
        votes = Counter()
        any_correct = False
        for j, (val, st) in enumerate(texts):
            status_tally[st] += 1
            tot += 1
            ok = (val is not None and val == gold)
            if ok:
                corr += 1
                any_correct = True
                if j == 0:
                    n_first_correct += 1
            if val is not None:
                votes[val] += 1
        if any_correct:
            n_pass_k += 1
        if votes:
            n_maj_scorable += 1
            if votes.most_common(1)[0][0] == gold:
                n_maj += 1
    k = len(per_problem[0]) if per_problem else 0
    out("  " + "-" * 68)
    out(f"  [{tag}]  n={n}  K={k}")
    out(f"  single-sample acc      : {100*corr/max(tot,1):.2f}%  ({corr}/{tot} samples)")
    out(f"  pass@1 (sample[0])     : {100*n_first_correct/n:.2f}%  ({n_first_correct}/{n})")
    if k > 1:
        out(f"  exec-verified self-cons: {100*n_maj/n:.2f}%  ({n_maj}/{n})   <- majority over EXECUTED answers")
        out(f"  pass@{k:<15}: {100*n_pass_k/n:.2f}%  ({n_pass_k}/{n})   <- latent ceiling")
    out(f"  code health            : {dict(status_tally)}")
    has_code = tot - status_tally.get("no_code", 0)
    ran_ok = status_tally.get("ok", 0) + status_tally.get("no_number", 0)
    out(f"    has-code={100*has_code/max(tot,1):.0f}%  ran-without-error={100*ran_ok/max(tot,1):.0f}%  "
        f"produced-number={100*status_tally.get('ok',0)/max(tot,1):.0f}%")
    out("  " + "-" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--k", type=int, default=32, help="samples/problem for self-consistency")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--think", action="store_true", help="let the model think before coding (default: no-think code-only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exec-cwd", default=None, help="cwd for the code subprocesses (scratch)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--dump", default=None, help="write per-problem generations+code+grade JSON here")
    args = ap.parse_args()

    fh = open(args.log, "a") if args.log else None

    def out(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    probs = load_problems(args.source, args.n_problems, seed=args.seed)
    golds = [g for _, g, _ in probs]
    think = args.think

    out("=" * 72)
    out(f"PoT eval  model={args.model}")
    out(f"  source={args.source} n={len(probs)} K={args.k} think={'ON' if think else 'off'} "
        f"max_new={args.max_new_tokens} shots={len(SHOTS)} seed={args.seed}")
    out("=" * 72)

    prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, q, think)) for q, _, _ in probs]
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)

    # Phase 1a: greedy pass@1
    sp_g = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)
    outs_g = llm.generate(prompts, sp_g)
    greedy_texts = [[o.outputs[0].text] for o in outs_g]

    # Phase 1b: K samples for self-consistency / pass@K
    if args.k > 1:
        sp_s = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                              top_k=args.top_k, max_tokens=args.max_new_tokens)
        outs_s = llm.generate(prompts, sp_s)
        samp_texts = [[c.text for c in o.outputs] for o in outs_s]

    # Phase 2: execute + grade on CPU (timeout-guarded subprocess pool)
    cwd = args.exec_cwd or "/tmp"
    g_per = grade_texts(greedy_texts, golds, cwd)
    report("GREEDY", g_per, golds, out)
    s_per = None
    if args.k > 1:
        s_per = grade_texts(samp_texts, golds, cwd)
        report(f"SAMPLED T={args.temperature}", s_per, golds, out)

    if args.dump:
        import json
        rows = []
        for pi, (q, gold, _) in enumerate(probs):
            gt = greedy_texts[pi][0]
            gv, gs = g_per[pi][0]
            row = {"question": q, "gold": gold, "greedy_text": gt,
                   "greedy_code": extract_code(gt), "greedy_val": gv, "greedy_status": gs}
            if s_per is not None:
                row["samples"] = [{"code": extract_code(samp_texts[pi][j]),
                                   "val": s_per[pi][j][0], "status": s_per[pi][j][1]}
                                  for j in range(len(samp_texts[pi]))]
            rows.append(row)
        json.dump(rows, open(args.dump, "w"))
        out(f"  dumped {len(rows)} problems -> {args.dump}")

    if fh:
        fh.close()


if __name__ == "__main__":
    main()
