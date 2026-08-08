#!/usr/bin/env python3
"""HumanEval pass@1 for the Argonne think models — verifies the v7 coding tier actually added a
code-generation capability that v3 lacked (2026-07-12, §26).

The shipped v3 had NO coding data; v7 adds code_magicoder. This grader measures whether that
transferred: prompt the chat model to complete each HumanEval function, extract the code, and run it
against the benchmark's unit tests in a sandboxed subprocess (timeout + no network). Reports pass@1.

Standard benchmark protocol (executing generated solutions against the provided tests IS the defined
HumanEval eval); execution is isolated in a short-lived subprocess with a hard timeout.
Uses the validated vLLM port (register() applies the transformers-5.x tokenizer shim, needed here too).
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from clean_eval import build_ids  # noqa: E402

PROMPT = ("Complete the following Python function. Return the COMPLETE function (including the "
          "signature) inside a single ```python code block.\n\n{code}")


def load_humaneval():
    from datasets import load_from_disk, load_dataset
    try:
        return list(load_from_disk("/project/rcc/youzhi/data/humaneval_ds"))
    except Exception:
        pass
    for name in ("openai_humaneval", "openai/humaneval"):
        try:
            return list(load_dataset(name, split="test"))
        except Exception:
            continue
    raise RuntimeError("could not load HumanEval")


def extract_code(text, entry_point, prompt_code):
    """Pull a runnable function definition out of the model reply."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    m = re.search(r"```(?:python)?\s*(.*?)```", body, re.S)
    code = m.group(1) if m else body
    if f"def {entry_point}" in code:
        return code
    # model returned only a body / partial -> graft onto the provided signature
    return prompt_code + "\n" + code


def run_one(program, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(program)
        f.flush()
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True,
                               timeout=timeout, text=True)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


HARNESS = (
    "import signal\n"
    "def _timeout(s,f): raise TimeoutError()\n"
    "signal.signal(signal.SIGALRM,_timeout); signal.alarm({t})\n"
    "{code}\n{test}\ncheck({entry})\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=0)   # 0 = all 164
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    fh = open(args.log, "a") if args.log else None

    def out(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    probs = load_humaneval()
    if args.n_problems and args.n_problems > 0:
        probs = probs[:args.n_problems]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_new_tokens,
                        stop_token_ids=[im_end])
    prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, PROMPT.format(code=p["prompt"]), think=True))
               for p in probs]
    outs = llm.generate(prompts, sp)

    n_pass = n_code = 0
    for p, o in zip(probs, outs):
        code = extract_code(o.outputs[0].text, p["entry_point"], p["prompt"])
        if f"def {p['entry_point']}" in code:
            n_code += 1
        program = HARNESS.format(t=args.timeout, code=code, test=p["test"], entry=p["entry_point"])
        if run_one(program, args.timeout):
            n_pass += 1
    n = len(probs)
    out("=" * 60)
    out(f"HumanEval  model={Path(args.model).name}  n={n}")
    out(f"  produced a function def : {100*n_code/n:.1f}%  ({n_code}/{n})")
    out(f"  PASS@1                  : {100*n_pass/n:.1f}%  ({n_pass}/{n})")
    out("=" * 60)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
