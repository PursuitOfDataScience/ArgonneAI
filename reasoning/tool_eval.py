#!/usr/bin/env python3
"""Tool-call FORMAT eval — verifies the v7 tool_calc tier taught valid <tool_call> emission
(2026-07-12, §26). Held-out synthetic arithmetic queries (fresh numbers, NOT in training) + a
calculator/python tool spec in the system message; the model should emit a parseable
<tool_call>{"name":..,"arguments":..}</tool_call> whose expression computes the correct answer.

Metrics: % emitting a valid tool_call JSON; % whose tool expression evaluates to gold; % that also
reach the right \\boxed answer. This is a CAPABILITY check (not the ship gate — that stays clean math).
"""
import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed  # noqa: E402
from clean_eval import build_ids         # noqa: E402

TOOLS_SPEC = (
    '[{"name": "calculator", "description": "Evaluate an arithmetic expression", '
    '"parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, '
    '"required": ["expression"]}}, '
    '{"name": "python", "description": "Run a short Python snippet and return stdout", '
    '"parameters": {"type": "object", "properties": {"code": {"type": "string"}}, '
    '"required": ["code"]}}]'
)
SYS = ("You may call tools. Function signatures are inside <tools></tools>:\n<tools>\n"
       + TOOLS_SPEC + "\n</tools>\nCall a tool with a JSON object inside <tool_call></tool_call> "
       "and read its <tool_response>.")

SAFE = re.compile(r"^[\d\s+\-*/().%]+$")


def safe_eval(expr):
    expr = expr.strip()
    if not SAFE.match(expr):
        return None
    try:
        v = eval(expr, {"__builtins__": {}}, {})   # arithmetic-only (regex-gated)
        return float(v)
    except Exception:
        return None


def gen_problems(n, seed=0):
    rng = random.Random(seed)
    probs, who, it = [], ["Ava", "the shop", "Ben", "the club"], ["pears", "pens", "chips", "seats"]
    for _ in range(n):
        k = rng.choice(["add", "sub", "mul", "div"])
        w, i = rng.choice(who), rng.choice(it)
        if k == "add":
            a, b = rng.randint(123, 899), rng.randint(117, 777); ans = a + b
            q = f"{w} had {a} {i} and got {b} more. How many {i} now?"
        elif k == "sub":
            a = rng.randint(400, 950); b = rng.randint(111, a - 1); ans = a - b
            q = f"{w} had {a} {i} and used {b}. How many {i} are left?"
        elif k == "mul":
            a, b = rng.randint(13, 39), rng.randint(12, 29); ans = a * b
            q = f"There are {a} rows with {b} {i} each. How many {i} total?"
        else:
            b = rng.randint(3, 19); ans = rng.randint(12, 44); a = b * ans
            q = f"{w} splits {a} {i} into {b} equal groups. How many {i} per group?"
        probs.append((q, str(ans)))
    return probs


def parse_tool_call(text):
    """Return (name, expr_or_code) from the first well-formed <tool_call> in text, else None."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        try:
            obj = ast.literal_eval(m.group(1))
        except Exception:
            return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    payload = args.get("expression") or args.get("code") or ""
    if name == "python" or "print(" in payload:
        pm = re.search(r"print\((.*)\)", payload, re.S)
        payload = pm.group(1) if pm else payload
    return name, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--show", type=int, default=0, help="print N raw generations for inspection")
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

    probs = gen_problems(args.n_problems, args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[im_end])

    def sys_user_ids(system, user):
        enc = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return [int(x) for x in enc]

    prompts = [TokensPrompt(prompt_token_ids=sys_user_ids(SYS, q)) for q, _ in probs]
    outs = llm.generate(prompts, sp)

    n = len(probs)
    n_call = n_expr_ok = n_boxed_ok = 0
    for idx, ((q, gold), o) in enumerate(zip(probs, outs)):
        t = o.outputs[0].text
        if idx < args.show:
            out(f"\n--- sample {idx} (gold={gold}) ---\n{t[:500]}\n---")
        pc = parse_tool_call(t)
        if pc is not None:
            n_call += 1
            val = safe_eval(pc[1])
            if val is not None and abs(val - float(gold)) < 1e-6:
                n_expr_ok += 1
        if extract_boxed(t) == gold:
            n_boxed_ok += 1
    out("=" * 60)
    out(f"TOOL-CALL eval  model={Path(args.model).name}  n={n}")
    out(f"  emitted valid <tool_call>     : {100*n_call/n:.1f}%  ({n_call}/{n})")
    out(f"  tool expression == gold       : {100*n_expr_ok/n:.1f}%  ({n_expr_ok}/{n})")
    out(f"  final \\boxed answer == gold   : {100*n_boxed_ok/n:.1f}%  ({n_boxed_ok}/{n})")
    out("=" * 60)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
