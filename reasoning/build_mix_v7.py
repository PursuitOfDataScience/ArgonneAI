#!/usr/bin/env python3
"""Build cot_sft_mix_v7 = the v6 termination-safe backbone + THREE additive capability tiers, to
improve v3's reasoning while adding tool-calling and coding (2026-07-12, §25 go-forward).

Rationale (honest, evidence-grounded):
- §25 localized the wall as v3's own generation + self-verification (base capability): self-STaR
  saturates, but distilling a STRONGER teacher's correct traces is the higher-EV single-card lever.
  -> `teacher_gsm`: Qwen3-4B's correct short solutions on gsm8k-TRAIN (contamination-safe), wrapped
     as single-turn <think> student traces.
- User directive: add good tool-calling + coding data to boost reasoning effort.
  -> `code_magicoder`: Magicoder-OSS-Instruct (decontaminated) problem->solution, NO-think (direct
     coding capability; keeps the no-think channel healthy, termination-safe).
  -> `tool_calc`: SYNTHESIZED calculator/python tool-use reasoning traces, correct-by-construction
     (every number computed in Python). Tool interaction lives INSIDE the <think> block as scratchpad
     (single assistant turn) so it teaches the <tool_call>/<tool_response> FORMAT without relying on
     cot-sft's dropped `role:tool` turns, and without teaching the model to hallucinate tool outputs
     at the top level. (Tool-use for arithmetic-offload did NOT move SVAMP/ASDiv historically, §PoT-
     refuted; this tier is a capability add kept MODEST so it can't crowd out math/general/termination.)

The v6 backbone stays DOMINANT (~70%) to protect the §23 native-termination win + the 4-quadrant
no-think axis. Eval gate stays clean SVAMP/ASDiv (disjoint from every tier here).

Output: /project/rcc/youzhi/data/cot_sft_mix_v7 (schema: messages, tier, num_tokens).
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_generate import extract_boxed  # noqa: E402

V6 = "/project/rcc/youzhi/data/cot_sft_mix_v6"
TEACHER = "/project/rcc/youzhi/data/teacher_qwen_gsm"
MAGICODER = "/project/rcc/youzhi/data/ise-uiuc_Magicoder-OSS-Instruct-75K/data-oss_instruct-decontaminated.jsonl"
OUT = "/project/rcc/youzhi/data/cot_sft_mix_v7"
TOK_SRC = "/project/rcc/youzhi/models/instruct/x_v6v2_040"

SEED = 20260712
TEACHER_CAP = 6000            # strong external-teacher math signal (~15%)
TEACHER_MAX_TOK = 512
CODE_CAP = 3000              # coding capability (~8%)
CODE_MAX_TOK = 1024
CODE_LANGS = {"python", "cpp", "java", "javascript", "typescript", "go", "rust", "c", "csharp", "ruby"}
TOOL_CAP = 2500             # tool-calling format (~7%)


def make_ntok(tok):
    def ntok(messages):
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                      enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return len(enc)
    return ntok


# --------------------------- teacher tier ---------------------------
def build_teacher(rng, ntok):
    if not Path(TEACHER).exists():
        print(f"[teacher] {TEACHER} MISSING -> skipping teacher tier (rerun after gen_teacher)")
        return []
    ds = load_from_disk(TEACHER)
    rows = []
    for r in ds:
        gold = r["gold"]
        # strip a trailing "The answer is \boxed{}" from the teacher solution -> put steps in <think>
        sol = r["solution"].strip()
        i = sol.rfind("The answer is")
        steps = sol[:i].strip() if i > 0 else sol
        if not steps:
            steps = sol
        content = f"<think>\n{steps}\n</think>\n\nThe answer is $\\boxed{{{gold}}}$."
        if extract_boxed(content) != gold:
            continue
        msgs = [{"role": "user", "content": r["question"]},
                {"role": "assistant", "content": content}]
        nt = ntok(msgs)
        if nt > TEACHER_MAX_TOK:
            continue
        rows.append({"messages": msgs, "tier": "teacher_gsm", "num_tokens": nt})
    rng.shuffle(rows)
    rows = rows[:TEACHER_CAP]
    print(f"[teacher] teacher_gsm kept {len(rows)} (<= {TEACHER_MAX_TOK} tok)")
    return rows


# --------------------------- code tier ---------------------------
def build_code(rng, ntok):
    if not Path(MAGICODER).exists():
        print(f"[code] {MAGICODER} MISSING -> skipping code tier")
        return []
    pool = []
    with open(MAGICODER) as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            lang = str(o.get("lang", "")).lower()
            prob, sol = o.get("problem", ""), o.get("solution", "")
            if lang not in CODE_LANGS or not prob or not sol:
                continue
            pool.append((prob.strip(), sol.strip()))
    rng.shuffle(pool)
    rows = []
    for prob, sol in pool:
        msgs = [{"role": "user", "content": prob},
                {"role": "assistant", "content": sol}]   # direct coding, no <think> (termination-safe)
        nt = ntok(msgs)
        if nt > CODE_MAX_TOK:
            continue
        rows.append({"messages": msgs, "tier": "code_magicoder", "num_tokens": nt})
        if len(rows) >= CODE_CAP:
            break
    print(f"[code] code_magicoder kept {len(rows)} (<= {CODE_MAX_TOK} tok) from {len(pool)} candidates")
    return rows


# --------------------------- tool tier (synthesized) ---------------------------
TOOLS_SPEC = (
    '[{"name": "calculator", "description": "Evaluate an arithmetic expression", '
    '"parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, '
    '"required": ["expression"]}}, '
    '{"name": "python", "description": "Run a short Python snippet and return stdout", '
    '"parameters": {"type": "object", "properties": {"code": {"type": "string"}}, '
    '"required": ["code"]}}]'
)


def _fmt(n):
    return str(int(n)) if float(n) == int(n) else str(round(float(n), 4))


def _call(tool, expr, result):
    if tool == "calculator":
        args = f'{{"expression": "{expr}"}}'
    else:
        args = f'{{"code": "print({expr})"}}'
    return (f'<tool_call>\n{{"name": "{tool}", "arguments": {args}}}\n</tool_call>\n'
            f'<tool_response>\n{_fmt(result)}\n</tool_response>')


def build_tool(rng, ntok):
    names_a = ["Maria", "Sam", "the store", "the school", "Alex", "the farmer", "Priya", "the team"]
    items = ["apples", "books", "coins", "boxes", "tickets", "marbles", "cookies", "stamps", "cards"]
    rows = []
    kinds = ["sub", "add", "mul", "div", "two_step", "pct"]
    attempts = 0
    while len(rows) < TOOL_CAP and attempts < TOOL_CAP * 4:
        attempts += 1
        k = rng.choice(kinds)
        who, it = rng.choice(names_a), rng.choice(items)
        tool = rng.choice(["calculator", "python"])
        if k == "add":
            a, b = rng.randint(10, 400), rng.randint(5, 300)
            ans = a + b
            q = f"{who} had {a} {it} and then got {b} more. How many {it} now?"
            expr = f"{a} + {b}"
            think = f"Add {a} and {b}. I'll use the {tool} tool.\n{_call(tool, expr, ans)}\nSo the total is {ans}."
        elif k == "sub":
            a = rng.randint(30, 500); b = rng.randint(5, a - 1)
            ans = a - b
            q = f"{who} had {a} {it} and used {b}. How many {it} are left?"
            expr = f"{a} - {b}"
            think = f"Subtract {b} from {a}. I'll use the {tool} tool.\n{_call(tool, expr, ans)}\nSo {ans} remain."
        elif k == "mul":
            a, b = rng.randint(3, 40), rng.randint(2, 25)
            ans = a * b
            q = f"There are {a} groups with {b} {it} each. How many {it} in total?"
            expr = f"{a} * {b}"
            think = f"Multiply {a} by {b}. I'll use the {tool} tool.\n{_call(tool, expr, ans)}\nSo there are {ans}."
        elif k == "div":
            b = rng.randint(2, 20); ans = rng.randint(2, 40); a = b * ans
            q = f"{who} splits {a} {it} equally into {b} groups. How many {it} per group?"
            expr = f"{a} // {b}"
            think = f"Divide {a} by {b}. I'll use the {tool} tool.\n{_call(tool, expr, ans)}\nSo {ans} per group."
        elif k == "pct":
            p = rng.choice([10, 15, 20, 25, 40, 50, 75]); base = rng.choice([20, 40, 60, 80, 120, 200])
            ans = p * base / 100
            q = f"What is {p}% of {base} {it}?"
            expr = f"{p} * {base} / 100"
            think = f"Compute {p}% of {base}. I'll use the {tool} tool.\n{_call(tool, expr, ans)}\nSo it is {_fmt(ans)}."
        else:  # two_step: a + b*c
            a = rng.randint(10, 100); b = rng.randint(2, 12); c = rng.randint(2, 15)
            step1 = b * c; ans = a + step1
            q = (f"{who} started with {a} {it}, then bought {b} packs with {c} {it} each. "
                 f"How many {it} in total?")
            think = (f"First find the packs total, then add. I'll use the {tool} tool.\n"
                     f"{_call(tool, f'{b} * {c}', step1)}\n"
                     f"{_call(tool, f'{a} + {step1}', ans)}\nSo the total is {ans}.")
        content = f"<think>\n{think}\n</think>\n\nThe answer is $\\boxed{{{_fmt(ans)}}}$."
        if extract_boxed(content) != extract_boxed(f"\\boxed{{{_fmt(ans)}}}"):
            continue
        msgs = [
            {"role": "system", "content":
             "You may call tools. Function signatures are inside <tools></tools>:\n<tools>\n"
             + TOOLS_SPEC + "\n</tools>\nCall a tool with a JSON object inside "
             "<tool_call></tool_call> and read its <tool_response>."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": content}]
        rows.append({"messages": msgs, "tier": "tool_calc", "num_tokens": ntok(msgs)})
    print(f"[tool] tool_calc synthesized {len(rows)} (attempts {attempts})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build code+tool only (teacher may be absent)")
    args = ap.parse_args()
    rng = random.Random(SEED)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_SRC, trust_remote_code=True)
    ntok = make_ntok(tok)

    print(f"Loading v6 backbone <- {V6}")
    v6 = load_from_disk(V6)
    backbone = [{"messages": r["messages"], "tier": r["tier"], "num_tokens": r["num_tokens"]} for r in v6]
    print(f"  v6 backbone: {len(backbone)} rows")

    teacher = build_teacher(rng, ntok)
    code = build_code(rng, ntok)
    tool = build_tool(rng, ntok)

    rows = backbone + teacher + code + tool
    rng.shuffle(rows)
    comp = Counter(r["tier"] for r in rows)
    print(f"\nTOTAL v7: {len(rows)}")
    for t in sorted(comp):
        print(f"  {t:<20}{comp[t]:>7}  ({100*comp[t]/len(rows):.1f}%)")
    print(f"  max num_tokens: {max(r['num_tokens'] for r in rows)}")

    if args.dry_run:
        print("\n[dry-run] NOT saving. Sample tool_calc + code rows:")
        for r in rows:
            if r["tier"] == "tool_calc":
                print("\n--- tool_calc ---\n", r["messages"][-1]["content"][:500]); break
        return
    Dataset.from_list(rows).save_to_disk(OUT)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
