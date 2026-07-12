#!/usr/bin/env python3
"""Tool-EXECUTION decode loop — makes v7's learned tool-calling actually useful (§26 → §27).

§26 finding: think_v7 emits 100% valid <tool_call> with 100% correct expressions, but because the
tool_calc tier baked the <tool_response> in-trace (single-turn), the model HALLUCINATES the response
at inference (e.g. 540+389 -> writes "939", actual 929) -> the arithmetic offload is illusory (55%
final acc). This is the serving-system fix the evidence points to (parallel to §25's external reranker):
a real agentic loop that STOPS at </tool_call>, EXECUTES the tool, injects the REAL <tool_response>, and
resumes -> so the model reasons on CORRECT tool outputs instead of its own hallucinated ones.

Compares, on held-out arithmetic word problems, the shipped-style greedy trace (model hallucinates its
own tool response) vs the tool-EXECUTED loop (real calculator/python results injected). If real execution
lifts final-answer accuracy, tool-calling is a genuine capability — delivered via a serving loop, not weights.

Batched, continuous rounds: each round generates all open sequences to `</tool_call>` (a regular-text
stop string) or natural end; sequences that called a tool get the REAL result appended; done sequences
(closed </think>+answer, or max turns) are set aside. vLLM + validated port (register() for the shim).
"""
import argparse
import re
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed  # noqa: E402
from tool_eval import SYS, gen_problems, parse_tool_call, safe_eval  # noqa: E402


def execute_tool(name, payload):
    """Run the (regex-gated arithmetic-only) tool call; return the string result or 'ERROR'."""
    v = safe_eval(payload)
    if v is None:
        return "ERROR: could not evaluate"
    return str(int(v)) if v == int(v) else str(round(v, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/project/rcc/youzhi/models/instruct/think_v7")
    ap.add_argument("--n-problems", type=int, default=150)
    ap.add_argument("--max-turns", type=int, default=4)      # max tool calls per problem
    ap.add_argument("--seg-tokens", type=int, default=256)   # tokens per generation segment
    ap.add_argument("--tail-tokens", type=int, default=200)  # final answer segment
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--show", type=int, default=2)
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

    def ids_for(q):
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return [int(x) for x in enc]

    # -------- (A) baseline: single greedy pass (model hallucinates its own tool response) --------
    base_ids = [ids_for(q) for q, _ in probs]
    spb = SamplingParams(n=1, temperature=0.0, max_tokens=args.seg_tokens * args.max_turns + args.tail_tokens,
                         stop_token_ids=[im_end])
    bouts = llm.generate([TokensPrompt(prompt_token_ids=p) for p in base_ids], spb)
    base_texts = [o.outputs[0].text for o in bouts]

    # -------- (B) tool-EXECUTED loop --------
    prompts = [list(p) for p in base_ids]         # accumulated token ids per problem
    texts = ["" for _ in probs]                   # accumulated generated text (for grading/display)
    done = [False] * len(probs)
    n_exec = [0] * len(probs)
    sp_seg = SamplingParams(n=1, temperature=0.0, max_tokens=args.seg_tokens,
                            stop=["</tool_call>"], include_stop_str_in_output=True,
                            stop_token_ids=[im_end])
    for turn in range(args.max_turns + 1):
        idx = [i for i in range(len(probs)) if not done[i]]
        if not idx:
            break
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in idx], sp_seg)
        for i, o in zip(idx, outs):
            seg = o.outputs[0].text
            seg_ids = list(o.outputs[0].token_ids)
            texts[i] += seg
            prompts[i] = prompts[i] + seg_ids
            if seg.rstrip().endswith("</tool_call>") and turn < args.max_turns:
                pc = parse_tool_call(seg)
                result = execute_tool(*pc) if pc else "ERROR: unparseable tool call"
                inject = f"\n<tool_response>\n{result}\n</tool_response>\n"
                texts[i] += inject
                prompts[i] = prompts[i] + tok.encode(inject, add_special_tokens=False)
                n_exec[i] += 1
            else:
                done[i] = True
    # final answer tail for any sequence still open (closed </think> but no boxed yet)
    open_i = [i for i in range(len(probs)) if extract_boxed(texts[i]) is None]
    if open_i:
        sp_tail = SamplingParams(n=1, temperature=0.0, max_tokens=args.tail_tokens, stop_token_ids=[im_end])
        touts = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in open_i], sp_tail)
        for i, o in zip(open_i, touts):
            texts[i] += o.outputs[0].text

    # -------- grade + report --------
    n = len(probs)
    base_ok = sum(1 for (q, g), t in zip(probs, base_texts) if extract_boxed(t) == g)
    tool_ok = sum(1 for (q, g), t in zip(probs, texts) if extract_boxed(t) == g)
    avg_exec = sum(n_exec) / n
    out("=" * 70)
    out(f"TOOL-EXECUTION decode  model={Path(args.model).name}  n={n}")
    out(f"  (A) greedy, self-hallucinated tool response : {100*base_ok/n:.1f}%  ({base_ok}/{n})")
    out(f"  (B) REAL tool-execution loop                : {100*tool_ok/n:.1f}%  ({tool_ok}/{n})")
    out(f"  lift from real execution                    : {100*(tool_ok-base_ok)/n:+.1f} pts")
    out(f"  avg tool calls executed / problem           : {avg_exec:.2f}")
    out("=" * 70)
    for i in range(min(args.show, n)):
        out(f"\n--- sample {i} (gold={probs[i][1]}) [execed {n_exec[i]}] ---\n{texts[i][:600]}\n---")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
