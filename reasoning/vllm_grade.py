"""Fast vLLM greedy GSM8K grader for checkpoint selection (§22). Loads one model, greedy-decodes
N problems (with-think), auto-grades the \\boxed{} answer vs gold, prints pass@1 + failure modes.
Uses the validated vLLM port (fast + fills HBM). Run one model per invocation."""
import argparse
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from star_generate import extract_boxed, norm, load_problems  # noqa: E402


def build_ids(tok, q, think=True):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=think)
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    if len(enc) > 0 and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(x) for x in enc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-think", action="store_true", help="grade the no-think direct channel")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    probs = load_problems(args.source, args.n_problems, seed=args.seed)
    think = not args.no_think
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=2048, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)  # greedy
    prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, q, think)) for q, _, _ in probs]
    outs = llm.generate(prompts, sp)
    fm = {"correct": 0, "wrong": 0, "unclosed": 0, "no_answer": 0}
    for (q, gold, tier), o in zip(probs, outs):
        text = o.outputs[0].text
        pred = extract_boxed(text)
        if think and "</think>" not in text:
            fm["unclosed"] += 1
        elif pred is None:
            fm["no_answer"] += 1
        elif pred == gold:
            fm["correct"] += 1
        else:
            fm["wrong"] += 1
    n = len(probs)
    print(f"MODEL {args.model}")
    print(f"  GSM8K greedy pass@1 ({'think' if think else 'no-think'}) "
          f": {100*fm['correct']/n:.2f}%  ({fm['correct']}/{n})   fm={fm}")


if __name__ == "__main__":
    main()
