"""IFEval (instruction following) through lm-eval's vLLM backend, WITH the model's chat template.

run_lmeval_vllm.py scores the likelihood suite with no chat template, which is right for comparing a
chat model against its base but says nothing about following instructions. IFEval is a generation
task whose 541 prompts carry verifiable constraints ("answer in all lowercase", "use at least 3
bullet points"), so the prompt must be rendered the way a user would send it: apply_chat_template.

A leading `<think>...</think>` block is stripped before scoring (lm-eval's think_end_token), so a
think model is judged on its answer and a chat model's empty think block does not trip a
"start with ..." check. A trace that never closes is scored whole, and fails.

  python reasoning/run_ifeval_vllm.py --model-path <HF dir> --out ifeval_<name>.json
Needs the google/IFEval dataset and nltk's punkt_tab (set NLTK_DATA when offline).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

METRICS = ["prompt_level_strict_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "inst_level_loose_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-gen-toks", type=int, default=1280, help="lm-eval's IFEval default")
    ap.add_argument("--think-end-token", default="</think>", help="'' scores the raw output")
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--samples-out", default="", help="also write every prompt, response and verdict here")
    args = ap.parse_args()

    import vllm_argonne
    vllm_argonne.register()  # register argonne2 with vLLM + AutoConfig + tokenizer shim
    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    lm = VLLM(pretrained=args.model_path, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              max_gen_toks=args.max_gen_toks, think_end_token=args.think_end_token or None)
    r = lm_eval.simple_evaluate(model=lm, tasks=["ifeval"], num_fewshot=0, apply_chat_template=True,
                                limit=args.limit, bootstrap_iters=0, log_samples=bool(args.samples_out))
    res = r["results"]["ifeval"]
    out = {"ifeval": res, "model_path": args.model_path, "max_gen_toks": args.max_gen_toks,
           "think_end_token": args.think_end_token, "chat_template": True}
    json.dump(out, open(args.out, "w"), indent=2)
    if args.samples_out:
        json.dump([{"prompt": s["doc"]["prompt"], "response": s["filtered_resps"][0],
                    "strict": s["prompt_level_strict_acc"], "loose": s["prompt_level_loose_acc"]}
                   for s in r["samples"]["ifeval"]], open(args.samples_out, "w"), indent=1)
    print("ifeval " + "  ".join(f"{m} {100 * res[m + ',none']:.2f}" for m in METRICS), flush=True)


if __name__ == "__main__":
    main()
