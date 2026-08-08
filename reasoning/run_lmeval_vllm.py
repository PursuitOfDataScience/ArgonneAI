"""lm-eval-harness via the VALIDATED vLLM backend (fast + 90% HBM) — the right way to benchmark
the custom argonne2 arch. Same task suite as run_lmeval.py (the published card), but uses
lm_eval's native VLLM model (continuous batching) instead of the bs=1 HF path (~10-50x faster).
Builds the vLLM engine ONCE and reuses it across tasks; isolates per-task failures."""
import argparse
import json
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# same suite as the published card
SUITE = [
    ("arc_challenge", 25), ("hellaswag", 10), ("mmlu", 5), ("truthfulqa_mc2", 0),
    ("winogrande", 5), ("gsm8k", 5),
    ("arc_easy", 0), ("piqa", 0), ("openbookqa", 0), ("commonsense_qa", 0),
    ("sciq", 0), ("boolq", 0), ("lambada_openai", 0), ("gpqa_main_n_shot", 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/project/rcc/youzhi/models/instruct/blend_star_a06")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default="report/lmeval_a06_vllm.json")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=float, default=None)
    args = ap.parse_args()

    import vllm_argonne
    vllm_argonne.register()  # register argonne2 with vLLM + AutoConfig + tokenizer shim
    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    lm = VLLM(pretrained=args.model_path, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len)

    suite = SUITE if not args.tasks else [(t, dict(SUITE).get(t, 0)) for t in args.tasks]
    results = {}
    for task, shots in suite:
        print(f"\n===== {task}  ({shots}-shot) [vLLM] =====", flush=True)
        try:
            r = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=shots,
                                        limit=args.limit, bootstrap_iters=0)
            res = r["results"].get(task, {})
            results[task] = res
            print(task, json.dumps(res), flush=True)
        except Exception as e:
            print(f"{task} FAILED: {type(e).__name__}: {e}", flush=True)
            results[task] = {"error": f"{type(e).__name__}: {e}"}
        json.dump(results, open(args.out, "w"), indent=2)  # incremental save
    print("\n===== ALL DONE =====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
