#!/usr/bin/env python3
"""Open LLM Leaderboard v1 (+GSM8K) for Argonne-3.0-think via lm-eval-harness.

Evaluates the local `soup_blend_a085` checkpoint (bf16 — byte-identical to the
published `PursuitOfDataScience/Argonne-3.0-think`). Notes on this custom model:

- The checkpoint config has NO `auto_map`, so `trust_remote_code` alone can't find
  the class. We `import model` (repo root) first, which self-registers the `argonne2`
  arch with AutoModel — then `from_pretrained` resolves it.
- The model has NO padding / attention-mask support, so we default to `batch_size=1`
  (right-padding is causal-safe in theory, but bs=1 removes all doubt; use --bs to test).
- GSM8K is generative → greedy (`do_sample=False`). Anti-repeat decoding CORRUPTS this
  model's arithmetic (thinking_training.md §18f), so no repetition penalty / no-repeat-ngram.

Each task runs in its own `simple_evaluate` so a failure (e.g. a generation quirk on
GSM8K) doesn't lose the multiple-choice results. Results are written incrementally.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "reasoning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import model as _model  # noqa: F401  registers the `argonne2` architecture

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


class ArgonneHFLM(HFLM):
    """HFLM whose generation path fits ArgonneModel's custom `generate`.

    lm-eval's default `_model_generate` passes `stopping_criteria`, `pad_token_id`,
    `use_cache`, etc. — but `ArgonneModel.generate` accepts only
    (input_ids, max_length, temperature, top_k, top_p, do_sample,
    repetition_penalty, no_repeat_ngram_size). We call it with just the supported
    args (GREEDY, no repetition penalty — §18f: anti-repeat decoding corrupts this
    model's arithmetic), and let lm-eval trim at the task's stop strings and extract
    the answer afterward. The model's own KV cache makes this fast.
    """

    def _model_generate(self, context, max_length, stop, **generation_kwargs):  # noqa: ARG002
        with torch.autocast(device_type=self.device.type,
                            dtype=self.mixed_precision_dtype,
                            enabled=self.mixed_precision_dtype is not None):
            return self.model.generate(input_ids=context, max_length=max_length,
                                       do_sample=False)

# Open LLM Leaderboard v1 canonical tasks + few-shot counts, + GSM8K,
# plus an EXPANDED set of standard (fast, loglikelihood) benchmarks. Pass
# --tasks to run a subset; unknown/failed tasks are isolated (see main()).
SUITE = [
    # --- Open LLM Leaderboard v1 + GSM8K ---
    ("arc_challenge", 25),
    ("hellaswag", 10),
    ("mmlu", 5),
    ("truthfulqa_mc2", 0),
    ("winogrande", 5),
    ("gsm8k", 5),
    # --- Expanded set (all 0-shot, loglikelihood multiple-choice = fast) ---
    ("arc_easy", 0),
    ("piqa", 0),
    ("openbookqa", 0),
    ("commonsense_qa", 0),
    ("sciq", 0),
    ("boolq", 0),
    ("mathqa", 0),          # multiple-choice math (complements generative GSM8K)
    ("logiqa", 0),
    ("lambada_openai", 0),  # language modeling (last-word accuracy)
    ("gpqa_main_n_shot", 0),  # graduate-level; expected ~random at this scale
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",
                    default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of task names to run (default: full suite)")
    ap.add_argument("--limit", type=float, default=None,
                    help="cap examples/task (smoke test)")
    ap.add_argument("--bs", default="1", help="batch size (1 = no padding; 'auto' or int)")
    ap.add_argument("--apply-chat-template", action="store_true",
                    help="wrap prompts in the chat template (chat-model-faithful, less v1-comparable)")
    ap.add_argument("--fewshot", type=int, default=None,
                    help="override the few-shot count for ALL requested tasks (e.g. 0 for chat/think-mode)")
    ap.add_argument("--out", default=str(ROOT / "report" / "lmeval_results.json"))
    args = ap.parse_args()

    bs = int(args.bs) if args.bs.isdigit() else args.bs
    print(f"loading {args.model_path} (bf16, bs={bs}, chat_template={args.apply_chat_template})",
          flush=True)
    lm = ArgonneHFLM(pretrained=args.model_path, dtype="bfloat16",
                     trust_remote_code=True, batch_size=bs)
    # enable the KV cache for generative tasks even if the config shipped use_cache=False
    try:
        lm._model.config.use_cache = True
    except Exception:
        pass

    suite = [(t, s) for (t, s) in SUITE if (not args.tasks or t in args.tasks)]
    out = {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for task, shots in suite:
        if args.fewshot is not None:
            shots = args.fewshot
        print(f"\n===== {task}  ({shots}-shot) =====", flush=True)
        try:
            r = simple_evaluate(
                model=lm, tasks=[task], num_fewshot=shots,
                limit=args.limit,
                apply_chat_template=args.apply_chat_template,
                fewshot_as_multiturn=args.apply_chat_template,
                gen_kwargs="do_sample=False",
                bootstrap_iters=0,
            )
            out[task] = r["results"].get(task, r["results"])
            print(task, json.dumps(out[task], default=str)[:800], flush=True)
        except Exception as e:
            traceback.print_exc()
            out[task] = {"ERROR": str(e)}
        json.dump(out, open(args.out, "w"), indent=2, default=str)
    print("\nDONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
