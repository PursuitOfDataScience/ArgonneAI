---
license: apache-2.0
language:
- en
library_name: transformers
base_model: PursuitOfDataScience/argonne-4.5-base-ctx13568
tags:
- text-generation
- causal-lm
- transformer
- argonne
- instruct
- chat
- sft
- dpo
datasets:
- HuggingFaceH4/ultrachat_200k
- argilla/dpo-mix-7k
pipeline_tag: text-generation
---

# Argonne 4.5-instruct

**A 2.06B-parameter chat model trained from scratch, with a 13,568-token context.** It is
[argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568)
after supervised fine-tuning on [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) and DPO on [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k). Its sibling
[Argonne-4.5-think](https://huggingface.co/PursuitOfDataScience/Argonne-4.5-think) continues from
the same run into a reasoning model; use that one for math word problems.

Measured plainly: it keeps most of the base's knowledge (8-task lm-eval mean 53.28 against the
base's 54.67), keeps using its long context, and follows formatting instructions poorly. On IFEval it
scores 11.83, against 26.25 for Qwen2.5-0.5B-Instruct and 66.36 for Llama-3.2-3B-Instruct on the same
harness.

## Evaluation

### Instruction following (IFEval)

541 prompts with verifiable constraints ("no commas", "at least 3 bullet points"), rendered with each
model's own chat template, greedy decoding, up to 1,280 new tokens, a leading think block stripped
before scoring. Same harness and software for every row.

| model | prompt strict | instruction strict | prompt loose | instruction loose |
|---|---:|---:|---:|---:|
| **argonne-4.5-instruct** | **11.83** | **20.62** | **13.12** | **22.42** |
| the same run before DPO (SFT only, not released) | 9.24 | 18.82 | 12.01 | 21.94 |
| [Argonne-4.5-think](https://huggingface.co/PursuitOfDataScience/Argonne-4.5-think) | 11.83 | 20.62 | 14.42 | 23.26 |
| [Argonne-4.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-4.0-think) | 14.60 | 24.22 | 16.27 | 26.14 |
| Qwen2.5-0.5B-Instruct | 26.25 | 35.97 | 28.47 | 38.37 |
| Llama-3.2-3B-Instruct | 66.36 | 75.66 | 70.79 | 79.38 |

DPO adds +2.59 on prompt-level strict over the SFT-only model (paired McNemar p 0.054). Neither
UltraChat nor the DPO pairs target verifiable format constraints, and it shows: misses include commas
where commas were banned and lists that repeat instead of stopping.

### General capability (lm-eval)

The base card's tasks, few-shot counts and metric rule (`acc_norm` for multiple choice, `acc` for
winogrande and mmlu), scored with no chat template so the columns compare.

| task | 4.5-base-ctx13568 | **4.5-instruct** |
|---|---:|---:|
| arc_challenge | 41.72 | 40.10 |
| arc_easy | 62.04 | 58.29 |
| hellaswag | 55.83 | 55.85 |
| piqa | 70.51 | 70.46 |
| sciq | 85.40 | 81.60 |
| openbookqa | 38.00 | 37.40 |
| winogrande *(acc)* | 57.85 | 56.59 |
| mmlu *(acc)* | 26.03 | 25.99 |
| **8-task mean** | **54.67** | **53.28** |
| truthfulqa_mc2 | 40.93 | 45.74 |
| boolq | 67.61 | 57.49 |
| gsm8k strict-match | 4.78 | 5.53 |

Chat tuning costs 1.39 on the 8-task mean, mostly sciq (−3.80) and arc_easy (−3.75), and adds 4.81 on
truthfulqa_mc2. boolq is the largest single loss (−10.12), and most of it came with DPO (62.05 after
SFT alone). gsm8k shows it is not a math model; use Argonne-4.5-think for that.

### Long context

Position-bucketed loss on held-out arXiv (40 windows of 24,576 tokens from proof-pile-2's **test**
split), nats per token, lower is better.

| token position | 4.5-base-ctx13568 | **4.5-instruct** |
|---|---:|---:|
| 0 to 1,024 | 1.642 | 1.834 |
| 1,024 to 2,048 | 1.354 | 1.519 |
| 2,048 to 4,096 | 1.210 | 1.358 |
| 4,096 to 8,192 | 0.979 | 1.113 |
| 8,192 to 13,568 | 0.854 | 0.977 |
| 13,568 to 20,480 *(past the window)* | 0.832 | 0.955 |

Loss still falls with position all the way to 13,568. The curve sits 0.12 to 0.19 nats above the
base's and the gap is largest in the first bucket, so this is chat tuning moving the model away from
arXiv prose rather than a loss of context.

## Training

| stage | data | detail |
|---|---|---|
| base | [argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568) | 145.21B tokens, context 13,568 |
| 1: SFT | [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 207,865 conversations, 1 epoch (12,991 steps, 247M tokens), length 4,096, effective batch 16, LR 2e-5, loss on the final assistant turn only |
| 2: DPO | [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) | 6,750 pairs, 1 epoch, length 4,096, effective batch 8, LR 1e-6, β 0.03 |

Stage 1 ran on 4 to 8 NVIDIA A100 40GB GPUs with a sharded optimizer, stage 2 on one. These are the
first two stages of Argonne-4.5-think as well: that model continues from this checkpoint.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PursuitOfDataScience/argonne-4.5-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
).cuda()

messages = [{"role": "user", "content": "Give me three tips for writing a clear email to a busy colleague."}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()

out = model.generate(ids, max_length=ids.shape[1] + 400, do_sample=False)
print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

For throughput, serve it with vLLM or SGLang instead of `.generate()`.

## Usage notes

- Load with `trust_remote_code=True`; `config.json` carries an `auto_map`, so no manual setup.
- **Render prompts with `enable_thinking=False`**, as in the snippet. The chat template writes every
  assistant turn with an empty `<think></think>` block in front, and stage 1 trained on that block as
  part of the reply, so the model learned to emit one;
  with `enable_thinking=False` the template puts that block in the prompt and the reply starts
  directly. Without it, replies usually open with `<think>\n\n</think>`, which `skip_special_tokens`
  does not remove.
- The custom `generate` takes `max_length` (total length), not `max_new_tokens`, and also accepts
  `repetition_penalty` and `no_repeat_ngram_size`. Greedy decoding can repeat a phrase.
- `eos_token_id` is **151645** (`<|im_end|>`). Checked on the staged release files: four prompts
  with **no** `eos_token_id` argument all stopped on their own, in 32 to 188 tokens.
- `lm_head.weight` is reported missing on load. Expected: the embeddings are tied.

## Limitations

- **Weak instruction following** (IFEval above): do not rely on it to honour format constraints.
- **Factual errors are common.** In the smoke test it explained Rayleigh scattering as light
  "reflecting off particles" and said a tuple's order "is not specified".
- Not a reasoning model (gsm8k 5.53); use Argonne-4.5-think for arithmetic word problems.
- Fine-tuned at 4,096 tokens. The long-context probe above shows it still uses its context, but
  long-input tasks were not evaluated.
- English only, from UltraChat. No safety alignment beyond what UltraChat and the preference data
  provide.

## Source code

Everything is on the GitHub `main` branch:
[PursuitOfDataScience/ArgonneAI](https://github.com/PursuitOfDataScience/ArgonneAI/tree/main).

| file | role |
|---|---|
| [`model.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py) | the `argonne2` architecture, identical to the copy in this repo |
| [`sft.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/sft.py) | stage 1 |
| [`dpo.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/dpo.py) | stage 2 |
| [`reasoning/run_ifeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_ifeval_vllm.py) | IFEval with the chat template, through vLLM |
| [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py) | lm-eval through vLLM |
| [`reasoning/exp_longctx_learning.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/exp_longctx_learning.py) | the long-context probe |
| [`reasoning/release_smoke.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/release_smoke.py) | the termination check run on these release files |

## Citation

```bibtex
@misc{argonne45instruct,
  title  = {Argonne 4.5-instruct},
  author = {Youzhi Yu},
  year   = {2026},
  url    = {https://huggingface.co/PursuitOfDataScience/argonne-4.5-instruct}
}
```
