---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- text-generation
- causal-lm
- transformer
- argonne
- pretrained
- base-model
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/finemath
- nick007x/github-code-2025
- nvidia/Nemotron-Competitive-Programming-v1
- nvidia/OpenMathReasoning
- a-m-team/AM-DeepSeek-R1-Distilled-1.4M
- open-r1/Mixture-of-Thoughts
- PursuitOfDataScience/0.5M-thinking
- nvidia/Nemotron-SFT-Agentic-v2
pipeline_tag: text-generation
---

# Argonne 4.5-base

**A 2.06B-parameter base model trained from scratch on 128.07B tokens, with a 1,024-token
context.** It is the Argonne 4.5 base *before* long-context extension.

Most people want [**argonne-4.5-base-ctx13568**](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568)
instead. It is the same training run, continued for one more stage (17.15B tokens at a
13,568-token context). This checkpoint is published for two reasons: it is the exact starting
point of that last stage, so the effect of context extension can be measured against it, and it
suits short-context work where the extra stage is not wanted.

This is a **base model**: no instruction tuning, no alignment, no safety filtering. It continues
text; it does not follow instructions.

## Which checkpoint should I use?

| If you want | Use |
|---|---|
| Knowledge and commonsense tasks, inputs longer than 1,024 tokens | [argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568) |
| Math and step-by-step reasoning at short context, a start for math fine-tuning, or a before/after control for long-context research | **argonne-4.5-base** (this repo) |
| A smaller model, or a trained context longer than 13,568 tokens | [argonne-4.0-base](https://huggingface.co/PursuitOfDataScience/argonne-4.0-base) (1.04B, 65,536-token context) |

**The context stage is a trade, measured:** it raises seven of the eight multiple-choice tasks (+1.36 on the 8-task
mean; mmlu fell 28.88 to 26.03) and cuts gsm8k from 20.62 to **4.78** (−15.85). It trained on long arXiv plus FineWeb-Edu with no
math or reasoning replay, so the anneal's step-by-step math is what it gave up. For math at short context, this checkpoint is the better start.

Both 4.5 checkpoints come from one run:

```text
stage 1  pretrain             110.00B tokens   context 1,024
stage 2  reasoning anneal      18.07B tokens   context 1,024    ->  argonne-4.5-base  (this repo)
stage 3  context extension     17.15B tokens   context 13,568   ->  argonne-4.5-base-ctx13568
```

## What changed from Argonne 4.0-base

The block design is 4.0's. What changed is the size, the token budget, and one fixed defect.

| | argonne-4.0-base | Argonne 4.5 |
|---|---|---|
| **Parameters** | 1.04B | **2.06B** |
| **Shape** | 1,536 hidden × 32 layers, 6 query / 2 KV heads | 2,560 hidden × 24 layers, 10 query / 2 KV heads |
| **Pretraining** | 38.03B tokens (one pass over the edu/math/code bins) | **110.00B tokens** (2.89 passes over the same bins) |
| **Reasoning anneal** | 18.07B tokens, **no learning-rate cooldown** (a launcher defect) | the same 18.07B-token corpus, **with** a cooldown |
| **Context extension** | to 13,568, then to 65,536 | to 13,568 (in argonne-4.5-base-ctx13568 only) |

## Model architecture

| Component | Specification |
|---|---|
| **Parameters** | 2,063,639,552 (~2.06B), of which 388,272,640 (18.8%) are the tied embedding |
| **Layers** | 24 transformer blocks |
| **Hidden size** | 2,560 |
| **Attention** | grouped-query: 10 query heads share 2 key/value heads |
| **Head dimension** | 256 |
| **Feed-forward** | SwiGLU, 7,040 intermediate |
| **Attention pattern** | full causal attention on every layer |
| **Normalization** | RMSNorm (eps 1e-6), plus QK-norm, V-norm and sandwich norms |
| **Position encoding** | RoPE, θ = 1,000,000 |
| **Output logits** | soft-capped at 15.0 (`15 · tanh(x / 15)`) |
| **Context length** | **1,024 tokens**, the only length this checkpoint was trained at |
| **Vocabulary** | 151,669 (Qwen3 tokenizer) |
| **Tied embeddings** | yes: the output head reuses the input embedding |
| **KV cache** | 48 KiB per token in bf16, so 48 MiB for a full 1,024-token window |

**Config switches that are off.** `config.json` carries switches from the 4.5 research phase,
all disabled in this model: `attn_pattern: null` (so `sliding_window_size: 2048` is never used),
`interleaved_local_attention: false`, `nope_global: false`, `attn_gate: false`,
`doc_mask: false`, `mtp_module_layers: 0`, `mtp_loss_weight: 0.0`, `z_loss_weight: 0.0`. Every
layer runs full causal attention with RoPE.

## Training

Two stages, both causal language modeling.

| | Stage 1: pretrain | Stage 2: reasoning anneal |
|---|---|---|
| **Script** | `pretrain.py` | `continue_pretrain.py` |
| **Optimizer steps** | 203,451 | 33,415 (global steps 203,452 to 236,866) |
| **Tokens** | 110.00B | 18.07B |
| **Cumulative tokens** | 110.00B | **128.07B** |
| **Sequence length** | 1,024 | 1,024 |
| **Batch** | 540,672 tokens/step (528 sequences) | 540,672 tokens/step (528 sequences) |
| **Peak learning rate** | 6.0e-4 | 2.0e-4 |
| **Final learning rate** | 6.0e-5 | 2.0e-5 |
| **Warmup** | 8,000 steps | none |
| **Schedule** | constant, then linear decay over the last 15% (30,517 steps) | constant, then linear decay over the last 15% (5,012 steps) |
| **Optimizer state** | new | reset at the start of the stage |

Shared by both stages:

| Item | Value |
|---|---|
| **Optimizer** | AdamW (β₁ 0.9, β₂ 0.95, weight decay 0.1), fp32 master weights |
| **Gradient clipping** | 0.4 |
| **Precision** | bf16 autocast. FP8 matmuls (torchao, tensorwise, including the output head) for the first ~55.8k pretraining steps (~30.1B tokens); plain bf16 after that (the A100s have no FP8 support, and it stayed off for the one later H100 stretch) |
| **Vocabulary padding** | 151,669 → 151,680 rows during training (FP8 alignment), trimmed on export |
| **Memory** | AdamW state sharded across GPUs on the 40 GB cards; gradient checkpointing |
| **Published weights** | bfloat16 safetensors, about 4.1 GB |

### Hardware

| Stage | GPUs |
|---|---|
| Pretrain, steps 0 to ~55.8k | 3× NVIDIA H100 (80 GB and 94 GB variants), FP8 on |
| Pretrain, remaining ~147.7k steps | NVIDIA A100 40GB, 4 to 24 GPUs per job, plus one stretch of ~5.6k steps on 4× NVIDIA H100 80GB; FP8 off |
| Reasoning anneal | NVIDIA A100 40GB, 4 or 8 GPUs |

The batch stayed at exactly 540,672 tokens on every GPU count, so moving between GPU types did not
change the recipe.

### Known training quirks

- **A sliding window slipped into the first ~950 anneal steps.** For global steps 203,452 to
  204,397 (about 2.8% of stage 2, ~0.51B tokens), a configuration bug in `continue_pretrain.py`
  gave the 12 odd-numbered layers a 256-token sliding window instead of full attention. It was
  caught from the startup log and fixed, and training continued from the step-204,397 checkpoint
  with full attention (the affected steps were kept, not redone). All of pretraining and the
  other 97% of the anneal used full causal attention, which is what the config describes.
- **Precision changed mid-pretraining**, from FP8 to bf16 matmuls, when the run moved from H100 to
  A100 GPUs (see above). The fp32 master weights were carried across unchanged.

## Training data

### Stage 1: pretrain (110.00B tokens)

| Source | Weight | Unique tokens | Tokens drawn | Passes |
|---|---:|---:|---:|---:|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 50% | 20.59B | 55.00B | 2.67 |
| [FineMath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 30% | 9.95B | 33.00B | 3.32 |
| GitHub code ([nick007x/github-code-2025](https://huggingface.co/datasets/nick007x/github-code-2025), above-2-stars subset) | 20% | 7.50B | 22.00B | 2.93 |
| **Total** | | **38.03B** | **110.00B** | **2.89** average |

These are the same token files as argonne-4.0-base's stage 1, built by
[`build_a4_data.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_a4_data.py).
They are not pre-blended: `pretrain.py` draws one source per micro-batch by weight, and the
realized mix over all 110B tokens was 50.007 / 29.997 / 19.996. Every source stayed under 4
passes; FineMath is the most repeated at 3.32.

### Stage 2: reasoning anneal (18.07B tokens)

One pass over **the same anneal corpus as argonne-4.0-base's stage 2**, built by
[`build_reasoning_corpus.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_reasoning_corpus.py).

| Tier | Share | Sources |
|---|---:|---|
| code | 45.4% | [nick007x/github-code-2025](https://huggingface.co/datasets/nick007x/github-code-2025) · [nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) |
| reasoning | 24.0% | [a-m-team/AM-DeepSeek-R1-Distilled-1.4M](https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M) · [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) · [PursuitOfDataScience/0.5M-thinking](https://huggingface.co/datasets/PursuitOfDataScience/0.5M-thinking) |
| math | 19.9% | [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) |
| general replay | 9.0% | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| tool use | 1.7% | [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) |

Shares are the tier composition of the full pool this corpus was sliced from, as measured for
the [argonne-4.0-base card](https://huggingface.co/PursuitOfDataScience/argonne-4.0-base).

- Reasoning and tool data keep their `<think>` and tool-call tags, and conversations are
  formatted as ChatML turns (`<|im_start|>` ... `<|im_end|>`). The base has seen that formatting,
  but it has had no instruction tuning.
- The corpus is decontaminated against common evaluation sets.
- The general-text replay tier is small (9.0%). The 4.0-base card names that tier as one likely
  contributor to 4.0-base's weak world knowledge; this run reused the corpus unchanged, so the
  same caveat applies.

## Training loss

![Training loss curve](plots/argonne4_5_loss_plot.png)

Loss, perplexity and LR against cumulative tokens across all three stages, stage boundaries dashed.
The faint trace is the raw logged loss, the solid line a rolling median. **Loss steps at stage
boundaries are changes of data mixture, not capability jumps:** the anneal's reasoning corpus is
lower-entropy than the pretrain mixture and stage 3's long arXiv sits between them, so cross-stage
loss values are not comparable.
This checkpoint is the end of stage 2 (the second dashed line); the third segment is its
continuation, argonne-4.5-base-ctx13568.

## Tokenizer

[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)'s tokenizer (151,669 tokens)
through the `Qwen2Tokenizer` class, bundled with the weights.

- `<|endoftext|>` (151643) marks the end of a document. Training packed documents with only this
  token between them. It is both the EOS and the pad token.
- **No BOS token** is added, which matches training.
- `<|im_start|>` / `<|im_end|>` (151644 / 151645) are ChatML turn markers, seen in the anneal's
  conversation data.
- `<think>` / `</think>` (151667 / 151668) mark reasoning traces, seen in the anneal data.
- `<tool_call>` / `</tool_call>` (151657 / 151658) mark tool calls, seen in the anneal's tool tier.

One inherited setting to ignore: the tokenizer's `model_max_length` reads 131,072, which does not
describe this model. No chat template is bundled, since this is a base model.

## Evaluation

Scored with the same harness, tasks and few-shot counts as the 4.0-base card; the 4.0-base,
Llama-3.2-1B and Qwen3-0.6B-Base columns are that card's numbers. This checkpoint's 1,024-token context left-truncates the longest few-shot prompts
(25-shot arc_challenge, some 5-shot mmlu subjects), as it would for any 1,024-token model.

| task | **4.5-base** | 4.5-base-ctx13568 | 4.0-base | Llama-3.2-1B | Qwen3-0.6B-Base |
|---|---:|---:|---:|---:|---:|
| arc_challenge | **40.10** | 41.72 | 36.26 | 34.90 | 44.88 |
| arc_easy | **60.06** | 62.04 | 56.19 | 59.93 | 57.87 |
| hellaswag | **51.14** | 55.83 | 45.44 | 60.36 | 53.61 |
| piqa | **70.02** | 70.51 | 67.74 | 73.50 | 69.80 |
| sciq | **83.80** | 85.40 | 79.80 | 89.90 | 91.30 |
| openbookqa | **36.20** | 38.00 | 31.80 | 36.20 | 34.60 |
| winogrande *(acc)* | **56.27** | 57.85 | 55.49 | 61.96 | 60.22 |
| mmlu *(acc)* | **28.88** | 26.03 | 26.15 | 31.41 | 52.49 |
| **8-task mean** | **53.31** | 54.67 | 49.86 | 56.02 | 58.10 |
| gsm8k strict-match | **20.62** | 4.78 | 7.51 | 1.82 | 49.28 |
| gsm8k flexible-extract | **21.15** | 5.53 | 7.88 | 2.27 | 50.04 |

Metric rule, the same as the 4.0-base card: `acc_norm` for the multiple-choice tasks, `acc` for
winogrande and mmlu, and gsm8k in both extraction modes. Harness:
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) through the vLLM
backend in [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py).
The vLLM port reproduces `model.py`'s greedy tokens on 4.5 weights (exactness gate, 8 prompts x 64 tokens).

**Past 1,024 tokens: effectively blind, as expected.** On held-out arXiv (proof-pile-2 test split, 40
windows of 24,576 tokens) its next-token loss is 2.21 nats inside its window and 5.4 to 6.1 nats at
every position past it; it is the 1,024-token control arm of the long-context table in the
[argonne-4.5-base-ctx13568 card](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568). The two
previous Argonne checkpoints trained only at 1,024 tokens (the stage-2 ancestors of 3.5-base and
4.0-base) were effectively blind past their window: about 2.2 to 2.6 nats/token on held-out arXiv
inside it, and 5 to 6 nats/token beyond it. This checkpoint behaves the same.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PursuitOfDataScience/argonne-4.5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "The three laws of thermodynamics are"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 128,  # total length: prompt plus new tokens
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

For throughput, serve it with vLLM. The repo has a vLLM port of this architecture
([`reasoning/vllm_argonne.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/vllm_argonne.py),
written for vLLM 0.11.2) that must be registered before the engine starts:

```bash
pip install vllm==0.11.2
git clone https://github.com/PursuitOfDataScience/ArgonneAI
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # keep the engine in-process so the registration holds
export PYTHONPATH="$PWD/ArgonneAI/reasoning:$PWD/ArgonneAI"
```

```python
import vllm_argonne
vllm_argonne.register()  # registers the Argonne architecture with transformers and vLLM

from vllm import LLM, SamplingParams

llm = LLM(model="PursuitOfDataScience/argonne-4.5-base", trust_remote_code=True,
          dtype="bfloat16", max_model_len=1024)
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
print(llm.generate(["The three laws of thermodynamics are"], params)[0].outputs[0].text)
```

## Usage notes

- **Load with `trust_remote_code=True`.** The architecture is custom (`model.py` ships in this
  repo), and `config.json` carries an `auto_map`, so `from_pretrained` needs nothing else.
- **`generate()` is the model's own method, not the transformers one.** It takes `max_length`
  (prompt plus new tokens), not `max_new_tokens`. It takes `input_ids` only: there is no
  `attention_mask` and no padding support, so generate one prompt at a time (or use vLLM). It
  samples by default (`do_sample=True`, `temperature=1.0`); pass `do_sample=False` for greedy
  output.
- **It stops at `<|endoftext|>`** (151643). If you prompt in ChatML style, pass
  `eos_token_id=151645` to stop at `<|im_end|>` instead.
- **`lm_head.weight` is reported as missing on load.** This is expected: the output head is tied
  to the input embedding and is filled from it.
- **Keep inputs within 1,024 tokens.** Once a sequence grows past 1,024, `generate()` silently
  keeps only the last 1,024 tokens, and the tokenizer will not warn you.

## Limitations

- **Base model.** No instruction following, dialogue ability or safety alignment. Outputs can be
  wrong, biased or unsafe.
- **1,024-token context.** Longer inputs are unsupported; use
  [argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568) for more.
- **Skewed token mix.** After a pretrain that is half educational web text, the anneal is mostly
  code, math and reasoning traces with a 9% general-text replay tier, so general-domain behavior
  may trail the target domains.
- **GSM8K carries a contamination asterisk for this line.** The anneal draws on
  OpenMathReasoning and GSM8K-style data, and train-versus-test exposure has not been audited
  for this base. Read a GSM8K number as a signal, not as a clean capability claim.
- **Scale.** 2.06B parameters on 128B tokens is far below frontier compute.
- **Mostly English.** The data is English web text, math and code.

## Source code

Built from the GitHub `main` branch:
[PursuitOfDataScience/ArgonneAI](https://github.com/PursuitOfDataScience/ArgonneAI/tree/main).
Family overview and training history:
[README, Argonne 4.5 section](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/README.md#argonne-45-base).

| File | Role |
|---|---|
| [`model.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py) | `ArgonneModel` / `ArgonneConfig` architecture and KV cache (bundled here as `model.py`) |
| [`pretrain.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/pretrain.py) | stage 1: pretraining loop and the per-micro-batch mixture sampler |
| [`continue_pretrain.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/continue_pretrain.py) | stage 2: the reasoning anneal |
| [`build_a4_data.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_a4_data.py) | builds the stage-1 token files |
| [`build_reasoning_corpus.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_reasoning_corpus.py) | builds the stage-2 corpus (tiering, decontamination, slicing) |
| [`reasoning/vllm_argonne.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/vllm_argonne.py) | vLLM port of this architecture |
| [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py) | the benchmark harness for the evaluation table |
| [`reasoning/plot_a45_loss.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/plot_a45_loss.py) | the loss figure above |

Continued as: [argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568)
(stage 3, context extension to 13,568 tokens).

## Citation

```bibtex
@misc{argonne45base1k,
  author = {PursuitOfDataScience},
  title = {Argonne 4.5-base},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/PursuitOfDataScience/argonne-4.5-base}
}
```
