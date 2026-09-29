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
- long-context
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
- EleutherAI/proof-pile-2
pipeline_tag: text-generation
---

# Argonne 4.5-base-ctx13568

**A 2.06B-parameter base model trained from scratch on 145.21B tokens, with a trained
13,568-token context.** It is the final base checkpoint of the Argonne 4.5 line.

Three stages in one run: a 110B-token pretrain, an 18B-token reasoning anneal, and a 17B-token
context extension that teaches the model to use 13,568 tokens of context instead of 1,024. The
checkpoint from just before that last stage is published separately as
[argonne-4.5-base](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base), so the
extension can be measured against the exact weights it started from.

This is a **base model**: no instruction tuning, no alignment, no safety filtering. It continues
text; it does not follow instructions.

## Which checkpoint should I use?

| If you want | Use |
|---|---|
| Knowledge and commonsense tasks, inputs up to 13,568 tokens | **argonne-4.5-base-ctx13568** (this repo) |
| Math and step-by-step reasoning at short context, a start for math fine-tuning, or a before/after control for long-context research | [argonne-4.5-base](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base) |
| A smaller model, or a trained context longer than 13,568 tokens | [argonne-4.0-base](https://huggingface.co/PursuitOfDataScience/argonne-4.0-base) (1.04B, 65,536-token context) |

For math at short context, start from the 1k checkpoint: stage 3 cut gsm8k from 20.62 to 4.78 while
raising seven of eight multiple-choice tasks (see the evaluation section).

```text
stage 1  pretrain             110.00B tokens   context 1,024
stage 2  reasoning anneal      18.07B tokens   context 1,024    ->  argonne-4.5-base
stage 3  context extension     17.15B tokens   context 13,568   ->  argonne-4.5-base-ctx13568  (this repo)
```

## What changed from Argonne 4.0-base

The block design is 4.0's. What changed is the size, the token budget, and one fixed defect.

| | argonne-4.0-base | argonne-4.5-base-ctx13568 |
|---|---|---|
| **Parameters** | 1.04B | **2.06B** |
| **Shape** | 1,536 hidden × 32 layers, 6 query / 2 KV heads | 2,560 hidden × 24 layers, 10 query / 2 KV heads |
| **Pretraining** | 38.03B tokens (one pass over the edu/math/code bins) | **110.00B tokens** (2.89 passes over the same bins) |
| **Reasoning anneal** | 18.07B tokens, **no learning-rate cooldown** (a launcher defect) | the same 18.07B-token corpus, **with** a cooldown |
| **Context extension** | 6.02B tokens at 13,568, then 3.00B at 65,536 | 17.15B tokens at 13,568 (75% long arXiv, 25% replay) |
| **Total tokens** | 65.12B | 145.21B |
| **Trained context** | 65,536 | 13,568 |

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
| **Position encoding** | RoPE, θ = 1,000,000, no RoPE scaling |
| **Output logits** | soft-capped at 15.0 (`15 · tanh(x / 15)`) |
| **Context length** | **13,568 tokens**, trained, not extrapolated |
| **Vocabulary** | 151,669 (Qwen3 tokenizer) |
| **Tied embeddings** | yes: the output head reuses the input embedding |
| **KV cache** | 48 KiB per token in bf16, so about 636 MiB for a full 13,568-token window |


**Config switches that are off.** `config.json` carries switches from the 4.5 research phase,
all disabled in this model: `attn_pattern: null` (so `sliding_window_size: 2048` is never used),
`interleaved_local_attention: false`, `nope_global: false`, `attn_gate: false`,
`doc_mask: false`, `mtp_module_layers: 0`, `mtp_loss_weight: 0.0`, `z_loss_weight: 0.0`. Every
layer runs full causal attention with RoPE.

## Training

Three stages, all causal language modeling.

| | Stage 1: pretrain | Stage 2: reasoning anneal | Stage 3: context extension |
|---|---|---|---|
| **Script** | `pretrain.py` | `continue_pretrain.py` | `continue_pretrain.py` |
| **Optimizer steps** | 203,451 | 33,415 (to global step 236,866) | 31,595, one pass (final global step 268,461) |
| **Tokens** | 110.00B | 18.07B | 17.15B |
| **Cumulative tokens** | 110.00B | 128.07B | **145.21B** |
| **Sequence length** | 1,024 | 1,024 | **13,568** |
| **Batch** | 540,672 tokens/step (528 sequences) | 540,672 tokens/step (528 sequences) | 542,720 tokens/step (40 sequences) |
| **Peak learning rate** | 6.0e-4 | 2.0e-4 | 2.0e-4 |
| **Final learning rate** | 6.0e-5 | 2.0e-5 | 2.0e-5 |
| **Warmup** | 8,000 steps | none | none |
| **Schedule** | constant, then linear decay over the last 15% (30,517 steps) | constant, then linear decay over the last 15% (5,012 steps) | constant, then linear decay over the last 15% (4,739 steps) |
| **Optimizer state** | new | reset at the start of the stage | reset at the start of the stage |

Stage 3 uses 40 sequences per step because 40 divides evenly across 4, 5, 8, 10, 20 or 40 GPUs;
tokens per step stayed within 0.4% of the earlier stages. It also computes the cross-entropy in
chunks of 2,048 rows, which is what lets a 13,568-token sequence fit on a 40 GB card.

Shared by all stages:

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
| Context extension | NVIDIA A100 40GB, 8 GPUs |

The batch stayed fixed within each stage on every GPU count, so moving between GPU types did not
change the recipe.

### Known training quirks

- **A sliding window slipped into the first ~950 anneal steps.** For global steps 203,452 to
  204,397 (about 2.8% of stage 2, ~0.51B tokens), a configuration bug in `continue_pretrain.py`
  gave the 12 odd-numbered layers a 256-token sliding window instead of full attention. It was
  caught from the startup log and fixed, and training continued from the step-204,397 checkpoint
  with full attention (the affected steps were kept, not redone). All of pretraining, the other
  97% of the anneal, and all of stage 3 used full causal attention, which is what the config
  describes.
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
Reasoning and tool data keep their `<think>` and tool-call tags, conversations are formatted as
ChatML turns, and the corpus is decontaminated against common evaluation sets. The general
replay tier is small (9.0%); the 4.0-base card names that tier as one likely contributor to
4.0-base's weak world knowledge, and this run reused the corpus unchanged.

### Stage 3: context extension (17.15B tokens)

| Part | Share | Tokens | What it is |
|---|---:|---:|---|
| Long arXiv | 75.0% | 12.86B | every document of at least 27,136 tokens in the [proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) arXiv training split (274,362 documents), each used once |
| General replay | 25.0% | 4.29B | FineWeb-Edu drawn from the same pool as pretraining, so it rehearses text the model has already seen (by design, to limit forgetting) |
| **Total** | | **17.15B** | one pass |

- **Why 27,136 tokens, twice the window.** A document exactly one window long still straddles a
  window boundary unless it happens to line up. Measured on this source, the chance that a
  13,568-token window lies entirely inside one document is 41.6% with no length filter, 53.0%
  keeping documents of at least 13,568 tokens, and 71.1% at 27,136.
- **About half the training windows are single-document long context.** In the final mix, 53.3%
  of windows lie inside one document (calculated from the document lengths), and a 200-window
  sample of the packed stream found 50.5% with no document boundary. The rest are replay windows
  packed with short general-text documents, or arXiv windows that span two papers.
- **Attention is not masked at document boundaries** (standard packing), so in those windows the
  model attends across unrelated documents.
- Built by [`reasoning/build_longctx_arxiv.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/build_longctx_arxiv.py)
  (tokenize), [`reasoning/filter_docbin_by_length.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/filter_docbin_by_length.py)
  (length filter) and
  [`build_reasoning_corpus.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_reasoning_corpus.py)
  (the 75/25 mix).

## Training loss

![Training loss curve](plots/argonne4_5_loss_plot.png)

Loss, perplexity and LR against cumulative tokens across all three stages, stage boundaries dashed.
The faint trace is the raw logged loss, the solid line a rolling median. **Loss steps at stage
boundaries are changes of data mixture, not capability jumps:** the anneal's reasoning corpus is
lower-entropy than the pretrain mixture and stage 3's long arXiv sits between them, so cross-stage
loss values are not comparable.

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

Scored with the same harness, tasks and few-shot counts as the other columns.

| task | **4.5-base-ctx13568** | 4.5-base | 4.0-base | Llama-3.2-1B | Qwen3-0.6B-Base |
|---|---:|---:|---:|---:|---:|
| arc_challenge | 41.72 | 40.10 | 36.26 | 34.90 | 44.88 |
| arc_easy | 62.04 | 60.06 | 56.19 | 59.93 | 57.87 |
| hellaswag | 55.83 | 51.14 | 45.44 | 60.36 | 53.61 |
| piqa | 70.51 | 70.02 | 67.74 | 73.50 | 69.80 |
| sciq | 85.40 | 83.80 | 79.80 | 89.90 | 91.30 |
| openbookqa | 38.00 | 36.20 | 31.80 | 36.20 | 34.60 |
| winogrande *(acc)* | 57.85 | 56.27 | 55.49 | 61.96 | 60.22 |
| mmlu *(acc)* | 26.03 | 28.88 | 26.15 | 31.41 | 52.49 |
| **8-task mean** | 54.67 | 53.31 | 49.86 | 56.02 | 58.10 |
| gsm8k strict-match | 4.78 | 20.62 | 7.51 | 1.82 | 49.28 |
| gsm8k flexible-extract | 5.53 | 21.15 | 7.88 | 2.27 | 50.04 |

Metric rule, the same as the 4.0-base card: `acc_norm` for the multiple-choice tasks, `acc` for
winogrande and mmlu, and gsm8k in both extraction modes. Harness:
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) through the vLLM
backend in [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py).
The vLLM port reproduces `model.py`'s greedy tokens on these weights (exactness gate, 8 of 8 prompts x 64 tokens exact).

The 4.5-base-ctx13568 vs 4.5-base columns are the cost or gain of stage 3 at short context.
**Stage 3 is a trade, measured:** it raises seven of the eight multiple-choice tasks (+1.36 on the 8-task
mean; mmlu fell 28.88 to 26.03) and cuts gsm8k from 20.62 to **4.78** (−15.85). Stage 3 trained on long arXiv plus FineWeb-Edu with no
math or reasoning replay, so the anneal's step-by-step math is what it gave up.
**Forgetting check:** per-tier held-out cross-entropy on the anneal's own held-out tails, before and after
stage 3 ([`reasoning/tier_ce_probe.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/tier_ce_probe.py),
1M tokens per tier). Stage 3 made every anneal tier worse:

| tier | 4.5-base | 4.5-base-ctx13568 | change in perplexity |
|---|---:|---:|---:|
| tool calls | 0.433 | 2.059 | +408% |
| reasoning traces (R1) | 2.424 | 3.514 | +197% |
| competitive programming | 0.836 | 1.682 | +133% |
| reasoning (Mixture-of-Thoughts) | 0.914 | 1.703 | +120% |
| thinking traces | 0.652 | 1.424 | +116% |
| GitHub code | 1.097 | 1.809 | +104% |
| math (OpenMathReasoning) | 0.722 | 0.964 | +27% |
| FineWeb-Edu (replayed in stage 3) | 2.521 | 2.390 | −12% |

So this checkpoint is the long-context model, not the reasoning one: for code, math, reasoning traces or
tool use at short context, argonne-4.5-base is the stronger start.

### Long context

- **Trained, not extrapolated.** Every stage-3 step ran at 13,568 tokens. RoPE θ stays at
  1,000,000 with no scaling.
- **Past 13,568 tokens: holds to about 20k, then degrades.** On held-out arXiv the loss at
  13,568 to 20,480 (0.832) is still at the level of 8,192 to 13,568 (0.854), and rises to 0.963 at
  20,480 to 24,576. `generate()` keeps only the last 13,568 tokens once a sequence grows past the window.
- **Memory is small.** The KV cache is 48 KiB per token in bf16, about 636 MiB per sequence for a
  full window, on top of about 4.1 GB of weights.

Position-bucketed loss on held-out arXiv, nats/token, lower is better. The 1k checkpoint is
the control: it is the same weights before stage 3, so the gap past position 1,024 is the
extension itself, and the gap in the first bucket is what stage 3 changed at short context.

| Token position | 4.5-base (ctx 1,024) | **4.5-base-ctx13568** (ctx 13,568) |
|---|---:|---:|
| 0 to 1,024 | 2.214 | **1.642** |
| 1,024 to 2,048 | 5.441 | **1.355** |
| 2,048 to 4,096 | 6.099 | **1.210** |
| 4,096 to 8,192 | 5.925 | **0.979** |
| 8,192 to 13,568 | 5.823 | **0.854** |
| 13,568 to 20,480 *(past training length)* | 5.784 | **0.832** |
| 20,480 to 24,576 *(past training length)* | 5.962 | **0.963** |

Probe: [`reasoning/exp_longctx_learning.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/exp_longctx_learning.py).
Held out for both checkpoints: 40 windows of 24,576 tokens from proof-pile-2's arXiv **test** split
(1,730 of its 7,840 documents are that long). Stage 3 trained on every long document in the arXiv
training split, so the training-split shards used for the 4.0-base card were not reused. The loss
falls with position all the way to 13,568, which is what real context extension looks like; the
1k checkpoint collapses right after its window. Part of the first-bucket gap (2.21 vs 1.64) is stage
3's arXiv exposure, the same domain as the test text, rather than context.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PursuitOfDataScience/argonne-4.5-base-ctx13568"

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

For long inputs and for throughput, serve it with vLLM. The repo has a vLLM port of this
architecture
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

llm = LLM(model="PursuitOfDataScience/argonne-4.5-base-ctx13568", trust_remote_code=True,
          dtype="bfloat16", max_model_len=13568)
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
- **Keep inputs within 13,568 tokens.** Past that, `generate()` silently keeps only the last
  13,568 tokens, and the tokenizer will not warn you.

## Limitations

- **Base model.** No instruction following, dialogue ability or safety alignment. Outputs can be
  wrong, biased or unsafe.
- **13,568-token context.** Past it the model is untrained: held-out arXiv loss holds to about
  20k tokens, then degrades. For a longer trained context, see
  [argonne-4.0-base](https://huggingface.co/PursuitOfDataScience/argonne-4.0-base) (65,536).
- **Skewed token mix.** After a pretrain that is half educational web text, the anneal is mostly
  code, math and reasoning traces with a 9% general-text replay tier, and stage 3 is three
  quarters arXiv. General-domain behavior may trail the target domains.
- **GSM8K carries a contamination asterisk for this line.** The anneal draws on
  OpenMathReasoning and GSM8K-style data, and train-versus-test exposure has not been audited
  for this base. Read a GSM8K number as a signal, not as a clean capability claim.
- **Scale.** 2.06B parameters on ~145B tokens is far below frontier compute.
- **Mostly English.** The data is English web text, math, code and arXiv papers.

## Source code

Built from the GitHub `main` branch:
[PursuitOfDataScience/ArgonneAI](https://github.com/PursuitOfDataScience/ArgonneAI/tree/main).
Family overview and training history:
[README, Argonne 4.5 section](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/README.md#argonne-45-base-ctx13568).

| File | Role |
|---|---|
| [`model.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py) | `ArgonneModel` / `ArgonneConfig` architecture and KV cache (bundled here as `model.py`) |
| [`pretrain.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/pretrain.py) | stage 1: pretraining loop and the per-micro-batch mixture sampler |
| [`continue_pretrain.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/continue_pretrain.py) | stages 2 and 3: the reasoning anneal and the context extension |
| [`build_a4_data.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_a4_data.py) | builds the stage-1 token files |
| [`build_reasoning_corpus.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_reasoning_corpus.py) | builds the stage-2 corpus and the stage-3 mix |
| [`reasoning/build_longctx_arxiv.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/build_longctx_arxiv.py) | tokenizes proof-pile-2 arXiv for stage 3 |
| [`reasoning/filter_docbin_by_length.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/filter_docbin_by_length.py) | keeps documents of at least 27,136 tokens |
| [`reasoning/vllm_argonne.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/vllm_argonne.py) | vLLM port of this architecture |
| [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py) | the benchmark harness for the evaluation table |
| [`reasoning/exp_longctx_learning.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/exp_longctx_learning.py) | the position-bucketed long-context probe |
| [`reasoning/build_arxiv_test_docbin.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/build_arxiv_test_docbin.py) | tokenizes the held-out arXiv test split for the long-context table |
| [`reasoning/plot_a45_loss.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/plot_a45_loss.py) | the loss figure above |
| [`reasoning/tier_ce_probe.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/tier_ce_probe.py) | per-tier held-out cross-entropy (the forgetting check) |

Starting point of stage 3:
[argonne-4.5-base](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base).

## Citation

```bibtex
@misc{argonne45base,
  author = {PursuitOfDataScience},
  title = {Argonne 4.5-base-ctx13568},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568}
}
```
