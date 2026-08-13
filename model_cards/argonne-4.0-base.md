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
pipeline_tag: text-generation
---

# Argonne 4.0-base

Argonne 4.0-base is a **1.04B-parameter** decoder-only transformer trained from scratch on
**65.12B tokens**, with a **65,536-token trained context**. It is the base checkpoint of the
Argonne 4.0 line.

It is not a bigger [argonne-3.5-base](https://huggingface.co/PursuitOfDataScience/argonne-3.5-base) —
it is a deliberately smaller one. 3.5-base is 2.88B parameters on 88.84B tokens of mostly web text;
4.0-base is **36% of the parameters on 73% of the tokens**, and spends those tokens on a
math/code-weighted mixture instead. The bet, taken from a 49-run iso-token campaign, is that for a
~1B model **data composition** is a larger lever than parameter count, and that the advantage grows
with scale rather than shrinking.

Read the evaluation section before using this model. The bet paid off in one direction and clearly
failed in another, and both are reported below with the numbers.

The architecture is 3.5's, re-shaped: grouped-query attention with QK-norm, V-norm, sandwich norms,
and a final logit softcap, at 1,536 hidden × 32 layers instead of 3,072 × 24.

This is a **base model**: no instruction tuning, no alignment, no safety filtering.

## Model architecture

| Component | Specification |
|-----------|---------------|
| **Parameters** | 1,038,492,672 (~1.04B) |
| **Layers** | 32 transformer blocks |
| **Hidden size** | 1,536 |
| **Attention heads** | 6 query / 2 key-value (GQA) |
| **Head dimension** | 256 (derived: 1,536 / 6) |
| **Feed-forward** | SwiGLU MLP, 4,096 intermediate dim |
| **Attention pattern** | **Full causal on every layer** (see the note below) |
| **Normalization** | RMSNorm with QK / V / sandwich norms |
| **Position encoding** | RoPE (θ = 1,000,000) |
| **Logit stabilization** | Final logit softcap = 15.0 |
| **Context length** | **65,536 tokens** (trained, not extrapolated) |
| **Vocabulary size** | 151,669 |
| **Tied embeddings** | Yes (input ↔ output) |

> **Attention pattern — read this if you compare against the 3.0/3.5 configs.** Those repos ship
> `interleaved_local_attention: true, local_attention_window: 256`. **This model was trained with
> full causal attention on every layer, and its config says so** (`interleaved_local_attention:
> false`). The reason is not a design change: `model.py` implements the sliding window *only* on the
> flash-attn-2 code path, the training environment has flash-attn-4 (which does not expose
> `flash_attn.flash_attn_interface`), so every training slice fell back to SDPA and logged
> `local_attention_window=256 is configured but IGNORED on this path`. The window has therefore
> never been active in any Argonne pretrain. Publishing the flags as-is would hand a 256-token
> window to any downstream user who happens to have flash-attn-2 installed — on weights that never
> saw one, at an advertised 65,536-token context. The release config describes the model that was
> actually trained.

Parameter-count footnote: training logs report `1,038,509,568` because FP8 requires the `lm_head`
input dim to be a multiple of 128, so the vocabulary is padded 151,669 → 151,680 during training
(11 extra rows × 1,536 = 16,896 parameters). Export trims the padding back off, so the published
checkpoint is 1,038,492,672.

## Training

Four stages, all causal language modeling, all on 3× NVIDIA H100/H200 GPUs with DDP.

| | 1 — pretrain | 2 — reasoning anneal | 3 — ctx extension | 4 — ctx extension |
|---|---|---|---|---|
| **Script** | `pretrain.py` | `continue_pretrain.py` | `continue_pretrain.py` | `continue_pretrain.py` |
| **Steps** | 50 → 72,827 | → 103,457 | → 109,622 | → **112,674** |
| **Tokens** | 38.03B | 18.07B | 6.02B | 3.00B |
| **Cumulative** | 38.03B | 56.10B | 62.12B | **65.12B** |
| **Sequence length** | 1,024 | 1,024 | **13,568** | **65,536** |
| **Batch / GPU** | 170 | 32 | 2 | 1 |
| **Grad accumulation** | 1 | 6 | 12 | 5 |
| **Effective batch** | 522,240 tok/step | 589,824 tok/step | 976,896 tok/step | 983,040 tok/step |
| **Peak LR** | 6.0e-4 | 2.0e-4 | 1.0e-4 | 1.0e-4 |
| **End LR** | 6.0e-5 | **2.0e-4 (no decay)** | 1.0e-5 | 1.0e-5 |
| **Warmup** | 8,000 steps | 0 | 0 | 0 |
| **Schedule** | WSD, cooldown 10,923 steps (0.15) | **constant — no cooldown** | WSD, cooldown 900 | WSD, cooldown 458 |

Shared across all four stages:

| Item | Value |
|------|-------|
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, weight decay 0.1), fused |
| **Gradient clipping** | 0.4 |
| **Precision** | FP8 (torchao tensorwise, including `lm_head`) under bf16 autocast; fp32 optimizer states |
| **Vocab padding** | 151,669 → 151,680 during training for the FP8 `lm_head` (trimmed on export) |
| **Cross-entropy** | Chunked, so the 151,680-wide fp32 logit transient never materialises in full |
| **`torch.compile`** | Enabled |
| **Gradient checkpointing** | Enabled |
| **Data parallel** | 3 GPUs (DDP) |
| **Total optimizer steps** | 112,674 |
| **Checkpoint dtype on Hub** | bfloat16 |
| **Weight format on Hub** | 5 sharded safetensors + index |

**Stage 2 ran at a constant LR with no cooldown**, and that is a defect, not a choice — the launcher
passed `cooldown 0`. It is stated here rather than smoothed over because stage 2 is 18.07B tokens,
28% of the whole run, and because the general-knowledge weakness in the evaluation section is the
place it would show up. Stages 1, 3 and 4 all ran a real cooldown.

## Training data

The pretrain mixture is the point of this model. It is **not** a pre-blended corpus: `pretrain.py`'s
`WeightedMultiLoader` samples one source per micro-batch by weight, so the ratio is decoupled from
the raw size of each source.

| Stage 1 source | Weight | Tokens available |
|---|---:|---:|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 50% | 20.59B |
| [FineMath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 30% | 9.95B |
| GitHub code (permissive) | 20% | 7.50B |
| **combined** | | **38.03B** (≈1× per source) |

| Stage | Corpus | Tokens |
|---|---|---|
| **1 — pretrain** | 50 / 30 / 20 edu / math / code, sampled per micro-batch | 38.03B |
| **2 — reasoning anneal** | code / math / reasoning / tool mixture with a general-web replay tier | 18.07B |
| **3 — ctx extension** | a **disjoint** slice of the same stage-2 composite, read at 13,568 | 6.02B |
| **4 — ctx extension** | 50% long arXiv (docs ≥ 32,768 tok) / 25% reasoning replay / 25% edu replay | 3.00B |

The stage-2/3 composite is built by
[`build_reasoning_corpus.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_reasoning_corpus.py);
stages 2 and 3 take **disjoint** slices so stage 3 is not a second epoch. Measured composition of
that pool (24.33B tokens, by tier):

| Tier | Share | Source |
|---|---:|---|
| code | 45.4% | [nick007x/github-code-2025](https://huggingface.co/datasets/nick007x/github-code-2025) · [nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) |
| reasoning | 24.0% | [a-m-team/AM-DeepSeek-R1-Distilled-1.4M](https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M) · [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) · [PursuitOfDataScience/0.5M-thinking](https://huggingface.co/datasets/PursuitOfDataScience/0.5M-thinking) |
| math | 19.9% | [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) |
| **general (replay)** | **9.0%** | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| tool | 1.7% | [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) |

Stage 4 is built by
[`build_phasec_data.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_phasec_data.py)
from [proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) arXiv, restricted to
documents long enough to fill the window: 78.4% of that corpus's tokens sit in documents longer than
13,568, which is what makes a longer window worth training. Half the mixture is replay, specifically
to limit forgetting; the tier-CE measurement below shows it only partly worked.

Reasoning traces keep their `<think>`/tool tags, so the base has seen that formatting before any
fine-tuning. The corpus is decontaminated against common evaluation sets, and stage 4's evaluation
shards (`arXiv_09*`) are held out from its training shards (`arXiv_0[0-8]*`).

**Tokenizer:** [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (151,669-token
vocab), via the `Qwen2Tokenizer` compatibility class. Bundled with the checkpoint.

## Training loss

![Training loss curve](plots/argonne4_0_loss_plot.png)

Loss, perplexity and LR against cumulative tokens across all four stages, stage boundaries marked.
Two things about this figure are easy to misread:

- **The sawtooth in stage 1 is the sampler, not the optimization.** Each step draws *one* of the
  three sources, and their entropies differ a lot — an edu step logs ≈2.7, a math or code step
  ≈1.0–1.5. Consecutive logged steps therefore alternate. The faint trace is raw per-step loss; the
  solid line is a rolling median.
- **Loss steps at stage boundaries are changes of data mixture, not capability jumps.** Cross-stage
  loss values are not comparable; the anneal and long-arXiv corpora are intrinsically lower-entropy
  than the pretrain mixture.

## Evaluation

All numbers below were measured on **the exact weights in this repo** (step 112,674, post-cooldown).
Every benchmark uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
through a vLLM backend that is token-for-token validated against `model.py` on this architecture
(6 query / 2 KV heads, head_dim 256).

Metric rule: `acc_norm` for the multiple-choice tasks, `acc` for winogrande and mmlu (which report no
normalised variant), and gsm8k separately in both extraction modes. One rule for every column —
mixing `acc` and `acc_norm` across arms is worth ~3.7 points of phantom regression on this suite.
Regenerate with
[`reasoning/release_table.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/release_table.py).

Two anchors, both 1B-class, both scored on the **same harness, tasks and few-shot counts** in the same
campaign — so this is a like-for-like table, not a comparison against published numbers.

| task | **Argonne 4.0-base** | stage 3 (ctx 13,568) | Llama-3.2-1B | Qwen3-0.6B-Base |
|---|---:|---:|---:|---:|
| arc_challenge | **36.26** | 35.75 | 34.90 | 44.88 |
| arc_easy | 56.19 | 54.88 | 59.93 | 57.87 |
| hellaswag | 45.44 | 43.85 | 60.36 | 53.61 |
| piqa | 67.74 | 67.08 | 73.50 | 69.80 |
| sciq | 79.80 | 77.80 | 89.90 | 91.30 |
| openbookqa | 31.80 | 32.20 | 36.20 | 34.60 |
| winogrande *(acc)* | 55.49 | 56.35 | 61.96 | 60.22 |
| **mmlu** *(acc)* | **26.15** | 24.73 | 31.41 | **52.49** |
| **8-task mean** | **49.86** | **49.08** | **56.02** | **58.10** |
| **gsm8k strict-match** | **7.51** | **9.70** | **1.82** | **49.28** |
| gsm8k flexible-extract | 7.88 | 10.16 | 2.27 | 50.04 |

Parameters / training tokens: **Argonne 4.0-base 1.04B / 65.12B** · Llama-3.2-1B 1.24B / ~9T ·
Qwen3-0.6B-Base 0.6B / ~36T. This model has seen **0.7% of Llama's and 0.2% of Qwen's** token budget,
so a deficit is expected. What matters is its *size and shape*, and one of the two comparisons is much
worse than the other.

**Against Llama-3.2-1B the data bet shows up where it was supposed to.** 4.0-base wins generative math
**7.51 vs 1.82 — 4.1×** — at 84% of the parameters, and edges arc_challenge. It loses the rest, most of
it on the commonsense/knowledge tasks (hellaswag −14.9, piqa −5.8, sciq −10.1).

**Against Qwen3-0.6B-Base it is behind on all nine cells**, by −8.24 on the 8-task mean, **−26.34 on
MMLU** and **−41.77 on gsm8k**, at 1.7× the parameters. Tokenizer is not a confound: 4.0-base pretrains
with Qwen3's tokenizer, so the two share it exactly. Qwen3-0.6B's 49.28 gsm8k invites its own
contamination question, but MMLU is far harder to game and shows −26.34, so discounting gsm8k entirely
does not rescue the comparison.

**Read stage 3 → 4.0-base as the cost of the context extension, and it is not free.** Stage 4 bought
+0.78 on the multiple-choice mean and the entire 65,536-token window, and paid **−2.19 on gsm8k**, the
only *generative* task in the suite. Two other instruments point the same way: per-tier held-out
cross-entropy is worse on 7 of 8 reasoning-anneal tiers after stage 4 (`reason_r1` +69% perplexity,
`code_github` +25%), despite stage 4 being 50% replay specifically to prevent that. **Stage 4 is a
domain trade — scientific text and long-context reach in exchange for generative reasoning — not a
free context extension.** Full analysis:
[`reasoning/thinking_training.md`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/thinking_training.md)
§39.

For reference, the same suite at step 112,412 — 262 steps and 0.26B tokens earlier, mid-cooldown —
gave an 8-task mean of 49.94 and mmlu 25.95. **Finishing the LR cooldown moved nothing** (−0.08 mean,
+0.20 mmlu). That is worth stating because §39 flagged its mid-cooldown numbers as a lower bound that
"may soften" the trade; measured on the finished stage, it did not.

### Long context — position-bucketed NLL on held-out arXiv

The 65,536-token window is the claim that most needs evidence, because on this architecture
RoPE θ=1e6 does **not** extrapolate unaided. The test is a paired A/B on identical held-out windows
against this model's own ancestors — the stage-2 checkpoint, which was trained *only* at 1,024
tokens, and the stage-3 checkpoint at 13,568. Lower is better.

All three arms scored on the **same 24 held-out windows** of 49,152 tokens each, from
[proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) arXiv shards
(`arXiv_09*`) that stage 4 **did not train on** (it trains on `arXiv_0[0-8]*`). Nats/token, lower
is better.

| Token position | stage 2 (ctx 1,024) | stage 3 (ctx 13,568) | **Argonne 4.0-base** (ctx 65,536) |
|---|---:|---:|---:|
| 0 – 1,024 | 2.561 | 2.401 | **1.970** |
| 1,024 – 2,048 | 5.086 | 2.061 | **1.671** |
| 2,048 – 4,096 | 5.933 | 1.705 | **1.386** |
| 4,096 – 8,192 | 6.051 | 1.427 | **1.139** |
| 8,192 – 13,568 | 5.964 | 1.242 | **0.980** |
| 13,568 – 20,480 | 5.920 | 1.197 | **0.924** |
| 20,480 – 24,576 | 6.025 | 1.220 | **0.905** |
| 24,576 – 32,768 | 5.990 | 1.198 | **0.864** |
| 32,768 – 40,960 | 6.171 | 1.316 | **0.896** |
| 40,960 – 49,152 | 6.075 | 1.300 | **0.820** |

**RoPE θ=1e6 does not extrapolate unaided on this architecture.** Stage 2 is coherent inside its
1,024-token training window (2.56) and effectively blind past it — 5.1 to 6.2 nats, flat, for 48,000
tokens. If you are tempted to assume a large RoPE base buys free context, this is the counterexample,
measured on this exact model's own ancestor.

**The window is real.** The released model falls monotonically with position out to 49,152 and is
better than stage 3 at every bucket, with no short-context tax: the 0–1,024 control bucket improves
too (1.970 vs 2.401).

**But do not read the whole gain as context extension, and the probe's own falsifiable test is why.**
A genuine extension predicts the gap over the previous stage *grows* with position. It does not — the
stage 3 → 4.0-base gap is **U-shaped**: −0.43 nats at 0–1,024, a minimum of −0.26 in the middle, and
−0.48 in the 40,960–49,152 tail. It is as large at position 0 as at position 49,000, and stage 4 did
not need to extend position 0–1,024. Combined with stage 4 making the reasoning-anneal tiers *worse*
(§39b), the honest attribution is **distribution, not length**: stage 4 moved the model toward arXiv.
The context extension proper is carried by stage 3, which is exactly where the 1,024-only arm's ~6-nat
plateau collapses to 1.2–2.1.

**At the full 65,536 training length** the same shape holds. Re-run on 10 windows of 65,536 tokens
(the window set is seeded, so all three arms see the same documents), the tail bucket
**49,152 – 65,536** reads: stage 2 **5.215** · stage 3 **1.097** · **Argonne 4.0-base 0.774**. The
advertised window is usable end to end, not just nominally configured.

Reproduce with
[`reasoning/exp_longctx_learning.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/exp_longctx_learning.py)
(`--arms a4_anneal,a4_phaseb,a4_phasec --docs 24 --eval_len 49152 --docbin_glob 'arXiv_09*.bin'`).
Caveat worth stating plainly: **the evaluation corpus is stage 4's own training domain**, held-out
shards notwithstanding, so its win here bounds usable context but should not be read as general
capability.

### Two-axis base gate — reported as a footnote, deliberately

A 35-item greedy few-shot probe (20 arithmetic/word-problem, 15 world-knowledge) used internally as a
**go/no-go gate** on whether a base is worth spending a reasoning recipe on. Measured on the released
weights, in two prompt formats:

| | standard | extension | pooled | gate |
|---|---:|---:|---:|---|
| **Math** | 17/20 | 17/20 | **34/40** | PASS (≥14/20) |
| **General** | 14/15 | 14/15 | **28/30** | PASS (≥14/15) |

**It clears — and it is the wrong instrument, which is the point of putting it here rather than at the
top.** The general axis has a ceiling of 15 and every a4 checkpoint ever tested reads 14 or 15 on it,
so it cannot separate them; two phase-C checkpoints 40 steps apart differ by 2 pooled math points,
which is this probe's measured ±2-item noise floor. On this gate the released model reads *ahead* of
the 2.88B base that produced Argonne-3.5-think (14/20 · 14/15) at 36% of the parameters — while the
benchmark table above shows a 2× MMLU gap and a 6.6× gsm8k gap against a 0.6B public base. **A
saturating gate cannot rank bases; it can only reject very bad ones.** Reproduce with
[`reasoning/a4_gate_probe.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/a4_gate_probe.py).

### What is weak, stated plainly

**World knowledge is the binding axis, and MMLU is at chance.** 26.15 against 25.0 for random
guessing on 4-way multiple choice. Whatever this model knows, MMLU cannot detect it. Qwen3-0.6B-Base
reads 52.49 on the identical harness at 58% of the parameters, so this is not a "1B models can't do
MMLU" result — it is specific to this base.

Two contributors are identifiable in the recipe, and both are recorded rather than guessed at:

1. **The general-web replay tier was 9.0% of the stage-2 pool** — 2.20B of 24.33B tokens, against
   45.4% code and 24.0% reasoning. Stage 2 is 18.07B tokens, 28% of the whole run, and it is
   overwhelmingly not general text. The 3.5 line hit measurable general-text regression on a
   similarly-capped tier (+0.246 nats of held-out FineWeb-Edu cross-entropy, 18% of the way into the
   anneal) and raised the pool for that reason; this run did not use the raised mixture.
2. **Stage 2 ran with no LR cooldown** (flat 2.0e-4 for all 18.07B tokens) because the launcher passed
   `cooldown 0`. An un-annealed stage is exactly where knowledge consolidation would be expected to
   suffer.

Neither is proven causal — a controlled run would need a raised replay tier and a cooled stage 2, and
that experiment has not been run. They are named because they are the two levers this recipe left on
the table, not as an excuse for the number.

**The internal base gate says the opposite, and the gate is wrong.** It rates this model 17/20 math ·
14/15 general — clearing with margin and *ahead* of the 2.88B base that produced Argonne-3.5-think
(14/20 · 14/15) at 36% of the parameters — while blind to a 2× MMLU gap and a 6.6× gsm8k gap against a
0.6B public base. Any "this base looks strong" claim about the 4.0 line traces to that instrument and
should be re-read against the benchmark table. This is the methodological result of the release round,
and it is why the gate is reported above as a footnote rather than as a headline.

**What this model is genuinely good for**, stated so the negatives above are not over-read: it is a
real 65,536-token model with measured long-document gains that no 0.6B checkpoint here has; it beats a
1.24B Llama base on generative math by 4×; it is trained end-to-end from scratch with a fully
published recipe; and it is a working substrate for reasoning post-training — the 4.0 line's reasoning
derivatives are built on it. It is not the strongest 1B-class base available, and this card does not
claim otherwise.

## Source code

Built from the GitHub main branch: https://github.com/PursuitOfDataScience/ArgonneAI/tree/main

| File | Role |
|---|---|
| [`model.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py) | `ArgonneModel` / `ArgonneConfig` architecture + KV cache (bundled here as `model.py`) |
| [`pretrain.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/pretrain.py) | stage 1 — DDP pretraining loop and the `WeightedMultiLoader` mixture sampler |
| [`continue_pretrain.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/continue_pretrain.py) | stages 2–4 — anneal and the two context extensions |
| [`build_a4_data.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_a4_data.py) | builds the stage-1 per-source token bins |
| [`build_reasoning_corpus.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_reasoning_corpus.py) | builds the stage-2/3 corpus (tiering, decontamination, disjoint slicing) |
| [`build_phasec_data.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/build_phasec_data.py) | builds the stage-4 long-document + replay mixture |
| [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py) | the benchmark harness used above |
| [`reasoning/vllm_argonne.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/vllm_argonne.py) | vLLM port of this architecture (validated token-for-token) |
| [`reasoning/exp_longctx_learning.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/exp_longctx_learning.py) | the position-bucketed long-context NLL probe |
| [`reasoning/tier_ce_probe.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/tier_ce_probe.py) | per-tier held-out cross-entropy (the forgetting check) |
| [`reasoning/plot_a4_loss.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/plot_a4_loss.py) | the loss figure above |
| [`ARGONNE4.0.md`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/ARGONNE4.0.md) | the recipe, ops and honest bounds for this line |
| [`reasoning/thinking_training.md`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/thinking_training.md) | the full lab notebook; §39 is this base's evaluation round |

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "PursuitOfDataScience/argonne-4.0-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

prompt = "Write a short paragraph about scientific computing at Argonne National Laboratory."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 128,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Usage notes

- Load with `trust_remote_code=True` so the custom `ArgonneModel` / `ArgonneConfig` classes
  (`model.py`) are registered. `config.json` ships an `auto_map`, so `from_pretrained` resolves them
  without manual registration.
- The custom `generate` method on `ArgonneModel` takes `max_length` (total sequence length), not
  `max_new_tokens`.
- Weights are 5 bf16 safetensor shards with a `model.safetensors.index.json` weight map.
- `lm_head.weight` is reported missing on load. Expected and benign — embeddings are tied
  (`tie_word_embeddings: true`), so `lm_head` reads from `embed_tokens`.
- The full 65,536-token context is usable, but `.generate()` is not the way to use it. For anything
  at that length prefer a real serving engine; a validated vLLM port of this exact architecture is
  in the repo ([`reasoning/vllm_argonne.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/vllm_argonne.py)),
  and it is ~10–50× faster than the HF decode loop on this model.
- Switch to greedy decoding (`do_sample=False`) for deterministic output.

## Limitations

- **Base model.** No instruction following, dialogue ability, or safety alignment. Outputs can be
  factually wrong, biased, or unsafe. It continues text; it does not answer questions.
- **World knowledge is this model's weakest axis by a wide margin** — see the evaluation section.
  Do not use it where factual recall matters.
- **Scale.** 1.04B parameters on 65.12B tokens is far below frontier compute, and roughly 0.2% of
  the token budget of the 1B-class open models it is compared against above. Expect corresponding
  quality.
- **The token distribution is not general web text.** Stages 2–4 are reasoning/code/math/arXiv-heavy
  by design — this base exists to be fine-tuned into a reasoner — so general-domain behavior
  reflects a 9%-of-mixture replay tier rather than a general-purpose pretrain.
- **Stage 2 ran without an LR cooldown** (18.07B tokens at a flat 2.0e-4). Treat that stage's
  artifact as un-annealed.
- **GSM8K carries a contamination asterisk for this line.** The stage-2 mixture draws on
  OpenMathReasoning and GSM8K-style data, and train-vs-test exposure was not audited for this base.
  The GSM8K number above is reported because it is the only *generative* task in the suite and it
  moves where the multiple-choice tasks do not — not as a clean capability claim.
- Trained with FP8 matmuls. Weights are published in bf16 and load normally, but exact reproduction
  of the training run requires the same torchao FP8 path.

## Citation

```bibtex
@misc{argonne40base,
  author = {PursuitOfDataScience},
  title = {Argonne 4.0-base},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/PursuitOfDataScience/argonne-4.0-base}
}
```
