# Argonne LLM Family

Author: Youzhi Yu

Training pipeline and release history for the Argonne causal LM family, trained from scratch on FineWeb-derived web text.

| Model | Params | Context | Training tokens | Hugging Face |
|-------|--------|---------|-----------------|--------------|
| [Argonne 3.5-think](#argonne-35-think) | 2.88B | 13,568 | ~88.84B + post-training | [Argonne-3.5-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.5-think) |
| [Argonne 3.5-base](#argonne-35-base) | 2.88B | **13,568** (trained) | ~88.84B | [argonne-3.5-base](https://huggingface.co/PursuitOfDataScience/argonne-3.5-base) |
| [Argonne 3.0](#argonne-30) | 2.88B | 1,024 (RoPE θ=1e6) | ~76.05B | [argonne-3.0-base](https://huggingface.co/PursuitOfDataScience/argonne-3.0-base) |
| [Argonne 2.5](#argonne-25) | 1.27B | 1,024 | ~76.05B | [Argonne2.5-base](https://huggingface.co/PursuitOfDataScience/Argonne2.5-base) |
| [Argonne 2.0](#argonne-20) | 4.9B | 4,096 | ~21.9B | — (not released) |
| [Argonne 1.5](#argonne-15) | 357M | 2,048 | ~15.45B | [Argonne-1.5](https://huggingface.co/PursuitOfDataScience/Argonne-1.5) |
| [Argonne 1.0](#argonne-10) | 276M | 2,048 | FineWeb-Edu | [Argonne-1.0](https://huggingface.co/PursuitOfDataScience/Argonne-1.0) |

---

# Argonne 3.5-think

The reasoning model of the 3.5 line, released as [`PursuitOfDataScience/Argonne-3.5-think`](https://huggingface.co/PursuitOfDataScience/Argonne-3.5-think). Built on [argonne-3.5-base](https://huggingface.co/PursuitOfDataScience/argonne-3.5-base); emits an explicit `<think>…</think>` trace then a `\boxed{}` answer.

## vs Argonne 3.0-think

![3.5-think vs 3.0-think](plots/a35think_vs_3p0.png)

Both models measured in a **single job**, same grader, same n=300 / K=8 / seed — not compared against previously-recorded numbers.

| clean, n=300, K=8 | Argonne 3.0-think | **Argonne 3.5-think** | delta |
|---|---|---|---|
| SVAMP greedy | 22.00 | **65.00** | **+43.0** |
| SVAMP self-consistency | 30.00 | **74.00** | **+44.0** |
| SVAMP pass@8 | 51.33 | **90.67** | +39.3 |
| ASDiv greedy | 32.33 | **73.00** | **+40.7** |
| ASDiv self-consistency | 39.33 | **82.67** | **+43.3** |
| ASDiv pass@8 | 60.00 | **92.33** | +32.3 |

Judged on **SVAMP/ASDiv**, which appear in no training stage of this line. **GSM8K is contaminated** for Argonne reasoning models and is never reported.

## The mechanism: termination

![termination](plots/a35think_termination.png)

The defining failure of the 3.0 line was non-termination — traces that never closed `</think>`, so no answer was emitted. Training only on short, closed, correct traces makes termination a property of the weights: `no_answer` falls 53.7% → 1.3% (SVAMP) and 59.7% → 2.0% (ASDiv), and budget-forcing — previously the only thing that helped — now adds ~1 point.

## What actually moved the number

![attribution](plots/a35think_attribution.png)

The base raises the **ceiling** (pass@8 58.7 → 74.0) while greedy stays flat; the short-trace mix converts that ceiling into a **floor** (+36.7 greedy); SFT breadth via the weight soup adds the last ~2 points. No RL, no distillation from a stronger teacher.

## Recipe

| stage | data | detail |
|---|---|---|
| 1 — SFT | UltraChat 200k | 207,865 rows, 1 epoch, effective batch 20 |
| 2 — DPO | argilla/dpo-mix-7k | 6,750 pairs, LR 1e-6, β=0.03 |
| 3 — CoT-SFT | short-trace mix, 26,428 rows all ≤768 tokens | 1 epoch, effective batch 12 |
| 4 — weight soup | — | 0.85 × CoT + 0.15 × DPO |

α = 0.85 is a measured knee, not a default: α = 0.70 reintroduces non-termination. Full build log, including the ablations that failed and two predictions that turned out wrong, is in [`reasoning/thinking_training.md`](reasoning/thinking_training.md) §32.

---

# Argonne 3.5-base

Argonne 3.5-base is a 2.88B-parameter decoder-only transformer, released as [`PursuitOfDataScience/argonne-3.5-base`](https://huggingface.co/PursuitOfDataScience/argonne-3.5-base). Same architecture as Argonne 3.0; what changed is the training recipe and a three-stage data curriculum that ends in a **trained** 13,568-token context.

## Training loss curve

![Argonne 3.5 loss curve](plots/argonne3_5_loss_plot.png)

Loss, perplexity, and LR against cumulative tokens across all three stages. The step down at each stage boundary is a change of data mixture, not a capability jump — the anneal and context-extension corpora are intrinsically lower-entropy than FineWeb, so cross-stage loss values are not comparable.

## Training details

| Item | Value |
|------|-------|
| **Stages** | Pretrain (`pretrain.py`) → reasoning anneal (`continue_pretrain.py`) → context extension (`continue_pretrain.py`) |
| **Total optimizer steps** | 321,062 |
| **Tokens processed** | ~88.84B (65.30B pretrain + 17.50B anneal + 6.02B context extension) |
| **Sequence length** | 1,024 (stages 1–2) → **13,568** (stage 3) |
| **Effective batch** | 233,472 → 270,336 tokens/step (stages 1–2); 488,448 tokens/step (stage 3) |
| **Peak learning rate** | 6e-4 pretrain / 2e-4 anneal / 1e-4 context extension; WSD, 8,000 warmup steps, cooldown to 0.1× in every stage |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, weight decay 0.1), grad clip 0.4 |
| **Precision** | FP8 (torchao tensorwise, incl. `lm_head`) under bf16 autocast, `torch.compile`, gradient checkpointing |
| **Final train loss** | 0.8923 (stage-3 slice average) |
| **Hardware** | 3× NVIDIA H100/H200 (DDP) |

Recipe changes vs 3.0: a real LR cooldown in every stage (3.0 ran the WSD stable phase with `cooldown = 0`), a higher peak LR (6e-4 vs 3e-4), a longer warmup (8,000 vs 1,000), tighter gradient clipping (0.4 vs 1.0), and FP8 matmuls. The tighter clip plus QK-norm are what make the higher LR stable.

## Training data

- Stage 1 — [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) + [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), 85/15 (~65.30B tokens)
- Stage 2 — code / math / reasoning / tool anneal with a FineWeb-Edu general-replay tier, built by [`build_reasoning_corpus.py`](build_reasoning_corpus.py) (~17.50B tokens)
- Stage 3 — a **disjoint** slice of the same composite, read at 13,568 tokens (~6.02B tokens)
- Tokenizer: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (151,669-token vocab)

## Context extension is trained, not extrapolated

RoPE θ=1e6 does **not** extrapolate unaided on this architecture. Position-bucketed NLL on held-out arXiv, comparing the stage-2 checkpoint against the final weights it seeded (lower is better):

| Token position | Stage 2 (ctx 1,024) | Argonne 3.5-base (ctx 13,568) |
|---|---|---|
| 0 – 1,024 | 2.194 | **2.161** |
| 1,024 – 2,048 | 5.536 | **1.860** |
| 4,096 – 8,192 | 5.895 | **1.320** |
| 8,192 – 13,568 | 5.961 | **1.207** |
| 13,568 – 20,480 | 5.965 | **1.122** |
| 20,480 – 24,576 | 5.938 | **1.096** |

The stage-2 model is coherent inside its 1,024-token window and effectively blind past it. The final model improves monotonically with position, keeps improving *beyond* its own 13,568 training length, and pays no short-context tax (the 0–1,024 control bucket is better than stage 2). Reproduce with [`reasoning/exp_longctx_learning.py`](reasoning/exp_longctx_learning.py).

## Base gate

A 35-item greedy few-shot probe (20 math, 15 world-knowledge) used as a go/no-go gate on whether a base is worth a reasoning recipe — **not** a capability benchmark (n is small, it saturates, and it has a measured ±2-item noise floor).

| Checkpoint | Math /20 | General /15 |
|---|---|---|
| Stage 2 (step 308,733) | 18 | 15 |
| step 320,885 | 18 | 15 |
| step 321,054 | 17 | 15 |
| step 321,062 (released) | 18 | 15 |

Both axes clear the ≥14/20 ∧ ≥14/15 gate, and context extension cost nothing on either. Reproduce with [`reasoning/probe_pretrain_ckpt.py`](reasoning/probe_pretrain_ckpt.py). No standard held-out benchmark suite has been run on this checkpoint yet.

---

# Argonne 3.0

Argonne 3.0-base is a 2.88B-parameter decoder-only transformer, released as [`PursuitOfDataScience/argonne-3.0-base`](https://huggingface.co/PursuitOfDataScience/argonne-3.0-base). It combines grouped-query attention with stability-oriented additions: QK-norm, V-norm, sandwich norms, interleaved local/global attention, and a final logit softcap.

## Training loss curve

![Argonne 3.0 loss curve](plots/argonne3_0_loss_plot.png)

## Model architecture

| Component | Specification |
|-----------|---------------|
| **Parameters** | 2,882,162,688 (~2.88B) |
| **Layers** | 24 transformer blocks |
| **Hidden size** | 3,072 |
| **Attention heads** | 12 query / 4 key-value (GQA) |
| **Head dimension** | 256 |
| **Feed-forward** | SwiGLU MLP, 8,192 intermediate dim |
| **Attention pattern** | Interleaved local/global causal attention (window 256, every other layer) |
| **Normalization** | RMSNorm with QK / V / sandwich norms |
| **Position encoding** | RoPE (θ = 1,000,000) |
| **Logit stabilization** | Final logit softcap = 15.0 |
| **Context length** | 1,024 tokens |
| **Vocabulary size** | 151,669 (Qwen3 tokenizer) |
| **Tied embeddings** | Yes (input ↔ output) |

## Training details

| Item | Value |
|------|-------|
| **Stages** | Pretrain (`pretrain.py`) → continued pretrain (`continue_pretrain.py`) |
| **Total optimizer steps** | 329,148 |
| **Tokens processed** | ~76.05B (20.84B stage 1 + 55.21B stage 2, one epoch each) |
| **Sequence length** | 1,024 |
| **Effective batch** | 233,472 tokens/step (38 per GPU × grad accum 2 × 3 GPUs) |
| **Peak learning rate** | 3e-4, WSD schedule, 1,000 warmup steps, min LR ratio 0.1 |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, weight decay 0.1), grad clip 1.0 |
| **Precision** | bf16 autocast, `torch.compile`, gradient checkpointing |
| **Final train loss** | 2.5168 |
| **Hardware** | 3× NVIDIA H200 (DDP) |

## Training data

- Stage 1: FineWeb shard (~20.84B tokens)
- Stage 2: FineWeb CC-MAIN-2025-21 dump (~55.21B tokens)
- Tokenizer: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (151,669-token vocab)

---

# Argonne 2.5

Argonne 2.5 is a 1.27B-parameter pretraining checkpoint, released as [`PursuitOfDataScience/Argonne2.5-base`](https://huggingface.co/PursuitOfDataScience/Argonne2.5-base).

## Training loss curve

![Argonne 2.5 loss curve](plots/argonne2_5_loss_curve.png)

## Model architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | 1,273,807,360 (~1.27B) |
| **Layers** | 28 transformer blocks |
| **Hidden size** | 1,792 |
| **Attention heads** | 14 query / 7 key-value (GQA) |
| **Head dimension** | 128 |
| **Feed-forward** | SwiGLU MLP, 4,864 intermediate dim |
| **Context length** | 1,024 tokens |
| **Vocabulary size** | 151,669 |
| **Normalization** | RMSNorm (ε = 1e-6) |
| **Position encoding** | RoPE (θ = 10,000) |

## Training details

| Item | Value |
|------|-------|
| **Total steps** | 425,975 |
| **Tokens processed** | ~76.05B |
| **Final train loss** | 2.6119 |
| **Sequence length** | 1,024 |
| **Effective batch** | 245,760 tokens (20 per GPU × grad accum 4 × 3 GPUs) |
| **Learning rate** | 3e-4, min LR ratio 0.1, 1,000 warmup steps |
| **Precision** | bf16 autocast, `torch.compile` |
| **Hardware** | 3× H200 (DDP) |

## Training data

- FineWeb and FineWeb-Edu
- Final stage training shard: 55.2B tokens; cumulative across the full run: 76.05B tokens

---

# Argonne 2.0

A 4.9B-parameter decoder-only transformer trained from scratch with a custom tensor-parallel implementation on a single DGX A100 node. Not released on Hugging Face.

## Training loss curve

![Argonne 2.0 training loss vs tokens](plots/training_loss_vs_tokens.png)

## Model architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | 4,918,072,800 (~4.9B) |
| **Layers** | 24 transformer blocks |
| **Hidden size** | 4,080 |
| **Attention heads** | 24 query / 8 key-value (GQA) |
| **Head dimension** | 170 |
| **Feed-forward** | SwiGLU MLP, ~10,880 intermediate dim |
| **Context length** | 4,096 tokens |
| **Vocabulary size** | 151,665 (Qwen2.5-3B-Instruct tokenizer) |
| **Normalization** | RMSNorm (ε = 1e-6) |
| **Position encoding** | RoPE |

## Training details

| Item | Value |
|------|-------|
| **Total steps** | 1,347,890 |
| **Tokens processed** | ~21.9B |
| **Final loss** | ~2.5–3.5 |
| **Learning rate** | 1e-4 peak → 1e-5 (cosine), 2,000 warmup steps |
| **Optimizer** | AdamW (fused), weight decay 0.1, grad clip 1.0 |
| **Parallelism** | Tensor parallelism across 8 GPUs (sharded attention + MLP, replicated embeddings/norms, async all-reduce) |
| **Hardware** | 1× DGX A100 (8× A100 80GB, NVLink) |

## Training data

FineWeb (CC-MAIN-2025-26): 250 parquet shards streamed sequentially, documents tokenized with BOS/EOS boundaries, quality-filtered, chunked into 4,096-token sequences.

---

# Argonne 1.5

A 357M-parameter model, released as [`PursuitOfDataScience/Argonne-1.5`](https://huggingface.co/PursuitOfDataScience/Argonne-1.5).

## Training loss curve

![Argonne 1.5 pretraining loss](plots/v1.5_pretraining_loss_plot.png)

## Improvements over Argonne 1.0

- `torch.compile` for pretraining speed; flash attention for ~2.6× memory efficiency (larger batches)
- More layers and attention heads; more efficient GPU utilization
- Integrated with the Hugging Face `AutoModel` class; better text-generation support

## Model and training

| Item | Value |
|------|-------|
| **Parameters** | 356,516,640 (~357M) |
| **Config** | 16 layers, 16 heads, 1,296 hidden, 2,048 context |
| **Tokens processed** | 15,453,927,424 (~15.45B, same data as 1.0) |
| **Total steps** | 80,000 |
| **Batch size** | 756 |
| **Training cost** | 1,248 GPU hours on 1 DGX node (8× A100 80GB) |

---

# Argonne 1.0

The first Argonne model: 276M parameters, released as [`PursuitOfDataScience/Argonne-1.0`](https://huggingface.co/PursuitOfDataScience/Argonne-1.0). See the [model card](https://huggingface.co/PursuitOfDataScience/Argonne-1.0#inference) for inference details.

## Training loss curve

![Argonne 1.0 pretraining loss](plots/pretrain_loss_20250303.png)

## Model and training

| Item | Value |
|------|-------|
| **Parameters** | 275,827,680 (~276M) |
| **Config** | 12 layers, 12 heads, 1,296 hidden, 2,048 context, dropout 0.1 |
| **Data** | [FineWeb-Edu (CC-MAIN-2024-10)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **Learning rate** | 3e-5 until step 62,000, then 5e-5 (batch 48 → 60 at the same step) |
| **Total steps** | 160,000 |
| **Training cost** | 1,440 GPU hours on 1 DGX node (8× A100 80GB) |

---

# Argonne LLM Training

Distributed PyTorch training pipeline for the Argonne causal LM using Qwen-family tokenizers and llm.c-style binary token data.

## Pipeline stages

```text
preprocess_data.py        parquet -> train.bin token shards
pretrain.py               stage 1: DDP pretraining from scratch
continue_pretrain.py      stage 2: continued pretraining on new data
midtraining.py            stage 3: long-context midtraining (FSDP/DDP, e.g. 13,568 ctx)
sft.py                    stage 4: supervised fine-tuning (chat data, e.g. UltraChat)
dpo.py                    stage 5: direct preference optimization
cot-sft.py                stage 6: chain-of-thought SFT (reasoning / <think> models)
```

## Project layout

```text
ArgonneAI/
├── model.py                 # ArgonneModel/ArgonneConfig + HF registration (model_type="argonne2")
├── pretrain.py              # Main DDP pretraining script
├── continue_pretrain.py     # Continued pretraining (new-data continuation)
├── midtraining.py           # Long-context midtraining (FSDP support)
├── sft.py                   # Supervised fine-tuning
├── dpo.py                   # DPO preference tuning
├── cot-sft.py               # Chain-of-thought SFT (HF Trainer-based)
├── preprocess_data.py       # Parquet -> train.bin converter
├── inference.py             # Text generation from a checkpoint or HF repo
├── push_model_to_hf.py      # Publish checkpoints to Hugging Face
└── eval_sft_quality.py      # SFT quality probes
```

## Checkpointing and outputs

- Checkpoints are written as `checkpoint_step_<N>.pt`, including model/optimizer/scheduler state, `global_step`, `tokens_processed`, and data position.
- On periodic checkpoints, rank 0 prints a sampled generation.
- At end of run, scripts save a final checkpoint plus `final_model/` (weights, tokenizer, config via `save_pretrained`).
- Wall-clock-aware training (`--wall_time`) saves and exits cleanly before SLURM time limits, enabling auto-resubmit slice chains.

## Model notes (`model.py`)

- Hugging Face-compatible (`ArgonneConfig`, `ArgonneModel`), registered for `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM`; `from_pretrained` self-heals rotary buffers and the embedding tie after loading.
- GQA, SwiGLU MLP, RMSNorm (+ QK/V/sandwich norms), RoPE, interleaved local/global attention, final logit softcap.
- Attention path selection: FlashAttention 2 (if available) → PyTorch SDPA → math fallback; the active path is logged once at startup.
- Training preset used by `pretrain.py` / `continue_pretrain.py` (Argonne 3 architecture): hidden size 3,072, 24 layers, 12 query heads, 4 KV heads, `max_position_embeddings = --block_size`.
