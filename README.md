# Argonne 2.0

A **4.9 billion parameter** decoder-only transformer language model trained from scratch using tensor parallelism on a single DGX A100 node.

## Training Loss Curve

![Training Loss vs Tokens](training_loss_vs_tokens.png)

The model was trained on **~22 billion tokens** from FineWeb (CC-MAIN-2025-26), achieving a final loss of approximately **2.5–3.5** after 1.35 million training steps.

## Model Architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | 4,918,072,800 (~4.9B) |
| **Layers** | 24 transformer blocks |
| **Hidden Size** | 4,080 |
| **Attention Heads** | 24 query heads / 8 key-value heads (Grouped-Query Attention) |
| **Head Dimension** | 170 |
| **Feed-Forward** | SwiGLU MLP (~10,880 intermediate dim) |
| **Context Length** | 4,096 tokens |
| **Vocabulary Size** | 151,665 (Qwen2.5-3B-Instruct tokenizer) |
| **Normalization** | RMSNorm (ε = 1e-6) |
| **Position Encoding** | Rotary Position Embeddings (RoPE) |
| **Precision** | bfloat16 mixed precision |

### Key Architectural Features

- **Grouped-Query Attention (GQA)**: Uses 24 query heads with 8 key-value heads (3:1 ratio), reducing memory bandwidth requirements while maintaining model quality.
- **SwiGLU Activation**: Employs the SwiGLU activation function in the MLP layers for improved training dynamics.
- **Flash Attention**: Leverages PyTorch's `scaled_dot_product_attention` for efficient attention computation on Ampere/Hopper GPUs.
- **RoPE**: Rotary position embeddings enable better length generalization compared to absolute positional encodings.

## Training Details

### Hardware Configuration

- **Node**: 1× DGX A100 (8× NVIDIA A100 80GB GPUs)
- **Parallelism**: Tensor parallelism across 8 GPUs
- **Interconnect**: NVLink for high-bandwidth GPU-to-GPU communication

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Total Training Steps** | 1,347,890 |
| **Tokens Processed** | ~21.9 billion |
| **Micro-batch Size** | 2–4 per GPU |
| **Gradient Accumulation** | 4 steps |
| **Effective Batch Size** | ~64–128 sequences |
| **Learning Rate** | 1e-4 (peak) → 1e-5 (cosine decay) |
| **Warmup Steps** | 2,000 |
| **Weight Decay** | 0.1 |
| **Gradient Clipping** | 1.0 |
| **Optimizer** | AdamW (fused) |

### Training Data

The model was trained on **Common Crawl CC-MAIN-2025-26** data:
- 250 Parquet shards streamed sequentially
- Documents tokenized with BOS/EOS boundary markers
- Aggressive filtering of low-quality content (high digit ratio, low alpha ratio)
- Chunked into 4,096-token sequences

### Tensor Parallelism Implementation

The training uses a custom tensor parallel implementation (`TensorParallelModel`) that:
- **Shards attention projections**: Q/K/V/O projections are split across GPUs
- **Shards MLP layers**: Gate, up, and down projections distributed across GPUs
- **Replicates embeddings and norms**: Token embeddings and layer norms remain replicated for simplicity
- **Async all-reduce**: Uses asynchronous collective operations for overlapping communication with computation

Each GPU holds 1/8th of the attention heads (3 Q heads, 1 KV head per GPU) and 1/8th of the MLP hidden dimension.

### Checkpointing

- Checkpoints saved every ~430 steps
- Per-block gradient checkpointing enabled to reduce memory footprint
- Automatic pruning of old checkpoints (keeps last 50)
- Resumable training with exact data position tracking

## Training Progress

### Loss Progression

| Milestone | Steps | Tokens | Loss |
|-----------|-------|--------|------|
| Start | 0 | 0 | ~9.3 |
| 1K steps | 1,000 | 16M | ~6.5 |
| 10K steps | 10,000 | 164M | ~5.5 |
| 100K steps | 100,000 | 1.6B | ~4.0 |
| 500K steps | 500,000 | 8.2B | ~3.5 |
| 1M steps | 1,000,000 | 16.4B | ~3.0 |
| Final | 1,347,890 | 21.9B | ~2.5–3.5 |

### Sample Generations

**At step 1,347,190:**
> "Long long time ago, 5,000 years ago, I have been told it is more than 10,000 times. It is not the same as the 10,000 years ago. You just have to have a very good reason for believing that you are a good person and that you are good..."

The model demonstrates emergent capabilities in generating coherent English text, though quality varies with the inherent noise of web-scale pretraining data.

## Repository Structure

```
ArgonneAI/
├── model.py                    # Model architecture (ArgonneConfig, ArgonneModel)
├── training.py                 # Fresh training with tensor parallelism
├── resume_pretrain_tensor.py   # Resume training from checkpoints
├── data_processing.py          # Tokenization and data loading utilities
├── training_utils.py           # Schedulers, checkpoint utilities
├── inference.py                # Inference utilities
├── model-distillation.py       # Knowledge distillation experiments
├── IMPLEMENTATION_NOTES.md     # Architectural decisions
├── TENSOR_PARALLEL_FIX.md      # Debugging notes for TP issues
└── TENSOR_PARALLEL_USAGE.md    # TP launch instructions
```
