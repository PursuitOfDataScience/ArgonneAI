# ArgonneAI (Argonne 2.0)

## Overview
This repository contains the training infrastructure for **Argonne 2.0**, a decoder-only transformer language model trained with **tensor parallelism** for efficient multi-GPU training. The codebase emphasizes large-scale pretraining with distributed data processing, a unified scratch/resume workflow, and robust checkpointing.

## Model Architecture
Argonne 2.0 is a 24-layer decoder-only transformer with the following specifications:

- **Layers:** 24 transformer blocks
- **Attention:** Grouped-Query Attention (GQA)
  - 24 query heads
  - 8 key/value heads
  - Head dimension: 170 (hidden_size 4080 / 24 heads)
  - Ratio: 3 query heads per KV head
- **Hidden Size:** 4,080 dimensions
- **Intermediate Size:** 11,008 (SwiGLU MLP with 8/3× expansion, rounded to 256)
- **Context Length:** 4,096 tokens
- **Position Embeddings:** RoPE (Rotary Position Embeddings) with θ = 500,000
- **Normalization:** RMSNorm (ε = 1e-6)
- **Dropout:** 0.0 (disabled for baseline pretraining)
- **Activation:** SwiGLU in feed-forward layers
- **Flash Attention:** Enabled when available
- **Vocabulary:** 151,936 tokens (Qwen2.5-3B-Instruct tokenizer)

**Key Design Choices:**
- **Untied Embeddings:** Input and output embeddings are separate to allow flexible output heads
- **Even Divisibility:** Hidden size (4080) divides evenly by number of heads (24), ensuring integer head dimensions (170)
- **GQA Efficiency:** 8 KV heads divide evenly by 8 GPUs for clean tensor parallelism sharding

## Tokenizer
Argonne 2.0 uses the **Qwen2.5-3B-Instruct** tokenizer with the following features:

- **Vocabulary Size:** 151,936 tokens
- **Special Tokens:**
  - Conversational: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`
  - Tool calling: `<tool_call>`, `<|object_ref_start|>`, `<|object_ref_end|>`
  - Multimodal: `<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`, `<|video_pad|>`
- **Max Length:** 131,072 tokens (expandable beyond 4,096 training window)
- **Chat Template:** Qwen format for tool-aware prompts

## Training Infrastructure

### Tensor Parallelism
Both `training.py` and `resume_pretrain_tensor.py` use **tensor parallelism** to distribute model layers across multiple GPUs:

**How It Works:**
1. Each layer (attention, MLP) is horizontally sharded across all GPUs
2. All GPUs process the same batch simultaneously
3. Partial results are combined via `all_reduce` after each layer
4. Embeddings, norms, and LM head are replicated on all GPUs

**Advantages:**
- ✅ **Full Parallelism:** All GPUs work on every layer simultaneously
- ✅ **Memory Efficiency:** Large layers are split across GPU memory
- ✅ **Synchronous Training:** No pipeline bubbles or complex staging
- ✅ **Scalable:** Works well with NCCL on 8 GPUs (tested configuration)

### Unified Training Workflow
`training.py` now wraps `resume_pretrain_tensor.py` directly. Launching a brand-new run and resuming after the wall-time limit execute the exact same code path, so every optimisation (multi-threaded streaming tokenisation, fused AdamW, asynchronous GPU prefetch, gradient accumulation, document-boundary handling, etc.) is available from the first step.

**Key behaviours:**
- 🧭 **Single source of truth:** All CLI arguments accepted by `resume_pretrain_tensor.py` are forwarded by `training.py`, guaranteeing identical hyper-parameters and defaults.
- 🚀 **Optimised from step zero:** The scratch run benefits from the same fused AdamW optimiser (`fused=True`), rewarm-up aware cosine scheduler, and cudagraph safeguards used while resuming.
- 🔁 **Clear role split:** `training.py` always starts from scratch; use `resume_pretrain_tensor.py` to pick up from checkpoints once the initial wall-time expires.

### Learning Rate Schedule
Both scripts use a cosine decay schedule with warmup:

- **Peak Learning Rate:** 1e-4 (stable for small-batch tensor parallelism)
- **Minimum Learning Rate:** 1e-5 (10x lower than peak)
- **Warmup Steps:** 2000 (gradual ramp to avoid early divergence)
- **Scheduler:** Cosine annealing with linear warmup

**Rationale:**
- Longer warmup protects against optimizer spikes when batch sizes are small
- Peak LR aligned with proven 3B-scale training runs (1e-4) to maintain stability
- Conservative schedule reduces loss oscillations seen with 2e-4 defaults

### Checkpoint System
Both training scripts save checkpoints every 300 steps with full resumability:

**Checkpoint Contents:**
- Model weights (FP16/BF16)
- Optimizer state (AdamW)
- Learning rate scheduler state
- Exact data position (file index, row, chunk offset)
- Training metadata (step, tokens processed, loss)
- Parallelism configuration (world size, rank)

**Resume Capability:**
- Automatically finds latest checkpoint
- Restores exact data position (no duplicate processing)
- Seamlessly continues training from any step
- Compatible across `training.py` and `resume_pretrain_tensor.py`

## Repository Structure
```
.
├── README.md
├── model.py
├── data_processing.py
├── training_utils.py
├── training.py
├── train_with_fsdp.py
├── resume_pretrain.py
├── resume_pretrain_tensor.py
└── TENSOR_PARALLEL_USAGE.md
```

- `README.md` - This file
- `model.py` - Defines the `ArgonneModel` transformer architecture and its `ArgonneConfig`, including rotary attention, RMSNorm layers, and backwards-compatible aliases for Argonne 1.x checkpoints.
- `data_processing.py` - Handles tokenizer loading/fallback training, streaming parquet datasets, cached offline preprocessing, and assembling fixed-length token chunks for training.
- `training_utils.py` - Provides shared helpers such as dataset shard discovery/logging and a cosine learning rate scheduler with warmup for use across entrypoints.
- `training.py` - Tensor-parallel scratch launcher that calls into `resume_pretrain_tensor.py`, ensuring the initial run and later resumes share the same optimisations and CLI surface.
- `train_with_fsdp.py` - Alternative Fully Sharded Data Parallel (FSDP) launcher for multi-GPU jobs that prefer sharded data parallelism over pipeline stages.
- `resume_pretrain.py` - Legacy resume script maintained for compatibility with earlier experiments; kept while we migrate to the unified `training.py` flow.
- `resume_pretrain_tensor.py` - Tensor-parallel training/resume entrypoint that hosts the full training loop used by both scratch launches and long-running resumes.
- `TENSOR_PARALLEL_USAGE.md` - Detailed usage instructions, examples, and troubleshooting for tensor parallelism.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.13+ with CUDA support
- NVIDIA GPU(s) with Tensor Cores (Volta, Turing, Ampere architecture)
- NCCL 2.12+ (for multi-GPU communication)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ArgonneAI/ArgonneAI.git
   cd ArgonneAI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Downloading Pretrained Checkpoints
Pretrained checkpoints are hosted on Argonne's internal storage. To access them, you must be on the Argonne network or connected via VPN.

1. Create a directory for checkpoints:
   ```bash
   mkdir -p ~/ArgonneAI/checkpoints
   ```
2. Download the latest checkpoint:
   ```bash
   cp /path/to/argonne/storage/checkpoints/streaming_checkpoint_step_XXXX.pth ~/ArgonneAI/checkpoints/
   ```

## Training a New Model
To train a new Argonne 2.0 model from scratch:

1. Prepare your data in Parquet format and upload to a shared location.
2. Set the `DATA_PATH` environment variable:
   ```bash
   export DATA_PATH=/path/to/your/data
   ```
3. Run the training script:
 ```bash
  torchrun --standalone --nproc_per_node=8 training.py \
    --tokenizer-path ../Qwen2.5-3B-Instruct \
    --data-glob $DATA_PATH/*.parquet \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4
  ```

`training.py` always starts from scratch and ignores `--checkpoint-path`; once the wall-time limit is reached, launch `resume_pretrain_tensor.py` with the same arguments to continue from the latest checkpoint.

## FAQ

**Q: What is Argonne 2.0?**  
A: Argonne 2.0 is a state-of-the-art language model developed by Argonne National Laboratory, designed for efficient training and inference using tensor parallelism.

**Q: How is Argonne 2.0 different from other models?**
A: Argonne 2.0 features a unified tensor-parallel training stack with an optimised scratch/resume workflow, multi-threaded streaming data ingestion, and a robust checkpoint system, enabling efficient training on large-scale data.

**Q: What are the hardware requirements for training Argonne 2.0?**  
A: Training Argonne 2.0 requires NVIDIA GPUs with Tensor Cores, CUDA, and NCCL for distributed training. A minimum of 8 GPUs is recommended for optimal performance.

**Q: How can I contribute to the ArgonneAI project?**  
A: We welcome contributions! Please submit a pull request or open an issue to discuss potential improvements or features.

**Q: Where can I find more documentation?**  
A: Additional documentation, including detailed usage instructions for tensor parallelism, can be found in the `TENSOR_PARALLEL_USAGE.md` file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Argonne National Laboratory
- The PyTorch team
- The NVIDIA NCCL team
