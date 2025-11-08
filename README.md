# ArgonneAI (Argonne 2.0)

## Overview
Argonne 2.0 is a ~2B parameter, 20-layer decoder-only transformer trained with **tensor parallelism**.
The training entrypoints – `training.py` for fresh runs and `resume_pretrain_tensor.py` for
continuing jobs – now share the same optimised data pipeline, fused optimiser configuration, and
command-line behaviour.  Both scripts stream large Parquet datasets, shard the model across GPUs,
and write resumable checkpoints that can be swapped between the two workflows.

## Model Architecture
- **Parameters:** ~2.0B
- **Layers:** 20 transformer blocks with rotary position embeddings (RoPE)
- **Hidden Size:** 3,072 (128 dimensions per attention head)
- **Attention:** Grouped-Query Attention (24 Q heads / 4 KV heads)
- **Feed-Forward:** SwiGLU MLP (≈8.2k hidden dim)
- **Context Length:** 4,096 tokens
- **Normalization:** RMSNorm (ε = 1e-6)
- **Activation:** SwiGLU
- **Flash Attention:** Enabled when available
- **Tokenizer:** Qwen2.5-3B-Instruct with pad token fallbacks

## Training Pipeline Highlights
- **Tensor parallel execution** – `TensorParallelModel` shards attention and MLP projections across
  all GPUs while keeping embeddings and norms replicated.  Collectives are issued outside compiled
  graphs to avoid cudagraph capture issues.
- **Streaming dataset ingestion** – the shared `streaming_token_generator` performs threaded
  batch tokenisation, optional BOS/EOS injection, and aggressively filters malformed rows.  Data can
  be resumed at chunk-level precision using the persisted `DataPosition` state.
- **Automatic batch-size tuning** – `training.py` performs an OOM-aware search over the micro-batch
  size while keeping `--gradient-accumulation-steps` constant.  Successful runs reuse the exact
  configuration inside `resume_pretrain_tensor.py`.
- **Fused AdamW + gradient scaling** – both entrypoints create a fused AdamW optimiser
  (`fused=True`) and guard mixed precision with autocast/GradScaler.  Gradients are cast back to the
  parameter dtype prior to clipping to keep the fused kernels happy.
- **Torch compile acceleration** – `torch.compile` is enabled by default and can be disabled with
  `--disable-compile` if a run needs to fall back to eager mode.
- **Asynchronous prefetch** – CPU-collected batches are staged onto GPU streams ahead of time so
  tensor-parallel workers stay saturated.
- **Document boundary control** – `--add-document-boundary-tokens` can be specified in either script
  to add BOS/EOS markers even for chat-style tokenisers that need manual resolution.

## Usage
### Prerequisites
- Python 3.9+
- PyTorch with CUDA + NCCL (tested on Ampere/Hopper GPUs)
- Parquet or Arrow shards accessible to every worker

### Starting a new run
```bash
torchrun --nproc_per_node=8 training.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct \
  --data-glob /datasets/cc-main/*.parquet \
  --initial-batch-size 512 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --add-document-boundary-tokens
```
`training.py` will back off the batch size automatically if an out-of-memory error is detected and
persist checkpoints every 350 steps.  Restarting the same command will reuse the discovered batch
size.

### Resuming from a checkpoint
```bash
torchrun --nproc_per_node=8 resume_pretrain_tensor.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct \
  --checkpoint-path pretrained \
  --total-steps 200000 \
  --batch-size 384 \
  --gradient-accumulation-steps 4 \
  --rewarmup-steps 100 \
  --add-document-boundary-tokens
```
`resume_pretrain_tensor.py` automatically discovers the latest `streaming_checkpoint_step_*.pth`
file when a directory is provided.  Optimiser state is intentionally reset and the scheduler applies
an LR re-warmup for stability.

### Key command-line flags
| Flag | Description |
| --- | --- |
| `--data-glob` | Glob for Parquet/Arrow shards.  Both scripts fall back to common defaults if empty. |
| `--tokenizer-path` | Directory containing the Hugging Face tokenizer files. |
| `--initial-batch-size` / `--batch-size` | Micro-batch size per step.  The training script auto-tunes on OOM. |
| `--gradient-accumulation-steps` | Number of micro-batches to accumulate before each optimiser step. |
| `--disable-gradient-checkpointing` | Disable per-block checkpointing (uses more memory, faster per step). |
| `--disable-compile` | Run without `torch.compile` (falls back to eager execution). |
| `--add-document-boundary-tokens` | Add BOS/EOS tokens to every document prior to chunking. |
| `--warmup-steps` / `--rewarmup-steps` | Number of scheduler warmup steps (resume script shortens this after checkpoint loads). |
| `--trust-remote-code` | Allow custom tokenizer implementations. |

Environment overrides:
- `RESUME_TOKENIZER_BATCH_ROWS` and `RESUME_TOKENIZER_WORKERS` tune the threaded tokenizer batcher
  used by both scripts.

## Checkpoints
Checkpoints are stored under `pretrained/` by default and contain:
- `model_state_dict` (tensor-parallel shard friendly)
- Optimiser + scheduler state
- Learning-rate ramp metadata
- `data_position` for exact resume locations
- Training metadata (step, loss, tokens processed, world size, batch size)

Old checkpoints are automatically pruned (keep-last-50) when new ones are written.  Files can be
restored interchangeably between the initial training run and the resume script.

## Repository Layout
```
.
├── README.md
├── data_processing.py
├── model.py
├── resume_pretrain_tensor.py
├── training.py
├── training_utils.py
├── IMPLEMENTATION_NOTES.md
├── TENSOR_PARALLEL_FIX.md
└── TENSOR_PARALLEL_USAGE.md
```

## Further Reading
- `TENSOR_PARALLEL_USAGE.md` – step-by-step launch instructions and troubleshooting tips.
- `IMPLEMENTATION_NOTES.md` – architectural decisions and historical context.

## License
MIT License.  See `LICENSE` for details.

## Acknowledgements
- Argonne National Laboratory
- PyTorch & TorchInductor teams
- NVIDIA NCCL team
