# Argonne LLM Training

LLM.c style training for the Argonne model using Qwen tokenizer and FineWeb data.

## File Structure

```
ArgonneAI/
├── train_llm_c.py       # Main training script (DDP multi-GPU)
├── preprocess_data.py    # Convert parquet data to binary format
├── model.py             # Argonne model architecture
├── run_full_training.sh # SLURM script to launch training
├── preprocess_job.sh    # SLURM script to preprocess data
└── report/              # Training logs (gitignored)
```

## Requirements

- Python with PyTorch
- Transformers library
- Flash Attention
- Qwen tokenizer (e.g., Qwen3-0.6B-Base)

## Quick Start

### 1. Preprocess Data

Convert parquet files to binary format:

```bash
python3 preprocess_data.py \
  --tokenizer_path /path/to/Qwen3-0.6B-Base \
  --data_dir /path/to/fineweb-parquets \
  --output_dir /path/to/output \
  --workers 32
```

Or use the SLURM script:
```bash
sbatch preprocess_job.sh
```

### 2. Train

```bash
torchrun --nproc_per_node=2 train_llm_c.py \
  --tokenizer_path /path/to/Qwen3-0.6B-Base \
  --data_path /path/to/train.bin \
  --checkpoint_dir /path/to/checkpoints \
  --lr 1e-4 \
  --batch_size 20 \
  --total_batch_size 81920 \
  --block_size 1024 \
  --precision bf16 \
  --flash_attention 1 \
  --torch_compile 1 \
  --gradient_checkpointing 1
```

Or use the SLURM script:
```bash
sbatch run_full_training.sh
```

## Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tokenizer_path` | Path to tokenizer | Required |
| `--data_path` | Path to training data (.bin) | Required |
| `--checkpoint_dir` | Directory for checkpoints | Required |
| `--lr` | Learning rate | Required |
| `--batch_size` | Batch size per GPU | Required |
| `--total_batch_size` | Total batch size in tokens | Required |
| `--block_size` | Sequence length | Required |
| `--precision` | Training precision (fp32/fp16/bf16) | bf16 |
| `--flash_attention` | Use flash attention (0/1) | 1 |
| `--torch_compile` | Use torch.compile (0/1) | 0 |
| `--torch_compile_mode` | torch.compile mode (default/reduce-overhead/max-autotune) | default |
| `--ddp_static_graph` | Enable DDP static_graph optimization when available (0/1) | 0 |
| `--ddp_gradient_as_bucket_view` | Enable DDP gradient_as_bucket_view when available (0/1) | 0 |
| `--ddp_broadcast_buffers` | Broadcast DDP buffers every forward pass (0/1) | 1 |
| `--gradient_checkpointing` | Use gradient checkpointing (0/1) | 1 |
| `--resume_from` | Resume from checkpoint file | None |
| `--wall_time` | Wall time in seconds (save checkpoint 3 min before limit) | 0 (disabled) |

## Model Architecture

- **Hidden Size**: 2048
- **Layers**: 16
- **Attention Heads**: 16
- **KV Heads**: 8 (GQA)
- Based on Qwen tokenizer (151936 vocab size)

## Features

- DistributedDataParallel (DDP) for multi-GPU training
- Flash Attention support
- torch.compile for speedup
- Gradient checkpointing for memory efficiency
- BF16/FP16/FP32 precision support
- Automatic checkpointing with wall time limit
- Resume training from checkpoint
