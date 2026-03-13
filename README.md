# Argonne LLM Training

LLM.c style training for the Argonne model using Qwen3 tokenizer and FineWeb-Edu data.

## Quick Start

```bash
# Submit training job (1 epoch on full dataset)
sbatch run_full_training.sh
```

Training will:
1. Load pre-tokenized binary data from `/project/rcc/youzhi/fineweb-binary-qwen3/train.bin`
2. Use Qwen3-0.6B-Base tokenizer
3. Train for 1 epoch (~318K steps)
4. Save checkpoints every 30 minutes to `/project/rcc/youzhi/llm.c/checkpoints/`
5. Auto-resume from latest checkpoint on re-run

## Configuration

Edit `run_full_training.sh` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 10 | Micro-batch size per GPU |
| `--total_batch_size` | 65536 | Total tokens per step |
| `--block_size` | 1024 | Sequence length |
| `--num_steps` | 1000000 | Max training steps |
| `--max_epochs` | 1 | Number of epochs |
| `--checkpoint_interval` | 1800 | Checkpoint interval (seconds) |
| `--precision` | bf16 | Training precision |
| `--flash_attention` | 1 | Use Flash Attention |
| `--gradient_checkpointing` | 1 | Enable gradient checkpointing |
| `--torch_compile` | 1 | Use torch.compile for speedup |

## Training Output

- Logs saved to `report/train_N.out` and `report/train_N.err` (N = run number)
- Checkpoints saved to `/project/rcc/youzhi/llm.c/checkpoints/checkpoint_step_*.pt`

## Data Format

The training script expects binary token files in llm.c format:
- Magic number: 20240801 (uint32)
- Header: 256 * 4 bytes (int32 metadata)
- Tokens: uint32 array

Preprocessed data is available at:
`/project/rcc/youzhi/fineweb-binary-qwen3/train.bin` (~83GB, ~20.8B tokens)

## Model Architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | ~1.37B |
| **Layers** | 16 transformer blocks |
| **Hidden Size** | 2,048 |
| **Attention Heads** | 16 query / 8 key-value (GQA) |
| **Vocabulary** | 151,936 (Qwen3 tokenizer) |

## Files

```
├── model.py              # Argonne model architecture
├── train_llm_c.py        # LLM.c style training script
├── run_full_training.sh  # Slurm batch script for full training
└── report/               # Training logs
```

## Requirements

- Python with PyTorch, transformers, tqdm
- NVIDIA GPU with CUDA
- Qwen3-0.6B-Base tokenizer (auto-downloaded from HuggingFace)
