# Argonne LLM Training - Changes and Results

## Overview
Successfully configured LLM.c-style training for the Argonne model on 1x A100-40GB GPU using Qwen3 tokenizer and FineWeb-edu data.

## Benchmark Results

| Exp | Config | Batch | Grad_Acc | Steps/s | Tokens/s | MFU |
|-----|--------|-------|----------|---------|----------|-----|
| 0 | baseline (no compile) | 20 | 3 | 0.233 | 15,292 | 40.50% |
| 3 | torch_compile | 16 | 4 | 0.375 | 24,576 | 65.10%* |
| 14 | torch_compile | 10 | 6 | 0.400 | 26,214 | 69.50% |

*Note: exp 3 was not reproducible in subsequent runs (OOM errors)

## Key Findings

1. **torch_compile provides significant speedup** (~70% MFU vs 40% baseline)
2. **Memory constraints** on A100-40GB limit batch sizes with torch_compile
3. **Stable configuration**: batch_size=10 with torch_compile works reliably
4. **Baseline stable**: batch_size=20 without torch_compile (40% MFU) is most reliable

## Configuration Files

### run_full_training.sh
- Uses batch_size=10 with torch_compile=1
- Total batch size: 65,536 tokens
- Checkpoint interval: 30 minutes
- Gradient checkpointing: enabled
- Precision: bf16

### train_llm_c.py
- Loads Qwen3-0.6B tokenizer from `/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base`
- Uses FineWeb binary data from `/project/rcc/youzhi/fineweb-binary-qwen3/train.bin`
- Saves checkpoints to `/project/rcc/youzhi/llm.c/checkpoints/`

## Model Architecture
- Vocab: 151,936
- Hidden: 2,048
- Layers: 16
- Heads: 16 (8 KV - GQA)
- Context: 1,024
- Parameters: 1.37B

## Usage

```bash
# Submit training job
cd /home/youzhi/ArgonneAI
sbatch run_full_training.sh

# Check progress
tail -f report/train_1.out
```

## Current Status
- Training can run successfully with torch_compile at batch_size=10
- Achieves ~26k tokens/sec throughput on 1x A100-40GB
- Auto-resumes from latest checkpoint
