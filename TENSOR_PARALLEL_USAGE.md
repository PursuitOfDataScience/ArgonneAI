# Tensor Parallelism Training Guide

## Overview

The `resume_pretrain_tensor.py` script implements **tensor parallelism** for distributed training of the Argonne language model. This parallelization strategy shards individual layers across multiple GPUs, allowing all GPUs to work on the same layer simultaneously.

## Quick Start

### Basic Usage

```bash
# Launch with 4 GPUs
torchrun --standalone --nproc_per_node=4 resume_pretrain_tensor.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct \
  --checkpoint-path pretrained/streaming_checkpoint_step_1000.pth
```

### Full Example with All Options

```bash
torchrun --standalone --nproc_per_node=8 resume_pretrain_tensor.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct \
  --checkpoint-path pretrained/streaming_checkpoint_step_5000.pth \
  --data-glob "../data/CC-MAIN-2025-26/*.parquet" \
  --batch-size 4 \
  --block-size 4096 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 2000 \
  --total-steps 160000 \
  --weight-decay 0.1
```

## How Tensor Parallelism Works

### Layer Sharding Strategy

1. **Column-Parallel Layers** (split output features):
   - Query, Key, Value projections in attention
   - Gate and Up projections in MLP
   - Each GPU computes a subset of output features

2. **Row-Parallel Layers** (split input features):
   - Output projection in attention
   - Down projection in MLP
   - Each GPU computes partial results that are summed via all-reduce

### Communication Pattern

```
GPU 0: Compute shard 0 → All-Reduce → Combined Result
GPU 1: Compute shard 1 → All-Reduce → Combined Result
GPU 2: Compute shard 2 → All-Reduce → Combined Result
GPU 3: Compute shard 3 → All-Reduce → Combined Result
```

Each layer uses synchronous all-reduce operations to combine partial results from all GPUs.

## Resuming from Existing Checkpoints

The tensor parallel script can resume from checkpoints created by:
- `resume_pretrain.py` (pipeline parallel)
- `training.py` (pipeline parallel)
- Other tensor parallel checkpoints

The script automatically:
1. Converts checkpoint state to match the current model
2. Restores data position tracking
3. Loads optimizer and scheduler states
4. Continues from the exact training step

## Performance Considerations

### When to Use Tensor Parallelism

**✓ Best for:**
- Models that fit in memory when sharded across GPUs
- High-bandwidth interconnects (NVLink, InfiniBand)
- Maximizing GPU utilization on single nodes

**✗ Not ideal for:**
- Models requiring extreme memory reduction
- Low-bandwidth network connections
- Multi-node setups with slow interconnects

### Expected Performance

These are approximate values that vary based on model size, batch size, and hardware configuration:

- **Communication Overhead:** ~10-20% (typical) due to all-reduce operations after each sharded layer
- **Scaling Efficiency:** 80-90% (typical on NVLink-connected GPUs like A100/H100 in single-node configurations)
- **Memory Usage:** More balanced across GPUs compared to pipeline parallelism

Performance characteristics depend heavily on:
- Hardware interconnect bandwidth (NVLink >> PCIe)
- Model architecture (number of layers, hidden size)
- Batch size and sequence length
- GPU compute capability

## Environment Variables

### Required for Distributed Training

```bash
# These are automatically set by torchrun
export WORLD_SIZE=4        # Total number of GPUs
export RANK=0              # Global rank (0 to WORLD_SIZE-1)
export LOCAL_RANK=0        # Local rank on this node
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

### Manual Launch (without torchrun)

```bash
# On each GPU, set these manually
CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 python resume_pretrain_tensor.py ...
CUDA_VISIBLE_DEVICES=1 RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 python resume_pretrain_tensor.py ...
CUDA_VISIBLE_DEVICES=2 RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 python resume_pretrain_tensor.py ...
CUDA_VISIBLE_DEVICES=3 RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 python resume_pretrain_tensor.py ...
```

## Command-Line Arguments

### Required Arguments
- `--tokenizer-path`: Path to the tokenizer directory (e.g., `../Qwen2.5-3B-Instruct`)

### Training Configuration
- `--batch-size`: Batch size per GPU (default: 4)
- `--block-size`: Sequence length (default: 4096)
- `--learning-rate`: Peak learning rate (default: 1e-4)
- `--min-learning-rate`: Minimum learning rate (default: 1e-5)
- `--warmup-steps`: Number of warmup steps (default: 2000)
- `--total-steps`: Total training steps (default: 4,000,000)
- `--weight-decay`: Weight decay coefficient (default: 0.1)

### Data Configuration
- `--data-glob`: Glob pattern for data files (default: `../data/CC-MAIN-2025-26/*.parquet`)
- `--no-streaming`: Load all data into memory (default: use streaming)
- `--num-proc`: Number of processes for data loading (default: 8)

### Checkpoint Configuration
- `--checkpoint-path`: Path to checkpoint file or directory (auto-selects latest if directory)

### Other Options
- `--trust-remote-code`: Allow loading tokenizers with custom code
- `--local_rank`: Local rank for distributed training (set by torchrun)
- `--teacher-device-map`: For distillation runs, prefer `local` (default) to pin the teacher to each rank's GPU
  instead of letting every process attempt an `auto` device map across all GPUs.

## Monitoring Training

### Progress Output

```
Step 1000 | Loss: 2.3456 | Tokens: 1,234,567,890
Current LR: 2.50e-04
Shard: 000_00042.parquet | File index: 42 | Position: 1234 | Chunk: 5

--- Generated text at step 1000 ---
Long long time ago, there was a kingdom...
```

### Checkpoints

Checkpoints are saved every 300 steps (streaming) or 2000 steps (non-streaming) to:
```
pretrained/tensor_parallel_checkpoint_step_XXX.pth
```

### Training Statistics

JSON files with training metrics are saved to:
```
stats/current_training_stats_tensor_parallel_step_XXX.json
stats/final_training_stats_tensor_parallel_TIMESTAMP.json
```

## Troubleshooting

### Out of Memory Errors

1. **Reduce batch size:**
   ```bash
   --batch-size 2
   ```

2. **Reduce sequence length:**
   ```bash
   --block-size 2048
   ```

3. **Use more GPUs:**
   ```bash
   --nproc_per_node=8  # instead of 4
   ```

### NCCL Errors

If you see NCCL communication errors:

1. **Check GPU connectivity:**
   ```bash
   nvidia-smi topo -m
   ```

2. **Set NCCL debug:**
   ```bash
   export NCCL_DEBUG=INFO
   ```

3. **Increase timeout:**
   ```bash
   export NCCL_SOCKET_TIMEOUT=3600
   ```

### Slow Training

1. **Verify NVLink usage:**
   - Tensor parallelism requires high-bandwidth interconnects
   - Check `nvidia-smi topo -m` for NVLink paths

2. **Profile communication:**
   ```bash
   export NCCL_DEBUG=INFO
   # Look for high latency in all-reduce operations
   ```

3. **Consider alternatives:**
   - Use pipeline parallelism for slower interconnects
   - Use FSDP for data parallelism across many nodes

## Comparison with Other Parallelism Strategies

| Strategy | Best Use Case | Memory Per GPU | Communication |
|----------|---------------|----------------|---------------|
| **Tensor Parallel** | Single node, high bandwidth | Medium | High (all-reduce) |
| **Pipeline Parallel** | Very large models | Low (staged) | Low (point-to-point) |
| **Data Parallel (FSDP)** | Multi-node scaling | Low (sharded) | Medium (gradient sync) |

## Advanced: Multi-Node Tensor Parallelism

For multi-node training, set up the environment variables and use `torchrun`:

### Node 0 (Master)
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  resume_pretrain_tensor.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct
```

### Node 1 (Worker)
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  resume_pretrain_tensor.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct
```

## Implementation Details

### Key Classes

- **`TensorParallelModel`**: Wrapper around `ArgonneModel` that implements tensor sharding
- **`DataPosition`**: Tracks dataset position for seamless resumption
- **`shard_tensor_parallel()`**: Shards linear layers across GPUs

### Synchronization Points

1. **After attention output projection:** `dist.all_reduce(attn_output)`
2. **After MLP down projection:** `dist.all_reduce(mlp_output)`
3. **During checkpoint saves:** `dist.barrier()` ensures all ranks complete before saving

### Checkpoint Compatibility

The script saves checkpoints that are **compatible** with:
- Standard `ArgonneModel` inference (after gathering shards)
- Other training scripts (after state dict conversion)
- HuggingFace `save_pretrained()` format (final model only)

## References

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- Megatron-LM Tensor Parallelism: https://github.com/NVIDIA/Megatron-LM
- Efficient Large-Scale Training: https://arxiv.org/abs/1909.08053
