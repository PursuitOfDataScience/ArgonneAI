# Tensor Parallelism Implementation Notes

## Overview

This document provides technical implementation details for the tensor parallelism support added to the ArgonneAI training infrastructure via `resume_pretrain_tensor.py`.

## Architecture

### Design Goals

1. **Mirror Original Logic**: Maintain exact workflow and structure of `resume_pretrain.py`
2. **Checkpoint Compatibility**: Support resuming from pipeline-parallel checkpoints
3. **Transparent Distribution**: Abstract GPU distribution from training loop
4. **Minimal Code Duplication**: Reuse existing utilities and data processing

### Key Components

#### 1. TensorParallelModel Class

Wraps the base `ArgonneModel` with tensor parallelism capabilities:

```python
class TensorParallelModel(torch.nn.Module):
    def __init__(self, base_model: ArgonneModel, world_size: int, rank: int)
    def forward(self, input_ids, labels=None, attention_mask=None)
```

**Responsibilities:**
- Shard linear layers across GPUs during initialization
- Manage device placement for each rank
- Insert all-reduce synchronization points
- Provide compatible interface with base model

#### 2. Tensor Sharding Function

```python
def shard_tensor_parallel(module, world_size: int, rank: int) -> None
```

**Strategy:**
- **Column-parallel layers** (split output features):
  - Query, Key, Value projections: Each GPU computes subset of attention heads
  - Gate and Up projections: Each GPU computes subset of intermediate features
  - Formula: `out_features_per_gpu = total_out_features // world_size`

- **Row-parallel layers** (split input features):
  - Output projection: Combines partial results from all heads
  - Down projection: Combines partial results from intermediate layer
  - Formula: `in_features_per_gpu = total_in_features // world_size`
  - Bias only on rank 0 (to avoid duplication in sum)

#### 3. Communication Pattern

```
Forward Pass:
┌─────────────────────────────────────────┐
│ Embeddings (replicated)                 │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ For each transformer block:             │
│  1. Attention:                           │
│     - Input norm (replicated)           │
│     - Q,K,V proj (column-parallel)      │
│     - Attention computation (sharded)   │
│     - Output proj (row-parallel)        │
│     - all_reduce() ← synchronization    │
│  2. MLP:                                 │
│     - Post norm (replicated)            │
│     - Gate/Up proj (column-parallel)    │
│     - SwiGLU activation (sharded)       │
│     - Down proj (row-parallel)          │
│     - all_reduce() ← synchronization    │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Final norm + LM head (replicated)       │
└─────────────────────────────────────────┘
```

## Implementation Details

### Distributed Initialization

```python
def init_tensor_parallel_group(world_size: int, rank: int) -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
```

**Environment Variables Required:**
- `WORLD_SIZE`: Total number of processes
- `RANK`: Global rank (0 to WORLD_SIZE-1)
- `LOCAL_RANK`: Local rank on this node
- `MASTER_ADDR`: Address of rank 0
- `MASTER_PORT`: Port for communication

### Layer Sharding Logic

**Column-Parallel Example (Q Projection):**
```python
# Original: [batch, seq, hidden] → [batch, seq, num_heads * head_dim]
# Sharded:  [batch, seq, hidden] → [batch, seq, num_heads_per_gpu * head_dim]

out_features = layer.out_features  # num_heads * head_dim
chunk_size = out_features // world_size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size

new_layer = nn.Linear(layer.in_features, chunk_size)
new_layer.weight.data = layer.weight.data[start_idx:end_idx].clone()
```

**Row-Parallel Example (Output Projection):**
```python
# Original: [batch, seq, num_heads * head_dim] → [batch, seq, hidden]
# Sharded:  [batch, seq, num_heads_per_gpu * head_dim] → [batch, seq, hidden]
# Then:     all_reduce([batch, seq, hidden]) across all GPUs

in_features = layer.in_features  # num_heads * head_dim
chunk_size = in_features // world_size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size

new_layer = nn.Linear(chunk_size, layer.out_features)
new_layer.weight.data = layer.weight.data[:, start_idx:end_idx].clone()
```

### Synchronization Points

All-reduce operations are inserted after row-parallel layers:

```python
# After attention output projection
attn_output = block.attn(normed, position_embeddings, attention_mask)
if self.world_size > 1:
    dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)

# After MLP down projection
mlp_output = block.mlp(normed)
if self.world_size > 1:
    dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM)
```

### Checkpoint Handling

**Loading from Pipeline-Parallel Checkpoint:**
1. Load full model state dict on CPU
2. Initialize base ArgonneModel
3. Load state dict into base model
4. Wrap with TensorParallelModel (triggers sharding)
5. Each rank keeps only its shard in GPU memory

**Saving Tensor-Parallel Checkpoint:**
1. Each rank saves its local state dict
2. Only rank 0 performs the actual save
3. State dict represents sharded parameters
4. Can be loaded back into sharded or full model

## Performance Characteristics

### Communication Analysis

For a model with:
- `L` layers
- `H` hidden size
- `B` batch size
- `S` sequence length
- `P` GPUs

**All-reduce volume per training step:**
- 2 all-reduces per transformer block (attention + MLP)
- Each all-reduce: `B × S × H` elements
- Total: `2 × L × B × S × H × sizeof(dtype)` bytes

**Communication vs Computation Ratio:**
- Computation: O(B × S × H² × L)
- Communication: O(B × S × H × L × P)
- Ratio improves with larger hidden size

### Memory Distribution

Each GPU holds:
```
Embeddings:        vocab_size × H
Per-layer weights: ~(4 × H² / P) for attention + (3 × H × intermediate / P) for MLP
Activations:       B × S × H
Gradients:         Same as weights
Optimizer states:  2× gradients (Adam: momentum + variance)
```

### Scaling Considerations

**Strong Scaling (fixed model size, more GPUs):**
- Communication overhead increases
- Computation per GPU decreases
- Sweet spot typically at 4-8 GPUs per node

**Weak Scaling (larger model, more GPUs):**
- Maintains computation/communication ratio
- Memory per GPU stays constant
- Can scale to very large models

## Comparison with Pipeline Parallelism

| Aspect | Tensor Parallel | Pipeline Parallel |
|--------|----------------|-------------------|
| **Memory** | Balanced across GPUs | Imbalanced (first/last heavy) |
| **Latency** | Low (all GPUs busy) | High (sequential stages) |
| **Communication** | High (all-reduce) | Low (point-to-point) |
| **Utilization** | High (synchronous) | Medium (bubble time) |
| **Complexity** | Medium | Low |

## Future Enhancements

Potential improvements for the tensor parallelism implementation:

1. **Sequence Parallelism**: Shard sequence dimension in addition to layer weights
2. **Expert Parallelism**: For mixture-of-experts models
3. **ZeRO Integration**: Combine with ZeRO optimizer sharding
4. **Overlapped Communication**: Pipeline all-reduce with computation
5. **Dynamic Sharding**: Adjust parallelism degree per layer

## Testing Checklist

When validating the implementation:

- [ ] Single-GPU execution (should work as baseline)
- [ ] Multi-GPU execution (2, 4, 8 GPUs)
- [ ] Resume from pipeline-parallel checkpoint
- [ ] Resume from tensor-parallel checkpoint
- [ ] Loss convergence matches pipeline-parallel
- [ ] Memory usage balanced across GPUs
- [ ] Communication efficiency (via NCCL_DEBUG)
- [ ] Gradient correctness (compare with single-GPU)
- [ ] Generation quality matches base model

## References

1. **Megatron-LM**: https://arxiv.org/abs/1909.08053
   - Original tensor parallelism paper for transformers
   
2. **GPipe**: https://arxiv.org/abs/1811.06965
   - Pipeline parallelism comparison
   
3. **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html
   - API reference for distributed primitives
   
4. **NCCL**: https://docs.nvidia.com/deeplearning/nccl/
   - Understanding collective communication

## Troubleshooting

### Common Issues

**1. NCCL Timeout Errors**
```bash
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_IB_TIMEOUT=50
```

**2. Gradient Mismatch**
- Verify all-reduce placements
- Check bias handling in row-parallel layers
- Ensure proper gradient scaling

**3. Memory Imbalance**
- Check embedding replication
- Verify shard sizes are equal
- Monitor with `nvidia-smi` during training

**4. Slow Training**
- Verify NVLink connectivity (`nvidia-smi topo -m`)
- Check NCCL algorithm selection (`NCCL_ALGO=Ring`)
- Profile with `nsys` or `torch.profiler`

## Conclusion

The tensor parallelism implementation provides an efficient alternative to pipeline parallelism for the ArgonneAI training infrastructure. It maintains full compatibility with existing checkpoints while offering better GPU utilization and memory balance in single-node configurations.
