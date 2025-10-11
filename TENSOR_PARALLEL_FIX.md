# Tensor Parallelism Fix Summary

## Issues Fixed

### 1. RuntimeError: Invalid Tensor Shape

**Problem:**
When running with 8 GPUs using tensor parallelism, the code crashed with:
```
RuntimeError: shape '[4, 4096, 24, 170]' is invalid for input of size 8355840
```

This occurred at line 237 in `model.py` in the attention forward pass:
```python
query = query.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
```

**Root Cause:**
The `shard_tensor_parallel()` function was sharding the linear layer weights (dividing them across 8 GPUs), but it wasn't updating the attention module's configuration attributes (`num_heads`, `num_kv_heads`). 

With 8 GPUs:
- Original: 24 heads with head_dim=170, output size = 4 * 4096 * 24 * 170 = 66,846,720
- After sharding: 24/8 = 3 heads per GPU, actual size = 4 * 4096 * 3 * 170 = 8,355,840

The tensor was correctly sharded to 1/8 size, but the `view()` operation was still trying to use `num_heads=24` instead of the sharded `num_heads=3`.

**Solution:**
Updated `shard_tensor_parallel()` in `resume_pretrain_tensor.py` to also update the attention module configuration after sharding:

```python
# Update attention module configurations after sharding
for name, attn_module in module.named_modules():
    if hasattr(attn_module, 'num_heads') and hasattr(attn_module, 'head_dim'):
        original_num_heads = attn_module.num_heads
        original_num_kv_heads = getattr(attn_module, 'num_kv_heads', original_num_heads)
        
        # Shard the heads across GPUs
        attn_module.num_heads = original_num_heads // world_size
        if hasattr(attn_module, 'num_kv_heads'):
            attn_module.num_kv_heads = original_num_kv_heads // world_size
        
        # Update num_key_value_groups if present
        if hasattr(attn_module, 'num_key_value_groups'):
            attn_module.num_key_value_groups = attn_module.num_heads // attn_module.num_kv_heads
```

### 2. Duplicate Print Statements (8x)

**Problem:**
All print statements were appearing 8 times (once per GPU rank) in the console output, making it hard to read.

**Root Cause:**
The code wasn't checking if the current process was the main process (rank 0) before printing.

**Solution:**
Added `is_main_process` checks throughout the codebase:

1. **Updated `init_tensor_parallel_group()`** - Only rank 0 prints initialization message
2. **Updated `TensorParallelModel.__init__()`** - Only rank 0 prints model sharding message  
3. **Updated `streaming_token_generator()`** - Added `rank` parameter and `is_main_process` checks for all prints
4. **Updated `cleanup_old_checkpoints()`** - Only rank 0 performs cleanup and prints
5. **Updated `_resolve_checkpoint_path()`** - Only rank 0 prints auto-selected checkpoint

All functions that needed rank awareness were updated to accept a `rank` parameter and check `is_main_process = (rank == 0)` before printing.

## Files Modified

- `resume_pretrain_tensor.py` - Main file with all the fixes

## Testing

To verify the fixes work, run:
```bash
torchrun --standalone --nproc_per_node=8 resume_pretrain_tensor.py --tokenizer-path ../Qwen2.5-3B-Instruct
```

Expected behavior:
- ✓ No `RuntimeError: shape '[4, 4096, 24, 170]' is invalid for input of size 8355840`
- ✓ Print statements appear only once instead of 8 times
- ✓ Training proceeds normally with tensor parallelism across 8 GPUs

## Technical Details

### Tensor Sharding Math
With 8 GPUs and 24 attention heads:
- Each GPU handles: 24 ÷ 8 = 3 heads
- Each head dimension: 170
- Per-GPU query output: batch_size × seq_len × (3 × 170) = 4 × 4096 × 510

The `view()` operation must use the sharded `num_heads=3` not the original `num_heads=24`.

### Distributed Training Best Practices
When using PyTorch distributed training with multiple processes:
- Always check `rank == 0` before I/O operations (printing, saving, etc.)
- File I/O should typically only happen on rank 0 to avoid race conditions
- Model/optimizer state needs to be consistent across all ranks
- Communication primitives like `all_reduce` synchronize across all ranks
