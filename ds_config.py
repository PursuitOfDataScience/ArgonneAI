import os
import json
import torch

def get_ds_config(batch_size=8, grad_accum=2, zero_stage=2, offload=False):
    """
    Optimized DeepSpeed configuration with automatic resource detection
    
    Args:
        batch_size: Per-GPU batch size (8 recommended for better memory usage)
        grad_accum: Gradient accumulation steps (helps increase effective batch size)
        zero_stage: ZeRO optimization stage (2 recommended for large-scale training)
        offload: Whether to offload optimizer states to CPU
        
    Returns:
        Dictionary containing optimized DeepSpeed configuration
    """
    # Auto-detect available resources
    # NOTE: DeepSpeed will actually handle this at runtime based on hostfile
    # We're just estimating for config generation
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", local_size))
    
    # For batch size calculation, use conservative estimate
    # Actual parallelism will be determined by DeepSpeed at runtime
    if world_size == 1:  # Single node fallback detection
        try:
            import torch.xpu
            local_gpus = torch.xpu.device_count()
            print(f"Auto-detected {local_gpus} Intel XPUs on this node")
        except (ImportError, AttributeError):
            try:
                local_gpus = torch.cuda.device_count()
                print(f"Auto-detected {local_gpus} NVIDIA GPUs on this node")
            except (ImportError, AttributeError):
                local_gpus = 1
                print("Could not detect GPUs, assuming single device")
        
        world_size = local_gpus
    
    print(f"Configuring for estimated world_size = {world_size}")
    
    # Calculate effective global batch size
    global_batch_size = batch_size * world_size * grad_accum
    
    # Initial learning rate based on batch size - square root scaling
    base_lr = 6e-4  # Base learning rate for batch size 2048
    scaled_lr = base_lr * (global_batch_size / 2048)**0.5
    
    # The train_batch_size will be automatically adjusted by DeepSpeed
    # based on the actual number of GPUs at runtime
    config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        
        # Optimizer settings with warmup and decay
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": scaled_lr,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8
            }
        },
        
        # Learning rate scheduler
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": scaled_lr,
                "warmup_num_steps": 2000,
                "total_num_steps": 100000
            }
        },
        
        # Mixed precision settings optimized for Intel XPUs
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 12,  # More conservative for stability
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        # ZeRO optimization - basic parameters that work for all stages
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,  # Adjusted for Intel network
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,     # Adjusted for Intel network
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        
        # Intel XPU specific settings
        "xpu": {
            "enabled": True,
            "memory_efficient_self_attention": True,
            "memory_efficient_linear": True,  # Enable memory-efficient linear ops
        },
        
        # Communication optimization
        "communication_data_type": "fp16",  # Reduces communication overhead
        
        # Performance monitoring
        "steps_per_print": 100,
        "wall_clock_breakdown": False,  # Setting to False to reduce overhead
        "timer_detailed_breakdown": False,  # Setting to False to reduce overhead
        
        # Additional performance settings
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
        },

        # Simplified profiler settings
        "flops_profiler": {
            "enabled": False,  # Disable by default for performance
        },
    }
    
    # Add stage3-specific parameters only when using stage 3
    if zero_stage == 3:
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6
    
    # Add CPU offloading if requested
    if offload and zero_stage > 0:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True
        }
    
    return config

def save_ds_config(filename="ds_config_optimized.json", **kwargs):
    """Save optimized DeepSpeed config to a file"""
    config = get_ds_config(**kwargs)
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)
    return filename

# Add aliases for backward compatibility
get_optimized_ds_config = get_ds_config
save_optimized_ds_config = save_ds_config

if __name__ == "__main__":
    # Generate a default optimized config
    save_ds_config()
    print(f"Created optimized DeepSpeed config file")
