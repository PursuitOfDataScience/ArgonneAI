import os
import math
import json
import time
import torch
import random
import argparse
import deepspeed
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from datetime import datetime

# Import your existing modules
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data
from model import ArgonneConfig, ArgonneModel
from datasets import Dataset

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_deepspeed_environment():
    """Set up the distributed environment for DeepSpeed"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    if local_rank == -1 or world_size == -1:
        print("Warning: LOCAL_RANK or WORLD_SIZE not found in environment. Using default values.")
        local_rank = 0
        world_size = 1
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

# Rest of the helper functions remain the same...
def streaming_token_generator(data_files, tokenizer, start_file_idx=0, start_position=0):
    # ... existing implementation ...
    # Implementation omitted for brevity
    pass

class DataPosition:
    # ... existing implementation ...
    # Implementation omitted for brevity
    pass

def set_seed(seed):
    # ... existing implementation ...
    # Implementation omitted for brevity
    pass
    
def train_with_deepspeed(
    data_path, 
    config_path,
    output_dir="./output",
    block_size=2048,
    lr=3e-5,
    use_streaming=False,
    max_steps=100000,
    save_steps=1000,
    warmup_steps=1000,
    seed=42,
    batch_size=32,
    zero_stage=3,
    local_rank=-1  # Add local_rank parameter
):
    """
    Train the model using DeepSpeed with ZeRO optimization.
    
    Args:
        data_path: Path to data files (glob pattern or list)
        config_path: Path to DeepSpeed config JSON
        output_dir: Directory to save model checkpoints
        block_size: Maximum sequence length
        lr: Learning rate
        use_streaming: Whether to stream data or load it all
        max_steps: Maximum number of training steps
        save_steps: Save checkpoint every N steps
        warmup_steps: Learning rate warmup steps
        seed: Random seed for reproducibility
        batch_size: Per-GPU batch size
        zero_stage: DeepSpeed ZeRO stage (1, 2, or 3)
        local_rank: Local rank for distributed training
    """
    # Setup global seed for reproducibility
    set_seed(seed)
    
    # Use provided local_rank if available, otherwise set up environment
    if local_rank >= 0:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_main_process = (local_rank == 0)
    else:
        local_rank, world_size = setup_deepspeed_environment()
        is_main_process = (local_rank == 0)
    
    # Print information about the environment
    if is_main_process:
        print(f"=== DeepSpeed Training ===")
        print(f"World Size: {world_size}")
        print(f"Local Rank: {local_rank}")
        print(f"Using ZeRO Stage: {zero_stage}")
        print(f"Output Directory: {output_dir}")
        
    # Create output directories
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        
    # Expand glob pattern for data files if needed
    if isinstance(data_path, str) and ('*' in data_path or '?' in data_path):
        import glob
        data_files = glob.glob(data_path)
        
        # Sort files numerically if they follow a pattern like train-00123-of-00456
        import re
        def get_file_number(filename):
            match = re.search(r'train-(\d+)-of', filename)
            if match:
                return int(match.group(1))
            return 0
            
        data_files = sorted(data_files, key=get_file_number)
    else:
        data_files = [data_path] if isinstance(data_path, str) else data_path
        
    if is_main_process:
        print(f"Found {len(data_files)} data files")
        
    # Load tokenizer
    hf_tokenizer = load_bpe_tokenizer()
    
    # Initialize the model configuration
    config = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=16,
        n_head=16,
        n_embd=1296,
        dropout=0.1
    )
    
    # Create the model
    model = ArgonneModel(config)
    
    # Load DeepSpeed config
    with open(config_path, 'r') as f:
        ds_config = json.load(f)
        
    # Update DeepSpeed config with command line arguments
    ds_config['zero_optimization']['stage'] = zero_stage
    
    # Set learning rate and batch size in config
    ds_config['optimizer']['params']['lr'] = lr
    
    # CRITICAL FIX: Convert ALL "auto" strings to actual integers
    # This prevents the TypeError: '>' not supported between instances of 'str' and 'int'
    if ds_config['train_micro_batch_size_per_gpu'] == 'auto':
        ds_config['train_micro_batch_size_per_gpu'] = batch_size
    
    if ds_config['train_batch_size'] == 'auto':
        # Calculate global batch size based on world size
        ds_config['train_batch_size'] = batch_size * world_size
    
    if ds_config['gradient_accumulation_steps'] == 'auto':
        ds_config['gradient_accumulation_steps'] = 1
    
    if is_main_process:
        print(f"DeepSpeed configuration:")
        print(f"  Micro batch size: {ds_config['train_micro_batch_size_per_gpu']}")
        print(f"  Train batch size: {ds_config['train_batch_size']}")
        print(f"  Gradient accumulation steps: {ds_config['gradient_accumulation_steps']}")
    
    # Initialize DeepSpeed
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=ds_config
    )
    
    # Rest of the function remains the same...
    # Implementation omitted for brevity
    
def main():
    parser = argparse.ArgumentParser(description="Train with DeepSpeed")
    parser.add_argument("--data", type=str, required=True, help="Path to data files (glob pattern)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--config", type=str, default="ds_config.json", help="DeepSpeed config")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every N steps")
    parser.add_argument("--streaming", action="store_true", help="Use streaming data")
    parser.add_argument("--zero_stage", type=int, default=3, choices=[1, 2, 3], help="DeepSpeed ZeRO stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Add local_rank argument for DeepSpeed compatibility
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    train_with_deepspeed(
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output_dir,
        lr=args.lr,
        use_streaming=args.streaming,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        zero_stage=args.zero_stage,
        local_rank=args.local_rank  # Pass local_rank to the function
    )

if __name__ == "__main__":
    main()
