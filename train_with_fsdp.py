import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import time
import glob
import json
import argparse
import functools
import warnings
import logging

# Silence PyTorch FX symbolic shape warnings
warnings.filterwarnings("ignore", message=".*xindex is not in var_ranges, defaulting to unknown range.*")
os.environ["TORCH_FX_WARN_ONCE"] = "1"  # Only show warnings once

# More compatible way to handle torch logging
try:
    # For newer PyTorch versions
    import torch._logging
    try:
        torch._logging.set_logs(fx=logging.WARNING)
    except TypeError:
        # For PyTorch versions without this parameter
        logging.getLogger("torch.fx").setLevel(logging.WARNING)
except (ImportError, AttributeError):
    # Fallback for older versions
    logging.getLogger("torch").setLevel(logging.WARNING)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from datasets import Dataset
from data_processing import collate_batch, load_bpe_tokenizer, train_bpe_tokenizer, load_nonstream_data, create_text_file_from_arrow
from model import ArgonneConfig, Block, CausalSelfAttention, ArgonneModel

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Simple data tracking class to keep position information
class DataPosition:
    def __init__(self, streaming=True):
        self.streaming = streaming
        self.current_file_idx = 0
        self.position_in_file = 0
        self.current_filename = None
        self.global_step = 0
    
    def update_streaming_position(self, file_idx, position, filename=None):
        self.current_file_idx = file_idx
        self.position_in_file = position
        if filename:
            self.current_filename = filename
    
    def reset_for_new_pass(self):
        self.current_file_idx = 0
        self.position_in_file = 0
    
    def get_state(self):
        return {
            "file_idx": self.current_file_idx,
            "position": self.position_in_file,
            "filename": self.current_filename,
            "global_step": self.global_step,
            "streaming": self.streaming
        }
    
    def restore_state(self, state_dict):
        self.current_file_idx = state_dict.get("file_idx", 0)
        self.position_in_file = state_dict.get("position", 0)
        self.current_filename = state_dict.get("filename", None)
        self.global_step = state_dict.get("global_step", 0)
        self.streaming = state_dict.get("streaming", True)

# Updated streaming token generator to use datasets library
def streaming_token_generator(data_files, tokenizer, start_file_idx=0, start_position=0):
    """
    Generate tokens from Arrow files in a streaming fashion.
    
    Args:
        data_files: List of Arrow file paths
        tokenizer: BPE tokenizer
        start_file_idx: Index of file to start from
        start_position: Position within starting file
        
    Returns:
        Generator yielding (tokens, file_idx, position) tuples
    """
    # Ensure we have valid file index
    if start_file_idx >= len(data_files):
        start_file_idx = 0
        start_position = 0
    
    # Process files sequentially
    for file_idx in range(start_file_idx, len(data_files)):
        try:
            # Load current file
            current_file = data_files[file_idx]
            dataset = Dataset.from_file(current_file)
            
            # Start from provided position in first file
            position = start_position if file_idx == start_file_idx else 0
            
            # Reset start position after first file
            if file_idx == start_file_idx:
                start_position = 0
            
            # Process all samples in the dataset
            while position < len(dataset):
                try:
                    # Get text from dataset
                    sample = dataset[position]
                    if 'text' not in sample:
                        position += 1
                        continue
                        
                    text = sample['text']
                    if not text or not isinstance(text, str):
                        position += 1
                        continue
                    
                    # Tokenize text
                    tokens = tokenizer.encode(text)
                    if not tokens or len(tokens) < 3:  # Skip very short sequences
                        position += 1
                        continue
                        
                    # Return tokens and position info
                    yield tokens, file_idx, position
                    position += 1
                    
                except IndexError:
                    # End of dataset
                    break
                except Exception as e:
                    print(f"Error processing sample at position {position} in file {file_idx}: {e}")
                    position += 1
                    continue
        except Exception as e:
            print(f"Error opening file {data_files[file_idx]}: {e}")
            continue
    
    # If we've gone through all files, signal completion
    yield [], -1, -1

def setup_fsdp(rank=0, world_size=1):
    """Initialize distributed training environment"""
    # Set environment variables if not already set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def setup_mixed_precision():
    """Setup mixed precision for FSDP"""
    return MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

def train_with_fsdp(
    rank, 
    world_size, 
    data_files, 
    use_streaming=True,
    use_compile=True,
    resume_from_checkpoint=None
):
    """
    Train model using PyTorch FSDP to efficiently utilize all available GPU resources.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        data_files: List of .arrow file paths
        use_streaming: Whether to use streaming mode or load all data in memory
        use_compile: Whether to use torch.compile() for model optimization
        resume_from_checkpoint: Path to checkpoint for resuming training
    """
    # Initialize the distributed environment
    setup_fsdp(rank, world_size)
    
    # Set random seed for reproducibility
    torch.manual_seed(42 + rank)
    
    # Log info from rank 0 only
    is_main_process = (rank == 0)
    
    if is_main_process:
        print(f"Training with FSDP using {world_size} GPUs")
        print(f"Using device: {torch.cuda.get_device_name()}")
        
    # Calculate batch size based on number of GPUs
    per_gpu_batch_size = 64  
    batch_size = per_gpu_batch_size * world_size
    min_batch_size = 8 * world_size
    
    if is_main_process:
        print(f"Starting with batch_size={batch_size} across {world_size} GPUs")
        print(f"Per-GPU batch size: {per_gpu_batch_size}")
    
    # Load or create tokenizer (only on main process to avoid conflicts)
    if is_main_process and not os.path.exists("bpe_tokenizer/vocab.json"):
        print("No existing tokenizer found. Creating one from Arrow files...")
        text_file_path = "all_text_for_tokenizer.txt"
        create_text_file_from_arrow(data_files, text_file_path)
        train_bpe_tokenizer(text_file_path, vocab_size=12000)
    
    # Wait for main process to complete tokenizer creation
    dist.barrier()
    
    # Now all processes load the tokenizer
    tokenizer = load_bpe_tokenizer()
    
    # Define model config
    model_config = ArgonneConfig(
        vocab_size=12000,
        block_size=2048,
        n_layer=16,
        n_head=16,
        n_embd=1296,
        dropout=0.1
    )
    
    # Load non-streaming data if needed (could be sharded in future)
    tokenized_data = None
    total_samples = 0
    if not use_streaming:
        if is_main_process:
            print("Loading complete dataset into memory...")
        
        # Simple sharding for non-streaming data loading
        tokenized_data = load_nonstream_data(
            data_files, 
            tokenizer, 
            model_config.block_size, 
            num_proc=max(1, min(8, world_size))
        )
        total_samples = len(tokenized_data)
        
        if is_main_process:
            print(f"Loaded {total_samples} tokenized samples in memory")
    
    # Initialize tracking variables
    data_position = DataPosition(streaming=use_streaming)
    total_tokens_processed = 0
    global_step = 0
    
    # For synchronizing between processes
    dist.barrier()
    
    # Resume from checkpoint if provided
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        if is_main_process:
            print(f"Loading checkpoint from {resume_from_checkpoint}")
        
        checkpoint = torch.load(resume_from_checkpoint, map_location=f"cuda:{rank}")
        
        if "data_position" in checkpoint:
            data_position.restore_state(checkpoint["data_position"])
            if is_main_process:
                print(f"Resuming from file {data_position.current_file_idx}, position {data_position.position_in_file}")
        
        total_tokens_processed = checkpoint.get("tokens_processed", 0)
        global_step = checkpoint.get("global_step", 0)
        data_position.global_step = global_step
        
        if is_main_process:
            print(f"Resuming from global_step={global_step}, tokens_processed={total_tokens_processed:,}")
    
    # Setup auto wrap policy for FSDP
    # Define the transformer layer modules to wrap
    transformer_layer_cls = {
        Block,  # The main transformer block in our model
        CausalSelfAttention,  # The self-attention layer
    }
    
    # Use transformer policy which is designed for transformer architectures
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
    
    # Main training loop with automatic batch size adjustment
    while True:
        try:
            if is_main_process:
                print(f"\n=== Training with batch_size={batch_size} ===")
            
            # Create fresh model
            model = ArgonneModel(model_config)
            
            # Move model to current GPU
            model = model.to(rank)
            
            # Load model state if resuming
            if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
                model_state_dict = checkpoint["model_state_dict"]
                
                # Handle compiled model state dict
                if any(k.startswith("_orig_mod.") for k in model_state_dict.keys()):
                    if is_main_process:
                        print("Converting compiled model state dict format...")
                    new_state_dict = {}
                    for k, v in model_state_dict.items():
                        if k.startswith("_orig_mod."):
                            new_key = k.replace("_orig_mod.", "")
                            new_state_dict[new_key] = v
                        else:
                            new_state_dict[k] = v
                    model_state_dict = new_state_dict
                
                # Load state - need to modify for FSDP state loading
                model.load_state_dict(model_state_dict)
            
            # Set up mixed precision
            mixed_precision = setup_mixed_precision()
            
            # Wrap model with FSDP
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=False),
                use_orig_params=True  # Required for torch.compile compatibility
            )
            
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
            
            # Apply torch.compile if available and requested
            if use_compile and hasattr(torch, 'compile'):
                try:
                    if is_main_process:
                        print("Applying torch.compile() for optimization...")
                    model = torch.compile(model)
                    if is_main_process:
                        print("Model successfully compiled")
                except Exception as e:
                    if is_main_process:
                        print(f"torch.compile() failed with error: {e}")
                        print("Continuing with standard model")
            
            # Training settings
            epochs = 3
            steps_per_epoch = 50000 // world_size  
            tokens_in_current_attempt = 0
            
            # Ensure all processes are ready before starting
            dist.barrier()
            
            # Start training
            for epoch in range(epochs):
                # Streaming mode
                if use_streaming:
                    if is_main_process:
                        print(f"==== STREAMING Epoch {epoch} with batch_size={batch_size} ====")
                    
                    # For streaming, each rank processes different batches
                    # Use data_position for rank 0, then other ranks continue
                    file_idx = data_position.current_file_idx
                    position = data_position.position_in_file
                    
                    # For other ranks, adjust starting position
                    if rank > 0:
                        position += rank * per_gpu_batch_size  # Offset starting position
                    
                    token_gen = streaming_token_generator(
                        data_files,
                        tokenizer,
                        start_file_idx=file_idx,
                        start_position=position
                    )
                    
                    step_in_epoch = 0
                    token_batch = []
                    
                    # Create progress bar only on main process
                    pbar = None
                    if is_main_process:
                        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}")

                    while step_in_epoch < steps_per_epoch:
                        try:
                            # Get next batch of tokens
                            tokens, file_idx, position = next(token_gen)
                            
                            # Check for end-of-data sentinel value
                            if file_idx == -1:
                                if is_main_process:
                                    print("Reached end of dataset. Restarting from beginning.")
                                data_position.reset_for_new_pass()
                                token_gen = streaming_token_generator(data_files, tokenizer)
                                continue
                                
                            token_batch.append(tokens)
                            
                            # Update position tracker (only on rank 0 for coordination)
                            if rank == 0:
                                data_position.update_streaming_position(
                                    file_idx, 
                                    position,
                                    data_files[file_idx]
                                )

                            if len(token_batch) == per_gpu_batch_size:  # Use per-GPU batch size
                                # Prepare input and target tensors
                                x_tens, y_tens = collate_batch(token_batch, model_config.block_size)
                                token_batch.clear()
                                
                                if x_tens is None:
                                    continue

                                # Count tokens
                                batch_tokens = x_tens.numel()
                                tokens_in_current_attempt += batch_tokens
                                
                                # Move tensors to current device
                                x_tens = x_tens.to(rank)
                                y_tens = y_tens.to(rank)
                                
                                # Forward and backward pass
                                optimizer.zero_grad()
                                outputs = model(x_tens, y_tens)
                                loss = outputs[1]  # Assuming model returns (logits, loss)
                                
                                # Backward pass with FSDP
                                loss.backward()
                                optimizer.step()

                                # Update step counters
                                global_step += 1
                                step_in_epoch += 1
                                data_position.global_step = global_step
                                
                                # Update progress bar on main process
                                if is_main_process and pbar:
                                    pbar.update(1)
                                
                                # Periodically log progress (from main process only)
                                if global_step % 50 == 0 and is_main_process:
                                    # Gather losses across all processes
                                    loss_tensor = torch.tensor([loss.item()], device=f"cuda:{rank}")
                                    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
                                    dist.all_gather(gathered_losses, loss_tensor)
                                    
                                    # Calculate average loss
                                    avg_loss = torch.mean(torch.stack(gathered_losses)).item()
                                    
                                    current_tokens = total_tokens_processed + tokens_in_current_attempt
                                    print(f"Step {global_step} | Loss: {avg_loss:.4f} | Tokens: {current_tokens:,}")
                                    print(f"File: {file_idx}/{len(data_files)}, Position: {position}")
                                    
                                    # Generate sample text on main process
                                    prompt_str = "Long long time ago, "
                                    token_ids = tokenizer.encode(prompt_str)
                                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(rank)
                                    
                                    # Unwrap the FSDP model for generation
                                    with FSDP.summon_full_params(model):
                                        generated = model.generate(
                                            prompt_tensor,
                                            max_new_tokens=50
                                        )
                                    
                                    generated_text = tokenizer.decode(generated[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                                # Save checkpoint from main process only
                                if global_step % 300 == 0 and is_main_process:
                                    current_tokens = total_tokens_processed + tokens_in_current_attempt
                                    
                                    # Get model state dict from full params
                                    with FSDP.summon_full_params(model):
                                        model_state = model.state_dict()
                                    
                                    checkpoint = {
                                        "global_step": global_step,
                                        "tokens_processed": current_tokens,
                                        "model_state_dict": model_state,
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "loss": loss.detach().float().item(),
                                        "data_position": data_position.get_state()
                                    }
                                    
                                    os.makedirs("pretrained", exist_ok=True)
                                    save_path = f"pretrained/fsdp_step_{global_step}.pth"
                                    torch.save(checkpoint, save_path)
                                    print(f"Checkpoint saved at step {global_step} -> {save_path}")
                                
                                # Synchronize processes
                                dist.barrier()

                        except StopIteration:
                            if is_main_process:
                                print("Reached end of dataset via StopIteration. Restarting data generator.")
                            data_position.reset_for_new_pass()
                            token_gen = streaming_token_generator(data_files, tokenizer)
                            break
                    
                    # Close progress bar
                    if is_main_process and pbar:
                        pbar.close()
                                
                # Non-streaming mode implementation would go here
                else:
                    # Similar implementation for non-streaming mode...
                    pass

            # If we reach here, training completed successfully
            total_tokens_processed += tokens_in_current_attempt
            
            if is_main_process:
                print(f"Training completed successfully with batch_size={batch_size}")
                print(f"Total tokens processed: {total_tokens_processed:,}")
            
            break
            
        except torch.cuda.OutOfMemoryError:
            # OOM - reduce batch size and retry
            torch.cuda.empty_cache()
            
            # Calculate new batch size (reduce per-GPU batch size)
            new_per_gpu_batch_size = max(per_gpu_batch_size // 2, min_batch_size // world_size)
            new_batch_size = new_per_gpu_batch_size * world_size
            
            if is_main_process:
                print(f"CUDA Out of Memory! Reducing batch size from {batch_size} to {new_batch_size}")
                
            if new_batch_size == batch_size or new_per_gpu_batch_size <= 1:
                if is_main_process:
                    print(f"Already at minimum batch size ({batch_size}). Training failed.")
                break
                
            batch_size = new_batch_size
            per_gpu_batch_size = new_per_gpu_batch_size
            
            # Short pause to ensure memory is freed
            time.sleep(5)
        
        except Exception as e:
            if is_main_process:
                print(f"Error during training: {str(e)}")
                if "Expected all tensors to be on the same device" in str(e):
                    print("Device mismatch detected. Check model distribution.")
            raise e

    # Save final model and stats (main process only)
    if is_main_process:
        # Save training stats
        with FSDP.summon_full_params(model):
            model_params = sum(p.numel() for p in model.parameters())
            
            training_stats = {
                "total_tokens": total_tokens_processed,
                "batch_size": batch_size,
                "epochs": epochs,
                "global_steps": global_step,
                "n_layer": model_config.n_layer,
                "n_head": model_config.n_head,
                "n_embd": model_config.n_embd,
                "model_params": model_params,
                "num_gpus_used": world_size
            }
            
            # Write stats to file
            os.makedirs("stats", exist_ok=True)
            with open("stats/fsdp_training_stats.json", "w") as f:
                json.dump(training_stats, f, indent=2)
            
            print(f"\n===== TRAINING SUMMARY =====")
            print(f"Total tokens processed: {total_tokens_processed:,}")
            print(f"Model parameters: {model_params:,}")
            print(f"GPUs used: {world_size}")
            print(f"Final batch size: {batch_size} (per GPU: {batch_size//world_size})")
            print(f"Training steps: {global_step}")
            print(f"Stats saved to: stats/fsdp_training_stats.json")

            # Save final model and tokenizer
            try:
                # Get the full model from the wrapped FSDP model
                # Convert to half precision for storage efficiency
                model_to_save = model.module.half()
                model_to_save.save_pretrained("Argonne_LLM_fsdp")
                tokenizer.save_pretrained("Argonne_LLM_fsdp")
                print("Training complete; model and tokenizer saved successfully.")
            except Exception as e:
                print(f"Failed to save final model: {e}")
    
    # Clean up distributed environment
    cleanup()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Argonne LLM using PyTorch FSDP")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", 
                    help="Use non-streaming mode (load all data in memory)")
    parser.add_argument("--no-compile", dest="compile", action="store_false", 
                    help="Disable torch.compile")
    parser.add_argument("--checkpoint", type=str, default=None, 
                    help="Resume from checkpoint path")
    parser.add_argument("--data-path", type=str, default="data/*.arrow", 
                    help="Path to data files (glob pattern)")
    
    parser.set_defaults(streaming=True, compile=True)
    args = parser.parse_args()
    
    # Process data files
    data_files = glob.glob(args.data_path)
    if not data_files:
        raise ValueError(f"No files matched the pattern '{args.data_path}'")
    
    # Sort files by their numerical order if applicable
    import re
    def get_file_number(filename):
        match = re.search(r'train-(\d+)-of', filename)
        if match:
            return int(match.group(1))
        return 0
    
    data_files = sorted(data_files, key=get_file_number)
    print(f"Files will be processed in numerical order. First file: {data_files[0]}")
    
    # Launch with torchrun
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Launching with FSDP on {world_size} GPUs")
        
        # Check if executed with torchrun
        if "LOCAL_RANK" in os.environ:
            # Get rank from environment variable
            rank = int(os.environ["LOCAL_RANK"])
            # Call training function
            train_with_fsdp(
                rank=rank,
                world_size=world_size,
                data_files=data_files,
                use_streaming=args.streaming,
                use_compile=args.compile,
                resume_from_checkpoint=args.checkpoint
            )
        else:
            print("FSDP training requires torchrun. Please use:")
            print(f"torchrun --nproc_per_node={world_size} train_with_fsdp.py [args]")
    else:
        print("Only one GPU detected. FSDP requires multiple GPUs.")
        print("Continuing with single GPU training...")
        # Call training function with single process
        train_with_fsdp(
            rank=0,
            world_size=1,
            data_files=data_files,
            use_streaming=args.streaming,
            use_compile=args.compile,
            resume_from_checkpoint=args.checkpoint
        )

if __name__ == "__main__":
    main()
