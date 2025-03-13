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

def streaming_token_generator(data_files, tokenizer, start_file_idx=0, start_position=0):
    """
    Enhanced token generator that supports resuming from a position.
    
    Args:
        data_files: List of .arrow files to stream from
        tokenizer: HuggingFace tokenizer to use
        start_file_idx: Index of the file to start processing from
        start_position: Position within the file to start from
        
    Yields:
        (tokens, file_idx, position): A tuple containing tokenized data and position info
    """
    file_idx = start_file_idx
    processed_count = 0
    
    while file_idx < len(data_files):
        try:
            file_path = data_files[file_idx]
            print(f"Streaming from file {file_idx}/{len(data_files)}: {file_path}")
            
            try:
                # Use datasets library instead of pyarrow.parquet
                dataset = Dataset.from_file(file_path)
                print(f"Successfully loaded dataset with {len(dataset)} rows")
                print(f"Dataset features: {list(dataset.features.keys())}")
            except Exception as file_error:
                print(f"ERROR: Could not read file {file_path}: {file_error}")
                print(f"Skipping problematic file and moving to next one.")
                file_idx += 1
                continue
                
            position = start_position  # Start from specified position
            # Reset start_position for future files
            start_position = 0
            
            # Process entries from current position
            while position < len(dataset):
                try:
                    item = dataset[position]
                    # Get the text field - most commonly 'text' but could be others
                    if 'text' in item and item['text'] and isinstance(item['text'], str):
                        text = item['text']
                        tokens = tokenizer.encode(text)
                        processed_count += 1
                        yield tokens, file_idx, position
                    
                except Exception as e:
                    print(f"Error processing item at position {position}: {e}")
                
                position += 1
            
            file_idx += 1
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            file_idx += 1
    
    print(f"Completed processing all available files. Processed {processed_count} samples.")
    
    # Return sentinel value instead of raising StopIteration
    return None, -1, -1

# Simple data tracking class
class DataPosition:
    def __init__(self, streaming=True):
        """Track dataset position during training"""
        self.streaming = streaming
        
        # For streaming mode
        self.current_file_idx = 0
        self.position_in_file = 0
        
        # For non-streaming mode
        self.shuffled_indices = None
        self.current_position = 0
        self.global_step = 0
        
        # Files processed tracking
        self.files_processed = set()
        
    def get_state(self):
        """Returns state dict for checkpointing"""
        return {
            "streaming": self.streaming,
            "current_file_idx": self.current_file_idx,
            "position_in_file": self.position_in_file,
            "current_position": self.current_position,
            "global_step": self.global_step,
            "files_processed": list(self.files_processed)
        }
        
    def restore_state(self, state):
        """Restore position from checkpoint"""
        if state is None:
            return
            
        self.streaming = state.get("streaming", self.streaming)
        self.current_file_idx = state.get("current_file_idx", 0)
        self.position_in_file = state.get("position_in_file", 0)
        self.current_position = state.get("current_position", 0)
        self.global_step = state.get("global_step", 0)
        self.files_processed = set(state.get("files_processed", []))
    
    def update_streaming_position(self, file_idx, position, file_path=None):
        """Update streaming position information"""
        self.current_file_idx = file_idx
        self.position_in_file = position
        if file_path:
            self.files_processed.add(file_path)
    
    def update_nonstreaming_position(self, position):
        """Update non-streaming position"""
        self.current_position = position

    def generate_shuffled_indices(self, total_samples):
        """Generate shuffled indices for non-streaming mode"""
        if self.shuffled_indices is None or len(self.shuffled_indices) != total_samples:
            # Use deterministic seed based on global step to ensure consistency across nodes
            seed = 42 + self.global_step
            rng = np.random.RandomState(seed)
            self.shuffled_indices = rng.permutation(total_samples).tolist()
        return self.shuffled_indices[self.current_position:]
    
    def reset_for_new_pass(self, total_samples=None):
        """Reset position for a new pass through the data"""
        self.global_step += 1
        if self.streaming:
            self.current_file_idx = 0
            self.position_in_file = 0
        else:
            self.current_position = 0
            if total_samples:
                # Ensure consistent shuffling across processes
                seed = 42 + self.global_step
                rng = np.random.RandomState(seed)
                self.shuffled_indices = rng.permutation(total_samples).tolist()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
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
    zero_stage=3
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
    """
    # Setup global seed for reproducibility
    set_seed(seed)
    
    # Setup DeepSpeed distributed environment
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
    
    if ds_config['train_micro_batch_size_per_gpu'] == 'auto':
        ds_config['train_micro_batch_size_per_gpu'] = batch_size
    
    # Initialize DeepSpeed
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=ds_config
    )
    
    # Initialize data position tracker
    data_position = DataPosition(streaming=use_streaming)
    
    # Prepare for training (streaming or non-streaming)
    tokenized_data = None
    total_samples = 0
    
    if not use_streaming:
        # Load all data into memory
        if is_main_process:
            print("Loading all data into memory (non-streaming mode)...")
        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size)
        total_samples = len(tokenized_data)
        if is_main_process:
            print(f"Loaded {total_samples} samples into memory")
    
    # Token counting
    tokens_processed = 0
    global_step = 0
    
    # Training loop
    if is_main_process:
        print(f"Starting training for {max_steps} steps")
        
    with tqdm(total=max_steps, desc="Training", disable=not is_main_process) as pbar:
        if use_streaming:
            ###############################
            # STREAMING MODE
            ###############################
            # Initialize token generator
            token_gen = streaming_token_generator(
                data_files, 
                hf_tokenizer, 
                start_file_idx=data_position.current_file_idx,
                start_position=data_position.position_in_file
            )
            
            token_buffer = []
            
            while global_step < max_steps:
                try:
                    tokens, file_idx, position = next(token_gen)
                    
                    # Check for end of dataset
                    if file_idx == -1:
                        if is_main_process:
                            print("Reached end of dataset. Starting from beginning.")
                        data_position.reset_for_new_pass()
                        token_gen = streaming_token_generator(data_files, hf_tokenizer)
                        continue
                        
                    token_buffer.append(tokens)
                    
                    # Update data position
                    data_position.update_streaming_position(
                        file_idx, position, data_files[file_idx] if file_idx < len(data_files) else None
                    )
                    
                    # Process batch when buffer is full
                    if len(token_buffer) == batch_size:
                        x_tens, y_tens = collate_batch(token_buffer, block_size)
                        token_buffer.clear()
                        
                        if x_tens is None:
                            continue
                            
                        # Count tokens
                        batch_tokens = x_tens.numel()
                        tokens_processed += batch_tokens
                        
                        # Forward pass
                        outputs = model_engine(x_tens, y_tens)
                        loss = outputs[1]  # Assuming the second return value is loss
                        
                        # Backward and optimize
                        model_engine.backward(loss)
                        model_engine.step()
                        
                        # Update step counter
                        global_step += 1
                        data_position.global_step = global_step
                        
                        if is_main_process:
                            pbar.update(1)
                        
                        # Logging
                        if is_main_process and global_step % 50 == 0:
                            print(f"Step: {global_step}, Loss: {loss.item():.4f}, Tokens: {tokens_processed:,}")
                            print(f"File: {file_idx}/{len(data_files)}, Position: {position}")
                            
                            # Generate sample text
                            if global_step % 500 == 0:
                                model_engine.eval()
                                prompt_str = "Long long time ago, "
                                token_ids = hf_tokenizer.encode(prompt_str)
                                prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(model_engine.device)
                                
                                with torch.no_grad():
                                    generated = model.generate(prompt_tensor, max_new_tokens=50)
                                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")
                                model_engine.train()
                        
                        # Save checkpoint
                        if global_step % save_steps == 0:
                            if is_main_process:
                                print(f"Saving checkpoint at step {global_step}")
                                
                            # DeepSpeed save checkpoint
                            checkpoint_dir = f"{output_dir}/checkpoints/step_{global_step}"
                            client_state = {
                                "global_step": global_step,
                                "tokens_processed": tokens_processed,
                                "data_position": data_position.get_state()
                            }
                            model_engine.save_checkpoint(checkpoint_dir, client_state=client_state)
                            
                            # Save a small metadata file to track progress
                            if is_main_process:
                                progress_data = {
                                    "step": global_step,
                                    "tokens": tokens_processed,
                                    "loss": loss.item() if loss else None,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "file_idx": file_idx,
                                    "position": position
                                }
                                with open(f"{output_dir}/progress.json", "w") as f:
                                    json.dump(progress_data, f, indent=2)
                
                except StopIteration:
                    if is_main_process:
                        print("Reached end of dataset via StopIteration. Starting from beginning.")
                    data_position.reset_for_new_pass()
                    token_gen = streaming_token_generator(data_files, hf_tokenizer)
        
        else:
            ###############################
            # NON-STREAMING MODE
            ###############################
            while global_step < max_steps:
                # Generate shuffled indices
                indices = data_position.generate_shuffled_indices(total_samples)
                
                if len(indices) < batch_size:
                    # Reset for next pass over data
                    data_position.reset_for_new_pass(total_samples)
                    continue
                    
                # Get batch indices
                batch_indices = indices[:batch_size]
                data_position.update_nonstreaming_position(
                    data_position.current_position + len(batch_indices)
                )
                
                # Get batch data
                batch_token_lists = [tokenized_data[i] for i in batch_indices]
                x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                
                if x_tens is None:
                    continue
                
                # Count tokens
                batch_tokens = x_tens.numel()
                tokens_processed += batch_tokens
                
                # Forward pass
                outputs = model_engine(x_tens, y_tens)
                loss = outputs[1]  # Assuming the second return value is loss
                
                # Backward and optimize
                model_engine.backward(loss)
                model_engine.step()
                
                # Update step counter
                global_step += 1
                data_position.global_step = global_step
                
                if is_main_process:
                    pbar.update(1)
                
                # Logging
                if is_main_process and global_step % 50 == 0:
                    print(f"Step: {global_step}, Loss: {loss.item():.4f}, Tokens: {tokens_processed:,}")
                    print(f"Position in dataset: {data_position.current_position}/{total_samples}")
                    
                    # Generate sample text
                    if global_step % 500 == 0:
                        model_engine.eval()
                        prompt_str = "Long long time ago, "
                        token_ids = hf_tokenizer.encode(prompt_str)
                        prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(model_engine.device)
                        
                        with torch.no_grad():
                            generated = model.generate(prompt_tensor, max_new_tokens=50)
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")
                        model_engine.train()
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    if is_main_process:
                        print(f"Saving checkpoint at step {global_step}")
                        
                    # DeepSpeed save checkpoint
                    checkpoint_dir = f"{output_dir}/checkpoints/step_{global_step}"
                    client_state = {
                        "global_step": global_step,
                        "tokens_processed": tokens_processed,
                        "data_position": data_position.get_state()
                    }
                    model_engine.save_checkpoint(checkpoint_dir, client_state=client_state)
                    
                    # Save a small metadata file to track progress
                    if is_main_process:
                        progress_data = {
                            "step": global_step,
                            "tokens": tokens_processed,
                            "loss": loss.item() if loss else None,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "position": data_position.current_position,
                            "total_samples": total_samples
                        }
                        with open(f"{output_dir}/progress.json", "w") as f:
                            json.dump(progress_data, f, indent=2)
    
    # Save final model
    if is_main_process:
        print(f"Training complete! Saving final model...")
        
        # Get model state dict from the last rank
        model_to_save = model_engine.module
        
        # Convert to half precision for storage
        model_to_save = model_to_save.half()
        
        # Save model and tokenizer in Hugging Face format
        model_to_save.save_pretrained(f"{output_dir}/final_model")
        hf_tokenizer.save_pretrained(f"{output_dir}/final_model")
        
        # Save training summary
        summary = {
            "total_steps": global_step,
            "total_tokens": tokens_processed,
            "model_config": {
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size,
                "block_size": config.block_size,
                "parameters": sum(p.numel() for p in model_to_save.parameters())
            },
            "training_config": {
                "batch_size": batch_size,
                "zero_stage": zero_stage,
                "learning_rate": lr,
                "world_size": world_size
            },
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Training completed successfully!")
        print(f"Total steps: {global_step}")
        print(f"Total tokens processed: {tokens_processed:,}")

def main():
    parser = argparse.ArgumentParser(description="Train with DeepSpeed")
    parser.add_argument("--data", type=str, required=True, help="Path to data files (glob pattern)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--config", type=str, default="ds_config.json", help="DeepSpeed config")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--streaming", action="store_true", help="Use streaming data")
    parser.add_argument("--zero_stage", type=int, default=3, choices=[1, 2, 3], help="DeepSpeed ZeRO stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
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
        zero_stage=args.zero_stage
    )

if __name__ == "__main__":
    main()
