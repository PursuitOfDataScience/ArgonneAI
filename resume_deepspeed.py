import os
import math
import json
import torch
import argparse
import deepspeed
import numpy as np
import glob
from tqdm import tqdm
import torch.distributed as dist
from datetime import datetime

# Import your existing modules
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data
from model import ArgonneConfig, ArgonneModel
from datasets import Dataset

# Import shared functions from deepspeed_training.py
from deepspeed_training import (
    setup_deepspeed_environment,
    streaming_token_generator,
    DataPosition,
    set_seed
)

def resume_training_with_deepspeed(
    data_path, 
    checkpoint_dir,
    config_path,
    output_dir="./output",
    block_size=2048,
    lr=3e-5,
    use_streaming=False,
    max_steps=100000,
    save_steps=1000,
    seed=42,
    batch_size=32,
    zero_stage=3
):
    """
    Resume training using DeepSpeed with ZeRO optimization.
    
    Args:
        data_path: Path to data files (glob pattern or list)
        checkpoint_dir: Directory containing DeepSpeed checkpoint
        config_path: Path to DeepSpeed config JSON
        output_dir: Directory to save new checkpoints
        block_size: Maximum sequence length
        lr: Learning rate
        use_streaming: Whether to stream data or load it all
        max_steps: Maximum number of training steps
        save_steps: Save checkpoint every N steps
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
        print(f"=== Resuming DeepSpeed Training ===")
        print(f"World Size: {world_size}")
        print(f"Local Rank: {local_rank}")
        print(f"Using ZeRO Stage: {zero_stage}")
        print(f"Output Directory: {output_dir}")
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        
    # Create output directories
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        
    # Expand glob pattern for data files if needed
    if isinstance(data_path, str) and ('*' in data_path or '?' in data_path):
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
    
    # Initialize DeepSpeed with the checkpoint
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Load model from checkpoint
    # Note: DeepSpeed's load_checkpoint handles model, optimizer, and lr_scheduler states
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=ds_config
    )
    
    # Check if checkpoint exists and load it
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    
    # Load checkpoint and client state
    _, client_state = model_engine.load_checkpoint(checkpoint_dir)
    
    # Initialize variables from checkpoint's client state
    if client_state is None:
        if is_main_process:
            print("Warning: No client state found in checkpoint. Starting with default values.")
        global_step = 0
        tokens_processed = 0
        data_position = DataPosition(streaming=use_streaming)
    else:
        global_step = client_state.get('global_step', 0)
        tokens_processed = client_state.get('tokens_processed', 0)
        
        # Load data position from checkpoint
        data_position = DataPosition(streaming=use_streaming)
        if 'data_position' in client_state:
            data_position.restore_state(client_state['data_position'])
    
    # Log restored state
    if is_main_process:
        print(f"Restored training state from checkpoint:")
        print(f"  Global step: {global_step}")
        print(f"  Tokens processed: {tokens_processed:,}")
        if use_streaming:
            print(f"  Current file: {data_position.current_file_idx}, position: {data_position.position_in_file}")
        else:
            print(f"  Current position: {data_position.current_position}")
    
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
    
    # Training loop - continue from the loaded step
    if is_main_process:
        print(f"Resuming training from step {global_step} until {max_steps}")
        
    with tqdm(total=max_steps, initial=global_step, desc="Training", disable=not is_main_process) as pbar:
        if use_streaming:
            ###############################
            # STREAMING MODE
            ###############################
            # Initialize token generator, starting from the restored position
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
    parser = argparse.ArgumentParser(description="Resume training with DeepSpeed")
    parser.add_argument("--data", type=str, required=True, help="Path to data files (glob pattern)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
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
    
    resume_training_with_deepspeed(
        data_path=args.data,
        checkpoint_dir=args.checkpoint,
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
