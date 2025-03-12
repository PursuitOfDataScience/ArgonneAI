import os
import torch
import deepspeed
from tqdm import tqdm
import time
import glob
import argparse
import json
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data
from model import ArgonneConfig, ArgonneModel
from datasets import Dataset
from ds_config_optimized import get_optimized_ds_config, save_optimized_ds_config

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Updated streaming token generator to use datasets library
def streaming_token_generator(data_files, tokenizer, start_file_idx=0, start_position=0):
    """
    Enhanced token generator that supports position tracking.
    
    Args:
        data_files: List of files to process
        tokenizer: HF tokenizer
        start_file_idx: Starting file index
        start_position: Starting position within file
        
    Yields:
        (tokens, file_idx, position): Tokenized data with position info
    """
    file_idx = start_file_idx
    processed_count = 0
    
    while file_idx < len(data_files):
        try:
            file_path = data_files[file_idx]
            print(f"Streaming from file {file_idx}/{len(data_files)}: {file_path}")
            
            try:
                dataset = Dataset.from_file(file_path)
                print(f"Successfully loaded dataset with {len(dataset)} rows")
            except Exception as file_error:
                print(f"ERROR: Could not read file {file_path}: {file_error}")
                print(f"Skipping problematic file and moving to next one.")
                file_idx += 1
                continue
                
            position = start_position  # Start from specified position
            start_position = 0
            
            while position < len(dataset):
                try:
                    item = dataset[position]
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
    return None, -1, -1

def create_argonne_model(batch_size=12):
    """Create the ArgonneAI model configuration"""
    config = ArgonneConfig(
        vocab_size=12000,
        block_size=2048,
        n_layer=16, 
        n_head=16,
        n_embd=1296,
        dropout=0.1
    )
    
    # Initialize the model
    model = ArgonneModel(config)
    return model

def train_with_deepspeed(args):
    """Main training function using DeepSpeed"""
    # Expand glob pattern to get data files
    data_files = glob.glob(args.data_path)
    if not data_files:
        raise ValueError(f"No files found matching pattern: {args.data_path}")
    
    # Sort the files numerically by extracting the number from the filename
    import re
    def get_file_number(filename):
        match = re.search(r'train-(\d+)-of', filename)
        if match:
            return int(match.group(1))
        return 0
    
    # Sort files by their numerical order
    data_files = sorted(data_files, key=get_file_number)
    print(f"Found {len(data_files)} data files. First file: {data_files[0]}")
    
    # Load or create tokenizer
    hf_tokenizer = load_bpe_tokenizer()
    
    # Create model
    model = create_argonne_model(batch_size=args.batch_size)
    
    # Set up parameters for DeepSpeed ZeRO
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        config=args.deepspeed_config if hasattr(args, "deepspeed_config") and args.deepspeed_config else None
    )
    
    # Get the device we'll use for tensor operations
    device = model_engine.device
    
    # Training loop settings
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    global_step = 0
    
    # Simple data position tracker
    current_file_idx = 0
    position_in_file = 0
    
    # Token counting
    total_tokens_processed = 0
    
    # Main training loop
    for epoch in range(epochs):
        print(f"\n==== Starting epoch {epoch} ====")
        token_gen = streaming_token_generator(
            data_files, 
            hf_tokenizer, 
            start_file_idx=current_file_idx,
            start_position=position_in_file
        )
        
        step_in_epoch = 0
        token_batch = []
        
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            while step_in_epoch < steps_per_epoch:
                try:
                    # Get next tokens
                    tokens, file_idx, position = next(token_gen)
                    
                    # Check for end-of-data sentinel value
                    if file_idx == -1:
                        print("Reached end of dataset. Restarting from beginning.")
                        token_gen = streaming_token_generator(data_files, hf_tokenizer)
                        continue
                    
                    # Update position tracking
                    current_file_idx = file_idx
                    position_in_file = position
                    
                    token_batch.append(tokens)
                    
                    if len(token_batch) == args.batch_size * model_engine.gradient_accumulation_steps():
                        # Prepare batch
                        x_tens, y_tens = collate_batch(token_batch, 2048)
                        token_batch.clear()
                        
                        if x_tens is None:
                            continue
                        
                        # Count tokens
                        batch_tokens = x_tens.numel()
                        total_tokens_processed += batch_tokens
                        
                        # Move tensors to device
                        x_tens = x_tens.to(device)
                        y_tens = y_tens.to(device)
                        
                        # Forward pass
                        outputs = model_engine(x_tens, y_tens)
                        loss = outputs[1]  # Get loss from model outputs
                        
                        # Backward pass
                        model_engine.backward(loss)
                        
                        # Update weights
                        model_engine.step()
                        
                        # Increment counters
                        global_step += 1
                        step_in_epoch += 1
                        pbar.update(1)
                        
                        # Logging
                        if global_step % args.log_interval == 0:
                            print(f"\nStep {global_step} | Loss: {loss.item():.4f} | Tokens: {total_tokens_processed:,}")
                            print(f"File: {file_idx}/{len(data_files)}, Position: {position}")
                            
                            # Generate sample text
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
                            
                            generated = model.generate(prompt_tensor, max_new_tokens=100)
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")
                        
                        # Save checkpoint
                        if global_step % args.save_interval == 0:
                            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            
                            # DeepSpeed checkpoint saving
                            model_engine.save_checkpoint(checkpoint_path)
                            hf_tokenizer.save_pretrained(checkpoint_path)
                            
                            # Save training state
                            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                                json.dump({
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "total_tokens": total_tokens_processed,
                                    "file_idx": current_file_idx,
                                    "position_in_file": position_in_file
                                }, f)
                            
                            print(f"Saved checkpoint at {checkpoint_path}")
                
                except StopIteration:
                    print("Reached end of dataset stream")
                    break
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_path)
    hf_tokenizer.save_pretrained(final_path)
    
    print("\n===== TRAINING SUMMARY =====")
    print(f"Total tokens processed: {total_tokens_processed:,}")
    print(f"Total training steps: {global_step}")
    print(f"Model saved to: {final_path}")

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed training for ArgonneAI model")
    
    # Basic training arguments
    parser.add_argument("--data_path", type=str, default="data/*.arrow", 
                        help="Path or glob pattern to data files")
    parser.add_argument("--output_dir", type=str, default="./deepspeed_output",
                        help="Output directory for checkpoints and final model")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=50000,
                        help="Number of training steps per epoch")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint saving interval (steps)")
    
    # DeepSpeed specific arguments
    parser.add_argument("--zero_stage", type=int, default=2,
                        help="ZeRO optimization stage (0, 1, 2, or 3)")
    parser.add_argument("--offload", action="store_true",
                        help="Offload optimizer states to CPU")
    
    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate DeepSpeed config if not provided
    if not args.deepspeed_config:
        config_path = save_optimized_ds_config(
            filename=os.path.join(args.output_dir, "ds_config_optimized.json"),
            batch_size=args.batch_size,
            grad_accum=2,  # Using recommended gradient accumulation steps
            zero_stage=args.zero_stage,
            offload=args.offload
        )
        args.deepspeed_config = config_path
        print(f"Generated optimized DeepSpeed config at {config_path}")
    
    # Launch training
    train_with_deepspeed(args)

if __name__ == "__main__":
    main()