import os
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import glob
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data
from model import ArgonneConfig, ArgonneModel
from datasets import Dataset

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define our own version of streaming token generator with position tracking support
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
        self.global_step = 0  # Track by global step instead of epoch
        
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
        # For backwards compatibility
        if "epoch" in state and "global_step" not in state:
            self.global_step = state.get("epoch", 0)
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
            self.shuffled_indices = torch.randperm(total_samples).tolist()
        return self.shuffled_indices[self.current_position:]
    
    def reset_for_new_pass(self, total_samples=None):
        """Reset position for a new pass through the data, now based on step not epoch"""
        self.global_step += 1
        if self.streaming:
            self.current_file_idx = 0
            self.position_in_file = 0
        else:
            self.current_position = 0
            if total_samples:
                self.shuffled_indices = torch.randperm(total_samples).tolist()

def resume_training(
    data_path="data/*.arrow",
    checkpoint_path=None,
    total_training_steps=160_000,
    block_size=2048,
    batch_size=320,
    lr=3e-5,
    use_streaming=False,   
    num_proc=8
):
    # Expand glob pattern to get actual data files
    if isinstance(data_path, str) and ('*' in data_path or '?' in data_path):
        data_files = glob.glob(data_path)
        
        # Sort the files numerically by extracting the number from the filename
        # This assumes filenames like "fineweb-edu-train-00158-of-00218.arrow"
        import re
        def get_file_number(filename):
            match = re.search(r'train-(\d+)-of', filename)
            if match:
                return int(match.group(1))
            return 0  # Default case
        
        # Sort files by their numerical order
        data_files = sorted(data_files, key=get_file_number)
        print(f"Files will be processed in numerical order. First file: {data_files[0]}")
    else:
        data_files = [data_path] if isinstance(data_path, str) else data_path
    
    print(f"Found {len(data_files)} data files")
    
    # 1) Load tokenizer
    hf_tokenizer = load_bpe_tokenizer()

    # 2) Build config & base model
    config = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=16,
        n_head=16,
        n_embd=1296,
        dropout=0.1
    )
    base_model = ArgonneModel(config)
    
    # 3) Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Resuming from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Convert compiled model state dict to regular model format
    if any(k.startswith("_orig_mod.") for k in ckpt["model_state_dict"].keys()):
        print("Detected compiled model checkpoint, converting parameter names...")
        new_state_dict = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("_orig_mod.") and "pipeline_stages" not in k:
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
        ckpt["model_state_dict"] = new_state_dict
        print("Checkpoint parameter names converted successfully")

    base_model.load_state_dict(ckpt["model_state_dict"])
    
    # 4) Distribute model BEFORE creating optimizer
    base_model.distribute_model()  # Make sure model is distributed across GPUs first
    model = base_model  # Keep reference to distributed model
    
    # 5) NOW create optimizer with already-distributed parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # 6) Load optimizer state
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    # 7) CRITICAL FIX: Move optimizer states to match parameter devices
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            # For each parameter, ensure its state is on the same device as the parameter
            if param in optimizer.state:
                for state_key, state_val in optimizer.state[param].items():
                    if isinstance(state_val, torch.Tensor):
                        optimizer.state[param][state_key] = state_val.to(param.device)

    # Get global step and token count from checkpoint
    global_step = ckpt.get("global_step", 0)
    total_tokens_processed = ckpt.get("tokens_processed", 0)
    print(f"Loaded checkpoint at global_step={global_step}, tokens_processed={total_tokens_processed:,}")
    
    # Initialize data position tracker and restore its state if available
    data_position = DataPosition(streaming=use_streaming)
        
    # Look for data position in checkpoint (handling both key names for backward compatibility)
    if "data_position" in ckpt:
        print("Found data position information in checkpoint - will resume from correct position")
        data_position.restore_state(ckpt.get("data_position"))
        print(f"Resuming from file {data_position.current_file_idx}, position {data_position.position_in_file}")
        # Set global step from loaded checkpoint
        data_position.global_step = global_step
    else:
        print("No data position information found in checkpoint - manually setting to start from file 0")
        data_position.current_file_idx = 0
        data_position.position_in_file = 0
        data_position.global_step = global_step  # Initialize with global_step from checkpoint
        print(f"Manually set to start from file {data_position.current_file_idx}, position {data_position.position_in_file}")

    # Log GPU info
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 8) GradScaler for CUDA
    scaler = torch.amp.GradScaler("cuda")

    # Try to apply torch.compile() if available
    if hasattr(torch, 'compile'):
        try:
            print("Applying torch.compile() to optimize model execution...")
            model = torch.compile(model)
            print("Model successfully compiled!")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Continuing with uncompiled model.")

    first_device = model.devices[0]

    # Token counting variables
    tokens_in_this_session = 0

    # 9) Decide streaming vs non-streaming
    if use_streaming:
        print(f"=== Resuming training from global step {global_step} in STREAMING mode ===")
        print(f"=== Will train until reaching {total_training_steps} steps ===")
        
        # Create a streaming generator that starts from the correct position
        token_gen = streaming_token_generator(
            data_files, 
            hf_tokenizer, 
            start_file_idx=data_position.current_file_idx,
            start_position=data_position.position_in_file
        )
        token_buffer = []
        end_of_data = False

        # Use tqdm to track global step progress toward total_training_steps
        with tqdm(initial=global_step, total=total_training_steps, desc="Training") as pbar:
            current_file_idx = data_position.current_file_idx
            while global_step < total_training_steps:
                try:
                    tokens, file_idx, position = next(token_gen)
                    
                    # Check for end-of-data sentinel value
                    if file_idx == -1:
                        print("Reached end of dataset. Restarting from beginning.")
                        # Update tracker for new pass
                        data_position.reset_for_new_pass()
                        print(f"Starting new data pass at step {global_step}")
                        token_gen = streaming_token_generator(data_files, hf_tokenizer)
                        continue
                        
                    token_buffer.append(tokens)
                    
                    # Update data tracker with current position
                    data_position.update_streaming_position(file_idx, position, data_files[file_idx])
                    
                    # If we've moved to a new file, log it
                    if file_idx != current_file_idx:
                        current_file_idx = file_idx
                        print(f"Now processing file {file_idx}/{len(data_files)}: {data_files[file_idx]}")

                    if len(token_buffer) == batch_size:
                        x_tens, y_tens = collate_batch(token_buffer, block_size)
                        token_buffer.clear()
                        if x_tens is None:
                            continue

                        # Count tokens in this batch
                        batch_tokens = x_tens.numel()
                        tokens_in_this_session += batch_tokens
                        
                        x_tens = x_tens.to(first_device)
                        y_tens = y_tens.to(first_device)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            logits, loss = model(x_tens, y_tens)
                            loss = loss.to(first_device)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1
                        data_position.global_step = global_step  # Update data position's step counter
                        pbar.update(1)

                        if global_step % 50 == 0:
                            current_total_tokens = total_tokens_processed + tokens_in_this_session
                            print(f"Step {global_step} | Loss: {loss.item():.4f} | Tokens: {current_total_tokens:,}")
                            print(f"File: {file_idx}/{len(data_files)}, Position: {position}")
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                            generated = model.generate(prompt_tensor, max_new_tokens=50)
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                        if global_step % 300 == 0:
                            current_total_tokens = total_tokens_processed + tokens_in_this_session
                            ckpt_dict = {
                                "global_step": global_step,
                                "tokens_processed": current_total_tokens,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item(),
                                "data_position": data_position.get_state()  # Save data position
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            save_path = f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                            torch.save(ckpt_dict, save_path)
                            print(f"Checkpoint saved @ step {global_step} -> {save_path}")
                            
                            # Also update the stats file
                            update_training_stats(
                                tokens=current_total_tokens,
                                batch_size=batch_size,
                                steps=global_step,
                                model=model,
                                n_layer=config.n_layer,
                                n_head=config.n_head,
                                n_embd=config.n_embd
                            )

                except StopIteration:
                    # This shouldn't happen with our updated generator approach using sentinel values
                    # But keep as a fallback
                    print("Reached end of dataset via StopIteration. Restarting data generator.")
                    data_position.reset_for_new_pass()
                    print(f"Starting new data pass at step {global_step}")
                    token_gen = streaming_token_generator(data_files, hf_tokenizer)
                    continue

    else:
        print(f"=== Resuming training from global step {global_step} in NON-STREAMING mode ===")
        print(f"=== Will train until reaching {total_training_steps} steps ===")

        # 1) Load entire data in memory
        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=num_proc)
        total_samples = len(tokenized_data)
        print(f"Total in-memory tokenized samples: {total_samples}")
        
        # Use tqdm to track global step progress toward total_training_steps
        with tqdm(initial=global_step, total=total_training_steps, desc="Training") as pbar:
            while global_step < total_training_steps:
                # Get shuffled indices from the data tracker
                batch_indices = data_position.generate_shuffled_indices(total_samples)
                
                if len(batch_indices) < batch_size:
                    # We've reached the end of this shuffled data
                    data_position.reset_for_new_pass(total_samples)
                    print(f"Starting new data pass at step {global_step}")
                    continue
                
                # Take the next batch_size indices
                batch_indices = batch_indices[:batch_size]
                data_position.update_nonstreaming_position(
                    data_position.current_position + len(batch_indices)
                )
                
                # Get batch token lists
                batch_token_lists = [tokenized_data[i] for i in batch_indices]
                
                x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                if x_tens is None:
                    continue

                # Count tokens in this batch
                batch_tokens = x_tens.numel()
                tokens_in_this_session += batch_tokens
                
                x_tens = x_tens.to(first_device)
                y_tens = y_tens.to(first_device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits, loss = model(x_tens, y_tens)
                    loss = loss.to(first_device)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                data_position.global_step = global_step  # Update data position's step counter
                pbar.update(1)
                
                if global_step % 50 == 0:
                    current_total_tokens = total_tokens_processed + tokens_in_this_session
                    print(f"Step {global_step} | Loss: {loss.item():.4f} | Tokens: {current_total_tokens:,}")
                    print(f"Position in dataset: {data_position.current_position}/{total_samples}")
                    prompt_str = "Long long time ago, "
                    token_ids = hf_tokenizer.encode(prompt_str)
                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                    generated = model.generate(prompt_tensor, max_new_tokens=50)
                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                if global_step % 2000 == 0:
                    current_total_tokens = total_tokens_processed + tokens_in_this_session
                    ckpt_dict = {
                        "global_step": global_step,
                        "tokens_processed": current_total_tokens,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "data_position": data_position.get_state()  # Save data position
                    }
                    os.makedirs("pretrained", exist_ok=True)
                    save_path = f"pretrained/non_streaming_checkpoint_step_{global_step}.pth"
                    torch.save(ckpt_dict, save_path)
                    print(f"Checkpoint saved @ step {global_step} -> {save_path}")
                    
                    # Also update the stats file
                    update_training_stats(
                        tokens=current_total_tokens,
                        batch_size=batch_size,
                        steps=global_step,
                        model=model,
                        n_layer=config.n_layer,
                        n_head=config.n_head,
                        n_embd=config.n_embd
                    )

    # Final token count calculation
    final_token_count = total_tokens_processed + tokens_in_this_session
    
    # Update final stats
    update_training_stats(
        tokens=final_token_count,
        batch_size=batch_size,
        steps=global_step,
        model=model,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        final=True
    )
    
    print(f"\n===== TRAINING SUMMARY =====")
    print(f"Total tokens processed: {final_token_count:,}")
    print(f"Final step count: {global_step}")
    print(f"Training complete!")

    # Perform final save at the end of training
    try:
        # Convert to FP16 or keep in FP32 as needed
        model = model.half()
        # Move entire model to CPU to avoid cross-device references
        model = model.to("cpu")
        # Disable safe_serialization to allow shared tensor references
        model.save_pretrained("Argonne_LLM", safe_serialization=False)
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print(f"Training completed at step {global_step}. Final model saved.")
    except Exception as e:
        print(f"Failed to save final model: {e}")


def update_training_stats(tokens, batch_size, steps, model, n_layer, n_head, n_embd, final=False):
    """Update the training statistics file with current information"""
    # Calculate model parameters
    model_params = sum(p.numel() for p in model.parameters())
    
    training_stats = {
        "total_tokens": tokens,
        "batch_size": batch_size,
        "global_steps": steps,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "model_params": model_params,
        "final_training": final
    }
    
    # Write stats to JSON file
    os.makedirs("stats", exist_ok=True)
    
    # For the final update, create a timestamped file
    if final:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats/final_training_stats_{timestamp}.json"
    else:
        filename = f"stats/current_training_stats_step_{steps}.json"
        
    with open(filename, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    if final:
        print(f"Final training stats saved to: {filename}")
    return filename


def main():
    resume_training(
        data_path="data/*.arrow",
        checkpoint_path="pretrained/streaming_checkpoint_step_33900.pth", # manually set
        total_training_steps=80_000,
        block_size=2048,
        batch_size=756,
        lr=5e-5,
        use_streaming=True, 
        num_proc=4
    )

if __name__ == "__main__":
    main()