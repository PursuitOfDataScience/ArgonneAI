import os
import torch
from tqdm import tqdm
import time
import glob
import json
from data_processing import collate_batch, load_bpe_tokenizer, train_bpe_tokenizer, load_nonstream_data, create_text_file_from_arrow
from model import ArgonneConfig, ArgonneModel
from datasets import Dataset

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Simple data tracking class to keep position information
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
        self.epoch = 0
        
        # Files processed tracking
        self.files_processed = set()
        
    def get_state(self):
        """Returns state dict for checkpointing"""
        return {
            "streaming": self.streaming,
            "current_file_idx": self.current_file_idx,
            "position_in_file": self.position_in_file,
            "current_position": self.current_position,
            "epoch": self.epoch,
            "files_processed": list(self.files_processed)
        }
    
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
    
    def next_epoch(self, total_samples=None):
        """Move to next epoch"""
        self.epoch += 1
        if self.streaming:
            self.current_file_idx = 0
            self.position_in_file = 0
        else:
            self.current_position = 0
            if total_samples:
                self.shuffled_indices = torch.randperm(total_samples).tolist()

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
    
    # To be consistent with resume_pretrain.py, return sentinel value instead of raising StopIteration
    return None, -1, -1

def train_model_parallel(data_files, use_streaming=False, use_compile=True):
    """
    data_files should be a list of actual .arrow file paths, e.g.
    ["data/file1.arrow", "data/file2.arrow", ...]
    
    Includes automatic batch size adjustment when OOM errors occur.
    
    Args:
        data_files: List of .arrow file paths
        use_streaming: Whether to use streaming mode or load all data in memory
        use_compile: Whether to use torch.compile() for model optimization
    """
    # Initial batch size settings
    initial_batch_size = 512  # initial batch size
    min_batch_size = 12  # Minimum acceptable batch size
    batch_size = initial_batch_size  # Current working batch size
    
    # 1) If no tokenizer, train it on text extracted from Arrow
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("No existing tokenizer found. Building a text file from Arrow and training one...")
        # Create a text file from Arrow files
        text_file_path = "all_text_for_tokenizer.txt"
        create_text_file_from_arrow(data_files, text_file_path)
        # Now train BPE on that text file
        train_bpe_tokenizer(text_file_path, vocab_size=12000)

    # Load the tokenizer we just created (or found)
    hf_tokenizer = load_bpe_tokenizer()

    epochs = 3
    block_size = 2048
    n_layer = 16
    n_head = 16
    n_embd = 1296
    dropout = 0.1

    config_model = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    
    # Load non-streaming dataset once, outside the retry loop
    tokenized_data = None
    if not use_streaming:
        print("=== Loading dataset in memory for a full pass approach ===")
        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=128)
        total_samples = len(tokenized_data)
        print(f"Total tokenized samples: {total_samples}")
    
    # Token counting variables
    total_tokens_processed = 0
    
    # Initialize data position tracker
    data_position = DataPosition(streaming=use_streaming)
    
    # Main training loop with batch size adjustment
    while True:
        print(f"\n=== Attempting training with batch_size = {batch_size} ===")
        
        try:
            # Initialize a fresh model for each attempt
            model = ArgonneModel(config_model)
            
            # Log GPU info before distribution
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Distribute model across GPUs
            model.distribute_model()  # chunks across all visible GPUs
            
            # Log the device placement
            print("\nModel distribution:")
            first_device = model.devices[0]
            print(f"Token embedding on device: {next(model.token_embedding.parameters()).device}")
            print(f"Position embedding on device: {model.position_embedding.device}")
            for i, stage in enumerate(model.pipeline_stages):
                device = next(stage.parameters()).device
                print(f"Pipeline stage {i} on device: {device}")
            print(f"Final LayerNorm on device: {next(model.ln_f.parameters()).device}")
            print(f"Head on device: {next(model.head.parameters()).device}")
            
            # Apply torch.compile() for speed optimization
            if use_compile:
                # Check if PyTorch version supports compile
                if hasattr(torch, 'compile'):
                    print("Applying torch.compile() to optimize model execution...")
                    try:
                        # Use default mode for a balance between compilation time and execution speed
                        model = torch.compile(model, mode="default")
                        print("Model compilation successful!")
                    except Exception as e:
                        print(f"Failed to compile model: {e}")
                        print("Continuing with uncompiled model.")
                else:
                    print("torch.compile() not available in this PyTorch version. Continuing with uncompiled model.")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1) 
            scaler = torch.amp.GradScaler("cuda")
            global_step = 0
            tokens_in_current_attempt = 0  # Track tokens in this training attempt
            first_device = model.devices[0]  # Store the first device for consistency

            if use_streaming:
                ########################################################
                # STREAMING MODE
                ########################################################
                steps_per_epoch = 50000
                
                for epoch in tqdm(range(epochs)):
                    print(f"==== STREAMING with batch_size={batch_size} ====")
                    # Pass tracking info to the generator
                    token_gen = streaming_token_generator(
                        data_files, 
                        hf_tokenizer, 
                        data_position.current_file_idx,
                        data_position.position_in_file
                    )
                    step_in_epoch = 0
                    token_batch = []

                    while step_in_epoch < steps_per_epoch:
                        try:
                            # Check for end-of-data sentinel value
                            tokens, file_idx, position = next(token_gen)
                            
                            # Check for end-of-data sentinel value
                            if file_idx == -1:
                                print("Reached end of dataset. Restarting from beginning.")
                                # Update tracker for new pass
                                data_position.next_epoch()
                                print(f"Starting new data pass")
                                token_gen = streaming_token_generator(data_files, hf_tokenizer)
                                continue
                                
                            token_batch.append(tokens)
                            
                            # Update position tracker with current file and position
                            data_position.update_streaming_position(
                                file_idx, 
                                position,
                                data_files[file_idx]
                            )

                            if len(token_batch) == batch_size:
                                x_tens, y_tens = collate_batch(token_batch, block_size)
                                token_batch.clear()
                                if x_tens is None:
                                    continue

                                # Count tokens processed in this batch
                                batch_tokens = x_tens.numel()
                                tokens_in_current_attempt += batch_tokens
                                
                                x_tens, y_tens = x_tens.to(first_device), y_tens.to(first_device)

                                optimizer.zero_grad()
                                with torch.amp.autocast("cuda"):
                                    logits, loss = model(x_tens, y_tens)
                                    # Ensure loss is on the first device
                                    loss = loss.to(first_device)

                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                                global_step += 1
                                step_in_epoch += 1
                                
                                if global_step % 50 == 0:
                                    print(f"Step {global_step} | Loss: {loss.item():.4f} | Tokens processed: {tokens_in_current_attempt:,}")
                                    print(f"File: {file_idx}/{len(data_files)}, Position: {position}")
                                    prompt_str = "Long long time ago, "
                                    token_ids = hf_tokenizer.encode(prompt_str)
                                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                                   
                                    generated = model.generate(prompt_tensor, max_new_tokens=100)
                                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                                if global_step % 300 == 0:
                                    # Include token count and data position in checkpoint
                                    checkpoint = {
                                        "epoch": epoch,
                                        "global_step": global_step,
                                        "batch_size": batch_size,
                                        "tokens_processed": tokens_in_current_attempt,
                                        "model_state_dict": model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "loss": loss.item(),
                                        "data_position": data_position.get_state()  # Save position
                                    }
                                    os.makedirs("pretrained", exist_ok=True)
                                    torch.save(checkpoint, f"pretrained/streaming_checkpoint_step_{global_step}.pth")
                                    print(f"Checkpoint saved at step {global_step} with data position tracking")

                        except StopIteration:
                            print("Reached end of dataset (stream) before finishing this epoch.")
                            # Update position tracker for next epoch
                            data_position.next_epoch()
                            break

            else:
                ########################################################
                # NON-STREAMING MODE: full pass each epoch
                ########################################################
                batches_per_epoch = total_samples // batch_size

                for epoch in tqdm(range(epochs)):
                    print(f"==== Starting epoch {epoch} (NON-STREAMING) with batch_size={batch_size} ====")
                    
                    # Get shuffled indices for this epoch
                    indices = data_position.generate_shuffled_indices(total_samples)
                    
                    # Update epoch in position tracker
                    data_position.epoch = epoch
                    
                    for batch_idx in tqdm(range(batches_per_epoch)):
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
                        
                        # Update position in tracker
                        data_position.update_nonstreaming_position(end_idx)
                        
                        batch_token_lists = tokenized_data[start_idx:end_idx]

                        x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                        if x_tens is None:
                            continue

                        # Count tokens processed in this batch
                        batch_tokens = x_tens.numel()
                        tokens_in_current_attempt += batch_tokens
                        
                        x_tens = x_tens.to(first_device)
                        y_tens = y_tens.to(first_device)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            logits, loss = model(x_tens, y_tens)
                            # Ensure loss is on the first device
                            loss = loss.to(first_device)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1

                        if global_step % 50 == 0:
                            print(f"global_step {global_step} | Loss: {loss.item():.4f} | Tokens processed: {tokens_in_current_attempt:,}")
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                            
                            generated = model.generate(prompt_tensor, max_new_tokens=50)
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                        if global_step % 2000 == 0:
                            # Include token count and data position in checkpoint
                            checkpoint = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "batch_size": batch_size,
                                "tokens_processed": tokens_in_current_attempt,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item(),
                                "data_position": data_position.get_state()  # Save position
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            torch.save(checkpoint, f"pretrained/non_streaming_checkpoint_step_{global_step}.pth")
                            print(f"Checkpoint saved at step {global_step} with data position tracking")
            
            # If we reach here, training completed successfully
            # Update total token count for successful training
            total_tokens_processed = tokens_in_current_attempt
            print(f"Training completed successfully with batch_size={batch_size}")
            print(f"Total tokens processed during training: {total_tokens_processed:,}")
            break
            
        except torch.cuda.OutOfMemoryError:
            # Free memory
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            
            # Reduce batch size
            new_batch_size = max(batch_size - 12, min_batch_size)
            
            if new_batch_size == batch_size:
                print(f"Already at minimum batch size ({min_batch_size}). Training failed.")
                break
                
            print(f"CUDA Out of Memory! Reducing batch size from {batch_size} to {new_batch_size}")
            batch_size = new_batch_size
            
            # Short pause to ensure memory is freed
            time.sleep(5)
        
        except RuntimeError as e:
            print(f"Runtime error occurred: {str(e)}")
            if "Expected all tensors to be on the same device" in str(e):
                print("\nDevice mismatch error detected. This might be due to improper tensor movement between pipeline stages.")
                print("Check the error message for which devices are mismatched and verify the model distribution.")
            raise e  # Re-raise to see the full stack trace

    # Save token count to a file for reporting
    training_stats = {
        "total_tokens": total_tokens_processed,
        "batch_size": batch_size,
        "epochs": epochs,
        "global_steps": global_step,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "model_params": sum(p.numel() for p in model.parameters())
    }
    
    # Write stats to JSON file
    os.makedirs("stats", exist_ok=True)
    with open("stats/training_stats.json", "w") as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"\n===== TRAINING SUMMARY =====")
    print(f"Total tokens processed: {total_tokens_processed:,}")
    print(f"Model parameters: {training_stats['model_params']:,}")
    print(f"Epochs completed: {epochs}")
    print(f"Final batch size: {batch_size}")
    print(f"Training steps: {global_step}")
    print(f"Stats saved to: stats/training_stats.json")

    # Save final model and tokenizer
    try:
        model = model.half() # Convert from FP32 to FP16
        model.save_pretrained("Argonne_LLM")
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print("Model-parallel training complete; model and tokenizer saved successfully.")
    except:
        print("Failed to save final model, likely due to OOM issues.")



def main():
    # Expand .arrow files via glob
    data_files = glob.glob("data/*.arrow")
    if not data_files:
        raise ValueError("No files matched the pattern 'data/*.arrow'")
    
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
    print(f"Files will be processed in order. First file: {data_files[0]}")
    
    # Added use_compile parameter with default True
    train_model_parallel(data_files=data_files, use_streaming=True, use_compile=True)

if __name__ == "__main__":
    main()