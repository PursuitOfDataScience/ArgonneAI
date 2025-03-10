import os
import torch
from tqdm import tqdm
import time
import glob
import json
from data_processing import collate_batch, load_bpe_tokenizer, train_bpe_tokenizer, load_nonstream_data, create_text_file_from_arrow, streaming_token_generator
from model import ArgonneConfig, ArgonneModel

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
                    token_gen = streaming_token_generator(data_files, hf_tokenizer)
                    step_in_epoch = 0
                    token_batch = []

                    while step_in_epoch < steps_per_epoch:
                        try:
                            tokens = next(token_gen)
                            token_batch.append(tokens)

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
                                    prompt_str = "Long long time ago, "
                                    token_ids = hf_tokenizer.encode(prompt_str)
                                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                                   
                                    generated = model.generate(prompt_tensor, max_new_tokens=100)
                                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                                if global_step % 300 == 0:
                                    # Include token count in checkpoint
                                    checkpoint = {
                                        "epoch": epoch,
                                        "global_step": global_step,
                                        "batch_size": batch_size,
                                        "tokens_processed": tokens_in_current_attempt,
                                        "model_state_dict": model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "loss": loss.item()
                                    }
                                    os.makedirs("pretrained", exist_ok=True)
                                    torch.save(checkpoint, f"pretrained/streaming_checkpoint_step_{global_step}.pth")
                                    print(f"Checkpoint saved at step {global_step}")

                        except StopIteration:
                            print("Reached end of dataset (stream) before finishing this epoch.")
                            break

            else:
                ########################################################
                # NON-STREAMING MODE: full pass each epoch
                ########################################################
                batches_per_epoch = total_samples // batch_size

                for epoch in tqdm(range(epochs)):
                    print(f"==== Starting epoch {epoch} (NON-STREAMING) with batch_size={batch_size} ====")

                    for batch_idx in tqdm(range(batches_per_epoch)):
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
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
                            # Include token count in checkpoint
                            checkpoint = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "batch_size": batch_size,
                                "tokens_processed": tokens_in_current_attempt,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item()
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            torch.save(checkpoint, f"pretrained/non_streaming_checkpoint_step_{global_step}.pth")
                            print(f"Checkpoint saved at step {global_step}")
            
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

    # Added use_compile parameter with default True
    train_model_parallel(data_files=data_files, use_streaming=True, use_compile=True)

if __name__ == "__main__":
    main()