import os
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import glob

# Import our modules
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data, streaming_token_generator
from model import ArgonneConfig, ArgonneModel
from triton_model import TritonArgonneModel, is_triton_supported
from triton_training_wrapper import convert_model_to_triton

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def resume_pretrain_with_triton(
    data_pattern="data/*.arrow",
    checkpoint_path=None,
    total_training_steps=80_000,
    block_size=2048,
    batch_size=512,
    lr=3e-5,
    use_streaming=True, 
    use_compile=True,
    num_proc=8
):
    """
    Resume pretraining with Triton acceleration
    
    Args:
        data_pattern: Glob pattern for data files
        checkpoint_path: Path to checkpoint to resume from
        total_training_steps: Target total training steps
        block_size: Context window size
        batch_size: Batch size
        lr: Learning rate
        use_streaming: Whether to use streaming data mode
        use_compile: Whether to use torch.compile()
        num_proc: Number of processes for data loading
    """
    # 1) Check if Triton is available
    triton_available = is_triton_supported()
    if not triton_available:
        print("WARNING: Triton is not available on this system. Will use standard PyTorch operations.")
        print("For Triton support, you need a GPU with compute capability >= 7.0")
        print("and to install Triton: pip install triton")
    else:
        print("✓ Triton acceleration is available and will be used")
        
    # 2) Load tokenizer
    hf_tokenizer = load_bpe_tokenizer()

    # 3) Expand data files pattern
    data_files = glob.glob(data_pattern)
    if not data_files:
        raise ValueError(f"No files matched the pattern '{data_pattern}'")
    print(f"Found {len(data_files)} data files")

    # 4) Validate checkpoint
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Resuming from: {checkpoint_path}")

    # 5) Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # 6) Build config & base model
    if "config" in ckpt:
        config = ArgonneConfig(**ckpt["config"])
    else:
        config = ArgonneConfig(
            vocab_size=12000,
            block_size=block_size,
            n_layer=16,
            n_head=16,
            n_embd=1296,
            dropout=0.1
        )
        
    # 7) Create standard model first
    base_model = ArgonneModel(config)
    
    # 8) Convert compiled model state dict if present
    if any(k.startswith("_orig_mod.") for k in ckpt["model_state_dict"].keys()):
        print("Detected compiled model checkpoint, converting parameter names...")
        new_state_dict = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("_orig_mod.") and "pipeline_stages" not in k:
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            elif not k.startswith("_orig_mod.pipeline_stages"):
                new_state_dict[k] = v
        ckpt["model_state_dict"] = new_state_dict
        print("Checkpoint parameter names converted successfully")

    # 9) Load state dict into base model
    base_model.load_state_dict(ckpt["model_state_dict"])
    
    # 10) Distribute model BEFORE creating optimizer
    base_model.distribute_model()  # Make sure model is distributed across GPUs first
    
    # 11) AFTER distribution, convert to Triton if available
    model = convert_model_to_triton(base_model) if triton_available else base_model
    
    # 12) NOW create optimizer with already-distributed parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # 13) Load optimizer state
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    # 14) CRITICAL FIX: Move optimizer states to match parameter devices
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param in optimizer.state:
                for state_key, state_val in optimizer.state[param].items():
                    if isinstance(state_val, torch.Tensor):
                        optimizer.state[param][state_key] = state_val.to(param.device)

    # 15) Get global step and token count from checkpoint
    global_step = ckpt.get("global_step", 0)
    total_tokens_processed = ckpt.get("tokens_processed", 0)
    print(f"Loaded checkpoint at global_step={global_step}, tokens_processed={total_tokens_processed:,}")

    # 16) Log GPU info
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 17) GradScaler for mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # 18) Try to apply torch.compile() if requested
    if use_compile and hasattr(torch, 'compile'):
        try:
            print("Applying torch.compile() to optimize model execution...")
            model = torch.compile(model)
            print("Model successfully compiled!")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Continuing with uncompiled model.")

    first_device = model.devices[0]

    # 19) Token counting variables
    tokens_in_this_session = 0

    # 20) Decide streaming vs non-streaming
    if use_streaming:
        print(f"=== Resuming training from global step {global_step} in STREAMING mode ===")
        print(f"=== Will train until reaching {total_training_steps} steps ===")
        
        # Train until we reach the target number of steps
        token_gen = streaming_token_generator(data_files, hf_tokenizer)
        token_buffer = []

        # Use tqdm to track global step progress toward total_training_steps
        with tqdm(initial=global_step, total=total_training_steps, desc="Training") as pbar:
            while global_step < total_training_steps:
                try:
                    tokens = next(token_gen)
                    token_buffer.append(tokens)

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
                        pbar.update(1)

                        if global_step % 50 == 0:
                            current_total_tokens = total_tokens_processed + tokens_in_this_session
                            print(f"Step {global_step} | Loss: {loss.item():.4f} | Tokens: {current_total_tokens:,}")
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
                                "triton_accelerated": triton_available
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            save_path = f"pretrained/triton_checkpoint_step_{global_step}.pth"
                            torch.save(ckpt_dict, save_path)
                            print(f"Checkpoint saved @ step {global_step} -> {save_path}")
                            
                            # Update stats
                            update_training_stats(
                                tokens=current_total_tokens,
                                batch_size=batch_size,
                                steps=global_step,
                                model=model,
                                triton_acceleration=triton_available
                            )

                except StopIteration:
                    print("Reached end of dataset stream. Restarting data generator.")
                    token_gen = streaming_token_generator(data_files, hf_tokenizer)
                    continue

    else:
        print(f"=== Resuming training from global step {global_step} in NON-STREAMING mode ===")
        print(f"=== Will train until reaching {total_training_steps} steps ===")

        # 1) Load entire data in memory. Possibly parallel map with `num_proc`.
        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=num_proc)
        total_samples = len(tokenized_data)
        print(f"Total in-memory tokenized samples: {total_samples}")

        # Calculate how many full passes we need
        batches_per_epoch = total_samples // batch_size
        remaining_steps = total_training_steps - global_step
        
        # Use tqdm to track global step progress toward total_training_steps
        with tqdm(initial=global_step, total=total_training_steps, desc="Training") as pbar:
            while global_step < total_training_steps:
                # Shuffle data for each epoch
                indices = torch.randperm(total_samples)
                
                for idx in range(0, total_samples, batch_size):
                    if global_step >= total_training_steps:
                        break
                        
                    # Get batch indices
                    batch_indices = indices[idx:min(idx + batch_size, total_samples)]
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
                    pbar.update(1)
                    
                    if global_step % 50 == 0:
                        current_total_tokens = total_tokens_processed + tokens_in_this_session
                        print(f"Step {global_step} | Loss: {loss.item():.4f} | Tokens: {current_total_tokens:,}")
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
                            "triton_accelerated": triton_available
                        }
                        os.makedirs("pretrained", exist_ok=True)
                        save_path = f"pretrained/triton_checkpoint_step_{global_step}.pth"
                        torch.save(ckpt_dict, save_path)
                        print(f"Checkpoint saved @ step {global_step} -> {save_path}")
                        
                        # Update stats
                        update_training_stats(
                            tokens=current_total_tokens,
                            batch_size=batch_size,
                            steps=global_step,
                            model=model,
                            triton_acceleration=triton_available
                        )

    # Final token count calculation
    final_token_count = total_tokens_processed + tokens_in_this_session
    
    # Update final stats
    update_training_stats(
        tokens=final_token_count,
        batch_size=batch_size,
        steps=global_step,
        model=model,
        triton_acceleration=triton_available,
        final=True
    )
    
    print(f"\n===== TRAINING SUMMARY =====")
    print(f"Total tokens processed: {final_token_count:,}")
    print(f"Final step count: {global_step}")
    print(f"Triton acceleration: {'Yes' if triton_available else 'No'}")
    print(f"Training complete!")

    # Perform final save at the end of training
    try:
        model = model.half()  # Convert to FP16
        model.save_pretrained("Argonne_LLM")
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print(f"Training completed at step {global_step}. Final model saved.")
    except Exception as e:
        print(f"Failed to save final model: {e}")


def update_training_stats(tokens, batch_size, steps, model, triton_acceleration=False, final=False):
    """Update the training statistics file with current information"""
    # Calculate model parameters
    model_params = sum(p.numel() for p in model.parameters())
    
    # Extract config information
    if hasattr(model, 'config'):
        config = model.config
        n_layer = getattr(config, 'n_layer', 16)
        n_head = getattr(config, 'n_head', 16)
        n_embd = getattr(config, 'n_embd', 1296)
    else:
        n_layer = 16
        n_head = 16
        n_embd = 1296
    
    training_stats = {
        "total_tokens": tokens,
        "batch_size": batch_size,
        "global_steps": steps,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "model_params": model_params,
        "final_training": final,
        "triton_accelerated": triton_acceleration
    }
    
    # Write stats to JSON file
    os.makedirs("stats", exist_ok=True)
    
    # For the final update, create a timestamped file
    if final:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats/final_training_stats_{timestamp}.json"
    else:
        filename = "stats/current_training_stats.json"
        
    with open(filename, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    if final:
        print(f"Final training stats saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Resume pretraining with Triton acceleration")
    
    parser.add_argument("--data_pattern", type=str, default="data/*.arrow",
                      help="Pattern for data files")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to checkpoint file to resume from")
    parser.add_argument("--total_steps", type=int, default=80_000,
                      help="Total number of training steps")
    parser.add_argument("--block_size", type=int, default=2048,
                      help="Context window size")
    parser.add_argument("--batch_size", type=int, default=320,
                      help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5,
                      help="Learning rate")
    parser.add_argument("--no_streaming", action="store_true",
                      help="Disable streaming mode")
    parser.add_argument("--no_compile", action="store_true",
                      help="Disable torch.compile")
    parser.add_argument("--num_proc", type=int, default=8,
                      help="Number of processes for data loading")
    
    args = parser.parse_args()
    
    resume_pretrain_with_triton(
        data_pattern=args.data_pattern,
        checkpoint_path=args.checkpoint,
        total_training_steps=args.total_steps,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        use_streaming=not args.no_streaming,
        use_compile=not args.no_compile,
        num_proc=args.num_proc
    )

if __name__ == "__main__":
    main()
