import os
import torch
import torch.nn as nn
import argparse
import time
import glob
import json
import warnings
import math
from tqdm import tqdm
import torch.nn.functional as F

# Make sure we can import our utils
import sys

# Import existing training functionality
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data, streaming_token_generator
from model import ArgonneConfig, ArgonneModel
from triton_kernels import is_triton_supported
from triton_model import TritonArgonneModel

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def convert_model_to_triton(model):
    """
    Convert a standard model to use Triton acceleration
    
    Args:
        model: The standard ArgonneModel to convert
        
    Returns:
        TritonArgonneModel if Triton is supported, otherwise the original model
    """
    if is_triton_supported():
        print("Converting model to use Triton acceleration...")
        
        # Check if model is already distributed
        if model.pipeline_stages is not None:
            # For already distributed models, we need special handling
            print("Model is already distributed - using specialized conversion")
            return TritonArgonneModel.from_model(model)
        else:
            # First convert to Triton model
            triton_model = TritonArgonneModel.from_model(model)
            return triton_model
    else:
        warnings.warn("Triton is not supported on this system. Using standard model.")
        return model

def training_step(model, batch_x, batch_y, optimizer, scaler, device, grad_clip=1.0):
    """
    Execute a single training step with gradient clipping and error handling
    
    Args:
        model: The model to train
        batch_x: Input tensor
        batch_y: Target tensor
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device for training
        grad_clip: Maximum norm for gradient clipping
        
    Returns:
        loss_val: Loss value (float)
    """
    # Ensure inputs are on the correct device
    x_tens = batch_x.to(device, non_blocking=True)
    y_tens = batch_y.to(device, non_blocking=True)
            
    # Zero gradients, using set_to_none for memory efficiency
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass with mixed precision
    with torch.amp.autocast("cuda"):
        logits, loss = model(x_tens, y_tens)
        # Ensure loss is on the correct device
        loss = loss.to(device)
    
    # Check for NaN/Inf values - early detection helps debug issues
    if torch.isnan(loss) or torch.isinf(loss):
        warnings.warn(f"Warning: NaN or Inf detected in loss: {loss.item()}. Consider reducing learning rate.")
        return float('nan')
    
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    
    # Unscale gradients for clipping
    scaler.unscale_(optimizer)
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Step optimizer and update scaler
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

def train_model_with_triton(
    data_files,
    use_streaming=False, 
    use_compile=False,  # Default to False for better stability
    batch_size=320,     # More moderate default batch size
    epochs=3,
    block_size=2048,
    n_layer=16,
    n_head=16,
    n_embd=1296,
    dropout=0.1,
    lr=1e-5,
    weight_decay=0.1,
    grad_clip=1.0,      # Add gradient clipping
    warmup_steps=100,   # Add LR warmup
    checkpoint_path=None
):
    """
    Trains a model with Triton acceleration.
    This function wraps the standard training process but uses Triton-accelerated model components.
    
    Args:
        data_files: List of .arrow file paths
        use_streaming: Whether to use streaming mode or load all data in memory
        use_compile: Whether to use torch.compile() for model optimization
        batch_size: Starting batch size - will adjust if OOM occurs
        epochs: Number of epochs for training
        block_size: Context window size
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension size
        dropout: Dropout probability
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        grad_clip: Maximum gradient norm
        warmup_steps: LR warmup steps
        checkpoint_path: Path to checkpoint to resume from
    """
    print(f"\n=== Training with Triton Acceleration ===")
    print(f"Batch Size: {batch_size}")
    print(f"Model: {n_layer} layers, {n_head} heads, {n_embd} embedding dim")
    
    # Check Triton availability
    triton_available = is_triton_supported()
    
    if triton_available:
        print("✓ Triton acceleration is available and will be used")
        # IMPORTANT: Disable torch.compile when using Triton to avoid memory access conflicts
        if use_compile:
            print("⚠️ Disabling torch.compile() to prevent conflicts with Triton acceleration")
            use_compile = False
    else:
        warnings.warn(
            "\nTriton is not supported on this system. Will train with standard PyTorch implementation.\n"
            "For Triton support, you need a GPU with compute capability >= 7.0 and to install triton:\n"
            "pip install triton\n"
        )
    
    # Load the tokenizer
    hf_tokenizer = load_bpe_tokenizer()

    # Create model config
    config_model = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    
    # Load dataset if not streaming
    tokenized_data = None
    if not use_streaming:
        print("=== Loading dataset in memory for a full pass approach ===")
        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=8)
        total_samples = len(tokenized_data)
        print(f"Total tokenized samples: {total_samples}")
    
    # Initialize model
    global_step = 0
    total_tokens_processed = 0
    
    if checkpoint_path:
        print(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle both standard and Triton model checkpoints
        if "config" in checkpoint:
            config_model = ArgonneConfig(**checkpoint["config"])
        
        # Create standard model first
        model = ArgonneModel(config_model)
        
        # Handle compiled model states
        if any(k.startswith("_orig_mod.") for k in checkpoint["model_state_dict"].keys()):
            print("Detected compiled model checkpoint, converting parameter names...")
            new_state_dict = {}
            for k, v in checkpoint["model_state_dict"].items():
                if k.startswith("_orig_mod.") and "pipeline_stages" not in k:
                    new_key = k.replace("_orig_mod.", "")
                    new_state_dict[new_key] = v
                elif not k.startswith("_orig_mod.pipeline_stages"):
                    new_state_dict[k] = v
            checkpoint["model_state_dict"] = new_state_dict
            
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Get training state
        global_step = checkpoint.get("global_step", 0)
        total_tokens_processed = checkpoint.get("tokens_processed", 0)
        
        print(f"Loaded checkpoint: step={global_step}, tokens={total_tokens_processed:,}")
    else:
        # Create new model
        model = ArgonneModel(config_model)
    
    # Log GPU info
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # IMPORTANT: If we're using Triton, convert BEFORE distributing
    if triton_available:
        model = convert_model_to_triton(model)
    
    # Verify tensors are on the expected devices
    print("=== Checking tensor device placement ===")
    # Check embedding placement
    embedding_device = next(model.token_embedding.parameters()).device
    print(f"Token embedding on device: {embedding_device}")
    
    # Now distribute model across GPUs
    model.distribute_model()
    
    # Set up learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        return 1.0  # Constant after warmup
    
    # Create optimizer with distributed parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.95)  # Slightly more stable beta values
    )
    
    # Create LR scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Load optimizer state if resuming
    if checkpoint_path and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Fix optimizer states to be on correct devices
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in optimizer.state:
                    for state_key, state_val in optimizer.state[param].items():
                        if isinstance(state_val, torch.Tensor):
                            optimizer.state[param][state_key] = state_val.to(param.device)
    
    # Apply torch.compile if requested - but only if Triton isn't being used
    if use_compile and hasattr(torch, 'compile'):
        try:
            print("Applying torch.compile() for additional optimizations...")
            model = torch.compile(model)
            print("Model successfully compiled!")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Continuing with uncompiled model.")
    
    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda")
    first_device = model.devices[0]
    tokens_in_this_session = 0
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Training loop
    if use_streaming:
        print(f"=== Starting streaming training with Triton acceleration ===")
        steps_per_epoch = 50000
        
        for epoch in range(epochs):
            token_gen = streaming_token_generator(data_files, hf_tokenizer)
            step_in_epoch = 0
            token_batch = []
            
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                while step_in_epoch < steps_per_epoch:
                    try:
                        tokens = next(token_gen)
                        token_batch.append(tokens)
                        
                        if len(token_batch) == batch_size:
                            x_tens, y_tens = collate_batch(token_batch, block_size)
                            token_batch.clear()
                            if x_tens is None:
                                continue
                            
                            # Count tokens
                            batch_tokens = x_tens.numel()
                            tokens_in_this_session += batch_tokens
                            
                            # Execute training step
                            loss_val = training_step(
                                model, x_tens, y_tens, optimizer, scaler, first_device, grad_clip
                            )
                            
                            # Update learning rate
                            scheduler.step()
                            current_lr = scheduler.get_last_lr()[0]
                            
                            global_step += 1
                            step_in_epoch += 1
                            pbar.update(1)
                            
                            if global_step % 50 == 0:
                                print(f"Step {global_step} | Loss: {loss_val:.4f} | LR: {current_lr:.2e} | "
                                     f"Tokens: {total_tokens_processed + tokens_in_this_session:,}")
                                
                                # Generate sample - but only if loss is not NaN
                                if not math.isnan(loss_val):
                                    try:
                                        # Temporarily switch model to eval mode
                                        model.eval()
                                        
                                        prompt_str = "Long long time ago, "
                                        token_ids = hf_tokenizer.encode(prompt_str)
                                        prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                                        
                                        # Generate with greedy decoding for stability
                                        with torch.no_grad():
                                            generated = model.generate(
                                                prompt_tensor, 
                                                max_new_tokens=50,
                                                temperature=0.8,
                                                sample=False  # Use greedy decoding for stability
                                            )
                                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")
                                        
                                        # Switch back to training mode
                                        model.train()
                                    except Exception as e:
                                        print(f"Generation error: {e}")
                            
                            if global_step % 300 == 0:
                                # Save checkpoint with tokens processed
                                checkpoint = {
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "batch_size": batch_size,
                                    "tokens_processed": total_tokens_processed + tokens_in_this_session,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "loss": loss_val,
                                    "lr": current_lr,
                                    "config": config_model.to_dict()
                                }
                                os.makedirs("pretrained", exist_ok=True)
                                torch.save(checkpoint, f"pretrained/triton_checkpoint_step_{global_step}.pth")
                                print(f"Checkpoint saved at step {global_step}")
                                
                    except StopIteration:
                        print("End of dataset stream reached. Restarting data generator.")
                        token_gen = streaming_token_generator(data_files, hf_tokenizer)
                        
    else:
        # Non-streaming mode
        print(f"=== Starting non-streaming training with Triton acceleration ===")
        batches_per_epoch = total_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle dataset indices for each epoch
            indices = torch.randperm(total_samples)
            
            with tqdm(total=batches_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx in range(batches_per_epoch):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    batch_token_lists = [tokenized_data[i] for i in batch_indices]
                    x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                    if x_tens is None:
                        continue
                    
                    # Count tokens
                    batch_tokens = x_tens.numel()
                    tokens_in_this_session += batch_tokens
                    
                    # Execute training step 
                    loss_val = training_step(
                        model, x_tens, y_tens, optimizer, scaler, first_device, grad_clip
                    )
                    
                    # Update learning rate
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    
                    global_step += 1
                    pbar.update(1)
                    
                    if global_step % 50 == 0:
                        print(f"Step {global_step} | Loss: {loss_val:.4f} | LR: {current_lr:.2e} | "
                             f"Tokens: {total_tokens_processed + tokens_in_this_session:,}")
                        
                        # Generate sample - but only if loss is not NaN
                        if not math.isnan(loss_val):
                            try:
                                # Temporarily switch model to eval mode
                                model.eval()
                                
                                prompt_str = "Long long time ago, "
                                token_ids = hf_tokenizer.encode(prompt_str)
                                prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                                
                                # Generate with greedy decoding for stability
                                with torch.no_grad():
                                    generated = model.generate(
                                        prompt_tensor, 
                                        max_new_tokens=50,
                                        temperature=0.8,
                                        sample=False  # Use greedy decoding for stability
                                    )
                                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")
                                
                                # Switch back to training mode
                                model.train()
                            except Exception as e:
                                print(f"Generation error: {e}")
                    
                    if global_step % 1000 == 0:
                        # Save checkpoint with tokens processed
                        checkpoint = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_size": batch_size,
                            "tokens_processed": total_tokens_processed + tokens_in_this_session,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss_val,
                            "lr": current_lr,
                            "config": config_model.to_dict()
                        }
                        os.makedirs("pretrained", exist_ok=True)
                        torch.save(checkpoint, f"pretrained/triton_checkpoint_step_{global_step}.pth")
                        print(f"Checkpoint saved at step {global_step}")
    
    # Training complete - update token count
    total_tokens_processed += tokens_in_this_session
    
    # Save final model
    try:
        model.eval()  # Switch to eval mode before saving
        model = model.half()  # Convert to FP16 for storage
        model.save_pretrained("Argonne_LLM_Triton")
        hf_tokenizer.save_pretrained("Argonne_LLM_Triton")
        
        # Save training stats
        training_stats = {
            "total_tokens": total_tokens_processed,
            "batch_size": batch_size,
            "epochs": epochs,
            "global_steps": global_step,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "model_params": sum(p.numel() for p in model.parameters()),
            "triton_accelerated": triton_available
        }
        
        with open(os.path.join("Argonne_LLM_Triton", "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=2)
            
        print(f"\n===== TRAINING SUMMARY =====")
        print(f"Total tokens processed: {total_tokens_processed:,}")
        print(f"Model parameters: {training_stats['model_params']:,}")
        print(f"Epochs completed: {epochs}")
        print(f"Final batch size: {batch_size}")
        print(f"Training steps: {global_step}")
        print(f"Triton acceleration: {'Yes' if triton_available else 'No'}")
        print(f"Model saved to: Argonne_LLM_Triton")
            
    except Exception as e:
        print(f"Failed to save final model: {e}")
        
    return model

def resume_triton_training(
    data_files,
    checkpoint_path,
    use_streaming=True,
    use_compile=False,
    batch_size=None,
    total_steps=None,
    lr=None,
    grad_clip=1.0,
    warmup_steps=100
):
    """
    Resume training with a Triton-accelerated model from a checkpoint.
    
    Args:
        data_files: List of data file paths
        checkpoint_path: Path to checkpoint to resume from
        use_streaming: Whether to use streaming mode
        use_compile: Whether to additionally use torch.compile
        batch_size: Batch size (will use checkpoint value if None)
        total_steps: Target total steps (will continue indefinitely if None)
        lr: Learning rate (will use checkpoint value if None)
        grad_clip: Maximum gradient norm
        warmup_steps: LR warmup steps
    """
    if checkpoint_path is None:
        raise ValueError("Must provide checkpoint_path to resume training")
        
    print(f"\n=== Resuming Training with Triton Acceleration ===")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get config from checkpoint or create default
    if "config" in checkpoint:
        config_model = ArgonneConfig(**checkpoint["config"])
    else:
        # Use parameters from checkpoint if available
        n_layer = checkpoint.get("n_layer", 16)
        n_head = checkpoint.get("n_head", 16)
        n_embd = checkpoint.get("n_embd", 1296)
        block_size = checkpoint.get("block_size", 2048)
        config_model = ArgonneConfig(
            vocab_size=12000,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.1
        )
    
    # Get other parameters from checkpoint
    orig_batch_size = checkpoint.get("batch_size", 512)
    if batch_size is None:
        batch_size = orig_batch_size
        
    # Get global step and token count
    global_step = checkpoint.get("global_step", 0)
    total_tokens_processed = checkpoint.get("tokens_processed", 0)
    
    # Get learning rate from checkpoint or use provided value
    checkpoint_lr = checkpoint.get("lr", 3e-5)
    if lr is None:
        lr = checkpoint_lr
        
    print(f"Resuming from step {global_step} with {total_tokens_processed:,} tokens processed")
    print(f"Using batch size: {batch_size}, learning rate: {lr}")
    
    # Continue with standard training using the extracted parameters
    return train_model_with_triton(
        data_files=data_files,
        use_streaming=use_streaming,
        use_compile=use_compile,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        lr=lr,
        grad_clip=grad_clip,
        warmup_steps=warmup_steps
    )
    
def main():
    """Parse command-line arguments and start training"""
    parser = argparse.ArgumentParser(description="Train or resume training with Triton acceleration")
    
    # Data and model configuration
    parser.add_argument("--data_pattern", type=str, default="data/*.arrow", 
                        help="Pattern for data files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Training configuration
    parser.add_argument("--streaming", action="store_true", 
                        help="Use streaming mode for data loading")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--batch_size", type=int, default=320,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Learning rate warmup steps")
    
    # Model architecture
    parser.add_argument("--block_size", type=int, default=2048,
                        help="Context window size")
    parser.add_argument("--n_layer", type=int, default=16,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=16,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=1296,
                        help="Embedding dimension")
    
    args = parser.parse_args()
    
    # Import math here since we need it in the script
    import math
    
    # Expand data file pattern
    data_files = glob.glob(args.data_pattern)
    if not data_files:
        raise ValueError(f"No files matched the pattern '{args.data_pattern}'")
    
    print(f"Found {len(data_files)} data files")
    
    # Check if we're resuming or starting fresh
    if args.checkpoint:
        resume_triton_training(
            data_files=data_files,
            checkpoint_path=args.checkpoint,
            use_streaming=args.streaming,
            use_compile=not args.no_compile,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps
        )
    else:
        train_model_with_triton(
            data_files=data_files,
            use_streaming=args.streaming,
            use_compile=not args.no_compile,
            batch_size=args.batch_size,
            epochs=args.epochs,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            lr=args.lr,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps
        )
        
if __name__ == "__main__":
    main()
