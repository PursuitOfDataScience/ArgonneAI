import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
import numpy as np
import multiprocessing
import json
import re
import logging
from datetime import datetime
import argparse



from mp_pretrain import ArgonneModelParallel, ArgonneConfig, load_bpe_tokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("instruct_finetuning")

def get_optimal_num_processes(requested_procs=None):
    """Determine optimal number of processes based on available CPU cores."""
    max_procs = multiprocessing.cpu_count()
    if requested_procs is None:
        # Use 75% of available cores by default
        return max(1, int(max_procs * 0.75))
    else:
        # Ensure we don't exceed available cores
        return min(max_procs, requested_procs)

# ...existing data preparation code...

def fix_pipeline_state_dict(state_dict):
    """Convert pipeline-parallel state_dict keys to match single-GPU model structure."""
    new_state_dict = {}
    
    # Pattern for pipeline stage keys
    pipeline_pattern = r'pipeline_stages\.(\d+)\.(\d+)\.(.*)'
    
    # First pass to detect structure
    processed_blocks = {}
    for key in state_dict.keys():
        if not key.startswith('pipeline_stages.'):
            continue
            
        match = re.match(pipeline_pattern, key)
        if match:
            gpu_idx, block_in_gpu_idx = int(match.group(1)), int(match.group(2))
            processed_blocks.setdefault(gpu_idx, set()).add(block_in_gpu_idx)
    
    # Determine blocks_per_gpu (maximum block index + 1)
    blocks_per_gpu = 1
    if processed_blocks:
        blocks_per_gpu = max(max(indices) for indices in processed_blocks.values()) + 1
        logger.info(f"Detected {blocks_per_gpu} blocks per GPU in the model")
    
    # Second pass to convert keys
    for key, value in state_dict.items():
        if key.startswith('pipeline_stages.'):
            match = re.match(pipeline_pattern, key)
            if match:
                gpu_idx, block_in_gpu_idx, rest = int(match.group(1)), int(match.group(2)), match.group(3)
                # Calculate global block index
                global_block_idx = gpu_idx * blocks_per_gpu + block_in_gpu_idx
                new_key = f'blocks.{global_block_idx}.{rest}'
                new_state_dict[new_key] = value
        else:
            # Copy other weights unchanged
            new_state_dict[key] = value
    
    return new_state_dict

def load_model_direct(model_path, device=None):
    """Load a model directly from files, avoiding from_pretrained issues."""
    logger.info(f"Loading model directly from {model_path}")
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config and model
    config = ArgonneConfig(**config_dict)
    model = ArgonneModelParallel(config)
    
    # Try to load weights
    model_bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_bin_path):
        logger.info(f"Loading weights from {model_bin_path}")
        state_dict = torch.load(model_bin_path, map_location="cpu")
        
        # Check if weights use pipeline structure
        has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
        if has_pipeline:
            logger.info("Converting pipeline structure to blocks structure...")
            state_dict = fix_pipeline_state_dict(state_dict)
        
        model.load_state_dict(state_dict, strict=False)
    else:
        # Try SafeTensors format
        try:
            from safetensors.torch import load_file
            model_safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(model_safetensors_path):
                logger.info(f"Loading weights from {model_safetensors_path}")
                state_dict = load_file(model_safetensors_path)
                
                # Check if weights use pipeline structure
                has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
                if has_pipeline:
                    logger.info("Converting pipeline structure to blocks structure...")
                    state_dict = fix_pipeline_state_dict(state_dict)
                
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")
        except ImportError:
            logger.warning("SafeTensors not available and PyTorch weights not found")
            raise
    
    # Move to device if specified
    if device is not None:
        model.to(device)
    
    # Add devices attribute needed by generate method
    if not hasattr(model, 'devices'):
        if device is not None:
            model.devices = [device]
        else:
            model.devices = [torch.device("cpu")]
    
    return model, config

def setup_ddp(local_rank):
    """Initialize distributed training environment."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logger.info(f"Initialized DDP: rank {rank}/{world_size} on device {device}")
        
        return device, rank, world_size
    else:
        logger.warning("No CUDA available, running on CPU only")
        return torch.device("cpu"), 0, 1

def cleanup_ddp():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def log_memory_usage(device):
    """Log GPU memory usage for the specified device."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, "
                   f"Reserved: {reserved:.2f}GB, "
                   f"Peak: {max_allocated:.2f}GB")

def prepare_oasst_data(data_dir, tokenizer, max_length=2048, num_proc=None):
    """Prepare and format the OpenAssistant data for instruction fine-tuning."""
    # Determine optimal number of processes
    num_proc = get_optimal_num_processes(num_proc)
    print(f"Using {num_proc} processes for data processing")
    
    # Load OpenAssistant datasets
    print(f"Loading OpenAssistant data from {data_dir}")
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    # Print column names and first example
    print(f"Dataset columns: {train_dataset.column_names}")
    print("\nExample data point:")
    print(train_dataset[0])
    
    # Define instruction format
    def format_instruction(example):
        """Format messages as a single instruction-following text."""
        # Check if the dataset has the newer conversation format
        if "conversation" in example:
            conversation_data = example.get("conversation", [])
            if not conversation_data:
                return {"formatted_text": "", "length": 0}
                
            conversation = ""
            for msg in conversation_data:
                role = msg.get("role", "").lower()
                text = msg.get("text", "")
                
                if role == "user":
                    conversation += f"USER: {text}\n\n"
                elif role == "assistant":
                    conversation += f"ASSISTANT: {text}\n\n"
        # Fall back to the original format
        else:
            messages = example.get("messages", [])
            if not messages:
                return {"formatted_text": "", "length": 0}
                
            conversation = ""
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                
                if role == "human" or role == "prompter":
                    conversation += f"USER: {content}\n\n"
                elif role == "assistant" or role == "gpt":
                    conversation += f"ASSISTANT: {content}\n\n"
                
        # Add tokenized version
        tokens = tokenizer.encode(
            conversation, 
            truncation=True, 
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        return {
            "formatted_text": conversation,
            "tokenized_text": tokens,
            "length": len(tokens)
        }
    
    # Process datasets (with multiprocessing)
    columns_to_remove = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_instruction, 
        remove_columns=columns_to_remove,
        num_proc=num_proc,
        desc="Processing training data"
    )
    valid_dataset = valid_dataset.map(
        format_instruction, 
        remove_columns=valid_dataset.column_names,
        num_proc=num_proc,
        desc="Processing validation data"
    )
    
    # Filter out empty examples or too short ones (at least 16 tokens)
    train_dataset = train_dataset.filter(lambda x: x["length"] > 16)
    valid_dataset = valid_dataset.filter(lambda x: x["length"] > 16)
    
    print(f"Processed train dataset size: {len(train_dataset)}")
    print(f"Processed validation dataset size: {len(valid_dataset)}")
    
    # Create final format
    def prepare_for_training(examples):
        """Convert tokenized examples to input_ids and labels."""
        input_ids = examples["tokenized_text"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
    
    train_dataset = train_dataset.map(
        prepare_for_training,
        num_proc=num_proc,
        desc="Preparing training tensors"
    )
    valid_dataset = valid_dataset.map(
        prepare_for_training,
        num_proc=num_proc,
        desc="Preparing validation tensors"
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    return {"train": train_dataset, "validation": valid_dataset}

def finetune_model(model_path, data_dir, output_dir="Argonne_LLM_Finetuned", 
                  batch_size=8, num_proc=None, local_rank=-1):
    """Fine-tune the pretrained model on instruction data with DDP for multi-GPU training."""
    # Setup distributed training if local_rank is provided
    if local_rank != -1:
        device, rank, world_size = setup_ddp(local_rank)
        is_main_process = (rank == 0)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        is_main_process = True
    
    logger.info(f"Using device: {device}")
    
    # Only create output directory in the main process
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer
    if is_main_process:
        logger.info(f"Loading tokenizer...")
    tokenizer = load_bpe_tokenizer()
    
    # Load datasets with multiprocessing (only in main process to avoid duplicated work)
    if is_main_process:
        dataset = prepare_oasst_data(data_dir, tokenizer, num_proc=num_proc)
        train_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
    
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(valid_dataset)}")
    else:
        # For non-main processes, load data after main process completes
        while not os.path.exists(os.path.join(data_dir, "train")) or \
              not os.path.exists(os.path.join(data_dir, "validation")):
            logger.info(f"Rank {rank}: Waiting for datasets to be prepared...")
            import time
            time.sleep(10)
        
        # Now load the pre-processed datasets
        train_dataset = load_from_disk(os.path.join(data_dir, "train"))
        valid_dataset = load_from_disk(os.path.join(data_dir, "validation"))
        
        logger.info(f"Rank {rank}: Loaded datasets")
    
    # Synchronize all processes after datasets are loaded
    if world_size > 1:
        dist.barrier()
    
    # Custom collate function
    def custom_data_collator(features):
        # Extract input_ids
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Get max length in this batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad sequences to the max length in this batch
        padded_input_ids = []
        padded_labels = []
        
        for ids, lbl in zip(input_ids, labels):
            # Padding for input_ids
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([ids, torch.ones(padding_length, dtype=torch.long) * tokenizer.pad_token_id])
            padded_input_ids.append(padded_ids)
            
            # Padding for labels (use -100 for padding positions to ignore in loss)
            padded_lbl = torch.cat([lbl, torch.ones(padding_length, dtype=torch.long) * -100])
            padded_labels.append(padded_lbl)
        
        # Stack tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }
        
        return batch
    
    # Training parameters
    learning_rate = 2e-5
    num_epochs = 3
    min_batch_size = 1
    log_every = 50  # Print loss every 50 steps
    
    # Try with decreasing batch size
    current_batch_size = batch_size
    
    while current_batch_size >= min_batch_size:
        model = None
        optimizer = None
        scaler = None
        
        try:
            logger.info(f"Rank {rank}: Attempting fine-tuning with batch_size={current_batch_size}")
            
            # Set initial training state
            global_step = 0
            best_val_loss = float('inf')
            
            # Load model
            logger.info(f"Rank {rank}: Loading pretrained model from {model_path}")
            model, config = load_model_direct(model_path, device)
            
            # Wrap model with DDP for distributed training
            if world_size > 1:
                model = DDP(model, device_ids=[local_rank])
                logger.info(f"Rank {rank}: Model wrapped with DDP")
            
            # Memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            
            # Setup optimizer with larger eps for stability
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-6)
            
            # Use mixed precision
            scaler = torch.amp.GradScaler('cuda')
            
            # Create data samplers for distributed training
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            ) if world_size > 1 else None
            
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            ) if world_size > 1 else None
            
            # Setup dataloaders
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=current_batch_size, 
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                collate_fn=custom_data_collator
            )
            
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, 
                batch_size=current_batch_size, 
                shuffle=False,
                sampler=valid_sampler,
                collate_fn=custom_data_collator
            )
            
            # Training loop
            for epoch in range(num_epochs):
                # Set epoch for sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                
                model.train()
                train_loss = 0.0
                train_steps = 0
                epoch_start_time = datetime.now()
                
                # Track losses for logging every 50 steps
                loss_history = []
                
                # Training
                progress_bar = tqdm(enumerate(train_dataloader), 
                                   total=len(train_dataloader), 
                                   desc=f"Epoch {epoch+1}/{num_epochs}",
                                   disable=not is_main_process)
                
                for step, batch in progress_bar:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Reset gradients at the start of each step
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        logits, loss = model(input_ids, labels)
                    
                    # No need to scale loss for gradient accumulation
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Add loss to tracking values
                    unscaled_loss = loss.item()  # No need to multiply back
                    train_loss += unscaled_loss
                    loss_history.append(unscaled_loss)
                    train_steps += 1
                    
                    # Update progress bar in main process
                    if is_main_process:
                        progress_bar.set_postfix({"loss": unscaled_loss})
                    
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Step optimizer and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                    
                    global_step += 1
                    
                    # Log losses every 50 steps
                    if global_step % log_every == 0:
                        # Print detailed loss information
                        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
                        
                        if world_size > 1:
                            # Gather losses from all processes
                            loss_tensor = torch.tensor([avg_loss], device=device)
                            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
                            dist.all_gather(gathered_losses, loss_tensor)
                            
                            if is_main_process:
                                all_losses = [l.item() for l in gathered_losses]
                                global_avg_loss = sum(all_losses) / len(all_losses)
                                logger.info(f"Step {global_step}, Epoch {epoch+1}/{num_epochs}, "
                                          f"Train Loss: {global_avg_loss:.4f}")
                        else:
                            logger.info(f"Step {global_step}, Epoch {epoch+1}/{num_epochs}, "
                                      f"Train Loss: {avg_loss:.4f}")
                        
                        # Reset loss history after logging
                        loss_history = []
                        
                        # Log memory usage
                        if is_main_process:
                            log_memory_usage(device)
                    
                    # Checkpointing (only in main process)
                    if is_main_process and global_step % 1000 == 0:
                        # Create checkpoint directory
                        checkpoint_dir = os.path.join(output_dir, f"instruct-checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        # Save model
                        if isinstance(model, DDP):
                            model.module.save_pretrained(checkpoint_dir, safe_serialization=False)
                        else:
                            model.save_pretrained(checkpoint_dir, safe_serialization=False)
                        
                        tokenizer.save_pretrained(checkpoint_dir)
                        logger.info(f"Checkpoint saved at {checkpoint_dir}")
                        
                        # Run validation to evaluate the model periodically
                        val_loss = evaluate_model(model, valid_dataloader, device, is_main_process, world_size, rank)
                        
                        if is_main_process:
                            logger.info(f"Step {global_step}: Validation Loss: {val_loss:.4f}")
                            
                            # Save best model
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                                
                                # Save the model
                                if isinstance(model, DDP):
                                    model.module.save_pretrained(output_dir, safe_serialization=False)
                                else:
                                    model.save_pretrained(output_dir, safe_serialization=False)
                                    
                                tokenizer.save_pretrained(output_dir)
                                logger.info(f"Best model saved to {output_dir}")
                
                # Synchronize processes at end of epoch
                if world_size > 1:
                    dist.barrier()
                
                # Calculate and report average loss across all processes
                if world_size > 1:
                    # Gather loss from all processes
                    avg_loss = train_loss / max(1, train_steps)
                    loss_tensor = torch.tensor([avg_loss], device=device)
                    all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
                    dist.all_gather(all_losses, loss_tensor)
                    
                    if is_main_process:
                        # Average losses from all processes
                        all_losses = [l.item() for l in all_losses]
                        avg_train_loss = sum(all_losses) / len(all_losses)
                        epoch_duration = datetime.now() - epoch_start_time
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Global Step {global_step}, "
                                  f"Average Training Loss: {avg_train_loss:.4f}, Duration: {epoch_duration}")
                else:
                    avg_train_loss = train_loss / max(1, train_steps)
                    epoch_duration = datetime.now() - epoch_start_time
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Global Step {global_step}, "
                              f"Average Training Loss: {avg_train_loss:.4f}, Duration: {epoch_duration}")
                
                # Full validation at the end of each epoch
                logger.info(f"Running full validation at the end of epoch {epoch+1}")
                val_loss = evaluate_model(model, valid_dataloader, device, is_main_process, world_size, rank)
                
                if is_main_process:
                    logger.info(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}")
                    
                    # Save if best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logger.info(f"New best validation loss: {best_val_loss:.4f}")
                        
                        # Save the model
                        if isinstance(model, DDP):
                            model.module.save_pretrained(output_dir, safe_serialization=False)
                        else:
                            model.save_pretrained(output_dir, safe_serialization=False)
                            
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f"Best model saved to {output_dir}")
            
            # Training completed successfully
            if is_main_process:
                logger.info(f"Fine-tuning completed successfully with batch_size={current_batch_size}")
                
                # Save the final model
                final_model_dir = os.path.join(output_dir, "final_model")
                os.makedirs(final_model_dir, exist_ok=True)
                
                logger.info(f"Saving final model to {final_model_dir}")
                if isinstance(model, DDP):
                    model.module.save_pretrained(final_model_dir, safe_serialization=False)
                else:
                    model.save_pretrained(final_model_dir, safe_serialization=False)
                    
                tokenizer.save_pretrained(final_model_dir)
                
                # Save training info
                config_data = {
                    "model_type": "Argonne_LLM",
                    "training_completed": True,
                    "final_training_loss": avg_train_loss if 'avg_train_loss' in locals() else None,
                    "final_validation_loss": val_loss if 'val_loss' in locals() else None,
                    "best_validation_loss": best_val_loss,
                    "training_completed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "base_model": model_path,
                    "fine_tuned_on": data_dir
                }
                
                with open(os.path.join(final_model_dir, "training_info.json"), "w") as f:
                    json.dump(config_data, f, indent=2)
                    
                logger.info(f"Model successfully saved in Hugging Face format at {final_model_dir}")
            
            # Synchronize all processes before breaking the loop
            if world_size > 1:
                dist.barrier()
            
            break
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Rank {rank}: CUDA out of memory with batch_size={current_batch_size}")
            # Clean up memory
            if 'model' in locals() and model is not None:
                del model
            if 'optimizer' in locals() and optimizer is not None:
                del optimizer
            if 'scaler' in locals() and scaler is not None:
                del scaler
                
            torch.cuda.empty_cache()
            
            # Reduce batch size
            current_batch_size = max(1, current_batch_size // 2)
            
            if current_batch_size < min_batch_size:
                logger.error(f"Rank {rank}: Batch size too small, can't proceed")
                break
                
            logger.info(f"Rank {rank}: Retrying with reduced batch_size={current_batch_size}")
            import time
            time.sleep(5)  # Give time for memory cleanup
            
            # Make sure all processes reduce batch size together
            if world_size > 1:
                dist.barrier()
    
    logger.info(f"Rank {rank}: Fine-tuning process completed")
    
    # Clean up distributed environment
    if world_size > 1:
        cleanup_ddp()

def evaluate_model(model, dataloader, device, is_main_process, world_size, rank):
    """Evaluate model on provided dataloader and return average loss."""
    model.eval()
    val_loss = 0.0
    val_steps = 0
    loss_history = []
    
    with torch.no_grad():
        for val_step, batch in enumerate(tqdm(dataloader, desc="Validation", 
                                             disable=not is_main_process)):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            try:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(input_ids, labels)
                
                # Track loss
                val_loss += loss.item()
                loss_history.append(loss.item())
                val_steps += 1
                
                # Log periodic validation loss (every 50 steps)
                if (val_step + 1) % 50 == 0 and is_main_process and loss_history:
                    avg_recent_val = sum(loss_history) / len(loss_history)
                    logger.info(f"Validation Step {val_step+1}, Loss: {avg_recent_val:.4f}")
                    loss_history = []
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Rank {rank}: Out of memory during validation: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    # Synchronize processes after validation
    if world_size > 1:
        dist.barrier()
    
    # Calculate and report average validation loss across all processes
    if world_size > 1:
        # Gather validation loss from all processes
        avg_val_loss_local = val_loss / max(1, val_steps)
        val_loss_tensor = torch.tensor([avg_val_loss_local], device=device)
        all_val_losses = [torch.zeros_like(val_loss_tensor) for _ in range(world_size)]
        dist.all_gather(all_val_losses, val_loss_tensor)
        
        # Also gather steps so we can weight accordingly
        val_steps_tensor = torch.tensor([val_steps], device=device)
        all_val_steps = [torch.zeros_like(val_steps_tensor) for _ in range(world_size)]
        dist.all_gather(all_val_steps, val_steps_tensor)
        
        if is_main_process:
            # Calculate weighted average based on steps per process
            all_val_losses = [l.item() for l in all_val_losses]
            all_val_steps = [s.item() for s in all_val_steps]
            total_val_steps = sum(all_val_steps)
            
            if total_val_steps > 0:
                avg_val_loss = sum(l * s for l, s in zip(all_val_losses, all_val_steps)) / total_val_steps
            else:
                avg_val_loss = float('inf')
                
            return avg_val_loss
    else:
        return val_loss / max(1, val_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Argonne LLM on instruction data")
    parser.add_argument("--model_path", type=str, default="./Argonne_LLM", 
                        help="Path to the pretrained model")
    parser.add_argument("--data_dir", type=str, default="./instruct_data/ProcessedOpenAssistant",
                        help="Directory containing the OpenAssistant data")
    parser.add_argument("--output_dir", type=str, default="./Argonne_LLM_Finetuned",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Initial batch size for training (per GPU)")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of processes to use for data processing")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1 for non-distributed)")
    
    args = parser.parse_args()
    
    # Create output directory (only in main process)
    if args.local_rank <= 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Start fine-tuning with DDP
    finetune_model(
        args.model_path, 
        args.data_dir, 
        args.output_dir, 
        batch_size=args.batch_size, 
        num_proc=args.num_proc,
        local_rank=args.local_rank
    )