import argparse
import json
import os
import time
import traceback
from typing import List, Optional, Tuple

import torch
from datasets import Dataset
from tqdm import tqdm

from data_processing import (
    collate_batch,
    load_nonstream_data,
    load_tokenizer,
    chunk_tokens,
)
from model import ArgonneConfig, ArgonneModel
from training_utils import (
    CosineWarmupScheduler,
    log_dataset_plan,
    resolve_data_files,
    validate_tokenizer_path,
)

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
        self.chunk_offset = 0
        
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
            "chunk_offset": self.chunk_offset,
            "current_position": self.current_position,
            "epoch": self.epoch,
            "files_processed": sorted(self.files_processed),
        }

    def update_streaming_position(self, file_idx, position, chunk_offset=0, file_path=None):
        """Update streaming position information"""
        self.current_file_idx = file_idx
        self.position_in_file = position
        self.chunk_offset = chunk_offset
        if file_path:
            self.files_processed.add(os.path.basename(file_path))
    
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
            self.chunk_offset = 0
        else:
            self.current_position = 0
            if total_samples:
                self.shuffled_indices = torch.randperm(total_samples).tolist()

# Updated streaming token generator to use datasets library
def streaming_token_generator(
    data_files,
    tokenizer,
    block_size,
    start_file_idx=0,
    start_position=0,
    start_chunk_offset=0,
):
    """
    Enhanced token generator that supports position tracking.
    
    Args:
        data_files: List of files to process
        tokenizer: HF tokenizer
        start_file_idx: Starting file index
        start_position: Starting position within file
        
    Yields:
        (tokens, file_idx, position, shard_name, chunk_idx): tokenized chunk with
        metadata used for resuming and logging
    """
    file_idx = start_file_idx
    start_file_idx = max(start_file_idx, 0)
    processed_count = 0

    while file_idx < len(data_files):
        try:
            file_path = data_files[file_idx]
            shard_name = os.path.basename(file_path)
            print(
                f"Streaming from shard {file_idx + 1}/{len(data_files)}: {shard_name}"
            )
            
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
                
            position = start_position if file_idx == start_file_idx else 0
            resume_position = position
            # Reset resume markers for future files
            resume_chunk_offset = start_chunk_offset if file_idx == start_file_idx else 0
            start_position = 0
            start_chunk_offset = 0

            # Process entries from current position
            while position < len(dataset):
                try:
                    item = dataset[position]
                    # Get the text field - most commonly 'text' but could be others
                    if 'text' in item and item['text'] and isinstance(item['text'], str):
                        text = item['text']
                        tokens = tokenizer.encode(text, add_special_tokens=False)

                        skip_chunks = (
                            resume_chunk_offset
                            if (file_idx == start_file_idx and position == resume_position)
                            else 0
                        )

                        for chunk_idx, chunk in enumerate(chunk_tokens(tokens, block_size)):
                            if chunk_idx < skip_chunks:
                                continue

                            processed_count += 1
                            yield chunk, file_idx, position, shard_name, chunk_idx

                except Exception as e:
                    print(f"Error processing item at position {position}: {e}")

                resume_chunk_offset = 0
                position += 1

            file_idx += 1

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            file_idx += 1

    print(f"Completed processing all available files. Processed {processed_count} samples.")

    # To be consistent with resume_pretrain.py, return sentinel value instead of raising StopIteration
    return None, -1, -1, "", -1

def train_model_parallel(
    data_files: List[str],
    tokenizer_path: str,
    *,
    use_streaming: bool = True,
    trust_remote_code: bool = False,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    weight_decay: float,
):
    """
    data_files should be a list of actual .arrow file paths, e.g.
    ["data/file1.arrow", "data/file2.arrow", ...]
    
    Includes automatic batch size adjustment when OOM errors occur.
    
    Args:
        data_files: List of .arrow file paths
        tokenizer_path: Local directory containing tokenizer files.
        use_streaming: Whether to use streaming mode or load all data in memory
        trust_remote_code: Allow tokenizers with custom code.
        learning_rate: Peak learning rate applied after warmup.
        min_learning_rate: Minimum learning rate reached at schedule boundaries.
        warmup_steps: Number of optimizer steps spent linearly ramping the LR.
        max_steps: Maximum number of optimizer steps before halting training.
        weight_decay: AdamW weight decay coefficient.
    """
    # Initial batch size settings
    # Based on recent runs the auto-tuner consistently converges to a
    # micro-batch size of 4 before training can proceed without OOMs.
    # Starting the search at this value avoids multiple failing retries
    # and makes the warmup phase much faster.
    initial_batch_size = 4  # initial batch size tuned from previous run logs
    min_batch_size = 2  # Minimum acceptable batch size
    batch_size = initial_batch_size  # Current working batch size

    # Binary search bookkeeping for automatic batch-size discovery
    largest_successful_batch = None  # highest batch size that trained without OOM
    smallest_failed_batch = None  # lowest batch size that triggered OOM

    # Record the originally requested scheduler hyperparameters so we can
    # adapt them when the batch size changes. The defaults assume an
    # eight-sample micro-batch, so we rescale the learning rates and the
    # warmup horizon when the tuner settles on a smaller batch. This keeps
    # the warmup token budget roughly constant and prevents the peak
    # learning rate from being too aggressive for the noisier gradients.
    reference_micro_batch = 8
    requested_base_lr = learning_rate
    requested_min_lr = min(min_learning_rate, learning_rate)
    requested_warmup_steps = warmup_steps

    schedule_base_lr = learning_rate
    schedule_min_lr = requested_min_lr
    schedule_warmup_steps = warmup_steps

    def _compute_schedule_params(active_batch_size: int) -> Tuple[float, float, int]:
        """Return lr schedule parameters adjusted for the active batch size."""

        if active_batch_size <= 0:
            return schedule_base_lr, schedule_min_lr, schedule_warmup_steps

        # Scale the learning rates by the square-root of the batch ratio but
        # never increase beyond the user provided values.
        batch_ratio = active_batch_size / max(1, reference_micro_batch)
        lr_scale = min(batch_ratio ** 0.5, 1.0)

        scaled_base = max(requested_base_lr * lr_scale, 1e-8)
        scaled_min = max(requested_min_lr * lr_scale, 0.0)

        # Keep the warmup token count close to the reference setting. When the
        # batch shrinks we stretch the warmup steps so the model still sees the
        # same number of tokens before hitting the peak learning rate.
        warmup_scale = max(1.0, reference_micro_batch / max(1, active_batch_size))
        scaled_warmup = max(int(round(requested_warmup_steps * warmup_scale)), 1)

        return scaled_base, scaled_min, scaled_warmup

    validate_tokenizer_path(tokenizer_path)
    hf_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)

    epochs = 3
    block_size = 4096

    if hf_tokenizer.pad_token is None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})

    current_max_length = getattr(hf_tokenizer, "model_max_length", None)
    target_max_length = max(block_size + 1, block_size * 2)
    if current_max_length is None or current_max_length < target_max_length:
        hf_tokenizer.model_max_length = target_max_length

    config_model = ArgonneConfig(
        vocab_size=len(hf_tokenizer),
        max_position_embeddings=block_size,
        hidden_size=4096,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=8,
        rope_theta=500000.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_flash_attention=True,
        tie_word_embeddings=False,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        eos_token_id=hf_tokenizer.eos_token_id,
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
    
    # Track the best batch size discovered during retries
    final_batch_size = None

    # Main training loop with batch size adjustment
    stop_training = False
    global_step = 0
    model = None
    optimizer = None
    scaler = None

    def handle_oom_error(error: Exception) -> bool:
        """Adjust batch size after an out-of-memory error.

        Returns True if training should retry with a smaller batch size,
        or False if training should abort because the minimum batch size was reached.
        """

        nonlocal batch_size, model, optimizer, scaler, largest_successful_batch, smallest_failed_batch

        error_message = str(error)
        print("CUDA Out of Memory detected during training attempt.")
        if error_message:
            print("Full error message:")
            print(error_message)

        tb = getattr(error, "__traceback__", None)
        if tb is not None:
            formatted_traceback = "".join(
                traceback.format_exception(type(error), error, tb)
            ).rstrip()
            if formatted_traceback:
                print("Traceback:")
                print(formatted_traceback)

        model = None
        optimizer = None
        scaler = None
        torch.cuda.empty_cache()

        # Update the binary-search upper bound with the newly failed batch size.
        if smallest_failed_batch is None or batch_size < smallest_failed_batch:
            smallest_failed_batch = batch_size

        if largest_successful_batch is None:
            candidate = batch_size // 2
        else:
            candidate = (largest_successful_batch + smallest_failed_batch) // 2

        candidate = max(candidate, min_batch_size)

        if candidate == batch_size:
            if candidate <= min_batch_size:
                print(
                    f"Already at minimum batch size ({min_batch_size}). Training failed."
                )
                return False

            candidate = max(candidate - 1, min_batch_size)

        print(
            "CUDA Out of Memory! Reducing batch size from "
            f"{batch_size} to {candidate} (search bounds: "
            f"best_success={largest_successful_batch}, failed={smallest_failed_batch})"
        )
        batch_size = candidate

        time.sleep(5)
        return True

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
            print(f"Token embedding on device: {model.embed_tokens.weight.device}")
            if model.pipeline_partitions:
                for i, (start, end, device) in enumerate(model.pipeline_partitions):
                    print(f"Pipeline stage {i}: layers {start} - {end - 1} on {device}")
            print(f"Final RMSNorm on device: {model.norm.weight.device}")
            print(f"Head on device: {model.lm_head.weight.device}")
            
            # Apply torch.compile() for speed optimization whenever possible
            if hasattr(torch, "compile"):
                print("Attempting to apply torch.compile() to the distributed model...")
                try:
                    model = torch.compile(model, mode="default")
                    print("Model compilation successful!")
                except Exception as e:
                    print(f"torch.compile failed: {e}")
                    if len(model.devices) > 1:
                        print(
                            "PyTorch can only lower a graph that lives on a single device. "
                            "Because the pipeline partitions span multiple GPUs, Dynamo "
                            "breaks the graph and falls back to eager mode."
                        )
                    print("Continuing with the eager model implementation.")
            else:
                print(
                    "torch.compile() not available in this PyTorch version. Continuing with uncompiled model."
                )
            
            schedule_base_lr, schedule_min_lr, schedule_warmup_steps = _compute_schedule_params(
                batch_size
            )

            if (
                schedule_base_lr != requested_base_lr
                or schedule_min_lr != requested_min_lr
                or schedule_warmup_steps != requested_warmup_steps
            ):
                print(f"\nAdjusted schedule for micro-batch size {batch_size}:")
                print(
                    "  - peak lr: "
                    f"{requested_base_lr:.6e} -> {schedule_base_lr:.6e}"
                )
                print(
                    "  - min  lr: "
                    f"{requested_min_lr:.6e} -> {schedule_min_lr:.6e}"
                )
                print(
                    "  - warmup steps: "
                    f"{requested_warmup_steps} -> {schedule_warmup_steps}"
                )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=schedule_base_lr, weight_decay=weight_decay
            )
            scaler = torch.amp.GradScaler("cuda")
            global_step = 0
            tokens_in_current_attempt = 0  # Track tokens in this training attempt
            first_device = model.devices[0]  # Store the first device for consistency

            scheduler = CosineWarmupScheduler(
                optimizer,
                base_lr=schedule_base_lr,
                warmup_steps=schedule_warmup_steps,
                max_steps=max_steps,
                min_lr=schedule_min_lr,
            )

            if use_streaming:
                ########################################################
                # STREAMING MODE
                ########################################################
                loss = None

                def run_training_step(
                    x_tens,
                    y_tens,
                    shard_name,
                    file_idx,
                    position,
                    chunk_offset,
                ):
                    nonlocal tokens_in_current_attempt, global_step, loss, stop_training

                    batch_tokens = x_tens.numel()
                    tokens_in_current_attempt += batch_tokens

                    current_lr = scheduler.step(global_step)

                    x_local = x_tens.to(first_device)
                    y_local = y_tens.to(first_device)

                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=x_local, labels=y_local)
                        loss = outputs.loss.to(first_device)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1

                    if global_step % 50 == 0 and loss is not None:
                        print(
                            f"Step {global_step} | Loss: {loss.item():.4f} | Tokens processed: {tokens_in_current_attempt:,}"
                        )
                        print(
                            "Shard: "
                            f"{shard_name} | File index: {file_idx} | Position: {position} | Chunk: {chunk_offset}"
                        )
                        print(f"Current LR: {current_lr:.6e}")
                        prompt_str = "Long long time ago, "
                        token_ids = hf_tokenizer.encode(prompt_str)
                        prompt_tensor = (
                            torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                        )

                        generated = model.generate(
                            prompt_tensor, max_length=prompt_tensor.shape[1] + 100
                        )
                        generated_text = hf_tokenizer.decode(generated[0].tolist())
                        print(
                            f"\n--- Generated text at step {global_step} ---\n{generated_text}\n"
                        )

                    if global_step % 300 == 0:
                        checkpoint = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_size": batch_size,
                            "tokens_processed": tokens_in_current_attempt,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": loss.item() if loss is not None else None,
                            "data_position": data_position.get_state(),
                        }
                        os.makedirs("pretrained", exist_ok=True)
                        torch.save(
                            checkpoint, f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                        )
                        print(
                            f"Checkpoint saved at step {global_step} with data position tracking"
                        )

                    if global_step >= max_steps:
                        print(
                            f"Reached configured max_steps={max_steps}. Stopping training loop."
                        )
                        stop_training = True
                        return

                for epoch in tqdm(range(epochs)):
                    print(f"==== STREAMING with batch_size={batch_size} ====")
                    token_gen = streaming_token_generator(
                        data_files,
                        hf_tokenizer,
                        block_size,
                        data_position.current_file_idx,
                        data_position.position_in_file,
                        data_position.chunk_offset,
                    )
                    token_batch: List[List[int]] = []
                    active_shard = None
                    last_meta: Optional[Tuple[str, int, int, int]] = None

                    while True:
                        try:
                            tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)
                        except StopIteration:
                            if token_batch:
                                x_tens, y_tens = collate_batch(token_batch, block_size)
                                token_batch.clear()
                                if x_tens is not None and last_meta is not None:
                                    run_training_step(
                                        x_tens,
                                        y_tens,
                                        last_meta[0],
                                        last_meta[1],
                                        last_meta[2],
                                        last_meta[3],
                                    )

                            print(
                                "Completed streaming pass for this epoch. Restarting for next epoch."
                            )
                            data_position.next_epoch()
                            break

                        if shard_name != active_shard:
                            print(
                                f"--> Now training on shard {file_idx + 1}/{len(data_files)}: {shard_name}"
                            )
                            active_shard = shard_name

                        token_batch.append(tokens)
                        meta = (shard_name, file_idx, position, chunk_idx)
                        last_meta = meta
                        data_position.update_streaming_position(
                            file_idx,
                            position,
                            chunk_idx,
                            data_files[file_idx],
                        )

                        if len(token_batch) < batch_size:
                            continue

                        x_tens, y_tens = collate_batch(token_batch, block_size)
                        token_batch.clear()
                        if x_tens is None:
                            continue

                        if last_meta is not None:
                            run_training_step(
                                x_tens,
                                y_tens,
                                last_meta[0],
                                last_meta[1],
                                last_meta[2],
                                last_meta[3],
                            )
                        if stop_training:
                            break

                    if stop_training:
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

                        current_lr = scheduler.step(global_step)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            outputs = model(input_ids=x_tens, labels=y_tens)
                            loss = outputs.loss.to(first_device)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1

                        if global_step % 50 == 0 and loss is not None:
                            print(
                                f"global_step {global_step} | Loss: {loss.item():.4f} | Tokens processed: {tokens_in_current_attempt:,}"
                            )
                            print(f"Current LR: {current_lr:.6e}")
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                            
                            generated = model.generate(prompt_tensor, max_length=prompt_tensor.shape[1] + 50)
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
                                "scheduler_state_dict": scheduler.state_dict(),
                                "loss": loss.item() if loss is not None else None,
                                "data_position": data_position.get_state()  # Save position
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            torch.save(checkpoint, f"pretrained/non_streaming_checkpoint_step_{global_step}.pth")
                            print(f"Checkpoint saved at step {global_step} with data position tracking")

                        if global_step >= max_steps:
                            print(
                                f"Reached configured max_steps={max_steps}. Stopping training loop."
                            )
                            stop_training = True
                            break

                    if stop_training:
                        break

            # If we reach here, training completed successfully
            # Update total token count for successful training
            total_tokens_processed = tokens_in_current_attempt
            final_batch_size = batch_size
            if (
                largest_successful_batch is None
                or batch_size > largest_successful_batch
            ):
                largest_successful_batch = batch_size
            print(f"Training completed successfully with batch_size={batch_size}")
            print(f"Total tokens processed during training: {total_tokens_processed:,}")
            print(
                "Maximum stable batch size discovered (set this manually for future runs): "
                f"{final_batch_size}"
            )
            break
            
        except torch.cuda.OutOfMemoryError as e:
            if handle_oom_error(e):
                continue
            break

        except RuntimeError as e:
            message = str(e)
            if "out of memory" in message.lower():
                if handle_oom_error(e):
                    continue
                break

            print(f"Runtime error occurred: {message}")
            if "Expected all tensors to be on the same device" in message:
                print("\nDevice mismatch error detected. This might be due to improper tensor movement between pipeline stages.")
                print("Check the error message for which devices are mismatched and verify the model distribution.")
            raise e  # Re-raise to see the full stack trace

    # Save token count to a file for reporting
    if final_batch_size is None:
        print("Training did not complete successfully. Exiting without saving stats or model.")
        return

    training_stats = {
        "total_tokens": total_tokens_processed,
        "batch_size": batch_size,
        "epochs": epochs,
        "global_steps": global_step,
        "num_layers": config_model.num_hidden_layers,
        "num_attention_heads": config_model.num_attention_heads,
        "hidden_size": config_model.hidden_size,
        "model_params": sum(p.numel() for p in model.parameters()),
        "max_batch_size": final_batch_size,
        "data_shards_seen": sorted(data_position.files_processed),
        "base_learning_rate": schedule_base_lr,
        "min_learning_rate": schedule_min_lr,
        "warmup_steps": schedule_warmup_steps,
        "max_steps": max_steps,
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
    if final_batch_size is not None:
        print(f"Maximum stable batch size: {final_batch_size}")
    print(f"Training steps: {global_step}")
    print(
        "Learning rate schedule: "
        f"warmup {schedule_warmup_steps} steps | peak {schedule_base_lr:.6e} | min {schedule_min_lr:.6e}"
    )
    print(f"Configured max steps: {max_steps}")
    if training_stats["data_shards_seen"]:
        print("Shards processed this run:")
        for shard_name in training_stats["data_shards_seen"]:
            print(f"  - {shard_name}")
    print(f"Stats saved to: stats/training_stats.json")

    # Save final model and tokenizer
    try:
        model = model.half()
        model = model.to("cpu")
        model.save_pretrained("Argonne_LLM", safe_serialization=False)
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print(f"Training completed. Final model saved.")
    except Exception as e:
        print(f"Failed to save final model: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Argonne 2 model on offline Arrow shards.")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=os.path.join("..", "data", "*.arrow"),
        help="Glob pattern for Arrow files (default: ../data/*.arrow)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Directory on disk containing the pretrained tokenizer to load (e.g., a LLaMA or Qwen tokenizer).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode and load Arrow files into memory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading tokenizers that require custom code.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW during cosine schedule warmup.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5,
        help="Minimum learning rate used at the start and end of the cosine schedule.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of optimizer steps used for linear warmup.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=160_000,
        help="Total optimizer steps to schedule before halting training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient passed to AdamW.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fallback_patterns = [os.path.join("data", "*.arrow")]
    if args.data_glob != os.path.join("..", "data", "*.arrow"):
        fallback_patterns.insert(0, os.path.join("..", "data", "*.arrow"))

    data_files, _ = resolve_data_files(
        args.data_glob, fallback_patterns=fallback_patterns
    )

    log_dataset_plan(data_files)

    train_model_parallel(
        data_files=data_files,
        tokenizer_path=args.tokenizer_path,
        use_streaming=not args.no_streaming,
        trust_remote_code=args.trust_remote_code,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
    )

if __name__ == "__main__":
    main()
