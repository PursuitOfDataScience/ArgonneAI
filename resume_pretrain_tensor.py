import argparse
import contextlib
import json
import os
import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from data_processing import (
    collate_batch,
    load_tokenizer,
)
from model import ArgonneConfig, ArgonneModel
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    cast_state_dict_to_dtype,
    log_dataset_plan,
    safe_torch_load,
    safe_torch_save,
    resolve_data_files,
    validate_tokenizer_path,
)

# Import all tensor parallel components from training.py
from training import (
    DataPosition,
    streaming_token_generator,
    init_tensor_parallel_group,
    TensorParallelModel,
    _ensure_gradient_dtype_matches_params,
    shard_attention_layer,
    shard_mlp_layer,
    shard_tensor_parallel_correctly,
    _gcd,
)


CHECKPOINT_PATTERN = re.compile(r"_step_(\d+)(?:_rank\d+)?\.pth$")


def streaming_token_generator_with_special_tokens(
    data_files: List[str],
    tokenizer,
    block_size: int,
    start_file_idx: int = 0,
    start_position: int = 0,
    start_chunk_offset: int = 0,
    rank: int = 0,
):
    """
    Streaming token generator that adds BOS/EOS tokens PER DOCUMENT.
    
    Documents are wrapped with special tokens:
        [<BOS>, ...document_tokens..., <EOS>]
    
    Multiple documents are concatenated and packed into chunks of block_size.
    Long documents naturally span multiple chunks (BOS only at start, EOS only at end).
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    
    # Check if tokenizer has special tokens
    has_special_tokens = (bos_id is not None and eos_id is not None)
    
    if not has_special_tokens and rank == 0:
        print("⚠ Tokenizer missing BOS/EOS tokens - falling back to no special tokens")
    
    token_buffer = []
    
    for file_idx in range(start_file_idx, len(data_files)):
        file_path = data_files[file_idx]
        shard_name = os.path.basename(file_path)
        
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            texts = table.column('text').to_pylist()
        except Exception as e:
            if rank == 0:
                print(f"⚠ Error reading {shard_name}: {e}")
            continue
        
        start_pos = start_position if file_idx == start_file_idx else 0
        
        for doc_idx in range(start_pos, len(texts)):
            text = texts[doc_idx]
            if not text or not isinstance(text, str):
                continue
            
            # Tokenize the document
            doc_tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if not doc_tokens:
                continue
            
            # Add BOS/EOS tokens if available
            if has_special_tokens:
                doc_tokens = [bos_id] + doc_tokens + [eos_id]
            
            # Add to buffer
            token_buffer.extend(doc_tokens)
            
            # Yield chunks when buffer is large enough
            while len(token_buffer) >= block_size:
                chunk = token_buffer[:block_size]
                token_buffer = token_buffer[block_size:]
                yield chunk, file_idx, doc_idx, shard_name, 0
        
        # Reset position for next file
        start_position = 0
    
    # Yield any remaining tokens
    if token_buffer:
        # Pad to block_size if needed
        if len(token_buffer) < block_size:
            # IMPROVED: Use EOS as padding, or pad_token, or fall back to 0
            pad_id = eos_id if eos_id is not None else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            token_buffer.extend([pad_id] * (block_size - len(token_buffer)))
        yield token_buffer[:block_size], len(data_files) - 1, -1, "final_chunk", 0


def cleanup_old_checkpoints(directory: str, keep: int = 3, rank: int = 0) -> None:
    """Keep only the most recent checkpoint files in a directory."""
    
    if rank != 0:
        return

    if keep <= 0:
        return

    if not os.path.isdir(directory):
        return

    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(directory):
        match = CHECKPOINT_PATTERN.search(name)
        if not match:
            continue

        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue

        step = int(match.group(1))
        candidates.append((step, path))

    if len(candidates) <= keep:
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, path in candidates[keep:]:
        try:
            os.remove(path)
            print(f"Removed old checkpoint: {os.path.basename(path)}")
        except OSError as exc:
            print(f"WARNING: Failed to remove checkpoint '{path}': {exc}")


def _resolve_checkpoint_path(checkpoint_path: Optional[str], rank: int = 0) -> Optional[str]:
    """
    Resolve the checkpoint path, auto-selecting the highest step if needed.
    Returns None if no valid checkpoint found or if checkpoint is incompatible.
    """
    if checkpoint_path and checkpoint_path.upper() == "NONE":
        if rank == 0:
            print("Explicitly starting from scratch (--checkpoint-path NONE)")
        return None
        
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    search_dirs = []
    if checkpoint_path and os.path.isdir(checkpoint_path):
        search_dirs.append(checkpoint_path)

    default_dir = os.path.join(os.getcwd(), "pretrained")
    if default_dir not in search_dirs:
        search_dirs.append(default_dir)

    candidates: List[Tuple[int, str]] = []
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            match = CHECKPOINT_PATTERN.search(name)
            if not match:
                continue
            step = int(match.group(1))
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path):
                candidates.append((step, full_path))

    if not candidates:
        if rank == 0:
            print("No checkpoint files found - starting from scratch")
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    latest_step, latest_path = candidates[0]
    if rank == 0:
        print(f"Found checkpoint '{os.path.basename(latest_path)}' (step {latest_step})")
    return latest_path


def reconstruct_full_weights_from_sharded(state_dict: dict, world_size: int, rank: int = 0) -> dict:
    """
    Reconstruct full model weights from a tensor-parallel sharded checkpoint.
    
    This handles checkpoints saved by training.py which contain already-sharded weights.
    """
    if rank == 0:
        print("Detected sharded checkpoint - reconstructing full weights...")
    
    # Check if this is actually a sharded checkpoint
    sample_key = "blocks.0.attn.q_proj.weight"
    if sample_key in state_dict:
        expected_full_size = 4080  # Full q_proj should be [4080, 4080]
        actual_size = state_dict[sample_key].shape[0]
        
        if actual_size == expected_full_size:
            if rank == 0:
                print("  Checkpoint contains full weights already - no reconstruction needed")
            return state_dict
    
    # This is a sharded checkpoint - we need to reconstruct
    # For tensor parallel checkpoints, we only have rank 0's shard
    # We'll expand it back to full size by replicating (for resuming from scratch)
    
    reconstructed = {}
    
    for key, value in state_dict.items():
        # Check if this is a sharded attention layer
        if '.attn.q_proj.weight' in key or '.attn.k_proj.weight' in key or '.attn.v_proj.weight' in key:
            # Column-parallel: out_features was sharded
            # Original: [full_out, in] -> Sharded: [shard_out, in]
            # Reconstruct by replicating world_size times
            full_out = value.shape[0] * world_size
            full_weight = value.repeat(world_size, 1)
            reconstructed[key] = full_weight
            if rank == 0 and 'blocks.0' in key:
                print(f"  Reconstructed {key}: {value.shape} -> {full_weight.shape}")
                
        elif '.attn.o_proj.weight' in key:
            # Row-parallel: in_features was sharded
            # Original: [out, full_in] -> Sharded: [out, shard_in]
            full_in = value.shape[1] * world_size
            full_weight = value.repeat(1, world_size)
            reconstructed[key] = full_weight
            if rank == 0 and 'blocks.0' in key:
                print(f"  Reconstructed {key}: {value.shape} -> {full_weight.shape}")
                
        elif '.mlp.gate_proj.weight' in key or '.mlp.up_proj.weight' in key:
            # Column-parallel MLP
            full_out = value.shape[0] * world_size
            full_weight = value.repeat(world_size, 1)
            reconstructed[key] = full_weight
            if rank == 0 and 'blocks.0' in key:
                print(f"  Reconstructed {key}: {value.shape} -> {full_weight.shape}")
                
        elif '.mlp.down_proj.weight' in key:
            # Row-parallel MLP
            full_in = value.shape[1] * world_size
            full_weight = value.repeat(1, world_size)
            reconstructed[key] = full_weight
            if rank == 0 and 'blocks.0' in key:
                print(f"  Reconstructed {key}: {value.shape} -> {full_weight.shape}")
                
        # Check for sharded biases
        elif '.attn.q_proj.bias' in key or '.attn.k_proj.bias' in key or '.attn.v_proj.bias' in key:
            full_bias = value.repeat(world_size)
            reconstructed[key] = full_bias
        elif '.mlp.gate_proj.bias' in key or '.mlp.up_proj.bias' in key:
            full_bias = value.repeat(world_size)
            reconstructed[key] = full_bias
        elif '.attn.o_proj.bias' in key or '.mlp.down_proj.bias' in key:
            # Row-parallel bias - only rank 0 has it, already full
            reconstructed[key] = value
        else:
            # Not a sharded parameter
            reconstructed[key] = value
    
    if rank == 0:
        print(f"✓ Reconstructed full weights from sharded checkpoint")
    
    return reconstructed


def check_checkpoint_compatibility(checkpoint_path: str, config: ArgonneConfig, rank: int = 0) -> bool:
    """
    Check if a checkpoint is compatible with the current model architecture.
    Returns True if compatible, False otherwise.
    """
    try:
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", {})
        
        # Check if checkpoint is from pipeline parallel training
        is_pipeline_checkpoint = ckpt.get("pipeline_parallel", False)
        is_tensor_checkpoint = ckpt.get("tensor_parallel", False)
        checkpoint_world_size = ckpt.get("world_size", 1)
        
        if rank == 0:
            if is_pipeline_checkpoint:
                print(f"✓ Found pipeline parallel checkpoint from training.py")
            elif is_tensor_checkpoint:
                print(f"✓ Found tensor parallel checkpoint (world_size={checkpoint_world_size})")
            
        # Check for naming convention compatibility
        has_q_proj = any('q_proj' in key for key in state_dict.keys())
        has_old_names = any('query.weight' in key for key in state_dict.keys())
        
        if has_old_names and not has_q_proj:
            if rank == 0:
                print("⚠ Checkpoint uses OLD architecture - INCOMPATIBLE")
            return False
        
        # Check model dimensions match (accounting for potential sharding)
        if "embed_tokens.weight" in state_dict:
            vocab_size, hidden_size = state_dict["embed_tokens.weight"].shape
            if hidden_size != config.hidden_size:
                if rank == 0:
                    print(f"⚠ Hidden size mismatch: checkpoint={hidden_size}, config={config.hidden_size}")
                return False
        
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"⚠ Error checking checkpoint compatibility: {e}")
        return False


def resume_training(
    data_glob: str,
    tokenizer_path: str,
    checkpoint_path: Optional[str] = None,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 4096,
    batch_size: int = 4,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    use_streaming: bool = True,
    num_proc: int = 8,
    trust_remote_code: bool = False,
    force_from_scratch: bool = False,
    rewarmup_steps: int = 100,
    add_special_tokens: bool = False,  # NEW parameter
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)
    init_tensor_parallel_group(world_size, rank)
    is_main_process = (rank == 0)
    
    if is_main_process:
        cleanup_old_checkpoints("pretrained", keep=50, rank=rank)
    
    # Resolve data files
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    fallback_patterns = [
        os.path.join("data", "CC-MAIN-2025-26", "*.parquet"),
        os.path.join("..", "data", "*.arrow"),
        os.path.join("data", "*.arrow"),
    ]
    if data_glob != default_data_glob:
        fallback_patterns.insert(0, default_data_glob)
    data_files, used_patterns = resolve_data_files(data_glob, fallback_patterns=fallback_patterns)
    
    if is_main_process:
        print(f"Found {len(data_files)} data files")
        log_dataset_plan(data_files)

    # Load tokenizer
    validate_tokenizer_path(tokenizer_path)
    hf_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})
    hf_tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)
    vocab_size = len(hf_tokenizer)

    # Build config - MUST MATCH training.py exactly
    config = ArgonneConfig(
        vocab_size=vocab_size,
        hidden_size=4080,
        max_position_embeddings=block_size,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=8,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_flash_attention=True,
        use_gradient_checkpointing=False,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        eos_token_id=hf_tokenizer.eos_token_id,
    )

    # Try to load checkpoint with compatibility check
    global_step = 0
    total_tokens_processed = 0
    load_checkpoint = False
    
    if force_from_scratch:
        if is_main_process:
            print("="*70)
            print("FORCED START FROM SCRATCH (--force-from-scratch)")
            print("="*70)
        resolved_checkpoint = None
    else:
        resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path, rank)
        
        if resolved_checkpoint:
            is_compatible = check_checkpoint_compatibility(resolved_checkpoint, config, rank)
            
            if not is_compatible:
                if is_main_process:
                    print("="*70)
                    print("INCOMPATIBLE CHECKPOINT DETECTED")
                    print("="*70)
                    print("The checkpoint uses a different model architecture.")
                    print("Starting training from scratch with the current architecture.")
                    print("="*70)
                resolved_checkpoint = None
            else:
                load_checkpoint = True
                if is_main_process:
                    print(f"✓ Checkpoint is compatible, will resume from: {resolved_checkpoint}")

    # Determine dtype
    supports_bf16 = False
    amp_dtype = torch.float32
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
        supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    # Create base model (keep parameters in FP32 for stable optimizer state)
    base_model = ArgonneModel(config)

    if load_checkpoint and resolved_checkpoint:
        # Load checkpoint
        ckpt = safe_torch_load(resolved_checkpoint, map_location="cpu", weights_only=True)
        
        # Check if checkpoint is from tensor parallel training
        checkpoint_world_size = ckpt.get("world_size", 1)
        is_tensor_checkpoint = ckpt.get("tensor_parallel", False)
        
        # Handle compiled model checkpoints
        if any(k.startswith("_orig_mod.") for k in ckpt["model_state_dict"].keys()):
            if is_main_process:
                print("Detected compiled model checkpoint, converting parameter names...")
            new_state_dict = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("_orig_mod."):
                    new_key = k.replace("_orig_mod.", "")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            ckpt["model_state_dict"] = new_state_dict
        
        # CRITICAL FIX: Reconstruct full weights if loading from sharded checkpoint
        if is_tensor_checkpoint and checkpoint_world_size > 1:
            if is_main_process:
                print(f"Reconstructing full weights from {checkpoint_world_size}-way sharded checkpoint...")
            ckpt["model_state_dict"] = reconstruct_full_weights_from_sharded(
                ckpt["model_state_dict"], 
                checkpoint_world_size, 
                rank
            )
        
        converted_state = cast_state_dict_to_dtype(ckpt["model_state_dict"], torch.float32)
        
        # Load weights
        try:
            base_model.load_state_dict(converted_state, strict=True)
            if is_main_process:
                print("✓ Loaded checkpoint with strict=True (exact match)")
        except RuntimeError as e:
            if is_main_process:
                print(f"⚠ Strict loading failed: {e}")
                print("  Attempting non-strict loading...")
            base_model.load_state_dict(converted_state, strict=False)
            if is_main_process:
                print("✓ Loaded checkpoint with strict=False (some keys may be missing)")
        
        # Get checkpoint info
        global_step = ckpt.get("global_step", 0)
        total_tokens_processed = ckpt.get("tokens_processed", 0)
        
        if is_main_process:
            print(f"✓ Loaded checkpoint from step {global_step}, tokens: {total_tokens_processed:,}")
            checkpoint_type = "pipeline parallel" if ckpt.get("pipeline_parallel") else "tensor parallel"
            print(f"  Checkpoint type: {checkpoint_type}")
    else:
        if is_main_process:
            print("="*70)
            print("STARTING FRESH TRAINING")
            print("="*70)
    
    # Create tensor parallel wrapper
    model = TensorParallelModel(base_model, world_size, rank)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay, 
        fused=False
    )
    
    # Setup scheduler
    min_lr = min(min_lr, lr)
    
    # CRITICAL: If resuming, use shorter re-warmup to stabilize
    effective_warmup = warmup_steps
    if load_checkpoint and resolved_checkpoint and global_step > 0:
        effective_warmup = rewarmup_steps  # Much shorter re-warmup
        if is_main_process:
            print(f"⚠ Using {rewarmup_steps}-step re-warmup for resume stability")
    
    scheduler = CosineWarmupScheduler(
        optimizer, 
        base_lr=lr, 
        warmup_steps=effective_warmup,  # Use re-warmup if resuming
        max_steps=total_training_steps, 
        min_lr=min_lr,
    )
    
    # Do NOT load optimizer state (incompatible across parallelism schemes)
    if load_checkpoint and resolved_checkpoint:
        if is_main_process:
            print("⚠ Note: Optimizer state not restored (incompatible with tensor parallelism)")
            print("  Using fresh AdamW optimizer - expect small loss spike initially")
            print(f"  Re-warming up learning rate over {effective_warmup} steps")

        # Don't load scheduler state - we're intentionally re-warming
        # The effective_warmup setting above handles the restart
    else:
        scheduler.step(global_step)
    
    if is_main_process:
        print(f"✓ Training setup complete")
        print(f"  - Starting step: {global_step}")
        print(f"  - Tokens processed: {total_tokens_processed:,}")
        print(f"  - Learning rate: {lr:.2e}")
        print(f"  - Tensor parallelism: world_size={world_size}")
    
    # Setup data position
    data_position = DataPosition(streaming=use_streaming)
    if load_checkpoint and resolved_checkpoint and "data_position" in ckpt:
        data_position.restore_state(ckpt.get("data_position"))
        if is_main_process:
            print(f"✓ Data position: file {data_position.current_file_idx}, position {data_position.position_in_file}")
    else:
        if is_main_process:
            print("  Starting from beginning of dataset")

    # Setup mixed precision
    use_grad_scaler = amp_dtype == torch.float16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    if is_main_process:
        if supports_bf16:
            print("✓ Using torch.bfloat16 autocast")
        elif amp_dtype == torch.float16:
            print("✓ Using torch.float16 autocast with GradScaler")

    first_device = model.device
    tokens_in_this_session = 0
    last_loss_value: Optional[float] = None

    # Training loop
    if use_streaming:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"RESUMING TRAINING WITH TENSOR PARALLELISM")
            print(f"{'='*70}")
            print(f"Step: {global_step} / {total_training_steps}")
            print(f"Learning rate: {lr:.2e}")
            print(f"Batch size: {batch_size}")
            print(f"World size: {world_size}")
            # NEW: Print special tokens mode
            if add_special_tokens:
                print(f"Special tokens: ENABLED (BOS/EOS per document)")
            else:
                print(f"Special tokens: DISABLED (backward compatible mode)")
            print(f"{'='*70}\n")

        # NEW: Choose generator based on flag
        if add_special_tokens:
            token_gen = streaming_token_generator_with_special_tokens(
                data_files, hf_tokenizer, block_size,
                data_position.current_file_idx, data_position.position_in_file, 
                data_position.chunk_offset, rank
            )
        else:
            token_gen = streaming_token_generator(
                data_files, hf_tokenizer, block_size,
                data_position.current_file_idx, data_position.position_in_file, 
                data_position.chunk_offset, rank
            )
        
        token_buffer: List[List[int]] = []
        active_shard: Optional[str] = None
        last_meta: Optional[Tuple[str, int, int, int]] = None

        pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training") if is_main_process else None
        
        try:
            while global_step < total_training_steps:
                try:
                    tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)

                    if file_idx == -1:
                        if is_main_process:
                            print("End of dataset - restarting")
                        data_position.next_epoch()
                        # NEW: Use same generator type on restart
                        if add_special_tokens:
                            token_gen = streaming_token_generator_with_special_tokens(
                                data_files, hf_tokenizer, block_size, rank=rank
                            )
                        else:
                            token_gen = streaming_token_generator(
                                data_files, hf_tokenizer, block_size, rank=rank
                            )
                        continue

                    token_buffer.append(tokens)
                    last_meta = (shard_name, file_idx, position, chunk_idx)
                    data_position.update_streaming_position(file_idx, position, chunk_idx, data_files[file_idx])

                    if shard_name != active_shard:
                        active_shard = shard_name
                        if is_main_process:
                            print(f"Processing shard {file_idx + 1}/{len(data_files)}: {shard_name}")

                    if len(token_buffer) < batch_size:
                        continue

                    x_tens, y_tens = collate_batch(token_buffer, block_size)
                    token_buffer.clear()
                    if x_tens is None or last_meta is None:
                        continue

                    batch_tokens = x_tens.numel()
                    tokens_in_this_session += batch_tokens
                    current_lr = scheduler.step(global_step)

                    x_local = x_tens.to(first_device)
                    y_local = y_tens.to(first_device)
                    optimizer.zero_grad(set_to_none=True)

                    autocast_context = torch.amp.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available() else contextlib.nullcontext()

                    with autocast_context:
                        outputs = model(input_ids=x_local)
                        logits = outputs.logits
                        loss_tensor = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y_local.view(-1),
                            ignore_index=-100,
                        )

                    last_loss_value = float(loss_tensor.detach().cpu().item())

                    if scaler is not None:
                        scaler.scale(loss_tensor).backward()
                        scaler.unscale_(optimizer)
                        _ensure_gradient_dtype_matches_params(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_tensor.backward()
                        _ensure_gradient_dtype_matches_params(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    global_step += 1
                    if pbar is not None:
                        pbar.update(1)

                    if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                        current_total_tokens = total_tokens_processed + tokens_in_this_session
                        print(f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {current_total_tokens:,} | LR: {current_lr:.6e}")

                    if global_step % 300 == 0:
                        current_total_tokens = total_tokens_processed + tokens_in_this_session
                        prompt_str = "Long long time ago, "
                        token_ids = hf_tokenizer.encode(prompt_str)
                        prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                        generated = model.generate(
                            prompt_tensor,
                            max_length=prompt_tensor.shape[1] + 100,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.9,
                        )

                        if is_main_process:
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                            model_state = cast_state_dict_to_dtype(model.base_model.state_dict(), amp_dtype)
                            checkpoint_state = {
                                "global_step": global_step,
                                "tokens_processed": current_total_tokens,
                                "model_state_dict": model_state,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "loss": last_loss_value,
                                "data_position": data_position.get_state(),
                                "model_dtype": str(amp_dtype),
                                "tensor_parallel": True,
                                "world_size": world_size,
                                "rank": rank,
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            save_path = f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                            safe_torch_save(checkpoint_state, save_path)
                            print(f"Checkpoint saved @ step {global_step} -> {save_path}")

                        update_training_stats(
                            tokens=current_total_tokens,
                            batch_size=batch_size,
                            steps=global_step,
                            model=model,
                            n_layer=config.n_layer,
                            n_head=config.n_head,
                            n_embd=config.n_embd,
                            base_lr=lr,
                            min_lr=min_lr,
                            warmup_steps=warmup_steps,
                            max_steps=total_training_steps,
                        )

                except StopIteration:
                    if is_main_process:
                        print("StopIteration - restarting dataset")
                    data_position.next_epoch()
                    # NEW: Use same generator type on restart
                    if add_special_tokens:
                        token_gen = streaming_token_generator_with_special_tokens(
                            data_files, hf_tokenizer, block_size, rank=rank
                        )
                    else:
                        token_gen = streaming_token_generator(
                            data_files, hf_tokenizer, block_size, rank=rank
                        )
                    continue
        finally:
            if pbar is not None:
                pbar.close()

    final_token_count = total_tokens_processed + tokens_in_this_session
    
    if is_main_process:
        print(f"\n===== TRAINING COMPLETE =====")
        print(f"Total tokens: {final_token_count:,}")
        print(f"Final step: {global_step}")

    if dist.is_initialized():
        dist.barrier()


def update_training_stats(
    tokens,
    batch_size,
    steps,
    model,
    n_layer,
    n_head,
    n_embd,
    *,
    base_lr: float | None = None,
    min_lr: float | None = None,
    warmup_steps: int | None = None,
    max_steps: int | None = None,
    final: bool = False,
):
    """Update the training statistics file with current information"""
    model_params = sum(p.numel() for p in model.parameters())
    
    training_stats = {
        "total_tokens": tokens,
        "batch_size": batch_size,
        "global_steps": steps,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "model_params": model_params,
        "final_training": final,
        "parallelism_type": "tensor_parallel"
    }

    if base_lr is not None:
        training_stats["base_learning_rate"] = base_lr
    if min_lr is not None:
        training_stats["min_learning_rate"] = min_lr
    if warmup_steps is not None:
        training_stats["warmup_steps"] = warmup_steps
    if max_steps is not None:
        training_stats["max_steps"] = max_steps
    
    os.makedirs("stats", exist_ok=True)
    
    if final:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats/final_training_stats_tensor_parallel_{timestamp}.json"
    else:
        filename = f"stats/current_training_stats_tensor_parallel_step_{steps}.json"
        
    with open(filename, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    if final:
        print(f"Final training stats saved to: {filename}")
    return filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume Argonne pretraining with Tensor Parallelism")
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=default_data_glob,
        help="Glob pattern for parquet shards",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Filesystem directory containing the pretrained tokenizer to reuse.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional path to checkpoint. Use 'NONE' to explicitly start from scratch.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_MAX_TRAINING_STEPS,
        help="Total number of training steps to run.",
    )
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="Minimum learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--rewarmup-steps",
        type=int,
        default=100,
        help="Number of re-warmup steps when resuming.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading tokenizers that require custom code.",
    )
    parser.add_argument(
        "--force-from-scratch",
        action="store_true",
        help="Force training from scratch, ignoring any checkpoints.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add BOS/EOS tokens per document (recommended for new training runs)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resume_training(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        total_training_steps=args.total_steps,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        rewarmup_steps=args.rewarmup_steps,
        weight_decay=args.weight_decay,
        use_streaming=not args.no_streaming,
        num_proc=args.num_proc,
        trust_remote_code=args.trust_remote_code,
        force_from_scratch=args.force_from_scratch,
        add_special_tokens=args.add_special_tokens,  # NEW: Pass through the flag
    )


if __name__ == "__main__":
    main()