import argparse
import contextlib
import json
import os
import re
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
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
    DEFAULT_MAX_TRAINING_STEPS,
    cast_state_dict_to_dtype,
    load_streaming_shard,
    log_dataset_plan,
    safe_torch_load,
    safe_torch_save,
    resolve_data_files,
    validate_tokenizer_path,
)


def _ensure_gradient_dtype_matches_params(model: torch.nn.Module) -> None:
    """Cast gradients to match their parameter's dtype/device for fused optimizers."""
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if grad.dtype != param.dtype or grad.device != param.device:
            param.grad = grad.to(device=param.device, dtype=param.dtype)


# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DataPosition:
    def __init__(self, streaming: bool = True):
        """Track dataset position during training"""
        self.streaming = streaming
        self.current_file_idx = 0
        self.position_in_file = 0
        self.chunk_offset = 0
        self.shuffled_indices: Optional[List[int]] = None
        self.current_position = 0
        self.epoch = 0
        self.files_processed = set()

    def get_state(self) -> dict:
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

    def restore_state(self, state: Optional[dict]) -> None:
        """Restore position information from checkpoint data."""
        if not state:
            return
        self.streaming = state.get("streaming", self.streaming)
        self.current_file_idx = state.get("current_file_idx", 0)
        self.position_in_file = state.get("position_in_file", 0)
        self.chunk_offset = state.get("chunk_offset", state.get("chunk_index", 0))
        self.current_position = state.get("current_position", 0)
        self.epoch = state.get("epoch", state.get("global_step", 0))
        files = state.get("files_processed", [])
        self.files_processed = {os.path.basename(f) for f in files}

    def update_streaming_position(self, file_idx: int, position: int, chunk_offset: int = 0, file_path: Optional[str] = None) -> None:
        """Update streaming position information"""
        self.current_file_idx = file_idx
        self.position_in_file = position
        self.chunk_offset = chunk_offset
        if file_path:
            self.files_processed.add(os.path.basename(file_path))

    def update_nonstreaming_position(self, position: int) -> None:
        """Update non-streaming position"""
        self.current_position = position

    def generate_shuffled_indices(self, total_samples: int) -> List[int]:
        """Generate shuffled indices for non-streaming mode"""
        if self.shuffled_indices is None or len(self.shuffled_indices) != total_samples:
            self.shuffled_indices = torch.randperm(total_samples).tolist()
        return self.shuffled_indices[self.current_position:]

    def next_epoch(self, total_samples: Optional[int] = None) -> None:
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


def streaming_token_generator(data_files: List[str], tokenizer, block_size: int, start_file_idx: int = 0, start_position: int = 0, start_chunk_offset: int = 0, rank: int = 0):
    """Generator with chunk-level resume support"""
    file_idx = max(start_file_idx, 0)
    processed_count = 0
    is_main_process = (rank == 0)
    initial_file_idx = file_idx
    initial_position = start_position
    initial_chunk_offset = start_chunk_offset

    while file_idx < len(data_files):
        try:
            file_path = data_files[file_idx]
            shard_name = os.path.basename(file_path)
            if is_main_process:
                print(f"Streaming from shard {file_idx + 1}/{len(data_files)}: {shard_name}")

            try:
                dataset = load_streaming_shard(file_path)
                if is_main_process:
                    print(f"Successfully loaded dataset with {len(dataset)} rows")
            except Exception as file_error:
                if is_main_process:
                    print(f"ERROR: Could not read file {file_path}: {file_error}")
                file_idx += 1
                continue

            if file_idx == initial_file_idx:
                position = initial_position
                resume_chunk_offset = initial_chunk_offset
                if is_main_process and (position > 0 or resume_chunk_offset > 0):
                    print(f"  >>> RESUMING from position {position}, chunk offset {resume_chunk_offset}")
            else:
                position = 0
                resume_chunk_offset = 0

            while position < len(dataset):
                try:
                    item = dataset[position]
                    if "text" in item and item["text"] and isinstance(item["text"], str):
                        text = item["text"]
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        for chunk_idx, chunk in enumerate(chunk_tokens(tokens, block_size)):
                            if file_idx == initial_file_idx and position == initial_position:
                                if chunk_idx < resume_chunk_offset:
                                    continue
                            processed_count += 1
                            yield chunk, file_idx, position, shard_name, chunk_idx
                except Exception as e:
                    if is_main_process:
                        print(f"Error processing item at position {position}: {e}")
                position += 1
            file_idx += 1
        except Exception as e:
            if is_main_process:
                print(f"Error processing file {file_path}: {e}")
            file_idx += 1

    if is_main_process:
        print(f"Completed processing all available files. Processed {processed_count} samples.")
    return None, -1, -1, "", -1


CHECKPOINT_PATTERN = re.compile(r"_step_(\d+)(?:_rank\d+)?\.pth$")


def cleanup_old_checkpoints(directory: str, keep: int = 3, rank: int = 0) -> None:
    """Keep only the most recent checkpoint files in a directory."""
    
    # Only rank 0 performs cleanup
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


def _resolve_checkpoint_path(checkpoint_path: Optional[str], rank: int = 0) -> str:
    """Resolve the checkpoint path, auto-selecting the highest step if needed."""
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
        search_desc = ", ".join(search_dirs)
        raise FileNotFoundError(f"No checkpoint files matching '*_step_*.pth' found in: {search_desc}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    latest_step, latest_path = candidates[0]
    if rank == 0:
        print(f"Auto-selected checkpoint '{os.path.basename(latest_path)}' (step {latest_step})")
    return latest_path


def init_tensor_parallel_group(world_size: int, rank: int) -> None:
    """Initialize distributed process group for tensor parallelism"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    if rank == 0:
        print(f"Initialized tensor parallel group: rank {rank}/{world_size}")


def shard_tensor_parallel(module, world_size: int, rank: int) -> None:
    """Shard linear layers across GPUs for tensor parallelism."""
    import torch.nn as nn
    
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Linear):
            is_column_parallel = any(suffix in name for suffix in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'])
            is_row_parallel = any(suffix in name for suffix in ['o_proj', 'down_proj'])
            
            if is_column_parallel:
                out_features = layer.out_features
                chunk_size = out_features // world_size
                start_idx = rank * chunk_size
                end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features
                
                new_layer = nn.Linear(layer.in_features, end_idx - start_idx, bias=layer.bias is not None)
                new_layer.weight.data = layer.weight.data[start_idx:end_idx].clone()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data[start_idx:end_idx].clone()
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(module.named_modules())[parent_name]
                    setattr(parent, child_name, new_layer)
                
            elif is_row_parallel:
                in_features = layer.in_features
                chunk_size = in_features // world_size
                start_idx = rank * chunk_size
                end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features
                
                new_layer = nn.Linear(end_idx - start_idx, layer.out_features, bias=layer.bias is not None)
                new_layer.weight.data = layer.weight.data[:, start_idx:end_idx].clone()
                if layer.bias is not None:
                    if rank == 0:
                        new_layer.bias.data = layer.bias.data.clone()
                    else:
                        new_layer.bias = None
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(module.named_modules())[parent_name]
                    setattr(parent, child_name, new_layer)
    
    for name, attn_module in module.named_modules():
        if hasattr(attn_module, 'num_heads') and hasattr(attn_module, 'head_dim'):
            original_num_heads = attn_module.num_heads
            original_num_kv_heads = getattr(attn_module, 'num_kv_heads', original_num_heads)
            attn_module.num_heads = original_num_heads // world_size
            if hasattr(attn_module, 'num_kv_heads'):
                attn_module.num_kv_heads = original_num_kv_heads // world_size
            if hasattr(attn_module, 'num_key_value_groups'):
                attn_module.num_key_value_groups = attn_module.num_heads // attn_module.num_kv_heads


class TensorParallelModel(torch.nn.Module):
    """Wrapper for ArgonneModel that implements tensor parallelism."""
    def __init__(self, base_model: ArgonneModel, world_size: int, rank: int):
        super().__init__()
        self.base_model = base_model
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.gradient_checkpointing = False
        self.base_model = self.base_model.to(self.device)
        shard_tensor_parallel(self.base_model, world_size, rank)
        if rank == 0:
            print(f"Model sharded for tensor parallelism across {world_size} GPUs")
    
    def _block_forward(self, block, hidden_states, position_embeddings, attention_mask):
        """Forward pass for a single block with tensor parallelism."""
        residual = hidden_states
        normed = block.input_norm(hidden_states)
        attn_output = block.attn(normed, position_embeddings, attention_mask)
        if self.world_size > 1:
            dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)
        hidden_states = residual + attn_output
        residual = hidden_states
        normed = block.post_norm(hidden_states)
        mlp_output = block.mlp(normed)
        if self.world_size > 1:
            dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM)
        hidden_states = residual + mlp_output
        return hidden_states
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass with tensor parallelism."""
        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        hidden_states = self.base_model.embed_tokens(input_ids)
        batch_size, seq_length = input_ids.shape
        cos, sin = self.base_model.rotary_emb(hidden_states, seq_length)
        position_embeddings = (cos, sin)
        
        for block in self.base_model.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._block_forward, block, hidden_states, position_embeddings, attention_mask, use_reentrant=False
                )
            else:
                hidden_states = self._block_forward(block, hidden_states, position_embeddings, attention_mask)
        
        hidden_states = self.base_model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(logits=logits, loss=loss)
    
    def generate(self, input_ids, max_length=1024, temperature=1.0, top_k=None, top_p=None, do_sample=True):
        """Generation method matching ArgonneModel interface"""
        return self.base_model.generate(input_ids, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample)
    
    def state_dict(self, *args, **kwargs):
        """Get state dict from base model"""
        return self.base_model.state_dict(*args, **kwargs)
    
    def parameters(self):
        """Get parameters from base model"""
        return self.base_model.parameters()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False


def is_tensor_parallel_checkpoint(state_dict: dict, config: ArgonneConfig) -> bool:
    """Detect if a checkpoint was saved from a tensor-parallel model."""
    expected_q_proj_out = config.num_attention_heads * (config.hidden_size // config.num_attention_heads)
    if "blocks.0.attn.q_proj.weight" in state_dict:
        actual_shape = state_dict["blocks.0.attn.q_proj.weight"].shape
        if actual_shape[0] < expected_q_proj_out:
            return True
    return False


def resume_training(
    data_glob: str,
    tokenizer_path: str,
    checkpoint_path: Optional[str] = None,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 4096,
    batch_size: int = 4,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    use_streaming: bool = True,
    num_proc: int = 8,
    trust_remote_code: bool = False,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)
    init_tensor_parallel_group(world_size, rank)
    is_main_process = (rank == 0)
    
    # Clean up old checkpoints at startup (only once)
    if is_main_process:
        cleanup_old_checkpoints("pretrained", keep=3, rank=rank)
    
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

    # Build config
    config = ArgonneConfig(
        vocab_size=vocab_size,
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

    # Load checkpoint
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path, rank)
    if is_main_process:
        print(f"Resuming from: {resolved_checkpoint}")

    ckpt = safe_torch_load(resolved_checkpoint, map_location="cpu", weights_only=True)
    stored_model_dtype = ckpt.get("model_dtype")

    # Determine dtype
    supports_bf16 = False
    amp_dtype = torch.float32
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
        supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()

        requested_dtype: Optional[torch.dtype] = None
        if isinstance(stored_model_dtype, str):
            if stored_model_dtype == str(torch.bfloat16):
                if supports_bf16:
                    requested_dtype = torch.bfloat16
                else:
                    if is_main_process:
                        print("Checkpoint was saved in bf16 but GPU doesn't support it. Using fp16.")
            elif stored_model_dtype == str(torch.float16):
                requested_dtype = torch.float16

        if requested_dtype is not None:
            amp_dtype = requested_dtype
        else:
            amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    target_dtype = amp_dtype if amp_dtype in (torch.float16, torch.bfloat16) else torch.float32

    # Handle compiled model checkpoints
    if any(k.startswith("_orig_mod.") for k in ckpt["model_state_dict"].keys()):
        if is_main_process:
            print("Detected compiled model checkpoint, converting parameter names...")
        new_state_dict = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("_orig_mod.") and "pipeline_stages" not in k:
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
        ckpt["model_state_dict"] = new_state_dict

    converted_state = cast_state_dict_to_dtype(ckpt["model_state_dict"], target_dtype)
    
    # Check if checkpoint is tensor-parallel
    is_tp_checkpoint = is_tensor_parallel_checkpoint(converted_state, config)
    
    # Create base model
    base_model = ArgonneModel(config)
    base_model.to(dtype=target_dtype)
    
    if is_tp_checkpoint:
        if is_main_process:
            print("="*70)
            print("DETECTED TENSOR-PARALLEL CHECKPOINT")
            print("="*70)
            print("This checkpoint contains sharded weights from tensor-parallel training.")
            print("Loading the sharded weights directly into the sharded model structure.")
            print("="*70)
        
        # First shard the model architecture
        model = TensorParallelModel(base_model, world_size, rank)
        
        # Then load the already-sharded weights
        # Note: This assumes the checkpoint contains weights compatible with current sharding
        missing_keys, unexpected_keys = model.base_model.load_state_dict(converted_state, strict=False)
        if is_main_process and (missing_keys or unexpected_keys):
            print(f"Loading checkpoint with missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
    else:
        if is_main_process:
            print("Detected regular (unsharded) checkpoint - loading full weights then sharding")
        base_model.load_state_dict(converted_state)
        model = TensorParallelModel(base_model, world_size, rank)
    
    model.gradient_checkpointing_enable()
    if is_main_process:
        print("Gradient checkpointing enabled")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=False)
    
    # Get checkpoint info
    global_step = ckpt.get("global_step", 0)
    total_tokens_processed = ckpt.get("tokens_processed", 0)
    
    # Setup scheduler
    min_lr = min(min_lr, lr)
    scheduler = CosineWarmupScheduler(optimizer, base_lr=lr, warmup_steps=warmup_steps, max_steps=total_training_steps, min_lr=min_lr)
    if "scheduler_state_dict" in ckpt and not is_tp_checkpoint:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        scheduler.step(global_step)
    
    if is_main_process:
        print(f"Resuming from step {global_step}, tokens processed: {total_tokens_processed:,}")
    
    # Setup data position
    data_position = DataPosition(streaming=use_streaming)
    if "data_position" in ckpt:
        data_position.restore_state(ckpt.get("data_position"))
        if is_main_process:
            print(f"Resuming from file {data_position.current_file_idx}, position {data_position.position_in_file}")
    else:
        if is_main_process:
            print("No data position found - starting from beginning")

    # Setup mixed precision
    use_grad_scaler = amp_dtype == torch.float16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    if is_main_process:
        if supports_bf16:
            print("Using torch.bfloat16 autocast")
        elif amp_dtype == torch.float16:
            print("Using torch.float16 autocast with GradScaler")

    first_device = model.device
    tokens_in_this_session = 0
    last_loss_value: Optional[float] = None

    # Training loop
    if use_streaming:
        if is_main_process:
            print(f"=== Resuming streaming training from step {global_step} ===")

        token_gen = streaming_token_generator(
            data_files, hf_tokenizer, block_size,
            data_position.current_file_idx, data_position.position_in_file, data_position.chunk_offset, rank
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
                        token_gen = streaming_token_generator(data_files, hf_tokenizer, block_size, rank=rank)
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
                        outputs = model(input_ids=x_local, labels=y_local)
                        loss_tensor = outputs.loss.to(first_device)

                    last_loss_value = float(loss_tensor.detach().cpu().item())

                    if scaler is not None:
                        scaler.scale(loss_tensor).backward()
                        scaler.unscale_(optimizer)
                        _ensure_gradient_dtype_matches_params(model)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_tensor.backward()
                        _ensure_gradient_dtype_matches_params(model)
                        optimizer.step()

                    global_step += 1
                    if pbar is not None:
                        pbar.update(1)

                    if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                        current_total_tokens = total_tokens_processed + tokens_in_this_session
                        print(f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {current_total_tokens:,} | LR: {current_lr:.6e}")

                    if global_step % 300 == 0 and is_main_process:
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
                            n_layer=config.num_hidden_layers,
                            n_head=config.num_attention_heads,
                            n_embd=config.hidden_size,
                            base_lr=lr,
                            min_lr=min_lr,
                            warmup_steps=warmup_steps,
                            max_steps=total_training_steps,
                        )

                except StopIteration:
                    if is_main_process:
                        print("StopIteration - restarting dataset")
                    data_position.next_epoch()
                    token_gen = streaming_token_generator(data_files, hf_tokenizer, block_size, rank=rank)
                    continue
        finally:
            if pbar is not None:
                pbar.close()

    else:
        if is_main_process:
            print(f"=== Resuming non-streaming training from step {global_step} ===")

        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=num_proc)
        total_samples = len(tokenized_data)
        if is_main_process:
            print(f"Total samples: {total_samples}")
        
        pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training") if is_main_process else None
        
        try:
            while global_step < total_training_steps:
                batch_indices = data_position.generate_shuffled_indices(total_samples)
                
                if len(batch_indices) < batch_size:
                    data_position.next_epoch(total_samples)
                    if is_main_process:
                        print(f"Starting new epoch at step {global_step}")
                    continue
                
                batch_indices = batch_indices[:batch_size]
                data_position.update_nonstreaming_position(data_position.current_position + len(batch_indices))
                batch_token_lists = [tokenized_data[i] for i in batch_indices]
                
                x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                if x_tens is None:
                    continue

                batch_tokens = x_tens.numel()
                tokens_in_this_session += batch_tokens
                current_lr = scheduler.step(global_step)

                x_local = x_tens.to(first_device)
                y_local = y_tens.to(first_device)
                optimizer.zero_grad(set_to_none=True)

                autocast_context = torch.amp.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available() else contextlib.nullcontext()

                with autocast_context:
                    outputs = model(input_ids=x_local, labels=y_local)
                    loss_tensor = outputs.loss.to(first_device)

                last_loss_value = float(loss_tensor.detach().cpu().item())

                if scaler is not None:
                    scaler.scale(loss_tensor).backward()
                    scaler.unscale_(optimizer)
                    _ensure_gradient_dtype_matches_params(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_tensor.backward()
                    _ensure_gradient_dtype_matches_params(model)
                    optimizer.step()

                global_step += 1
                if pbar is not None:
                    pbar.update(1)

                if global_step % 50 == 0 and is_main_process:
                    current_total_tokens = total_tokens_processed + tokens_in_this_session
                    print(f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {current_total_tokens:,}")

                if global_step % 2000 == 0 and is_main_process:
                    current_total_tokens = total_tokens_processed + tokens_in_this_session
                    prompt_str = "Long long time ago, "
                    token_ids = hf_tokenizer.encode(prompt_str)
                    token_ids = hf_tokenizer.encode(prompt_str)
                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                    generated = model.generate(
                        prompt_tensor,
                        max_length=prompt_tensor.shape[1] + 50,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                    )
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
                    save_path = f"pretrained/non_streaming_checkpoint_step_{global_step}.pth"
                    safe_torch_save(checkpoint_state, save_path)
                    print(f"Checkpoint saved @ step {global_step} -> {save_path}")

                    update_training_stats(
                        tokens=current_total_tokens,
                        batch_size=batch_size,
                        steps=global_step,
                        model=model,
                        n_layer=config.num_hidden_layers,
                        n_head=config.num_attention_heads,
                        n_embd=config.hidden_size,
                        base_lr=lr,
                        min_lr=min_lr,
                        warmup_steps=warmup_steps,
                        max_steps=total_training_steps,
                    )
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
    
    # Write stats to JSON file
    os.makedirs("stats", exist_ok=True)
    
    # For the final update, create a timestamped file
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
        help=(
            "Glob pattern for parquet shards "
            "(default: ../data/CC-MAIN-2025-26/*.parquet)"
        ),
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
        help=(
            "Optional path to a specific checkpoint file or directory. "
            "If omitted or a directory is provided, the script automatically "
            "selects the checkpoint with the highest step value in the 'pretrained' directory."
        ),
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
        default=3e-4,
        help="Peak learning rate used after warmup.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5,
        help="Minimum learning rate applied at the beginning and end of training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of optimizer steps reserved for linear LR warmup.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode and load parquet shards into memory.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient for AdamW.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading tokenizers that require custom code.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by torch.distributed.launch)",
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
        weight_decay=args.weight_decay,
        use_streaming=not args.no_streaming,
        num_proc=args.num_proc,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
