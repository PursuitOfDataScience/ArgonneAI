import argparse
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import pickle
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Deque, Dict, List, Optional, Tuple

# CRITICAL: Disable cudagraphs via environment variable BEFORE importing torch
# This is the most reliable way to prevent cudagraph capture in compiled models
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from data_processing import collate_batch
from model import ArgonneConfig, ArgonneModel
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    cast_state_dict_to_dtype,
    log_dataset_plan,
    safe_torch_load,
    safe_torch_save,
)

# Import all tensor parallel components from training.py
from training import (
    DEFAULT_DATA_GLOB,
    DataPosition,
    TensorParallelModel,
    _ensure_gradient_dtype_matches_params,
    _maybe_enable_compilation,
    average_gradients,
    broadcast_parameters,
    resolve_data_parallel_source_rank,
    load_tokenizer_and_build_config,
    resolve_training_data,
    setup_distributed_environment,
    shard_attention_layer,
    shard_mlp_layer,
    shard_tensor_parallel_correctly,
    streaming_token_generator,
    _gcd,
)

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_GATHER_OBJECT_SUPPORTS_DEVICE = "device" in inspect.signature(dist.gather_object).parameters

CHECKPOINT_PATTERN = re.compile(r"_step_(\d+)(?:_rank\d+)?\.pth$")


COLUMN_PARALLEL_WEIGHT_SUFFIXES = (
    ".attn.q_proj.weight",
    ".attn.k_proj.weight",
    ".attn.v_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
)

ROW_PARALLEL_WEIGHT_SUFFIXES = (
    ".attn.o_proj.weight",
    ".mlp.down_proj.weight",
)

COLUMN_PARALLEL_BIAS_SUFFIXES = (
    ".attn.q_proj.bias",
    ".attn.k_proj.bias",
    ".attn.v_proj.bias",
    ".mlp.gate_proj.bias",
    ".mlp.up_proj.bias",
)

KV_COLUMN_PARALLEL_WEIGHT_SUFFIXES = (
    ".attn.k_proj.weight",
    ".attn.v_proj.weight",
)

KV_COLUMN_PARALLEL_BIAS_SUFFIXES = (
    ".attn.k_proj.bias",
    ".attn.v_proj.bias",
)


def _state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cpu_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def _optimizer_state_to_cpu(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Recursively move optimizer state tensors to CPU for safe serialization."""

    if not isinstance(state_dict, Mapping):
        return {}

    cpu_state: Dict[str, object] = {}
    for key, value in state_dict.items():
        if key == "state" and isinstance(value, Mapping):
            state_map: Dict[int, Dict[str, object]] = {}
            for param_id, inner in value.items():
                if not isinstance(inner, Mapping):
                    continue
                param_state: Dict[str, object] = {}
                for inner_key, inner_value in inner.items():
                    if isinstance(inner_value, torch.Tensor):
                        param_state[inner_key] = inner_value.detach().cpu()
                    else:
                        param_state[inner_key] = inner_value
                try:
                    state_map[int(param_id)] = param_state
                except (TypeError, ValueError):
                    continue
            cpu_state["state"] = state_map
        elif isinstance(value, Mapping):
            cpu_state[key] = _optimizer_state_to_cpu(value)
        else:
            cpu_state[key] = value

    if "param_groups" in state_dict:
        cpu_state["param_groups"] = state_dict["param_groups"]

    return cpu_state


def _normalize_optimizer_state_structure(state: Mapping[str, object]) -> Dict[str, object]:
    """Ensure optimizer state dict uses integer parameter ids and copies nested values."""

    normalized: Dict[str, object] = {}

    state_block = state.get("state") if isinstance(state, Mapping) else None
    if isinstance(state_block, Mapping):
        normalized_state: Dict[int, Dict[str, object]] = {}
        for param_id, inner in state_block.items():
            if not isinstance(inner, Mapping):
                continue
            try:
                pid = int(param_id)
            except (TypeError, ValueError):
                continue
            inner_copy: Dict[str, object] = {}
            for inner_key, inner_value in inner.items():
                inner_copy[inner_key] = inner_value
            normalized_state[pid] = inner_copy
        normalized["state"] = normalized_state
    else:
        normalized["state"] = {}

    param_groups = state.get("param_groups") if isinstance(state, Mapping) else None
    if isinstance(param_groups, list):
        normalized["param_groups"] = [dict(group) for group in param_groups]
    else:
        normalized["param_groups"] = []

    defaults = state.get("defaults") if isinstance(state, Mapping) else None
    if isinstance(defaults, Mapping):
        normalized["defaults"] = dict(defaults)

    for key, value in state.items():
        if key in {"state", "param_groups", "defaults"}:
            continue
        normalized[key] = value

    return normalized


def _gather_object_to_cpu(
    obj: object,
    gather_list: Optional[List[Optional[object]]],
    *,
    dst: int,
    group: Optional[dist.ProcessGroup],
) -> None:
    """Gather ``obj`` across ``group`` while forcing CPU buffers to avoid CUDA OOM."""

    if not dist.is_initialized():
        if gather_list is not None:
            for idx in range(len(gather_list)):
                gather_list[idx] = obj if idx == 0 else None
        return

    cpu_device = torch.device("cpu")

    if _GATHER_OBJECT_SUPPORTS_DEVICE:
        dist.gather_object(
            obj,
            gather_list,
            dst=dst,
            group=group,
            device=cpu_device,
        )
        return

    if group is None:
        rank_in_group = dist.get_rank()
        world_size = dist.get_world_size()
        dst_rank = dst
    else:
        rank_in_group = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        try:
            dst_rank = dist.get_group_rank(group, dst)
        except ValueError:
            dst_rank = dst

    # ``gather`` with NCCL requires CUDA tensors, so emulate gather_object by
    # serializing locally and broadcasting one rank at a time. This keeps all
    # buffers on CPU while avoiding NCCL's GPU allocations for coalesced
    # gathers.
    if gather_list is not None and rank_in_group == dst_rank:
        if len(gather_list) < world_size:
            raise ValueError(
                "gather_list must be sized to the tensor-parallel world size"
            )
        for idx in range(len(gather_list)):
            gather_list[idx] = None

    for src_rank in range(world_size):
        # Only the source rank seeds the object; others supply a placeholder.
        payload: List[Optional[object]]
        if rank_in_group == src_rank:
            payload = [obj]
        else:
            payload = [None]

        dist.broadcast_object_list(payload, src=src_rank, group=group)

        if rank_in_group != dst_rank:
            payload[0] = None
            continue

        received = payload[0]
        if gather_list is not None:
            if src_rank < len(gather_list):
                gather_list[src_rank] = received


def _deduplicate_kv_tensor(
    tensor: torch.Tensor,
    *,
    config: Optional[ArgonneConfig],
    is_bias: bool,
) -> torch.Tensor:
    """Average replicated KV heads when tensor parallelism exceeds num_kv_heads."""

    if config is None:
        return tensor

    expected_kv_out = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    if expected_kv_out <= 0:
        return tensor

    current = tensor.shape[0]
    if current == expected_kv_out:
        return tensor

    if current % expected_kv_out != 0:
        return tensor

    replication = current // expected_kv_out
    if replication <= 1:
        return tensor

    head_dim = config.hidden_size // config.num_attention_heads

    if is_bias:
        reshaped = tensor.reshape(replication, config.num_key_value_heads, head_dim)
        averaged = reshaped.mean(dim=0)
        return averaged.reshape(-1)

    in_features = tensor.shape[1]
    reshaped = tensor.reshape(replication, config.num_key_value_heads, head_dim, in_features)
    averaged = reshaped.mean(dim=0)
    return averaged.reshape(expected_kv_out, in_features)


def merge_tensor_parallel_shards(
    shard_states: Sequence[Mapping[str, torch.Tensor]],
    *,
    config: Optional[ArgonneConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct full weights from tensor-parallel shards."""

    shard_list: List[Mapping[str, torch.Tensor]] = list(shard_states)
    if not shard_list:
        return {}

    all_keys = set()
    for shard in shard_list:
        all_keys.update(shard.keys())

    full_state: Dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        values = [shard.get(key) for shard in shard_list]
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if not tensors:
            continue

        if key.endswith(COLUMN_PARALLEL_WEIGHT_SUFFIXES):
            merged = torch.cat(tensors, dim=0)
            if key.endswith(KV_COLUMN_PARALLEL_WEIGHT_SUFFIXES):
                merged = _deduplicate_kv_tensor(merged, config=config, is_bias=False)
            full_state[key] = merged
        elif key.endswith(COLUMN_PARALLEL_BIAS_SUFFIXES):
            merged = torch.cat(tensors, dim=0)
            if key.endswith(KV_COLUMN_PARALLEL_BIAS_SUFFIXES):
                merged = _deduplicate_kv_tensor(merged, config=config, is_bias=True)
            full_state[key] = merged
        elif key.endswith(ROW_PARALLEL_WEIGHT_SUFFIXES):
            # Skip ranks that do not hold this shard (bias-less linears)
            ordered = [value for value in values if isinstance(value, torch.Tensor)]
            full_state[key] = torch.cat(ordered, dim=1)
        else:
            # Parameters replicated on all ranks (embeddings, norms, lm_head, biases)
            full_state[key] = tensors[0]

    return full_state




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
        checkpoint_world_size = int(
            ckpt.get("tensor_parallel_world_size")
            or ckpt.get("world_size", 1)
        )
        
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
    gradient_accumulation_steps: int = 4,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    use_streaming: bool = True,
    num_proc: int = 8,
    trust_remote_code: bool = False,
    force_from_scratch: bool = False,
    rewarmup_steps: int = 100,
    use_gradient_checkpointing: bool = True,
    add_document_tokens: bool = False,
    enable_compile: bool = True,
):
    dist_ctx = setup_distributed_environment()
    is_main_process = dist_ctx.is_main_process
    rank = dist_ctx.rank
    tensor_world_size = dist_ctx.tensor_parallel_size
    tensor_parallel_rank = dist_ctx.tensor_parallel_rank
    grad_accum_steps = max(1, int(gradient_accumulation_steps))

    if is_main_process:
        print("=" * 70)
        print("RESUMING PRETRAINING WITH TENSOR + DATA PARALLELISM")
        print("=" * 70)
        print(
            "Total GPUs: %d (%d node(s) × %d GPU(s) each)"
            % (
                dist_ctx.world_size,
                dist_ctx.num_nodes,
                dist_ctx.tensor_parallel_size,
            )
        )
        if dist_ctx.data_parallel_size > 1:
            print(f"Data parallel replicas: {dist_ctx.data_parallel_size}")

    if is_main_process:
        cleanup_old_checkpoints("pretrained", keep=50, rank=dist_ctx.rank)

    # Resolve data files
    data_files, _ = resolve_training_data(data_glob)

    if is_main_process:
        print(f"Found {len(data_files)} data files")
        log_dataset_plan(data_files)

    if dist_ctx.data_parallel_size > 1:
        local_data_files = data_files[dist_ctx.data_parallel_rank :: dist_ctx.data_parallel_size]
    else:
        local_data_files = list(data_files)

    if dist_ctx.tensor_parallel_rank == 0:
        print(
            f"Node {dist_ctx.node_rank}: assigned {len(local_data_files)} shard(s) for training"
        )

    if not local_data_files:
        raise RuntimeError(
            "No data shards assigned to rank %d (data_parallel_rank=%d)."
            % (dist_ctx.rank, dist_ctx.data_parallel_rank)
        )

    # Load tokenizer
    hf_tokenizer, config = load_tokenizer_and_build_config(
        tokenizer_path,
        block_size=block_size,
        trust_remote_code=trust_remote_code,
        use_gradient_checkpointing=use_gradient_checkpointing,
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
        resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path, dist_ctx.rank)

        if resolved_checkpoint:
            is_compatible = check_checkpoint_compatibility(
                resolved_checkpoint, config, dist_ctx.rank
            )
            
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

    checkpoint_world_size = tensor_world_size
    checkpoint_tensor_shards: Optional[List[Dict[str, torch.Tensor]]] = None
    checkpoint_optimizer_shards: Optional[List[Dict[str, object]]] = None
    legacy_optimizer_state: Optional[Dict[str, object]] = None
    saved_scheduler_state: Optional[Dict[str, object]] = None
    load_shards_after_wrap = False

    if load_checkpoint and resolved_checkpoint:
        ckpt = safe_torch_load(resolved_checkpoint, map_location="cpu", weights_only=True)

        saved_grad_accum = ckpt.get("gradient_accumulation_steps")
        if (
            isinstance(saved_grad_accum, int)
            and saved_grad_accum > 0
            and saved_grad_accum != grad_accum_steps
            and is_main_process
        ):
            print(
                "⚠ Checkpoint was created with gradient accumulation=%d (current setting=%d)"
                % (saved_grad_accum, grad_accum_steps)
            )

        checkpoint_world_size = int(
            ckpt.get("tensor_parallel_world_size")
            or ckpt.get("world_size", 1)
        )
        is_tensor_checkpoint = ckpt.get("tensor_parallel", False)

        raw_state_dict = ckpt.get("model_state_dict", {}) or {}
        raw_shard_list = ckpt.get("tensor_parallel_shards")
        raw_optimizer_shards = ckpt.get("tensor_parallel_optimizer_states")
        legacy_optimizer_state = ckpt.get("optimizer_state_dict")
        saved_scheduler_state = ckpt.get("scheduler_state_dict")

        def _normalize_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not state_dict:
                return {}
            needs_conversion = any(k.startswith("_orig_mod.") for k in state_dict.keys())
            if not needs_conversion:
                return dict(state_dict)
            if is_main_process:
                print("Detected compiled model checkpoint, converting parameter names...")
            converted: Dict[str, torch.Tensor] = {}
            for key, value in state_dict.items():
                new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
                converted[new_key] = value
            return converted

        raw_state_dict = _normalize_keys(raw_state_dict)

        shard_states: Optional[List[Dict[str, torch.Tensor]]] = None
        if isinstance(raw_shard_list, (list, tuple)) and raw_shard_list:
            shard_states = []
            for shard in raw_shard_list:
                normalized = _normalize_keys(shard)
                shard_states.append(cast_state_dict_to_dtype(_state_dict_to_cpu(normalized), torch.float32))

        if isinstance(raw_optimizer_shards, (list, tuple)) and raw_optimizer_shards:
            checkpoint_optimizer_shards = []
            for shard in raw_optimizer_shards:
                if isinstance(shard, Mapping):
                    checkpoint_optimizer_shards.append(dict(shard))
                else:
                    checkpoint_optimizer_shards.append({})

        model_state_cpu: Dict[str, torch.Tensor] = {}
        if raw_state_dict:
            model_state_cpu = cast_state_dict_to_dtype(_state_dict_to_cpu(raw_state_dict), torch.float32)

        if shard_states and len(shard_states) != checkpoint_world_size:
            raise RuntimeError(
                "Checkpoint provides tensor_parallel_shards but the list length does not match the saved world size."
            )

        sample_key = "blocks.0.attn.q_proj.weight"
        if model_state_cpu and sample_key in model_state_cpu:
            expected = config.hidden_size
            if model_state_cpu[sample_key].shape[0] != expected and not shard_states:
                # OLD CHECKPOINT: Has sharded weights but no shard metadata
                # Replicate the shard to reconstruct full weights
                if is_main_process:
                    print("⚠ Loading OLD checkpoint with sharded weights (no tensor_parallel_shards metadata)")
                    print("  Reconstructing full weights by replicating rank-0 shard...")
                
                shard_size = model_state_cpu[sample_key].shape[0]
                if expected % shard_size != 0:
                    raise RuntimeError(
                        f"Cannot reconstruct full weights: expected size {expected} is not divisible by shard size {shard_size}"
                    )
                
                # Replicate the shard to create pseudo-shards for all ranks
                replicated_shards = []
                for _ in range(checkpoint_world_size):
                    replicated_shards.append(dict(model_state_cpu))
                
                if is_main_process:
                    print(f"  Merging {len(replicated_shards)} replicated shards into full weights...")

                model_state_cpu = merge_tensor_parallel_shards(
                    replicated_shards, config=config
                )

        if shard_states and checkpoint_world_size == tensor_world_size:
            load_shards_after_wrap = True
            checkpoint_tensor_shards = shard_states
        else:
            if shard_states and checkpoint_world_size != tensor_world_size:
                if is_main_process:
                    print(
                        "Merging tensor-parallel shards from checkpoint to rebuild full weights "
                        "(checkpoint world size=%d, target world size=%d)..."
                        % (checkpoint_world_size, tensor_world_size)
                    )
                model_state_cpu = merge_tensor_parallel_shards(shard_states, config=config)

            if not model_state_cpu:
                raise RuntimeError(
                    "Checkpoint does not contain full weights to load after merging tensor parallel shards."
                )

            try:
                base_model.load_state_dict(model_state_cpu, strict=True)
                if is_main_process:
                    print("✓ Loaded checkpoint with strict=True (exact match)")
            except RuntimeError as e:
                if is_main_process:
                    print(f"⚠ Strict loading failed: {e}")
                    print("  Attempting non-strict loading...")
                base_model.load_state_dict(model_state_cpu, strict=False)
                if is_main_process:
                    print("✓ Loaded checkpoint with strict=False (some keys may be missing)")

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
    model = TensorParallelModel(
        base_model,
        tensor_world_size,
        tensor_parallel_rank,
        dist_ctx.local_rank,
        dist_ctx.tensor_parallel_group,
    )

    if load_shards_after_wrap and checkpoint_tensor_shards:
        if len(checkpoint_tensor_shards) != tensor_world_size:
            raise RuntimeError(
                "tensor_parallel_shards length does not match current world size; cannot load shard weights."
            )
        shard_state = checkpoint_tensor_shards[tensor_parallel_rank]
        try:
            model.base_model.load_state_dict(shard_state, strict=True)
            if is_main_process:
                print("✓ Loaded tensor-parallel shard weights with strict=True")
        except RuntimeError as shard_err:
            if is_main_process:
                print(f"⚠ Strict shard loading failed: {shard_err}")
                print("  Falling back to strict=False for shard weights")
            model.base_model.load_state_dict(shard_state, strict=False)
            if is_main_process:
                print("✓ Loaded tensor-parallel shard weights with strict=False")

    dp_broadcast_src = resolve_data_parallel_source_rank(
        group=dist_ctx.data_parallel_group,
        default_rank=dist_ctx.rank,
    )
    broadcast_parameters(
        model.parameters(),
        group=dist_ctx.data_parallel_group,
        src=dp_broadcast_src,
    )
    broadcast_parameters(
        model.base_model.buffers(),
        group=dist_ctx.data_parallel_group,
        src=dp_broadcast_src,
    )

    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"✓ Model contains {total_params:,} parameters")

    _maybe_enable_compilation(
        model,
        enable_compile=enable_compile,
        is_main_process=is_main_process,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=True
    )

    optimizer_state_loaded = False
    optimizer_state_warning: Optional[str] = None
    saved_lr_ramp_state = None
    if load_checkpoint and resolved_checkpoint:
        saved_lr_ramp_state = ckpt.get("lr_ramp_state")

        if checkpoint_optimizer_shards and checkpoint_world_size != tensor_world_size:
            if is_main_process:
                print(
                    "⚠ Cannot restore optimizer state: checkpoint tensor parallel world size=%d (current=%d)"
                    % (checkpoint_world_size, tensor_world_size)
                )
            optimizer_state_warning = (
                "checkpoint tensor parallel world size does not match current topology"
            )
        elif checkpoint_optimizer_shards and checkpoint_world_size == tensor_world_size:
            shard_idx = tensor_parallel_rank
            if 0 <= shard_idx < len(checkpoint_optimizer_shards):
                shard_state = checkpoint_optimizer_shards[shard_idx]
                if isinstance(shard_state, Mapping) and shard_state:
                    try:
                        normalized_state = _normalize_optimizer_state_structure(shard_state)
                        optimizer.load_state_dict(normalized_state)
                        optimizer_state_loaded = True
                    except Exception as opt_err:
                        if is_main_process:
                            print(
                                "⚠ Failed to restore optimizer state shard for rank %d: %s"
                                % (tensor_parallel_rank, opt_err)
                            )
                        optimizer_state_warning = (
                            "optimizer shard existed but failed to load; see error above"
                        )
        elif legacy_optimizer_state and tensor_world_size == 1:
            try:
                normalized_state = _normalize_optimizer_state_structure(legacy_optimizer_state)
                optimizer.load_state_dict(normalized_state)
                optimizer_state_loaded = True
            except Exception as opt_err:
                if is_main_process:
                    print(f"⚠ Failed to restore optimizer state: {opt_err}")
                optimizer_state_warning = "legacy optimizer state was incompatible"
        elif not checkpoint_optimizer_shards and not legacy_optimizer_state:
            optimizer_state_warning = (
                "checkpoint did not include tensor-parallel optimizer shards"
            )

        if optimizer_state_loaded and is_main_process:
            print("✓ Restored optimizer state from checkpoint")

    # Setup scheduler
    min_lr = min(min_lr, lr)

    # CRITICAL: If resuming, use shorter re-warmup to stabilize
    effective_warmup = warmup_steps
    if load_checkpoint and resolved_checkpoint and global_step > 0:
        if optimizer_state_loaded:
            effective_warmup = warmup_steps
        else:
            effective_warmup = rewarmup_steps  # Much shorter re-warmup
            if is_main_process and rewarmup_steps > 0:
                print(f"⚠ Using {rewarmup_steps}-step re-warmup for resume stability")
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=lr,
        warmup_steps=effective_warmup,  # Use re-warmup if resuming
        max_steps=total_training_steps,
        min_lr=min_lr,
    )

    lr_ramp_tracker = None
    if (
        load_checkpoint
        and resolved_checkpoint
        and global_step > 0
        and rewarmup_steps > 0
        and not optimizer_state_loaded
    ):
        steps_completed = 0
        stored_start_lr = float(min_lr)
        if isinstance(saved_lr_ramp_state, dict):
            stored_total = int(saved_lr_ramp_state.get("total_steps", rewarmup_steps))
            stored_completed = int(saved_lr_ramp_state.get("steps_completed", 0))
            if 0 <= stored_completed < stored_total:
                steps_completed = max(0, stored_completed)
            stored_start_lr = float(saved_lr_ramp_state.get("start_lr", stored_start_lr))

        def _preview_lr_value(step: int) -> float:
            preview = getattr(scheduler, "preview_lr", None)
            if callable(preview):
                try:
                    return float(preview(step))
                except Exception:
                    pass

            lr_for_step = getattr(scheduler, "_lr_for_step", None)
            if callable(lr_for_step):
                try:
                    return float(lr_for_step(step))
                except Exception:
                    pass

            last_lr = getattr(scheduler, "last_lr", None)
            if last_lr is not None:
                try:
                    return float(last_lr)
                except Exception:
                    pass

            return float(lr)

        target_step = global_step + 1
        target_lr = _preview_lr_value(target_step)
        start_lr = float(stored_start_lr)
        if isinstance(saved_lr_ramp_state, dict):
            start_lr = float(saved_lr_ramp_state.get("start_lr", start_lr))

        completed_steps = min(steps_completed, rewarmup_steps)
        ramp_total = max(rewarmup_steps, 1)

        def _compute_progress(step_count: int) -> float:
            if ramp_total <= 0:
                return start_lr
            fraction = min(max(step_count / ramp_total, 0.0), 1.0)
            return start_lr + (target_lr - start_lr) * fraction

        current_ramp_lr = _compute_progress(completed_steps)

        lr_ramp_tracker = {
            "steps_completed": completed_steps,
            "total_steps": rewarmup_steps,
            "start_lr": start_lr,
            "target_lr": target_lr,
            "last_step": global_step,
            "current_lr": current_ramp_lr,
        }

        if is_main_process:
            print(
                "⚠ Using %d-step LR ramp from %.6e to %.6e after optimizer reset"
                % (rewarmup_steps, start_lr, target_lr)
            )
            if lr_ramp_tracker["steps_completed"] > 0:
                print(
                    "  Continuing ramp progress: %d/%d steps already applied"
                    % (
                        lr_ramp_tracker["steps_completed"],
                        lr_ramp_tracker["total_steps"],
                    )
                )
                print(
                    "  Resuming ramp from LR %.6e"
                    % lr_ramp_tracker["current_lr"]
                )

    if load_checkpoint and resolved_checkpoint:
        if optimizer_state_loaded:
            if isinstance(saved_scheduler_state, Mapping):
                try:
                    scheduler.load_state_dict(saved_scheduler_state)
                    if is_main_process:
                        print("✓ Restored scheduler state from checkpoint")
                except Exception as sched_err:
                    if is_main_process:
                        print(f"⚠ Failed to restore scheduler state: {sched_err}")
                    scheduler.step(global_step)
            else:
                scheduler.step(global_step)
        else:
            if is_main_process:
                print("⚠ Note: Optimizer state not restored (incompatible with tensor parallelism)")
                if optimizer_state_warning:
                    print(f"  Reason: {optimizer_state_warning}")
                print("  Using fresh AdamW optimizer - expect small loss spike initially")
                if not checkpoint_optimizer_shards:
                    print(
                        "  Future checkpoints saved with this script will store tensor-parallel optimizer shards"
                    )
                if rewarmup_steps > 0:
                    print(f"  Re-warming up learning rate over {effective_warmup} steps")
            # When not restoring optimizer state we intentionally avoid loading scheduler state
    else:
        scheduler.step(global_step)
    
    if is_main_process:
        print(f"✓ Training setup complete")
        print(f"  - Starting step: {global_step}")
        print(f"  - Tokens processed: {total_tokens_processed:,}")
        print(f"  - Learning rate: {lr:.2e}")
        print(
            "  - Topology: tensor=%d, data=%d, total GPUs=%d"
            % (
                tensor_world_size,
                dist_ctx.data_parallel_size,
                dist_ctx.world_size,
            )
        )
    
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
    using_cuda = torch.cuda.is_available()
    prefetch_stream = torch.cuda.Stream(device=first_device) if using_cuda else None
    max_prefetch_batches = 6 if using_cuda else 2
    prefetched_batches: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque()
    prefetch_warmup_done = not using_cuda

    def enqueue_batch(x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> None:
        if not using_cuda:
            prefetched_batches.append((x_cpu.to(first_device), y_cpu.to(first_device)))
            return

        assert prefetch_stream is not None
        with torch.cuda.stream(prefetch_stream):
            x_gpu = x_cpu.to(first_device, non_blocking=True)
            y_gpu = y_cpu.to(first_device, non_blocking=True)
        prefetched_batches.append((x_gpu, y_gpu))

    def pop_prefetched() -> Tuple[torch.Tensor, torch.Tensor]:
        if using_cuda and prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        return prefetched_batches.popleft()

    micro_step = 0
    tokens_in_this_session = 0
    last_loss_value: Optional[float] = None
    current_lr = lr

    optimizer.zero_grad(set_to_none=True)

    # Training loop
    if use_streaming:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"RESUMING TRAINING WITH TENSOR PARALLELISM")
            print(f"{'='*70}")
            print(f"Step: {global_step} / {total_training_steps}")
            print(f"Learning rate: {lr:.2e}")
            print(f"Batch size: {batch_size}")
            print(
                "Topology: tensor=%d, data=%d, total GPUs=%d"
                % (
                    tensor_world_size,
                    dist_ctx.data_parallel_size,
                    dist_ctx.world_size,
                )
            )
            print(f"{'='*70}\n")

        # Initialize the streaming generator (optionally adding document boundary tokens)
        token_gen = streaming_token_generator(
            local_data_files,
            hf_tokenizer,
            block_size,
            data_position.current_file_idx,
            data_position.position_in_file,
            data_position.chunk_offset,
            rank,
            add_document_tokens=add_document_tokens,
        )
        
        token_buffer: List[List[int]] = []
        active_shard: Optional[str] = None
        last_meta: Optional[Tuple[str, int, int, int]] = None

        prompt_seed_text = "Long long time ago, "
        prompt_token_ids = hf_tokenizer.encode(prompt_seed_text)
        prompt_tensor_device = first_device if using_cuda else model.device
        cached_prompt_tensor = torch.tensor(
            prompt_token_ids, dtype=torch.long, device=prompt_tensor_device
        ).unsqueeze(0)

        pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training") if is_main_process else None

        try:
            while global_step < total_training_steps:
                try:
                    tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)

                    if file_idx == -1:
                        if is_main_process:
                            print("End of dataset - restarting")
                        data_position.next_epoch()
                        # Restart with same generator type
                        token_gen = streaming_token_generator(
                            local_data_files,
                            hf_tokenizer,
                            block_size,
                            rank=rank,
                            add_document_tokens=add_document_tokens,
                        )
                        continue

                    token_buffer.append(tokens)
                    last_meta = (shard_name, file_idx, position, chunk_idx)
                    data_position.update_streaming_position(
                        file_idx,
                        position,
                        chunk_idx,
                        local_data_files[file_idx],
                    )

                    if shard_name != active_shard:
                        active_shard = shard_name
                        if is_main_process:
                            print(
                                f"Processing shard {file_idx + 1}/{len(local_data_files)}: {shard_name}"
                            )

                    if len(token_buffer) < batch_size:
                        continue

                    x_tens, y_tens = collate_batch(token_buffer, block_size)
                    token_buffer.clear()
                    if x_tens is None or last_meta is None:
                        continue

                    x_local: torch.Tensor
                    y_local: torch.Tensor

                    if using_cuda:
                        enqueue_batch(x_tens, y_tens)
                        if not prefetch_warmup_done:
                            if len(prefetched_batches) < max_prefetch_batches:
                                continue
                            prefetch_warmup_done = True
                        x_local, y_local = pop_prefetched()
                    else:
                        enqueue_batch(x_tens, y_tens)
                        x_local, y_local = pop_prefetched()

                    batch_tokens = x_local.numel()
                    tokens_in_this_session += batch_tokens

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

                    loss_for_backward = loss_tensor / grad_accum_steps

                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

                    micro_step += 1

                    if micro_step >= grad_accum_steps:
                        next_step = global_step + 1
                        current_lr = scheduler.step(next_step)

                        ramp_active = (
                            lr_ramp_tracker is not None
                            and lr_ramp_tracker.get("steps_completed", 0)
                            < lr_ramp_tracker.get("total_steps", 0)
                        )

                        if ramp_active:
                            completed = lr_ramp_tracker["steps_completed"]
                            total = max(1, lr_ramp_tracker["total_steps"])
                            start_lr = float(lr_ramp_tracker.get("start_lr", min_lr))
                            target_lr = float(
                                lr_ramp_tracker.get("target_lr", current_lr)
                            )
                            ramp_fraction = min((completed + 1) / total, 1.0)
                            ramp_lr = start_lr + (target_lr - start_lr) * ramp_fraction
                            if target_lr >= start_lr:
                                ramp_lr = min(target_lr, max(min_lr, ramp_lr))
                            else:
                                ramp_lr = max(target_lr, min(min_lr, ramp_lr))

                            current_lr = scheduler.override_step(next_step, ramp_lr)
                            lr_ramp_tracker["steps_completed"] = min(
                                completed + 1,
                                lr_ramp_tracker["total_steps"],
                            )
                            lr_ramp_tracker["start_lr"] = start_lr
                            lr_ramp_tracker["target_lr"] = target_lr
                            lr_ramp_tracker["current_lr"] = current_lr
                            lr_ramp_tracker["last_step"] = next_step

                            if (
                                is_main_process
                                and lr_ramp_tracker["steps_completed"]
                                >= lr_ramp_tracker["total_steps"]
                            ):
                                print(
                                    "✓ LR ramp complete: reached target LR %.6e"
                                    % target_lr
                                )
                        elif lr_ramp_tracker is not None:
                            lr_ramp_tracker["current_lr"] = current_lr
                            lr_ramp_tracker["last_step"] = next_step

                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        _ensure_gradient_dtype_matches_params(model)
                        average_gradients(
                            model.parameters(),
                            group=dist_ctx.data_parallel_group,
                            world_size=dist_ctx.data_parallel_size,
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        _ensure_gradient_dtype_matches_params(model)
                        average_gradients(
                            model.parameters(),
                            group=dist_ctx.data_parallel_group,
                            world_size=dist_ctx.data_parallel_size,
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    micro_step = 0

                    if pbar is not None:
                        pbar.update(1)

                    if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                        current_total_tokens = total_tokens_processed + tokens_in_this_session
                        print(
                            f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {current_total_tokens:,} | LR: {current_lr:.6e}"
                        )

                    if global_step % 4611 == 0:
                        current_total_tokens = total_tokens_processed + tokens_in_this_session

                        # Generate on all ranks to keep collectives in sync
                        prompt_tensor = cached_prompt_tensor.to(first_device)
                        generated = model.generate(
                            prompt_tensor,
                            max_length=prompt_tensor.shape[1] + 100,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.9,
                        )

                        local_model_state = cast_state_dict_to_dtype(
                            model.base_model.state_dict(), amp_dtype
                        )
                        local_optimizer_state = _normalize_optimizer_state_structure(
                            _optimizer_state_to_cpu(optimizer.state_dict())
                        )

                        tensor_parallel_shards_to_save: Optional[
                            List[Dict[str, torch.Tensor]]
                        ] = None
                        tensor_parallel_optimizer_to_save: Optional[
                            List[Dict[str, object]]
                        ] = None

                        if (
                            dist.is_initialized()
                            and dist_ctx.tensor_parallel_group is not None
                            and dist_ctx.tensor_parallel_size > 1
                        ):
                            gather_root = dist.get_global_rank(
                                dist_ctx.tensor_parallel_group, 0
                            )

                            shard_gather: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None
                            optimizer_gather: Optional[List[Optional[Dict[str, object]]]] = None

                            if dist_ctx.rank == gather_root:
                                shard_gather = [None] * dist_ctx.tensor_parallel_size
                                optimizer_gather = [None] * dist_ctx.tensor_parallel_size

                            _gather_object_to_cpu(
                                local_model_state,
                                shard_gather,
                                dst=gather_root,
                                group=dist_ctx.tensor_parallel_group,
                            )
                            _gather_object_to_cpu(
                                local_optimizer_state,
                                optimizer_gather,
                                dst=gather_root,
                                group=dist_ctx.tensor_parallel_group,
                            )

                            if dist_ctx.rank == gather_root:
                                tensor_parallel_shards_to_save = [
                                    dict(shard) if isinstance(shard, Mapping) else {}
                                    for shard in (shard_gather or [])
                                ]
                                tensor_parallel_optimizer_to_save = [
                                    _normalize_optimizer_state_structure(state)
                                    if isinstance(state, Mapping)
                                    else {}
                                    for state in (optimizer_gather or [])
                                ]
                        else:
                            tensor_parallel_shards_to_save = [local_model_state]
                            tensor_parallel_optimizer_to_save = [local_optimizer_state]

                        if is_main_process:
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                            shards_to_save = tensor_parallel_shards_to_save or [local_model_state]
                            optimizer_states_to_save = (
                                tensor_parallel_optimizer_to_save
                                or [local_optimizer_state]
                            )

                            model_state = shards_to_save[0]
                            if len(shards_to_save) == tensor_world_size:
                                try:
                                    model_state = merge_tensor_parallel_shards(
                                        shards_to_save, config=config
                                    )
                                except Exception as merge_err:
                                    print(
                                        "⚠ Failed to merge tensor-parallel shards for full checkpoint: %s"
                                        % merge_err
                                    )

                            checkpoint_state = {
                                "global_step": global_step,
                                "tokens_processed": current_total_tokens,
                                "model_state_dict": model_state,
                                "optimizer_state_dict": optimizer_states_to_save[0],
                                "scheduler_state_dict": scheduler.state_dict(),
                                "lr_ramp_state": (
                                    dict(lr_ramp_tracker)
                                    if lr_ramp_tracker is not None
                                    else None
                                ),
                                "gradient_accumulation_steps": grad_accum_steps,
                                "loss": last_loss_value,
                                "data_position": data_position.get_state(),
                                "model_dtype": str(amp_dtype),
                                "tensor_parallel": True,
                                "tensor_parallel_world_size": tensor_world_size,
                                "data_parallel_world_size": dist_ctx.data_parallel_size,
                                "global_world_size": dist_ctx.world_size,
                                "rank": rank,
                                "tensor_parallel_rank": tensor_parallel_rank,
                                "data_parallel_rank": dist_ctx.data_parallel_rank,
                            }
                            checkpoint_state["tensor_parallel_shards"] = shards_to_save
                            checkpoint_state["tensor_parallel_optimizer_states"] = (
                                optimizer_states_to_save
                            )
                            os.makedirs("pretrained", exist_ok=True)
                            save_path = (
                                f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                            )
                            safe_torch_save(checkpoint_state, save_path)
                            print(f"Checkpoint saved @ step {global_step} -> {save_path}")

                except StopIteration:
                    if is_main_process:
                        print("StopIteration - restarting dataset")
                    data_position.next_epoch()
                    # Restart with same generator type
                    token_gen = streaming_token_generator(
                        local_data_files,
                        hf_tokenizer,
                        block_size,
                        rank=rank,
                        add_document_tokens=add_document_tokens,
                    )
                    continue
        finally:
            if using_cuda and prefetch_stream is not None:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
            prefetched_batches.clear()
            prefetch_warmup_done = not using_cuda
            if pbar is not None:
                pbar.close()

    final_token_count = total_tokens_processed + tokens_in_this_session
    
    if is_main_process:
        print(f"\n===== TRAINING COMPLETE =====")
        print(f"Total tokens: {final_token_count:,}")
        print(f"Final step: {global_step}")

    if dist.is_initialized():
        dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume Argonne pretraining with Tensor Parallelism")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=DEFAULT_DATA_GLOB,
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
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
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
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing to speed up training (requires more GPU memory).",
    )
    parser.add_argument(
        "--add-document-boundary-tokens",
        action="store_true",
        help=(
            "Prepend the tokenizer BOS token and append the EOS token to each document "
            "before chunking."
        ),
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile acceleration.",
    )
    # NOTE: torch.compile is enabled by default; use --disable-compile to opt out.
    # REMOVED: --checkpoint-segments argument
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        rewarmup_steps=args.rewarmup_steps,
        weight_decay=args.weight_decay,
        use_streaming=not args.no_streaming,
        num_proc=args.num_proc,
        trust_remote_code=args.trust_remote_code,
        force_from_scratch=args.force_from_scratch,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        add_document_tokens=args.add_document_boundary_tokens,
        enable_compile=not args.disable_compile,
    )


if __name__ == "__main__":
    main()
