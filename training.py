import argparse
import contextlib
import os
import re
import time
import traceback
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Dict, List, Optional, Tuple

# CRITICAL: Disable cudagraphs via environment variable BEFORE importing torch
# This is the most reliable way to prevent cudagraph capture in compiled models
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

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
    load_streaming_shard,
    log_dataset_plan,
    resolve_data_files,
    safe_torch_load,
    safe_torch_save,
    validate_tokenizer_path,
)

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def _state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Move all tensors in a state dict to CPU."""
    cpu_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def _ensure_gradient_dtype_matches_params(model: torch.nn.Module) -> None:
    """Cast gradients to match their parameter's dtype/device for fused optimizers."""
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if grad.dtype != param.dtype or grad.device != param.device:
            param.grad = grad.to(device=param.device, dtype=param.dtype)


def _gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor"""
    while b:
        a, b = b, a % b
    return a


@torch._dynamo.disable
def _allreduce_non_compiled(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for all-reduce that prevents torch.compile from tracing through it.
    
    This is critical for tensor parallelism with torch.compile because:
    1. Cudagraphs cannot capture collective operations
    2. The all-reduce needs to happen outside the compiled graph
    
    The @torch._dynamo.disable decorator ensures this function is never compiled.
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


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


def streaming_token_generator(
    data_files: List[str],
    tokenizer,
    block_size: int,
    start_file_idx: int = 0,
    start_position: int = 0,
    start_chunk_offset: int = 0,
    rank: int = 0,
):
    """Sequence-packing generator that concatenates documents separated by the
    tokenizer's EOS token into sequences of exactly ``block_size`` tokens.

    Documents are tokenized on the fly using a thread pool for CPU prefetching.
    When a document is truncated at the end of a packed sequence the remaining
    tokens carry over to the beginning of the next sequence.  This means every
    produced sequence is exactly ``block_size`` tokens and no padding is needed.

    Yields: (packed_tokens, file_idx, position, shard_name, seq_counter)
    """

    import gc

    file_idx = max(start_file_idx, 0)
    processed_count = 0
    is_main_process = (rank == 0)

    # Resolve EOS token id from the tokenizer (never hardcode)
    eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        # Fallback: try to resolve from eos_token string
        eos_token_str = getattr(tokenizer, "eos_token", None)
        if eos_token_str:
            try:
                eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_str)
                if not isinstance(eos_token_id, int) or eos_token_id < 0:
                    eos_token_id = None
            except Exception:
                eos_token_id = None

    if eos_token_id is None:
        raise RuntimeError(
            "Sequence packing requires an EOS token but the tokenizer does not "
            "provide one.  Please use a tokenizer with eos_token defined."
        )

    if is_main_process:
        eos_display = getattr(tokenizer, "eos_token", None) or eos_token_id
        print(f"✓ Sequence packing enabled (block_size={block_size}, eos={eos_display})")

    initial_file_idx = file_idx
    initial_position = start_position
    initial_chunk_offset = start_chunk_offset
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    logical_cores = os.cpu_count() or 1

    env_rows = os.environ.get("RESUME_TOKENIZER_BATCH_ROWS")
    if env_rows is not None:
        try:
            TOKENIZER_BATCH_ROWS = max(16, int(env_rows))
        except ValueError:
            TOKENIZER_BATCH_ROWS = 256
    else:
        TOKENIZER_BATCH_ROWS = max(128, min(512, logical_cores * 4))

    env_workers = os.environ.get("RESUME_TOKENIZER_WORKERS")
    if env_workers is not None:
        try:
            tokenizer_workers = max(1, int(env_workers))
        except ValueError:
            tokenizer_workers = max(1, logical_cores // 2 or 1)
    else:
        tokenizer_workers = max(4, min(32, logical_cores))

    max_pending_batches = max(8, tokenizer_workers * 4)

    def _tokenize_texts(batch_texts: List[str]) -> List[List[int]]:
        filtered_inputs: List[str] = []
        skip_mask: List[bool] = []

        for text in batch_texts:
            keep_text = True
            if len(text) > 50:
                alpha_count = sum(c.isalpha() or c.isspace() for c in text)
                digit_count = sum(c.isdigit() for c in text)
                if digit_count / len(text) > 0.5 or alpha_count / len(text) < 0.3:
                    keep_text = False

            if keep_text:
                filtered_inputs.append(text)
                skip_mask.append(False)
            else:
                filtered_inputs.append("")
                skip_mask.append(True)

        try:
            encoded_batch = tokenizer(
                filtered_inputs,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            if isinstance(encoded_batch, Mapping):
                token_batches = encoded_batch.get("input_ids", [])
            else:
                token_batches = encoded_batch
        except Exception as batch_error:
            if is_main_process:
                print(
                    f"Tokenizer batch failed: {batch_error}. Falling back to per-item encoding.",
                    flush=True,
                )
            token_batches = []
            for idx, text in enumerate(filtered_inputs):
                if skip_mask[idx]:
                    token_batches.append([])
                    continue
                try:
                    token_batches.append(
                        tokenizer.encode(text, add_special_tokens=False)
                    )
                except Exception as single_error:
                    token_batches.append([])
                    if is_main_process:
                        print(
                            f"  Tokenizer error for text len={len(text)}: {single_error}",
                            flush=True,
                        )

        if len(token_batches) < len(skip_mask):
            token_batches.extend([[] for _ in range(len(skip_mask) - len(token_batches))])

        normalized_batches: List[List[int]] = []
        for skip, tokens in zip(skip_mask, token_batches):
            if skip:
                normalized_batches.append([])
            elif isinstance(tokens, Sequence):
                normalized_batches.append(list(tokens))
            else:
                normalized_batches.append([])
        return normalized_batches

    pending_batches: Deque[Tuple[Future, List[Tuple[int, str]]]] = deque()

    def _clear_pending() -> None:
        while pending_batches:
            future, _ = pending_batches.popleft()
            future.cancel()

    # ---- Sequence packing state ----
    # Token buffer that accumulates tokens across documents until we have
    # exactly ``block_size`` tokens to yield.
    packing_buffer: List[int] = []
    seq_counter = 0  # monotonically increasing counter for yielded sequences
    # Track how many packed sequences we need to skip for resume
    sequences_to_skip = start_chunk_offset

    with ThreadPoolExecutor(max_workers=tokenizer_workers, thread_name_prefix="tokenizer") as tokenizer_pool:
        while file_idx < len(data_files):
            dataset = None

            try:
                file_path = data_files[file_idx]
                shard_name = os.path.basename(file_path)
                if is_main_process:
                    print(f"Streaming from shard {file_idx + 1}/{len(data_files)}: {shard_name}")

                try:
                    dataset = load_streaming_shard(file_path)
                    if is_main_process:
                        print(f"Successfully loaded dataset with {len(dataset)} rows")
                    consecutive_errors = 0

                except Exception as file_error:
                    consecutive_errors += 1
                    if is_main_process:
                        print(f"ERROR: Could not read file {file_path}: {file_error}")
                        print(f"Consecutive errors: {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}")

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        if is_main_process:
                            print(
                                "⚠ Too many consecutive errors. Forcing cleanup and restarting..."
                            )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        consecutive_errors = 0

                    file_idx += 1
                    continue

                # Determine starting position within this shard
                if file_idx == initial_file_idx:
                    position = initial_position
                    if is_main_process and position > 0:
                        print(
                            f"  >>> RESUMING from position {position}, seq_offset {sequences_to_skip}"
                        )
                else:
                    position = 0

                _clear_pending()

                while position < len(dataset) or pending_batches:
                    # Fill the pending queue with tokenization futures
                    while position < len(dataset) and len(pending_batches) < max_pending_batches:
                        batch_records: List[Tuple[int, str]] = []

                        while len(batch_records) < TOKENIZER_BATCH_ROWS and position < len(dataset):
                            try:
                                item = dataset[position]
                            except Exception as item_error:
                                if is_main_process:
                                    print(
                                        f"Error reading item at position {position}: {item_error}"
                                    )
                                position += 1
                                continue

                            text = item.get("text") if isinstance(item, Mapping) else None
                            if not text or not isinstance(text, str):
                                position += 1
                                continue

                            batch_records.append((position, text))
                            position += 1

                        if not batch_records:
                            continue

                        batch_texts = [text for _, text in batch_records]
                        future = tokenizer_pool.submit(_tokenize_texts, batch_texts)
                        pending_batches.append((future, batch_records))

                    if not pending_batches:
                        break

                    future, batch_records = pending_batches[0]
                    try:
                        token_batches = future.result()
                    except Exception as future_error:
                        if is_main_process:
                            print(
                                f"Tokenizer future failed: {future_error}. Dropping batch.",
                                flush=True,
                            )
                        pending_batches.popleft()
                        continue

                    pending_batches.popleft()

                    for (record_position, _text), tokens in zip(batch_records, token_batches):
                        if not tokens:
                            continue

                        # Append document tokens + EOS to the packing buffer
                        packing_buffer.extend(tokens)
                        packing_buffer.append(eos_token_id)

                        # Yield as many full packed sequences as possible
                        while len(packing_buffer) >= block_size:
                            packed_seq = packing_buffer[:block_size]
                            packing_buffer = packing_buffer[block_size:]

                            # Handle resume: skip sequences we've already processed
                            if sequences_to_skip > 0:
                                sequences_to_skip -= 1
                                continue

                            processed_count += 1
                            seq_counter += 1
                            yield packed_seq, file_idx, record_position, shard_name, seq_counter

                _clear_pending()
                file_idx += 1

            finally:
                if dataset is not None:
                    del dataset
                dataset = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Flush any remaining tokens in the packing buffer (last partial sequence)
    # Only yield if we have enough tokens to be useful (at least half block_size)
    if len(packing_buffer) >= block_size // 2 and sequences_to_skip <= 0:
        # Pad the last sequence with EOS tokens to reach block_size
        while len(packing_buffer) < block_size:
            packing_buffer.append(eos_token_id)
        packed_seq = packing_buffer[:block_size]
        processed_count += 1
        seq_counter += 1
        yield packed_seq, file_idx - 1, -1, "", seq_counter

    if is_main_process:
        print(f"Completed processing all available files. Produced {processed_count} packed sequences.")

    return None, -1, -1, "", -1


CHECKPOINT_PATTERN = re.compile(r"_step_(\d+)(?:_rank\d+)?\.pth$")


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


def init_tensor_parallel_group(world_size: int, rank: int) -> None:
    """Initialize distributed process group for tensor parallelism.

    When running on multiple nodes, this also creates two sub-groups:
    * **TP group** – ranks on the *same* node (same ``local_rank`` set)
    * **DP group** – ranks with the *same* ``local_rank`` across nodes

    The groups are stored as module-level globals so that ``TensorParallelModel``
    and the training loops can reference them.  For single-node runs the TP
    group is simply the world group and the DP group is ``None``.
    """
    global _TP_GROUP, _DP_GROUP, _TP_SIZE, _DP_SIZE, _TP_RANK, _DP_RANK

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    num_nodes = world_size // local_world_size

    if num_nodes > 1:
        # ---------- TP groups (intra-node) ----------
        for node_idx in range(num_nodes):
            tp_ranks = list(range(node_idx * local_world_size, (node_idx + 1) * local_world_size))
            group = dist.new_group(tp_ranks)
            if rank in tp_ranks:
                _TP_GROUP = group
                _TP_RANK = tp_ranks.index(rank)

        # ---------- DP groups (inter-node, same local_rank) ----------
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        for lr in range(local_world_size):
            dp_ranks = [node_idx * local_world_size + lr for node_idx in range(num_nodes)]
            group = dist.new_group(dp_ranks)
            if lr == local_rank:
                _DP_GROUP = group
                _DP_RANK = dp_ranks.index(rank)

        _TP_SIZE = local_world_size
        _DP_SIZE = num_nodes
    else:
        # Single-node: TP spans the whole world, no DP needed
        _TP_GROUP = None  # None means "use default world group"
        _DP_GROUP = None
        _TP_SIZE = world_size
        _DP_SIZE = 1
        _TP_RANK = rank
        _DP_RANK = 0

    if rank == 0:
        print(f"Initialized process groups: tp_size={_TP_SIZE}, dp_size={_DP_SIZE}, "
              f"world_size={world_size}")


# Module-level globals for process groups
_TP_GROUP: Optional[dist.ProcessGroup] = None
_DP_GROUP: Optional[dist.ProcessGroup] = None
_TP_SIZE: int = 1
_DP_SIZE: int = 1
_TP_RANK: int = 0
_DP_RANK: int = 0


def get_tp_group() -> Optional[dist.ProcessGroup]:
    """Return the tensor-parallel process group (None = world group)."""
    return _TP_GROUP


def get_dp_group() -> Optional[dist.ProcessGroup]:
    """Return the data-parallel process group (None = no DP)."""
    return _DP_GROUP


def get_tp_size() -> int:
    return _TP_SIZE


def get_dp_size() -> int:
    return _DP_SIZE


def get_tp_rank() -> int:
    return _TP_RANK


def get_dp_rank() -> int:
    return _DP_RANK


def shard_attention_layer(layer: torch.nn.Module, world_size: int, rank: int) -> None:
    """Shard attention Q, K, V, and output projection across tensor parallel dimension."""
    import torch.nn as nn
    
    # Store original values
    original_num_heads = layer.num_heads
    original_num_kv_heads = layer.num_kv_heads
    original_head_dim = layer.head_dim
    
    # Shard Q, K, V projections (column-parallel)
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        if hasattr(layer, proj_name):
            old_proj = getattr(layer, proj_name)
            out_features = old_proj.out_features
            in_features = old_proj.in_features
            
            chunk_size = out_features // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features
            
            new_proj = nn.Linear(in_features, end_idx - start_idx, bias=old_proj.bias is not None)
            new_proj.weight.data = old_proj.weight.data[start_idx:end_idx].clone()
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data[start_idx:end_idx].clone()
            
            setattr(layer, proj_name, new_proj)
    
    # Output projection (row-parallel)
    if hasattr(layer, 'o_proj'):
        old_proj = layer.o_proj
        in_features = old_proj.in_features
        out_features = old_proj.out_features
        
        chunk_size = in_features // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features
        
        new_proj = nn.Linear(end_idx - start_idx, out_features, bias=old_proj.bias is not None)
        new_proj.weight.data = old_proj.weight.data[:, start_idx:end_idx].clone()
        
        if old_proj.bias is not None:
            if rank == 0:
                new_proj.bias.data = old_proj.bias.data.clone()
            else:
                new_proj.bias = None
        
        setattr(layer, 'o_proj', new_proj)
    
    # Get actual sharded dimensions
    actual_q_out = layer.q_proj.out_features
    actual_k_out = layer.k_proj.out_features
    actual_v_out = layer.v_proj.out_features
    
    # Update layer attributes for sharded dimensions
    layer.num_heads = original_num_heads // world_size
    layer.head_dim = original_head_dim
    layer.num_kv_heads = original_num_kv_heads // world_size
    layer.num_key_value_groups = layer.num_heads // layer.num_kv_heads


def shard_mlp_layer(mlp: torch.nn.Module, world_size: int, rank: int) -> None:
    """Shard MLP layers across tensor parallel dimension."""
    import torch.nn as nn
    
    # SwiGLUMLP uses: gate_proj, up_proj (column-parallel), down_proj (row-parallel)
    for proj_name in ['gate_proj', 'up_proj']:
        if hasattr(mlp, proj_name):
            old_proj = getattr(mlp, proj_name)
            out_features = old_proj.out_features
            in_features = old_proj.in_features
            
            chunk_size = out_features // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features
            
            new_proj = nn.Linear(in_features, end_idx - start_idx, bias=old_proj.bias is not None)
            new_proj.weight.data = old_proj.weight.data[start_idx:end_idx].clone()
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data[start_idx:end_idx].clone()
            
            setattr(mlp, proj_name, new_proj)
    
    # down_proj: row-parallel (split input)
    if hasattr(mlp, 'down_proj'):
        old_proj = mlp.down_proj
        in_features = old_proj.in_features
        out_features = old_proj.out_features
        
        chunk_size = in_features // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features
        
        new_proj = nn.Linear(end_idx - start_idx, out_features, bias=old_proj.bias is not None)
        new_proj.weight.data = old_proj.weight.data[:, start_idx:end_idx].clone()
        
        if old_proj.bias is not None:
            if rank == 0:
                new_proj.bias.data = old_proj.bias.data.clone()
            else:
                new_proj.bias = None
        
        setattr(mlp, 'down_proj', new_proj)


def shard_tensor_parallel_correctly(model: ArgonneModel, world_size: int, rank: int) -> None:
    """Properly shard the model for tensor parallelism."""
    if rank == 0:
        print(f"Sharding model for tensor parallelism (world_size={world_size}, rank={rank})")
    
    # Iterate through blocks and shard their components
    for block_idx, block in enumerate(model.blocks):
        # Shard attention
        if hasattr(block, 'attn'):
            shard_attention_layer(block.attn, world_size, rank)
        
        # Shard MLP
        if hasattr(block, 'mlp'):
            shard_mlp_layer(block.mlp, world_size, rank)
    
    if rank == 0:
        # Print summary once (values are identical across all blocks)
        sample_attn = model.blocks[0].attn
        print(
            f"  Per-rank: num_heads={sample_attn.num_heads}, "
            f"num_kv_heads={sample_attn.num_kv_heads}, "
            f"head_dim={sample_attn.head_dim}, "
            f"Q={sample_attn.q_proj.out_features}, "
            f"K={sample_attn.k_proj.out_features}, "
            f"V={sample_attn.v_proj.out_features}"
        )
        print(f"✓ Successfully sharded {len(model.blocks)} transformer blocks")


class TensorParallelModel(torch.nn.Module):
    """Wrapper for ArgonneModel that implements tensor parallelism.

    When running with hybrid TP+DP, ``world_size`` and ``rank`` refer to the
    *tensor-parallel* group (i.e. intra-node), and ``tp_group`` is the NCCL
    sub-group used for all-reduce.  ``local_rank`` is used to select the CUDA
    device.
    """
    def __init__(self, base_model: ArgonneModel, world_size: int, rank: int,
                 tp_group: Optional[dist.ProcessGroup] = None,
                 local_rank: Optional[int] = None):
        super().__init__()
        self.base_model = base_model
        self.world_size = world_size
        self.rank = rank
        self.tp_group = tp_group
        _local_rank = local_rank if local_rank is not None else rank
        self.device = torch.device(f"cuda:{_local_rank}")
        self.gradient_checkpointing = False
        
        # Move model to device first
        self.base_model = self.base_model.to(self.device)
        
        # Then shard it
        shard_tensor_parallel_correctly(self.base_model, world_size, rank)
        
        if rank == 0:
            print(f"✓ Model ready for tensor parallel training on {world_size} GPUs")
    
    def _block_forward(self, block, hidden_states, position_embeddings, attention_mask=None):
        """Forward pass for a single block with tensor parallelism."""
        # Attention with residual
        residual = hidden_states
        normed = block.input_norm(hidden_states)
        attn_output = block.attn(normed, position_embeddings, attention_mask)

        attn_output = attn_output.contiguous()

        attn_reduce: Optional[dist.Work] = None
        if self.world_size > 1:
            attn_reduce = dist.all_reduce(attn_output, op=dist.ReduceOp.SUM, async_op=True, group=self.tp_group)

        if attn_reduce is not None:
            attn_reduce.wait()

        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        normed = block.post_norm(hidden_states)
        mlp_output = block.mlp(normed)

        mlp_output = mlp_output.contiguous()

        mlp_reduce: Optional[dist.Work] = None
        if self.world_size > 1:
            mlp_reduce = dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM, async_op=True, group=self.tp_group)

        if mlp_reduce is not None:
            mlp_reduce.wait()

        hidden_states = residual + mlp_output

        return hidden_states
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass with correct tensor parallelism."""
        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Embeddings (replicated on all ranks)
        b, t = input_ids.size()
        hidden_states = self.base_model.embed_tokens(input_ids)
        
        # Get rotary embeddings
        cos, sin = self.base_model.rotary_emb(hidden_states, t)
        position_embeddings = (cos, sin)
        
        # Process through blocks with simple per-block gradient checkpointing
        if self.gradient_checkpointing and self.training:
            for block in self.base_model.blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._block_forward, 
                    block, 
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    use_reentrant=False
                )
        else:
            # No checkpointing
            for block in self.base_model.blocks:
                hidden_states = self._block_forward(block, hidden_states, position_embeddings, attention_mask)
        
        # Final layer norm and output head (replicated)
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
        """Distributed text generation that respects tensor parallel sharding."""
        was_training = self.training
        self.eval()

        # Determine the global rank that is TP-rank 0 in our group for broadcasts
        if self.tp_group is not None:
            tp_global_ranks = dist.get_process_group_ranks(self.tp_group)
            tp_src_global = tp_global_ranks[0]
        else:
            tp_src_global = 0

        try:
            if not torch.is_tensor(input_ids):
                raise TypeError("input_ids must be a torch.Tensor")

            input_ids = input_ids.to(self.device)

            # Make sure all TP ranks start with the same prompt
            if self.world_size > 1 and dist.is_initialized():
                prompt_length = torch.tensor([input_ids.shape[1]], device=self.device, dtype=torch.long)
                dist.broadcast(prompt_length, src=tp_src_global, group=self.tp_group)

                if input_ids.shape[1] != int(prompt_length.item()):
                    new_prompt = torch.zeros(input_ids.shape[0], prompt_length.item(), dtype=input_ids.dtype, device=self.device)
                    if self.rank == 0:
                        new_prompt.copy_(input_ids[:, :prompt_length.item()])
                    dist.broadcast(new_prompt, src=tp_src_global, group=self.tp_group)
                    input_ids = new_prompt
                else:
                    dist.broadcast(input_ids, src=tp_src_global, group=self.tp_group)

            generated = input_ids

            use_autocast = self.device.type == "cuda"
            amp_dtype = None
            if use_autocast:
                weight = self.base_model.embed_tokens.weight
                if weight.is_floating_point():
                    amp_dtype = weight.dtype

            with torch.no_grad():
                while generated.shape[1] < max_length:
                    context_window = generated[:, -self.base_model.config.max_position_embeddings :]
                    autocast_context = (
                        torch.amp.autocast("cuda", dtype=amp_dtype)
                        if use_autocast and amp_dtype is not None
                        else contextlib.nullcontext()
                    )

                    with autocast_context:
                        outputs = self.forward(context_window)
                        logits = outputs.logits[:, -1, :]

                    logits = logits / temperature

                    next_token: torch.Tensor
                    if do_sample:
                        if top_k is not None:
                            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
                        if top_p is not None:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits = logits.masked_fill(indices_to_remove, float("-inf"))

                        if self.rank == 0:
                            probs = F.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.empty((generated.size(0), 1), dtype=torch.long, device=self.device)
                    else:
                        if self.rank == 0:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        else:
                            next_token = torch.empty((generated.size(0), 1), dtype=torch.long, device=self.device)

                    if self.world_size > 1 and dist.is_initialized():
                        dist.broadcast(next_token, src=tp_src_global, group=self.tp_group)

                    generated = torch.cat([generated, next_token], dim=-1)

                    if generated.shape[1] >= max_length:
                        break

            return generated
        finally:
            if was_training:
                self.train()
    
    def state_dict(self, *args, **kwargs):
        """Get state dict from base model"""
        return self.base_model.state_dict(*args, **kwargs)
    
    def parameters(self):
        """Get parameters from base model"""
        return self.base_model.parameters()
    
    def gradient_checkpointing_enable(self):
        """Enable per-block gradient checkpointing."""
        self.gradient_checkpointing = True
        if self.rank == 0:
            print(f"✓ Gradient checkpointing enabled (per-block)")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False


def _execute_training_attempt(
    *,
    config: ArgonneConfig,
    data_files: List[str],
    tokenizer,
    block_size: int,
    batch_size: int,
    grad_accum_steps: int,
    total_training_steps: int,
    world_size: int,
    rank: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    weight_decay: float,
    amp_dtype: torch.dtype,
    supports_bf16: bool,
    use_gradient_checkpointing: bool,
    is_main_process: bool,
    checkpoint_interval: int = 100,
) -> Tuple[int, int]:
    """Run a single end-to-end training attempt.

    If any rank hits CUDA OOM during the **first forward/backward pass**
    (i.e. before the first successful optimizer step), we broadcast the
    error to all ranks via ``dist.all_reduce`` so that every rank exits
    cleanly and the outer loop can retry with a smaller batch size.

    After at least one successful step, OOM is re-raised as a hard error
    because the optimizer / NCCL state may already be inconsistent.
    """

    tp_group = get_tp_group()
    dp_group = get_dp_group()
    tp_size = get_tp_size()
    dp_size = get_dp_size()
    tp_rank = get_tp_rank()
    dp_rank = get_dp_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Auto-adjust grad accumulation for multi-node DP so effective batch stays constant
    grad_accum_steps_original = grad_accum_steps
    grad_accum_steps = max(1, grad_accum_steps // dp_size)
    if is_main_process:
        print(f"  DP auto-adjusted grad_accum: {grad_accum_steps_original} -> {grad_accum_steps} (dp_size={dp_size})")

    base_model = ArgonneModel(config)
    model = TensorParallelModel(base_model, tp_size, tp_rank,
                                 tp_group=tp_group, local_rank=local_rank)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=True,
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=lr,
        warmup_steps=warmup_steps,
        max_steps=total_training_steps,
        min_lr=min_lr,
    )
    scheduler.step(0)

    use_grad_scaler = amp_dtype == torch.float16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    if is_main_process:
        effective_tokens_per_step = batch_size * block_size * grad_accum_steps * dp_size
        print(f"✓ Model initialized with tensor parallelism")
        print(f"  - Learning rate: {lr:.2e}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Grad accumulation: {grad_accum_steps}")
        print(f"  - TP size: {tp_size} (intra-node)")
        print(f"  - DP size: {dp_size} (inter-node, {dp_size} node(s))")
        print(f"  - World size: {world_size}")
        print(f"  - Effective tokens/step: {effective_tokens_per_step:,} (across all {dp_size} DP rank(s))")
        if supports_bf16:
            print("✓ Using torch.bfloat16 autocast")
        elif amp_dtype == torch.float16:
            print("✓ Using torch.float16 autocast with GradScaler")

    first_device = model.device
    using_cuda = torch.cuda.is_available()
    prefetch_stream = torch.cuda.Stream(device=first_device) if using_cuda else None
    prefetched_batches: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque()
    prefetch_warmup_done = not using_cuda
    max_prefetch_batches = 6 if using_cuda else 2

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

    data_position = DataPosition(streaming=True)

    # Shard data files across DP ranks so each node processes different data
    if dp_size > 1:
        dp_data_files = data_files[dp_rank::dp_size]
        if is_main_process:
            print(f"  DP data sharding: node {dp_rank} gets {len(dp_data_files)}/{len(data_files)} files")
    else:
        dp_data_files = data_files

    token_gen = streaming_token_generator(
        dp_data_files,
        tokenizer,
        block_size,
        rank=rank,
    )

    token_buffer: List[List[int]] = []
    active_shard: Optional[str] = None

    prompt_seed_text = "Long long time ago, "
    prompt_token_ids = tokenizer.encode(prompt_seed_text)
    prompt_tensor_device = first_device if using_cuda else model.device
    cached_prompt_tensor = torch.tensor(
        prompt_token_ids, dtype=torch.long, device=prompt_tensor_device
    ).unsqueeze(0)

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    global_step = 0
    tokens_processed = 0
    last_loss_value: Optional[float] = None
    current_lr = lr

    if is_main_process:
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Target steps: {total_training_steps}")
        print(f"{'='*70}\n")

    pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training") if is_main_process else None

    # Flag tensor used to broadcast OOM status across ranks.
    # 0 = OK, 1 = at least one rank hit OOM.
    oom_flag = torch.zeros(1, dtype=torch.int32, device=first_device)
    completed_one_step = False  # tracks if at least one optimizer step succeeded

    # ---- Probing step: run one dummy forward+backward to detect OOM early ----
    # This ensures all ranks fail together before entering the main loop.
    # First, do a cheap memory headroom check to avoid a mid-NCCL-op OOM
    # (which would deadlock the other ranks).
    if is_main_process:
        print("Running memory probe (1 forward + backward on dummy batch)...")

    probe_oom = False

    # Conservative memory check: estimate if batch can fit before touching NCCL
    if using_cuda:
        torch.cuda.synchronize(first_device)
        free_mem = torch.cuda.mem_get_info(first_device)[0]
        # Rough estimate: each sample needs ~(block_size * hidden_size * num_layers * 10) bytes
        # for activations + gradients with gradient checkpointing and bf16
        bytes_per_sample = block_size * config.hidden_size * config.num_hidden_layers * 10
        estimated_need = batch_size * bytes_per_sample
        headroom_ratio = free_mem / max(estimated_need, 1)
        if is_main_process:
            print(f"  Free GPU memory: {free_mem / 1e9:.2f} GB, estimated need: {estimated_need / 1e9:.2f} GB (ratio: {headroom_ratio:.2f})")
        if headroom_ratio < 0.5:
            # Almost certainly won't fit — skip the probe to avoid NCCL deadlock
            probe_oom = True
            if is_main_process:
                print(f"  ✗ Insufficient headroom (ratio {headroom_ratio:.2f} < 0.5) — skipping probe")

    if not probe_oom:
        try:
            dummy_x = torch.randint(0, config.vocab_size, (batch_size, block_size - 1), device=first_device)
            dummy_y = torch.randint(0, config.vocab_size, (batch_size, block_size - 1), device=first_device)
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                probe_out = model(input_ids=dummy_x)
                probe_loss = F.cross_entropy(
                    probe_out.logits.view(-1, probe_out.logits.size(-1)),
                    dummy_y.view(-1),
                    ignore_index=-100,
                )
            (probe_loss / grad_accum_steps).backward()
            optimizer.zero_grad(set_to_none=True)
            del dummy_x, dummy_y, probe_out, probe_loss
            torch.cuda.empty_cache()
            if is_main_process:
                print(f"✓ Memory probe passed for batch_size={batch_size}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as probe_err:
            err_msg = str(probe_err)
            if isinstance(probe_err, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg.lower():
                probe_oom = True
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            else:
                raise

    # All ranks agree on probe result
    oom_flag.fill_(1 if probe_oom else 0)
    dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
    if oom_flag.item() > 0:
        if is_main_process:
            print(f"✗ Memory probe FAILED for batch_size={batch_size} — triggering OOM retry")
        raise torch.cuda.OutOfMemoryError(
            f"Memory probe failed for batch_size={batch_size}"
        )
    # ---- end probing step ----------------------------------------------------

    try:
        while global_step < total_training_steps:
            try:
                tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)

                if file_idx == -1:
                    if is_main_process:
                        print("End of dataset - restarting")
                    data_position.next_epoch()
                    token_gen = streaming_token_generator(
                        dp_data_files,
                        tokenizer,
                        block_size,
                        rank=rank,
                    )
                    continue

                token_buffer.append(tokens)
                data_position.update_streaming_position(
                    file_idx, position, chunk_idx,
                    dp_data_files[file_idx] if file_idx < len(dp_data_files) else ""
                )

                if shard_name != active_shard:
                    active_shard = shard_name
                    if is_main_process:
                        print(f"Processing shard {file_idx + 1}/{len(dp_data_files)}: {shard_name}")

                if len(token_buffer) < batch_size:
                    continue

                x_tens, y_tens = collate_batch(token_buffer, block_size)
                token_buffer.clear()
                if x_tens is None:
                    continue

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

                autocast_context = (
                    torch.amp.autocast("cuda", dtype=amp_dtype)
                    if torch.cuda.is_available()
                    else contextlib.nullcontext()
                )

                # ---- OOM-safe forward / backward --------------------------------
                # Each rank tries the forward+backward locally.  If it OOMs, it
                # sets a local flag.  Then ALL ranks do an all_reduce(MAX) on the
                # flag so that every rank learns whether *any* rank OOMed.
                local_oom = False
                try:
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
                except (torch.cuda.OutOfMemoryError, RuntimeError) as fwd_err:
                    err_msg = str(fwd_err)
                    if isinstance(fwd_err, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg.lower():
                        local_oom = True
                        if is_main_process:
                            print(f"⚠ Rank {rank} hit CUDA OOM during forward/backward")
                        # Free as much as possible so the all_reduce below succeeds
                        del x_local, y_local
                        torch.cuda.empty_cache()
                    else:
                        raise

                # Collective OOM check — every rank participates
                oom_flag.fill_(1 if local_oom else 0)
                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)

                if oom_flag.item() > 0:
                    # At least one rank OOMed
                    if not completed_one_step:
                        # No step succeeded yet → safe to exit for batch-size retry
                        if is_main_process:
                            print("OOM detected (collective) before first successful step — aborting for retry")
                        raise torch.cuda.OutOfMemoryError(
                            "Collective OOM detected during batch-size search"
                        )
                    else:
                        # Already made progress — can't safely retry
                        raise RuntimeError(
                            "CUDA OOM after training already started. "
                            "Reduce batch size or gradient accumulation and restart."
                        )
                # ---- end OOM-safe block ------------------------------------------

                tokens_processed += batch_tokens * dp_size

                micro_step += 1

                if micro_step >= grad_accum_steps:
                    # ---- DP gradient all-reduce across nodes ----
                    if dp_group is not None and dp_size > 1:
                        for param in model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)

                    current_lr = scheduler.step(global_step)

                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        _ensure_gradient_dtype_matches_params(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        _ensure_gradient_dtype_matches_params(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    micro_step = 0
                    completed_one_step = True

                    if pbar is not None:
                        pbar.update(1)

                    if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                        print(
                            f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {tokens_processed:,} ({dp_size} node(s)) | LR: {current_lr:.6e}"
                        )

                    if global_step % checkpoint_interval == 0:
                        prompt_tensor = cached_prompt_tensor.to(first_device)
                        generated = model.generate(
                            prompt_tensor,
                            max_length=prompt_tensor.shape[1] + 100,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.9,
                        )

                        if is_main_process:
                            generated_text = tokenizer.decode(generated[0].tolist())
                            print(
                                f"\n--- Generated text at step {global_step} ---\n{generated_text}\n"
                            )

                        # Save checkpoint from DP rank 0 node only
                        # (all TP ranks on that node write their shard)
                        if dp_rank == 0:
                            import gc as _gc
                            os.makedirs("pretrained", exist_ok=True)

                            # Each TP rank saves its own model shard
                            shard_path = f"pretrained/.shard_rank{tp_rank}_step{global_step}.pt"
                            local_state = cast_state_dict_to_dtype(
                                _state_dict_to_cpu(model.base_model.state_dict()), amp_dtype
                            )
                            safe_torch_save(local_state, shard_path)
                            del local_state

                            # Each TP rank saves its own optimizer shard
                            opt_shard_path = f"pretrained/.opt_rank{tp_rank}_step{global_step}.pt"
                            safe_torch_save(_state_dict_to_cpu(optimizer.state_dict()), opt_shard_path)

                            _gc.collect()

                            # Wait for all TP ranks on this node
                            if tp_group is not None:
                                dist.barrier(group=tp_group)
                            else:
                                dist.barrier()

                            if is_main_process:
                                shard_list = []
                                opt_shards = []
                                for r in range(tp_size):
                                    rpath = f"pretrained/.shard_rank{r}_step{global_step}.pt"
                                    shard_list.append(safe_torch_load(rpath, map_location="cpu", weights_only=True))
                                    opath = f"pretrained/.opt_rank{r}_step{global_step}.pt"
                                    opt_shards.append(safe_torch_load(opath, map_location="cpu", weights_only=True))

                                checkpoint_state = {
                                    "global_step": global_step,
                                    "tokens_processed": tokens_processed,
                                    "model_state_dict": {},
                                    "tensor_parallel_shards": shard_list,
                                    "optimizer_shards": opt_shards,
                                    # Keep for backward compat (rank 0 only)
                                    "optimizer_state_dict": opt_shards[0] if opt_shards else {},
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "lr_ramp_state": None,
                                    "gradient_accumulation_steps": grad_accum_steps,
                                    "loss": last_loss_value,
                                    "data_position": data_position.get_state(),
                                    "model_dtype": str(amp_dtype),
                                    "tensor_parallel": True,
                                    "world_size": tp_size,
                                    "dp_world_size": dp_size,
                                    "rank": rank,
                                    "batch_size": batch_size,
                                }
                                save_path = (
                                    f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                                )
                                safe_torch_save(checkpoint_state, save_path)
                                print(f"Checkpoint saved @ step {global_step} -> {save_path}")
                                del checkpoint_state, shard_list, opt_shards

                                # Clean up temp shard files
                                for r in range(tp_size):
                                    rpath = f"pretrained/.shard_rank{r}_step{global_step}.pt"
                                    opath = f"pretrained/.opt_rank{r}_step{global_step}.pt"
                                    try:
                                        os.remove(rpath)
                                    except OSError:
                                        pass
                                    try:
                                        os.remove(opath)
                                    except OSError:
                                        pass

                            if tp_group is not None:
                                dist.barrier(group=tp_group)
                            else:
                                dist.barrier()
                            _gc.collect()
                            torch.cuda.empty_cache()

                        # All-world barrier to sync all DP ranks
                        dist.barrier()

            except StopIteration:
                if is_main_process:
                    print("StopIteration - restarting dataset")
                data_position.next_epoch()
                token_gen = streaming_token_generator(
                    dp_data_files,
                    tokenizer,
                    block_size,
                    rank=rank,
                )
                continue
    finally:
        if using_cuda and prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        prefetched_batches.clear()
        if pbar is not None:
            pbar.close()

    if is_main_process:
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total tokens: {tokens_processed:,} (across {dp_size} node(s))")
        print(f"Final step: {global_step}")

    if dist.is_initialized():
        dist.barrier()

    return global_step, tokens_processed


def train_from_scratch_tensor_parallel(
    data_glob: str,
    tokenizer_path: str,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 4096,
    initial_batch_size: int = 4,
    min_batch_size: int = 1,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    trust_remote_code: bool = False,
    gradient_accumulation_steps: int = 4,
    disable_gradient_checkpointing: bool = False,
    checkpoint_interval: int = 100,
):
    """Train model from scratch with tensor parallelism (intra-node) and
    data parallelism (inter-node) with automatic batch size tuning."""

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)
    init_tensor_parallel_group(world_size, rank)
    is_main_process = (rank == 0)

    tp_size = get_tp_size()
    dp_size = get_dp_size()

    if is_main_process:
        print("=" * 70)
        print("STARTING FRESH TRAINING FROM SCRATCH (TP + DP)")
        print("=" * 70)
        print(f"World size: {world_size} GPUs")
        print(f"TP size: {tp_size} (intra-node)")
        print(f"DP size: {dp_size} (inter-node)")
        print(f"Rank: {rank}")

    cleanup_old_checkpoints("pretrained", keep=50, rank=rank)

    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    fallback_patterns = [
        os.path.join("data", "CC-MAIN-2025-26", "*.parquet"),
        os.path.join("..", "data", "*.arrow"),
        os.path.join("data", "*.arrow"),
    ]
    if data_glob != default_data_glob:
        fallback_patterns.insert(0, default_data_glob)
    data_files, _ = resolve_data_files(
        data_glob, fallback_patterns=fallback_patterns
    )

    if is_main_process:
        print(f"Found {len(data_files)} data files")
        log_dataset_plan(data_files)

    validate_tokenizer_path(tokenizer_path)
    hf_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})
    hf_tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)
    vocab_size = len(hf_tokenizer)

    config = ArgonneConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        max_position_embeddings=block_size,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_flash_attention=True,
        use_gradient_checkpointing=not disable_gradient_checkpointing,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        eos_token_id=hf_tokenizer.eos_token_id,
    )

    supports_bf16 = False
    amp_dtype = torch.float32
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
        supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    grad_accum_steps = max(1, int(gradient_accumulation_steps))

    batch_size = initial_batch_size
    largest_successful_batch = None
    smallest_failed_batch = None

    while True:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"ATTEMPTING TRAINING WITH BATCH_SIZE = {batch_size}")
            print(f"{'='*70}")

        try:
            global_step, tokens_processed = _execute_training_attempt(
                config=config,
                data_files=data_files,
                tokenizer=hf_tokenizer,
                block_size=block_size,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                total_training_steps=total_training_steps,
                world_size=world_size,
                rank=rank,
                lr=lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                amp_dtype=amp_dtype,
                supports_bf16=supports_bf16,
                use_gradient_checkpointing=not disable_gradient_checkpointing,
                is_main_process=is_main_process,
                checkpoint_interval=checkpoint_interval,
            )

            if largest_successful_batch is None or batch_size > largest_successful_batch:
                largest_successful_batch = batch_size

            if is_main_process:
                print(f"Optimal batch size: {batch_size}")
                print("Checkpoints saved to: pretrained/")
                print(
                    "Resume with: torchrun --nproc_per_node=%d resume_pretrain_tensor.py --tokenizer-path %s"
                    % (world_size, tokenizer_path)
                )

            break

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_message = str(e)
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_message.lower()

            if not is_oom:
                raise

            if is_main_process:
                print(f"\n{'='*70}")
                print("CUDA OUT OF MEMORY DETECTED (collective)")
                print(f"{'='*70}")
                print(f"Error: {error_message}")

            # Aggressive cleanup — the model / optimizer from the failed
            # attempt are already out of scope, but fragments may linger.
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Barrier is safe here because _execute_training_attempt uses
            # all_reduce to ensure ALL ranks exit together on OOM.
            if dist.is_initialized():
                dist.barrier()

            if smallest_failed_batch is None or batch_size < smallest_failed_batch:
                smallest_failed_batch = batch_size

            if largest_successful_batch is None:
                new_batch_size = batch_size // 2
            else:
                new_batch_size = (largest_successful_batch + smallest_failed_batch) // 2

            new_batch_size = max(new_batch_size, min_batch_size)

            if new_batch_size == batch_size:
                if batch_size <= min_batch_size:
                    if is_main_process:
                        print(f"\n{'='*70}")
                        print(f"FATAL: Already at minimum batch size ({min_batch_size})")
                        print("Cannot reduce further. Training failed.")
                        print(f"{'='*70}")
                    raise RuntimeError(
                        f"Training failed even with minimum batch size {min_batch_size}"
                    )
                new_batch_size = max(batch_size - 1, min_batch_size)

            if is_main_process:
                print(f"\n{'='*70}")
                print("REDUCING BATCH SIZE")
                print(f"{'='*70}")
                print(f"Previous: {batch_size}")
                print(f"New: {new_batch_size}")
                print(
                    f"Search bounds: success={largest_successful_batch}, failed={smallest_failed_batch}"
                )
                print("Retrying in 5 seconds...")
                print(f"{'='*70}\n")

            batch_size = new_batch_size
            time.sleep(5)
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Train Argonne model from scratch with tensor parallelism")
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
        help="Directory containing the pretrained tokenizer.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_MAX_TRAINING_STEPS,
        help="Total number of training steps.",
    )
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        default=4,
        help="Initial batch size to try (will be reduced on OOM).",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=2,
        help="Minimum batch size before giving up.",
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
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Base number of micro-batches to accumulate before each optimizer step. Auto-adjusted by DP size.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing to speed up training (requires more GPU memory).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save a checkpoint and generate sample text every N optimizer steps.",
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
    train_from_scratch_tensor_parallel(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        total_training_steps=args.total_steps,
        block_size=args.block_size,
        initial_batch_size=args.initial_batch_size,
        min_batch_size=args.min_batch_size,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        trust_remote_code=args.trust_remote_code,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        disable_gradient_checkpointing=args.disable_gradient_checkpointing,
        checkpoint_interval=args.checkpoint_interval,
    )



if __name__ == "__main__":
    main()
 