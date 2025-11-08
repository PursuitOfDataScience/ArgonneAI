import argparse
import contextlib
import os
import re
import socket
import time
import traceback
import types
from dataclasses import dataclass
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Iterable, List, Optional, Tuple

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
    chunk_tokens,
)
from model import ArgonneConfig, ArgonneModel
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    cast_state_dict_to_dtype,
    determine_document_boundary_tokens,
    load_streaming_shard,
    log_dataset_plan,
    resolve_data_files,
    safe_torch_save,
    validate_tokenizer_path,
)

DEFAULT_DATA_GLOB = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
DEFAULT_FALLBACK_PATTERNS = [
    os.path.join("data", "CC-MAIN-2025-26", "*.parquet"),
    os.path.join("..", "data", "*.arrow"),
    os.path.join("data", "*.arrow"),
]

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


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
def _allreduce_non_compiled(
    tensor: torch.Tensor,
    *,
    async_op: bool = False,
    group: Optional[dist.ProcessGroup] = None,
):
    """Run all-reduce outside any compiled graph, optionally asynchronously."""

    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op, group=group)
    if async_op:
        return work
    return tensor


@torch._dynamo.disable
def _wait_for_work(work) -> None:
    """Wait on a distributed work handle outside compiled regions."""

    if work is not None:
        work.wait()


def _maybe_enable_compilation(
    model: torch.nn.Module,
    *,
    enable_compile: bool,
    is_main_process: bool,
) -> bool:
    """Optionally wrap ``model.forward`` with ``torch.compile`` for speedups."""

    if not enable_compile:
        if is_main_process:
            print("✓ torch.compile disabled (flag provided)")
        return False

    if not hasattr(torch, "compile"):
        if is_main_process:
            print("⚠ torch.compile requested but not available; running in eager mode")
        return False

    try:
        compiled_forward = torch.compile(model.forward.__func__)
        model.forward = types.MethodType(compiled_forward, model)
        if is_main_process:
            print("✓ torch.compile enabled for model forward pass")
        return True
    except Exception as exc:
        if is_main_process:
            print(f"⚠ torch.compile failed ({exc}); falling back to eager mode")
        return False


def resolve_training_data(data_glob: str) -> Tuple[List[str], List[str]]:
    """Resolve dataset files using the shared fallback patterns."""

    fallback_patterns = list(DEFAULT_FALLBACK_PATTERNS)
    if data_glob != DEFAULT_DATA_GLOB:
        fallback_patterns.insert(0, DEFAULT_DATA_GLOB)
    return resolve_data_files(data_glob, fallback_patterns=fallback_patterns)


def load_tokenizer_and_build_config(
    tokenizer_path: str,
    *,
    block_size: int,
    trust_remote_code: bool,
    use_gradient_checkpointing: bool,
):
    """Load tokenizer and construct the shared Argonne configuration."""

    validate_tokenizer_path(tokenizer_path)
    hf_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)

    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})

    hf_tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)

    config = ArgonneConfig(
        vocab_size=len(hf_tokenizer),
        hidden_size=3072,
        max_position_embeddings=block_size,
        num_hidden_layers=20,
        num_attention_heads=24,
        num_key_value_heads=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_flash_attention=True,
        use_gradient_checkpointing=use_gradient_checkpointing,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        eos_token_id=hf_tokenizer.eos_token_id,
    )

    return hf_tokenizer, config


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
    add_document_tokens: bool = False,
):
    """Generator with chunk-level resume support and optional BOS/EOS injection."""

    import gc

    file_idx = max(start_file_idx, 0)
    processed_count = 0
    is_main_process = (rank == 0)

    (
        bos_token_id,
        eos_token_id,
        bos_token_str,
        eos_token_str,
    ) = determine_document_boundary_tokens(tokenizer)

    def _ensure_token_id(
        token_id: Optional[int],
        token_str: Optional[str],
        default_candidates: Sequence[Optional[str]],
    ) -> Tuple[Optional[int], Optional[str]]:
        """Best-effort resolution of chat-style special tokens."""

        if token_id is not None:
            return token_id, token_str

        candidates: List[str] = []
        if token_str:
            candidates.append(token_str)
        for candidate in default_candidates:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            try:
                candidate_id = tokenizer.convert_tokens_to_ids(candidate)
            except Exception:
                candidate_id = None

            if isinstance(candidate_id, int) and candidate_id >= 0:
                return candidate_id, candidate

            try:
                encoded = tokenizer.encode(candidate, add_special_tokens=False)
            except Exception:
                encoded = None

            if encoded and len(encoded) == 1:
                return encoded[0], candidate

        return None, token_str

    fallback_bos_tokens: List[Optional[str]] = [
        getattr(tokenizer, "bos_token", None),
        "<|im_start|>",
        "<s>",
        "[CLS]",
    ]
    fallback_eos_tokens: List[Optional[str]] = [
        getattr(tokenizer, "eos_token", None),
        "<|im_end|>",
        "</s>",
        "[SEP]",
    ]

    additional_tokens = getattr(tokenizer, "additional_special_tokens", None) or []
    for token in additional_tokens:
        if token not in fallback_bos_tokens:
            fallback_bos_tokens.append(token)
        if token not in fallback_eos_tokens:
            fallback_eos_tokens.append(token)

    bos_token_id, bos_token_str = _ensure_token_id(
        bos_token_id,
        bos_token_str,
        fallback_bos_tokens,
    )
    eos_token_id, eos_token_str = _ensure_token_id(
        eos_token_id,
        eos_token_str,
        fallback_eos_tokens,
    )
    document_tokens_enabled = bool(add_document_tokens)

    if document_tokens_enabled:
        missing_tokens = []
        if bos_token_id is None:
            missing_tokens.append("BOS")
        if eos_token_id is None:
            missing_tokens.append("EOS")

        if missing_tokens:
            if is_main_process:
                print(
                    "⚠ Unable to add document boundary tokens: missing "
                    + ", ".join(missing_tokens)
                    + " token id(s) in tokenizer"
                )
            document_tokens_enabled = False
        elif is_main_process:
            bos_display = bos_token_str or bos_token_id
            eos_display = eos_token_str or eos_token_id
            print(
                "✓ Adding BOS/EOS tokens to each document "
                f"(bos={bos_display}, eos={eos_display})"
            )

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

    pending_batches: Deque[Tuple[Future, List[Tuple[int, str]], List[bool]]] = deque()

    def _clear_pending() -> None:
        while pending_batches:
            future, _, _ = pending_batches.popleft()
            future.cancel()

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

                if file_idx == initial_file_idx:
                    position = initial_position
                    resume_chunk_offset = initial_chunk_offset
                    if is_main_process and (position > 0 or resume_chunk_offset > 0):
                        print(
                            f"  >>> RESUMING from position {position}, chunk offset {resume_chunk_offset}"
                        )
                else:
                    position = 0
                    resume_chunk_offset = 0

                _clear_pending()

                while position < len(dataset) or pending_batches:
                    while position < len(dataset) and len(pending_batches) < max_pending_batches:
                        batch_records: List[Tuple[int, str]] = []
                        batch_resume_flags: List[bool] = []

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

                            resume_mid_document = (
                                document_tokens_enabled
                                and file_idx == initial_file_idx
                                and position == initial_position
                                and resume_chunk_offset > 0
                            )

                            batch_records.append((position, text))
                            batch_resume_flags.append(resume_mid_document)
                            position += 1

                        if not batch_records:
                            continue

                        batch_texts = [text for _, text in batch_records]
                        future = tokenizer_pool.submit(_tokenize_texts, batch_texts)
                        pending_batches.append((future, batch_records, batch_resume_flags))

                    if not pending_batches:
                        break

                    future, batch_records, batch_resume_flags = pending_batches[0]
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

                    for (record_position, text), resume_mid_document, tokens in zip(
                        batch_records,
                        batch_resume_flags,
                        token_batches,
                    ):
                        if not tokens:
                            continue

                        if document_tokens_enabled:
                            tokens = [bos_token_id, *tokens, eos_token_id]

                            if resume_mid_document:
                                tokens = tokens[1:]

                        if len(tokens) < 10:
                            continue

                        for chunk_idx, chunk in enumerate(chunk_tokens(tokens, block_size)):
                            if file_idx == initial_file_idx and record_position == initial_position:
                                if chunk_idx < resume_chunk_offset:
                                    continue

                            processed_count += 1
                            yield chunk, file_idx, record_position, shard_name, chunk_idx

                _clear_pending()
                file_idx += 1

            finally:
                if dataset is not None:
                    del dataset
                dataset = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if is_main_process:
        print(f"Completed processing all available files. Processed {processed_count} samples.")

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


@dataclass
class DistributedContext:
    """Container describing the distributed topology for training."""

    rank: int
    local_rank: int
    world_size: int
    tensor_parallel_size: int
    tensor_parallel_rank: int
    tensor_parallel_group: dist.ProcessGroup
    data_parallel_size: int
    data_parallel_rank: int
    data_parallel_group: Optional[dist.ProcessGroup]
    num_nodes: int
    node_rank: int
    is_main_process: bool
    hostname: str

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.local_rank}")


def setup_distributed_environment() -> DistributedContext:
    """Initialize global, tensor-parallel, and data-parallel process groups."""

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    inferred_local = os.environ.get("LOCAL_WORLD_SIZE")
    if inferred_local is not None:
        try:
            tensor_parallel_size = max(1, int(inferred_local))
        except ValueError:
            tensor_parallel_size = torch.cuda.device_count()
    else:
        tensor_parallel_size = torch.cuda.device_count()

    if tensor_parallel_size <= 0:
        tensor_parallel_size = 1

    if world_size % tensor_parallel_size != 0:
        raise RuntimeError(
            "Global world size %d is not divisible by tensor parallel size %d"
            % (world_size, tensor_parallel_size)
        )

    num_nodes = world_size // tensor_parallel_size

    tensor_parallel_group: Optional[dist.ProcessGroup] = None
    tensor_parallel_rank = 0
    for node_idx in range(num_nodes):
        ranks = list(
            range(node_idx * tensor_parallel_size, (node_idx + 1) * tensor_parallel_size)
        )
        group = dist.new_group(ranks)
        if rank in ranks:
            tensor_parallel_group = group
            tensor_parallel_rank = ranks.index(rank)

    if tensor_parallel_group is None:
        raise RuntimeError("Failed to initialize tensor parallel process group")

    data_parallel_size = num_nodes
    data_parallel_group: Optional[dist.ProcessGroup] = None
    data_parallel_rank = 0
    if data_parallel_size > 1:
        for local_idx in range(tensor_parallel_size):
            ranks = [local_idx + node_idx * tensor_parallel_size for node_idx in range(num_nodes)]
            group = dist.new_group(ranks)
            if rank in ranks:
                data_parallel_group = group
                data_parallel_rank = ranks.index(rank)
    else:
        data_parallel_group = None
        data_parallel_rank = 0

    hostname = socket.gethostname()
    is_main_process = rank == 0

    gathered_hostnames: List[str] = [""] * world_size
    dist.all_gather_object(gathered_hostnames, hostname)

    if is_main_process:
        unique_hosts = sorted(set(gathered_hostnames))
        print("=" * 70)
        print(
            "Distributed initialization complete: %d GPUs across %d node(s)"
            % (world_size, len(unique_hosts))
        )
        print(
            "  - Tensor parallel size: %d GPUs per node" % tensor_parallel_size
        )
        print(
            "  - Data parallel size: %d node(s)" % max(1, data_parallel_size)
        )
        print("Hosts participating:")
        for idx, host in enumerate(unique_hosts):
            host_gpu_count = sum(1 for name in gathered_hostnames if name == host)
            print(f"    • Node {idx}: {host} ({host_gpu_count} rank(s))")
        print("=" * 70)

    if tensor_parallel_rank == 0:
        print(
            f"[Node {rank // tensor_parallel_size}] Host {hostname} active with {tensor_parallel_size} GPU(s)"
        )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_group=tensor_parallel_group,
        data_parallel_size=max(1, data_parallel_size),
        data_parallel_rank=data_parallel_rank,
        data_parallel_group=data_parallel_group,
        num_nodes=num_nodes,
        node_rank=rank // tensor_parallel_size,
        is_main_process=is_main_process,
        hostname=hostname,
    )


def broadcast_parameters(
    parameters: Iterable[torch.Tensor],
    *,
    group: Optional[dist.ProcessGroup],
    src: int = 0,
) -> None:
    """Broadcast tensors from ``src`` rank to the provided group."""

    if group is None:
        return

    group_world_size = dist.get_world_size(group=group)
    if group_world_size <= 1:
        return

    for tensor in parameters:
        dist.broadcast(tensor, src=src, group=group)


def average_gradients(
    parameters: Iterable[torch.nn.Parameter],
    *,
    group: Optional[dist.ProcessGroup],
    world_size: int,
) -> None:
    """Average gradients across the provided data parallel ``group``."""

    if group is None or world_size <= 1:
        return

    for param in parameters:
        grad = param.grad
        if grad is None:
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
        grad.div_(world_size)


def shard_attention_layer(layer: torch.nn.Module, world_size: int, rank: int) -> None:
    """Shard attention Q, K, V, and output projection across tensor parallel dimension."""
    import torch.nn as nn
    
    # Store original values
    original_num_heads = layer.num_heads
    original_num_kv_heads = layer.num_kv_heads
    original_head_dim = layer.head_dim
    
    if rank == 0:
        print(f"  Original: num_heads={original_num_heads}, num_kv_heads={original_num_kv_heads}, head_dim={original_head_dim}")
    
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
    
    if rank == 0:
        print(f"  Sharded: num_heads={layer.num_heads}, num_kv_heads={layer.num_kv_heads}, head_dim={layer.head_dim}")
        print(f"  Dims: Q={actual_q_out}, K={actual_k_out}, V={actual_v_out}")
        print(f"  Groups: {layer.num_key_value_groups}")


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
        print(f"✓ Successfully sharded {len(model.blocks)} transformer blocks")


class TensorParallelModel(torch.nn.Module):
    """Wrapper for ArgonneModel that implements tensor parallelism."""

    def __init__(
        self,
        base_model: ArgonneModel,
        world_size: int,
        rank: int,
        local_rank: int,
        tensor_parallel_group: dist.ProcessGroup,
    ):
        super().__init__()
        self.base_model = base_model
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.tensor_parallel_group = tensor_parallel_group
        self.gradient_checkpointing = False

        # Move model to device first
        self.base_model = self.base_model.to(self.device)

        # Then shard it
        shard_tensor_parallel_correctly(self.base_model, world_size, rank)

        if rank == 0:
            print(
                f"✓ Model ready for tensor parallel training on {world_size} GPU(s) per node"
            )
    
    def _block_forward(self, block, hidden_states, position_embeddings, attention_mask=None):
        """Forward pass for a single block with tensor parallelism."""
        # Attention with residual
        residual = hidden_states
        normed = block.input_norm(hidden_states)
        attn_output = block.attn(normed, position_embeddings, attention_mask)

        attn_output = attn_output.contiguous()

        attn_reduce: Optional[dist.Work] = None
        if self.world_size > 1:
            attn_reduce = _allreduce_non_compiled(
                attn_output, async_op=True, group=self.tensor_parallel_group
            )

        if attn_reduce is not None:
            _wait_for_work(attn_reduce)

        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        normed = block.post_norm(hidden_states)
        mlp_output = block.mlp(normed)

        mlp_output = mlp_output.contiguous()

        mlp_reduce: Optional[dist.Work] = None
        if self.world_size > 1:
            mlp_reduce = _allreduce_non_compiled(
                mlp_output, async_op=True, group=self.tensor_parallel_group
            )

        if mlp_reduce is not None:
            _wait_for_work(mlp_reduce)

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

        try:
            if not torch.is_tensor(input_ids):
                raise TypeError("input_ids must be a torch.Tensor")

            input_ids = input_ids.to(self.device)

            # Make sure all ranks start with the same prompt
            if self.world_size > 1 and dist.is_initialized():
                prompt_length = torch.tensor(
                    [input_ids.shape[1]], device=self.device, dtype=torch.long
                )
                dist.broadcast(prompt_length, src=0, group=self.tensor_parallel_group)

                if input_ids.shape[1] != int(prompt_length.item()):
                    new_prompt = torch.zeros(input_ids.shape[0], prompt_length.item(), dtype=input_ids.dtype, device=self.device)
                    if self.rank == 0:
                        new_prompt.copy_(input_ids[:, : prompt_length.item()])
                    dist.broadcast(new_prompt, src=0, group=self.tensor_parallel_group)
                    input_ids = new_prompt
                else:
                    dist.broadcast(input_ids, src=0, group=self.tensor_parallel_group)

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
                        dist.broadcast(next_token, src=0, group=self.tensor_parallel_group)

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
    dist_ctx: DistributedContext,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    weight_decay: float,
    amp_dtype: torch.dtype,
    supports_bf16: bool,
    add_document_tokens: bool,
    use_gradient_checkpointing: bool,
    enable_compile: bool,
) -> Tuple[int, int]:
    """Run a single end-to-end training attempt."""

    base_model = ArgonneModel(config)
    model = TensorParallelModel(
        base_model,
        dist_ctx.tensor_parallel_size,
        dist_ctx.tensor_parallel_rank,
        dist_ctx.local_rank,
        dist_ctx.tensor_parallel_group,
    )

    broadcast_parameters(
        model.parameters(), group=dist_ctx.data_parallel_group, src=0
    )
    broadcast_parameters(
        model.base_model.buffers(), group=dist_ctx.data_parallel_group, src=0
    )

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    total_params = sum(p.numel() for p in model.parameters())

    _maybe_enable_compilation(model, enable_compile=enable_compile, is_main_process=is_main_process)

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

    is_main_process = dist_ctx.is_main_process

    if is_main_process:
        print(f"✓ Model initialized with tensor parallelism")
        print(f"  - Parameters: {total_params:,}")
        print(f"  - Learning rate: {lr:.2e}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Grad accumulation: {grad_accum_steps}")
        print(
            "  - Topology: %d tensor x %d data = %d total GPUs"
            % (
                dist_ctx.tensor_parallel_size,
                dist_ctx.data_parallel_size,
                dist_ctx.world_size,
            )
        )
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
    token_gen = streaming_token_generator(
        data_files,
        tokenizer,
        block_size,
        rank=dist_ctx.rank,
        add_document_tokens=add_document_tokens,
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

    try:
        while global_step < total_training_steps:
            try:
                tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)

                if file_idx == -1:
                    if is_main_process:
                        print("End of dataset - restarting")
                    data_position.next_epoch()
                    token_gen = streaming_token_generator(
                        data_files,
                        tokenizer,
                        block_size,
                        rank=dist_ctx.rank,
                        add_document_tokens=add_document_tokens,
                    )
                    continue

                token_buffer.append(tokens)
                data_position.update_streaming_position(
                    file_idx, position, chunk_idx, data_files[file_idx]
                )

                if shard_name != active_shard:
                    active_shard = shard_name
                    if is_main_process:
                        print(f"Processing shard {file_idx + 1}/{len(data_files)}: {shard_name}")

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
                tokens_processed += batch_tokens

                autocast_context = (
                    torch.amp.autocast("cuda", dtype=amp_dtype)
                    if torch.cuda.is_available()
                    else contextlib.nullcontext()
                )

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
                    current_lr = scheduler.step(global_step)

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
                        print(
                            f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {tokens_processed:,} | LR: {current_lr:.6e}"
                        )

                    if global_step % 350 == 0:
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

                            model_state = cast_state_dict_to_dtype(
                                model.base_model.state_dict(), amp_dtype
                            )

                            checkpoint_state = {
                                "global_step": global_step,
                                "tokens_processed": tokens_processed,
                                "model_state_dict": model_state,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "lr_ramp_state": None,
                                "gradient_accumulation_steps": grad_accum_steps,
                                "loss": last_loss_value,
                                "data_position": data_position.get_state(),
                                "model_dtype": str(amp_dtype),
                                "tensor_parallel": True,
                                "tensor_parallel_world_size": dist_ctx.tensor_parallel_size,
                                "data_parallel_world_size": dist_ctx.data_parallel_size,
                                "global_world_size": dist_ctx.world_size,
                                "rank": dist_ctx.rank,
                                "tensor_parallel_rank": dist_ctx.tensor_parallel_rank,
                                "data_parallel_rank": dist_ctx.data_parallel_rank,
                                "batch_size": batch_size,
                            }
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
                token_gen = streaming_token_generator(
                    data_files,
                    tokenizer,
                    block_size,
                    rank=dist_ctx.rank,
                    add_document_tokens=add_document_tokens,
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
        print(f"Total tokens: {tokens_processed:,}")
        print(f"Final step: {global_step}")

    if dist.is_initialized():
        dist.barrier()

    return global_step, tokens_processed


def train_from_scratch_tensor_parallel(
    data_glob: str,
    tokenizer_path: str,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 4096,
    initial_batch_size: int = 512,
    min_batch_size: int = 4,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    trust_remote_code: bool = False,
    gradient_accumulation_steps: int = 4,
    disable_gradient_checkpointing: bool = False,
    add_document_tokens: bool = False,
    enable_compile: bool = True,
):
    """Train model from scratch with tensor parallelism and automatic batch size tuning."""

    dist_ctx = setup_distributed_environment()
    is_main_process = dist_ctx.is_main_process

    if is_main_process:
        print("=" * 70)
        print("STARTING FRESH TRAINING FROM SCRATCH (TENSOR PARALLEL)")
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

    cleanup_old_checkpoints("pretrained", keep=50, rank=dist_ctx.rank)

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

    hf_tokenizer, config = load_tokenizer_and_build_config(
        tokenizer_path,
        block_size=block_size,
        trust_remote_code=trust_remote_code,
        use_gradient_checkpointing=not disable_gradient_checkpointing,
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
                data_files=local_data_files,
                tokenizer=hf_tokenizer,
                block_size=block_size,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                total_training_steps=total_training_steps,
                dist_ctx=dist_ctx,
                lr=lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                amp_dtype=amp_dtype,
                supports_bf16=supports_bf16,
                add_document_tokens=add_document_tokens,
                use_gradient_checkpointing=not disable_gradient_checkpointing,
                enable_compile=enable_compile,
            )

            if largest_successful_batch is None or batch_size > largest_successful_batch:
                largest_successful_batch = batch_size

            if is_main_process:
                print(f"Optimal batch size: {batch_size}")
                print("Checkpoints saved to: pretrained/")
                print(
                    "Resume with the same topology (tensor=%d, data=%d, total GPUs=%d)"
                    % (
                        dist_ctx.tensor_parallel_size,
                        dist_ctx.data_parallel_size,
                        dist_ctx.world_size,
                    )
                )

            break

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_message = str(e)
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_message.lower()

            if not is_oom:
                raise

            if is_main_process:
                print(f"\n{'='*70}")
                print("CUDA OUT OF MEMORY DETECTED")
                print(f"{'='*70}")
                print(f"Error: {error_message}")
                if hasattr(e, "__traceback__"):
                    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                    print("".join(tb_lines))

            torch.cuda.empty_cache()

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
        default=512,
        help="Initial batch size to try (will be reduced on OOM).",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=4,
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
        help="Number of micro-batches to accumulate before each optimizer step.",
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
            "Prepend the tokenizer BOS token and append the EOS token to each document before chunking."
        ),
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile acceleration.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    # Removed --checkpoint-segments argument (always use per-block checkpointing for stability)
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
        add_document_tokens=args.add_document_boundary_tokens,
        enable_compile=not args.disable_compile,
    )



if __name__ == "__main__":
    main()
 
