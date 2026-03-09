"""Resume pretraining with Data Parallelism (DDP).

This script can:
  1. Load a checkpoint from a previous tensor-parallel (TP) run, merge the
     sharded weights into a full model, and continue training with DDP.
  2. Load its own DDP checkpoints and resume seamlessly.

Key differences from resume_pretrain_tensor.py:
  - Uses ``torch.nn.parallel.DistributedDataParallel`` instead of custom
    ``TensorParallelModel``.  Each GPU holds a *full* copy of the model.
  - The streaming data generator is *per-rank*: each rank skips sequences that
    belong to other ranks (round-robin sharding by global step counter) so
    every rank sees disjoint data.
  - Automatic batch-size probing on the first launch (start at 16, reduce by 4
    on OOM, minimum 2).  Once a working batch size is found it is persisted in
    the checkpoint so subsequent runs skip the probing.
"""

import argparse
import contextlib
import gc
import os
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Dict, List, Optional, Tuple

# Disable cudagraphs before importing torch
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    safe_torch_load,
    safe_torch_save,
    resolve_data_files,
    validate_tokenizer_path,
)

# Enable TF32 precision on Ampere/Hopper GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------- Checkpoint filename patterns ------------------------------------

# Matches both TP and DP checkpoint filenames
CHECKPOINT_PATTERN = re.compile(r"_step_(\d+)(?:_rank\d+)?\.pth$")

# DP-specific checkpoint pattern (for dp_pretrained/ directory)
DP_CHECKPOINT_PATTERN = re.compile(r"dp_checkpoint_step_(\d+)\.pth$")

# TP checkpoint pattern (for pretrained/ directory)
TP_CHECKPOINT_PATTERN = re.compile(r"streaming_checkpoint_step_(\d+)\.pth$")

# ---------- TP shard merging helpers ----------------------------------------

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


def _state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cpu_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def merge_tensor_parallel_shards(
    shard_states: Sequence[Mapping[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Reconstruct full weights from tensor-parallel shards."""
    shard_list: List[Mapping[str, torch.Tensor]] = list(shard_states)
    if not shard_list:
        return {}

    all_keys: set = set()
    for shard in shard_list:
        all_keys.update(shard.keys())

    full_state: Dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        values = [shard.get(key) for shard in shard_list]
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        if not tensors:
            continue

        if key.endswith(COLUMN_PARALLEL_WEIGHT_SUFFIXES):
            full_state[key] = torch.cat(tensors, dim=0)
        elif key.endswith(COLUMN_PARALLEL_BIAS_SUFFIXES):
            full_state[key] = torch.cat(tensors, dim=0)
        elif key.endswith(ROW_PARALLEL_WEIGHT_SUFFIXES):
            ordered = [v for v in values if isinstance(v, torch.Tensor)]
            full_state[key] = torch.cat(ordered, dim=1)
        else:
            # Replicated parameters (embeddings, norms, lm_head)
            full_state[key] = tensors[0]

    return full_state


# ---------- Streaming data generator (identical to TP script) ---------------

def streaming_token_generator(
    data_files: List[str],
    tokenizer,
    block_size: int,
    start_file_idx: int = 0,
    start_position: int = 0,
    start_chunk_offset: int = 0,
    rank: int = 0,
):
    """Sequence-packing generator.

    Yields: (packed_tokens, file_idx, position, shard_name, seq_counter)
    """
    file_idx = max(start_file_idx, 0)
    processed_count = 0
    is_main_process = (rank == 0)

    eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
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
                    token_batches.append(tokenizer.encode(text, add_special_tokens=False))
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
    packing_buffer: List[int] = []
    seq_counter = 0

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
                    if is_main_process and position > 0:
                        print(f"  >>> RESUMING from row position {position}")
                else:
                    position = 0

                _clear_pending()

                while position < len(dataset) or pending_batches:
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
                                f"Tokenizer worker exception: {future_error}. Skipping batch.",
                                flush=True,
                            )
                        token_batches = [[] for _ in batch_records]
                    finally:
                        pending_batches.popleft()

                    for (record_position, _text), tokens in zip(batch_records, token_batches):
                        if not tokens:
                            continue

                        packing_buffer.extend(tokens)
                        packing_buffer.append(eos_token_id)

                        while len(packing_buffer) >= block_size:
                            packed_seq = packing_buffer[:block_size]
                            packing_buffer = packing_buffer[block_size:]

                            processed_count += 1
                            seq_counter += 1
                            yield packed_seq, file_idx, record_position, shard_name, seq_counter

                if is_main_process:
                    print(f"Finished processing {shard_name}, cleaning up resources...")

                _clear_pending()
                del dataset
                dataset = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                time.sleep(0.1)
                file_idx += 1

            except Exception as e:
                if is_main_process:
                    print(f"Error processing file {data_files[file_idx]}: {e}")

                if dataset is not None:
                    del dataset
                    dataset = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _clear_pending()
                file_idx += 1

    # Flush remaining tokens (last partial sequence)
    if len(packing_buffer) >= block_size // 2:
        while len(packing_buffer) < block_size:
            packing_buffer.append(eos_token_id)
        packed_seq = packing_buffer[:block_size]
        processed_count += 1
        seq_counter += 1
        yield packed_seq, file_idx - 1, -1, "", seq_counter

    if is_main_process:
        print(f"Completed processing all available files. Produced {processed_count} packed sequences.")

    return


# ---------- DataPosition (same as in training.py) ---------------------------

class DataPosition:
    def __init__(self, streaming=True):
        self.streaming = streaming
        self.current_file_idx = 0
        self.position_in_file = 0
        self.chunk_offset = 0
        self.current_position = 0
        self.epoch = 0
        self.files_processed: set = set()

    def get_state(self):
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
        if not state:
            return
        self.streaming = state.get("streaming", self.streaming)
        self.current_file_idx = state.get("current_file_idx", 0)
        self.position_in_file = state.get("position_in_file", 0)
        self.chunk_offset = state.get("chunk_offset", state.get("chunk_index", 0))
        self.current_position = state.get("current_position", 0)
        self.epoch = state.get("epoch", 0)
        files = state.get("files_processed", [])
        self.files_processed = {os.path.basename(f) for f in files}

    def update_streaming_position(self, file_idx, position, chunk_offset=0, file_path=None):
        self.current_file_idx = file_idx
        self.position_in_file = position
        self.chunk_offset = chunk_offset
        if file_path:
            self.files_processed.add(os.path.basename(file_path))

    def next_epoch(self, total_samples=None):
        self.epoch += 1
        if self.streaming:
            self.current_file_idx = 0
            self.position_in_file = 0
            self.chunk_offset = 0
        else:
            self.current_position = 0


# ---------- Checkpoint helpers ----------------------------------------------

def cleanup_old_checkpoints(directory: str, keep: int = 50, rank: int = 0) -> None:
    """Keep only the most recent checkpoint files in a directory."""
    if rank != 0:
        return
    if keep <= 0 or not os.path.isdir(directory):
        return

    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(directory):
        match = DP_CHECKPOINT_PATTERN.search(name)
        if not match:
            match = CHECKPOINT_PATTERN.search(name)
        if not match:
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
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


def _find_latest_dp_checkpoint(search_dir: str, rank: int = 0) -> Optional[str]:
    """Find the latest DP checkpoint in *search_dir*."""
    if not os.path.isdir(search_dir):
        return None
    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(search_dir):
        match = DP_CHECKPOINT_PATTERN.search(name)
        if match:
            path = os.path.join(search_dir, name)
            if os.path.isfile(path):
                candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if rank == 0:
        print(f"Found DP checkpoint '{os.path.basename(candidates[0][1])}' (step {candidates[0][0]})")
    return candidates[0][1]


def _find_latest_tp_checkpoint(search_dir: str, rank: int = 0) -> Optional[str]:
    """Find the latest TP checkpoint in *search_dir*."""
    if not os.path.isdir(search_dir):
        return None
    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(search_dir):
        match = TP_CHECKPOINT_PATTERN.search(name)
        if match:
            path = os.path.join(search_dir, name)
            if os.path.isfile(path):
                candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if rank == 0:
        print(f"Found TP checkpoint '{os.path.basename(candidates[0][1])}' (step {candidates[0][0]})")
    return candidates[0][1]


def _resolve_checkpoint(checkpoint_path: Optional[str], rank: int = 0):
    """Resolve the checkpoint to load.

    Priority:
      1. Explicit path (file or directory with DP checkpoints).
      2. dp_pretrained/ directory (own DP checkpoints).
      3. pretrained/ directory (TP checkpoints to convert).

    Returns (path, checkpoint_type) where checkpoint_type is 'dp' or 'tp'.
    """
    if checkpoint_path and checkpoint_path.upper() == "NONE":
        if rank == 0:
            print("Explicitly starting from scratch (--checkpoint-path NONE)")
        return None, None

    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Determine type by peeking
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
        if ckpt.get("data_parallel", False):
            return checkpoint_path, "dp"
        else:
            return checkpoint_path, "tp"

    # Check dp_pretrained/ first (own checkpoints)
    dp_dir = os.path.join(os.getcwd(), "dp_pretrained")
    dp_path = _find_latest_dp_checkpoint(dp_dir, rank)
    if dp_path:
        return dp_path, "dp"

    # Then check explicit directory if provided
    if checkpoint_path and os.path.isdir(checkpoint_path):
        dp_path = _find_latest_dp_checkpoint(checkpoint_path, rank)
        if dp_path:
            return dp_path, "dp"
        tp_path = _find_latest_tp_checkpoint(checkpoint_path, rank)
        if tp_path:
            return tp_path, "tp"

    # Fall back to pretrained/ (TP checkpoints)
    tp_dir = os.path.join(os.getcwd(), "pretrained")
    tp_path = _find_latest_tp_checkpoint(tp_dir, rank)
    if tp_path:
        return tp_path, "tp"

    if rank == 0:
        print("No checkpoint files found - starting from scratch")
    return None, None


# ---------- OOM-safe memory probe -------------------------------------------

def _memory_probe(model, config, batch_size, block_size, amp_dtype, grad_accum_steps, device, scaler):
    """Run one dummy forward+backward to check if batch_size fits in memory.

    Returns True if the probe passed, False if OOM.
    """
    try:
        dummy_x = torch.randint(0, config.vocab_size, (batch_size, block_size - 1), device=device)
        dummy_y = torch.randint(0, config.vocab_size, (batch_size, block_size - 1), device=device)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            out = model(input_ids=dummy_x)
            loss = F.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)),
                dummy_y.view(-1),
                ignore_index=-100,
            )
        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Cleanup
        if isinstance(model, DDP):
            model.module.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        del dummy_x, dummy_y, out, loss, scaled_loss
        torch.cuda.empty_cache()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e)
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg.lower():
            if isinstance(model, DDP):
                model.module.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False
        raise


# ---------- Save DP checkpoint ----------------------------------------------

def _save_dp_checkpoint(
    model,
    optimizer,
    scheduler,
    global_step,
    total_tokens,
    last_loss,
    data_position,
    batch_size,
    grad_accum_steps,
    amp_dtype,
    world_size,
    rank,
    lr_ramp_tracker,
    save_dir="dp_pretrained",
):
    """Save a full DP checkpoint from rank 0."""
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    # Get full model state dict (unwrap DDP)
    if isinstance(model, DDP):
        model_state = _state_dict_to_cpu(model.module.state_dict())
    else:
        model_state = _state_dict_to_cpu(model.state_dict())

    model_state = cast_state_dict_to_dtype(model_state, amp_dtype)

    checkpoint_state = {
        "global_step": global_step,
        "tokens_processed": total_tokens,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "lr_ramp_state": dict(lr_ramp_tracker) if lr_ramp_tracker is not None else None,
        "gradient_accumulation_steps": grad_accum_steps,
        "loss": last_loss,
        "data_position": data_position.get_state(),
        "model_dtype": str(amp_dtype),
        "data_parallel": True,
        "tensor_parallel": False,
        "world_size": world_size,
        "rank": 0,
        "batch_size": batch_size,
    }

    save_path = os.path.join(save_dir, f"dp_checkpoint_step_{global_step}.pth")
    safe_torch_save(checkpoint_state, save_path)
    print(f"DP checkpoint saved @ step {global_step} -> {save_path}")
    del checkpoint_state, model_state


# ---------- Main training function ------------------------------------------

def resume_training_dp(
    data_glob: str,
    tokenizer_path: str,
    checkpoint_path: Optional[str] = None,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 4096,
    initial_batch_size: int = 32,
    min_batch_size: int = 2,
    batch_size_step: int = 4,
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
    wall_time: int = 43200,
    use_torch_compile: bool = True,
):
    # ---- Distributed setup -------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    grad_accum_steps = max(1, int(gradient_accumulation_steps))

    # Wall-time checkpoint
    job_start_time = time.monotonic()
    wall_time_deadline = job_start_time + wall_time - 300  # 5 min buffer
    wall_time_checkpoint_saved = False
    if is_main_process:
        print(f"Wall time: {wall_time}s ({wall_time / 3600:.1f}h), "
              f"will save final checkpoint ~5 min before termination")

    if is_main_process:
        cleanup_old_checkpoints("dp_pretrained", keep=50, rank=rank)

    # ---- Resolve data files ------------------------------------------------
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

    # ---- Load tokenizer ----------------------------------------------------
    validate_tokenizer_path(tokenizer_path)
    hf_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)

    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})
    hf_tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)
    vocab_size = len(hf_tokenizer)

    # ---- Model config ------------------------------------------------------
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
        use_gradient_checkpointing=use_gradient_checkpointing,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        eos_token_id=hf_tokenizer.eos_token_id,
    )

    # ---- Resolve checkpoint ------------------------------------------------
    global_step = 0
    total_tokens_processed = 0
    load_checkpoint = False
    ckpt = None
    ckpt_type = None  # 'dp' or 'tp'
    resolved_checkpoint = None
    saved_batch_size = None  # If loaded from a DP checkpoint

    if force_from_scratch:
        if is_main_process:
            print("=" * 70)
            print("FORCED START FROM SCRATCH (--force-from-scratch)")
            print("=" * 70)
    else:
        resolved_checkpoint, ckpt_type = _resolve_checkpoint(checkpoint_path, rank)
        if resolved_checkpoint:
            load_checkpoint = True
            if is_main_process:
                print(f"✓ Will resume from {ckpt_type.upper()} checkpoint: {resolved_checkpoint}")

    # ---- Determine dtype ---------------------------------------------------
    supports_bf16 = False
    amp_dtype = torch.float32
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
        supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    # ---- Create model and load weights -------------------------------------
    model = ArgonneModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"✓ Model contains {total_params:,} parameters")

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

        if ckpt_type == "dp":
            # DP checkpoint: full model_state_dict directly
            model_state = ckpt.get("model_state_dict", {})
            if not model_state:
                raise RuntimeError("DP checkpoint does not contain model_state_dict")

            # Normalize _orig_mod keys if present
            if any(k.startswith("_orig_mod.") for k in model_state):
                model_state = {
                    k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k: v
                    for k, v in model_state.items()
                }

            model_state = cast_state_dict_to_dtype(_state_dict_to_cpu(model_state), torch.float32)

            try:
                model.load_state_dict(model_state, strict=True)
                if is_main_process:
                    print("✓ Loaded DP checkpoint weights with strict=True")
            except RuntimeError as e:
                if is_main_process:
                    print(f"⚠ Strict loading failed: {e}, trying non-strict...")
                model.load_state_dict(model_state, strict=False)
                if is_main_process:
                    print("✓ Loaded DP checkpoint weights with strict=False")

            saved_batch_size = ckpt.get("batch_size")

        elif ckpt_type == "tp":
            # TP checkpoint: need to merge shards
            raw_state_dict = ckpt.get("model_state_dict", {}) or {}
            raw_shard_list = ckpt.get("tensor_parallel_shards")

            def _normalize_keys(sd):
                if not sd:
                    return {}
                if any(k.startswith("_orig_mod.") for k in sd.keys()):
                    return {
                        (k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k): v
                        for k, v in sd.items()
                    }
                return dict(sd)

            raw_state_dict = _normalize_keys(raw_state_dict)

            shard_states = None
            if isinstance(raw_shard_list, (list, tuple)) and raw_shard_list:
                shard_states = [
                    cast_state_dict_to_dtype(_state_dict_to_cpu(_normalize_keys(s)), torch.float32)
                    for s in raw_shard_list
                ]

            model_state_cpu: Dict[str, torch.Tensor] = {}

            if shard_states:
                if is_main_process:
                    print(f"Merging {len(shard_states)} tensor-parallel shards into full weights...")
                model_state_cpu = merge_tensor_parallel_shards(shard_states)
            elif raw_state_dict:
                model_state_cpu = cast_state_dict_to_dtype(
                    _state_dict_to_cpu(raw_state_dict), torch.float32
                )
            else:
                raise RuntimeError("TP checkpoint contains neither model_state_dict nor tensor_parallel_shards")

            try:
                model.load_state_dict(model_state_cpu, strict=True)
                if is_main_process:
                    print("✓ Loaded merged TP checkpoint weights with strict=True")
            except RuntimeError as e:
                if is_main_process:
                    print(f"⚠ Strict loading failed: {e}, trying non-strict...")
                model.load_state_dict(model_state_cpu, strict=False)
                if is_main_process:
                    print("✓ Loaded merged TP checkpoint weights with strict=False")

            del model_state_cpu

        global_step = ckpt.get("global_step", 0)
        total_tokens_processed = ckpt.get("tokens_processed", 0)

        if is_main_process:
            print(f"✓ Loaded checkpoint from step {global_step}, tokens: {total_tokens_processed:,}")
    else:
        if is_main_process:
            print("=" * 70)
            print("STARTING FRESH TRAINING")
            print("=" * 70)

    # ---- Gradient checkpointing --------------------------------------------
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main_process:
            print("✓ Gradient checkpointing enabled")

    # ---- Move to device and wrap with DDP ----------------------------------
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process:
        print(f"✓ Model wrapped with DistributedDataParallel on {world_size} GPUs")

    # ---- torch.compile (best-effort) ---------------------------------------
    compiled = False
    if use_torch_compile:
        try:
            ddp_model = torch.compile(ddp_model)
            compiled = True
            if is_main_process:
                print("✓ torch.compile enabled (will compile on first forward pass)")
        except Exception as compile_err:
            if is_main_process:
                print(f"⚠ torch.compile failed: {compile_err}")
                print("  Continuing without compilation")
    else:
        if is_main_process:
            print("  torch.compile disabled by flag")

    # ---- Create optimizer --------------------------------------------------
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=True,
    )

    # ---- Batch-size probing ------------------------------------------------
    #  If we have a DP checkpoint with a saved batch_size, use it directly.
    #  Otherwise probe starting from initial_batch_size.
    use_grad_scaler = amp_dtype == torch.float16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    batch_size: int
    if saved_batch_size is not None and isinstance(saved_batch_size, int) and saved_batch_size > 0:
        batch_size = saved_batch_size
        if is_main_process:
            print(f"✓ Using batch size from DP checkpoint: {batch_size}")
    else:
        # Probe for the right batch size
        batch_size = initial_batch_size
        if is_main_process:
            print(f"Probing batch sizes starting from {batch_size}...")

        def _run_batch_probe(bs):
            """Run memory probe, returns True/False for OOM. Re-raises non-OOM errors."""
            local_ok = _memory_probe(ddp_model, config, bs, block_size, amp_dtype, grad_accum_steps, device, scaler)
            ok_tensor = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device)
            dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
            return ok_tensor.item() == 1

        try:
            while batch_size >= min_batch_size:
                if is_main_process:
                    print(f"  Trying batch_size={batch_size}...", end=" ", flush=True)

                if _run_batch_probe(batch_size):
                    if is_main_process:
                        print(f"✓ batch_size={batch_size} fits!")
                    break
                else:
                    if is_main_process:
                        print(f"✗ OOM")
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size -= batch_size_step
                    if batch_size < min_batch_size:
                        batch_size = min_batch_size
                        if is_main_process:
                            print(f"  Trying minimum batch_size={batch_size}...", end=" ", flush=True)
                        if _run_batch_probe(batch_size):
                            if is_main_process:
                                print(f"✓ batch_size={batch_size} fits!")
                            break
                        else:
                            raise RuntimeError(
                                f"Cannot fit even minimum batch size {min_batch_size} into GPU memory."
                            )
        except Exception as probe_err:
            err_msg = str(probe_err)
            is_oom = isinstance(probe_err, (torch.cuda.OutOfMemoryError, RuntimeError)) and "out of memory" in err_msg.lower()
            if not is_oom and compiled:
                # torch.compile caused a non-OOM failure during the first real
                # forward pass — disable compilation and retry the probe loop.
                compiled = False
                if is_main_process:
                    print(f"\n⚠ torch.compile error during probe: {probe_err}")
                    print("  Disabling torch.compile and retrying...")
                # Unwrap compiled model: rebuild DDP from the underlying module
                base_module = ddp_model
                # torch.compile wraps with OptimizedModule; peel it
                if hasattr(base_module, "_orig_mod"):
                    base_module = base_module._orig_mod
                if isinstance(base_module, DDP):
                    pass  # already DDP
                else:
                    base_module = DDP(model, device_ids=[local_rank], output_device=local_rank)
                ddp_model = base_module

                # Re-create optimizer on the unwrapped model
                optimizer = torch.optim.AdamW(
                    ddp_model.parameters(), lr=lr, weight_decay=weight_decay, fused=True,
                )
                gc.collect()
                torch.cuda.empty_cache()

                # Re-run the probe loop without compile
                batch_size = initial_batch_size
                while batch_size >= min_batch_size:
                    if is_main_process:
                        print(f"  Trying batch_size={batch_size}...", end=" ", flush=True)
                    if _run_batch_probe(batch_size):
                        if is_main_process:
                            print(f"✓ batch_size={batch_size} fits!")
                        break
                    else:
                        if is_main_process:
                            print(f"✗ OOM")
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_size -= batch_size_step
                        if batch_size < min_batch_size:
                            batch_size = min_batch_size
                            if is_main_process:
                                print(f"  Trying minimum batch_size={batch_size}...", end=" ", flush=True)
                            if _run_batch_probe(batch_size):
                                if is_main_process:
                                    print(f"✓ batch_size={batch_size} fits!")
                                break
                            else:
                                raise RuntimeError(
                                    f"Cannot fit even minimum batch size {min_batch_size} into GPU memory."
                                )
            else:
                raise

        if is_main_process:
            print(f"✓ Selected batch_size={batch_size}")

    # ---- Restore optimizer state -------------------------------------------
    saved_lr_ramp_state = None
    optimizer_restored = False

    if load_checkpoint and ckpt is not None and ckpt_type == "dp":
        saved_optim = ckpt.get("optimizer_state_dict")
        if saved_optim:
            try:
                optimizer.load_state_dict(saved_optim)
                optimizer_restored = True
                if is_main_process:
                    print("✓ Optimizer state restored from DP checkpoint")
            except Exception as opt_err:
                if is_main_process:
                    print(f"⚠ Could not restore optimizer state: {opt_err}")
                    print("  Using fresh AdamW optimizer")
        saved_lr_ramp_state = ckpt.get("lr_ramp_state")
    elif load_checkpoint and ckpt is not None and ckpt_type == "tp":
        # TP checkpoint: optimizer states are sharded for TP layout,
        # they don't match DDP parameter shapes → start fresh optimizer
        if is_main_process:
            print("⚠ TP checkpoint: optimizer states are sharded for TP layout — using fresh optimizer")
            print(f"  LR ramp over {rewarmup_steps} steps will stabilize training")

    # ---- Setup scheduler ---------------------------------------------------
    min_lr = min(min_lr, lr)

    effective_warmup = warmup_steps
    if load_checkpoint and resolved_checkpoint and global_step > 0:
        effective_warmup = rewarmup_steps
        if is_main_process:
            print(f"⚠ Using {rewarmup_steps}-step re-warmup for resume stability")

    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=lr,
        warmup_steps=effective_warmup,
        max_steps=total_training_steps,
        min_lr=min_lr,
    )

    # ---- LR ramp tracker for fresh optimizer after TP→DP conversion --------
    lr_ramp_tracker = None
    if load_checkpoint and resolved_checkpoint and global_step > 0 and rewarmup_steps > 0:
        steps_completed = 0
        if isinstance(saved_lr_ramp_state, dict):
            stored_total = int(saved_lr_ramp_state.get("total_steps", rewarmup_steps))
            stored_completed = int(saved_lr_ramp_state.get("steps_completed", 0))
            if 0 <= stored_completed < stored_total:
                steps_completed = max(0, stored_completed)

        lr_ramp_tracker = {
            "steps_completed": min(steps_completed, rewarmup_steps),
            "total_steps": rewarmup_steps,
        }

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
            return float(lr)

        # Skip ramp if optimizer was fully restored from DP checkpoint
        if optimizer_restored:
            lr_ramp_tracker = None
            if is_main_process:
                print("  Skipping LR ramp (optimizer fully restored from DP checkpoint)")
        else:
            if is_main_process:
                target_lr = _preview_lr_value(global_step)
                print(
                    "⚠ Using %d-step LR ramp from %.6e to %.6e after optimizer reset"
                    % (rewarmup_steps, min_lr, target_lr)
                )

    if not optimizer_restored:
        scheduler.step(global_step)

    if is_main_process:
        print(f"✓ Training setup complete")
        print(f"  - Starting step: {global_step}")
        print(f"  - Tokens processed: {total_tokens_processed:,}")
        print(f"  - Learning rate: {lr:.2e}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Gradient accumulation: {grad_accum_steps}")
        print(f"  - Data parallelism: world_size={world_size}")
        print(f"  - torch.compile: {'enabled' if compiled else 'disabled'}")

    # ---- Setup data position -----------------------------------------------
    data_position = DataPosition(streaming=use_streaming)
    if load_checkpoint and ckpt is not None and "data_position" in ckpt:
        data_position.restore_state(ckpt.get("data_position"))
        if is_main_process:
            print(f"✓ Data position: file {data_position.current_file_idx}, position {data_position.position_in_file}")
    else:
        if is_main_process:
            print("  Starting from beginning of dataset")

    # ---- Mixed precision info ----------------------------------------------
    if is_main_process:
        if supports_bf16:
            print("✓ Using torch.bfloat16 autocast")
        elif amp_dtype == torch.float16:
            print("✓ Using torch.float16 autocast with GradScaler")

    # ---- Prefetch setup ----------------------------------------------------
    using_cuda = torch.cuda.is_available()
    prefetch_stream = torch.cuda.Stream(device=device) if using_cuda else None
    max_prefetch_batches = 6 if using_cuda else 2
    prefetched_batches: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque()
    prefetch_warmup_done = not using_cuda

    def enqueue_batch(x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> None:
        if not using_cuda:
            prefetched_batches.append((x_cpu.to(device), y_cpu.to(device)))
            return
        assert prefetch_stream is not None
        with torch.cuda.stream(prefetch_stream):
            x_gpu = x_cpu.to(device, non_blocking=True)
            y_gpu = y_cpu.to(device, non_blocking=True)
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

    # ---- Training loop (streaming) -----------------------------------------
    if use_streaming:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"RESUMING TRAINING WITH DATA PARALLELISM (DDP)")
            print(f"{'='*70}")
            print(f"Step: {global_step} / {total_training_steps}")
            print(f"Batch size per GPU: {batch_size}")
            print(f"World size: {world_size}")
            print(f"Effective batch size: {batch_size * world_size * grad_accum_steps}")
            print(f"{'='*70}\n")

        # Each rank runs its own streaming generator from the same data
        # position.  The packed sequences are assigned round-robin across ranks
        # to avoid duplicated data.
        token_gen = streaming_token_generator(
            data_files,
            hf_tokenizer,
            block_size,
            data_position.current_file_idx,
            data_position.position_in_file,
            data_position.chunk_offset,
            rank,
        )

        token_buffer: List[List[int]] = []
        active_shard: Optional[str] = None
        last_meta: Optional[Tuple[str, int, int, int]] = None
        # Counter used for round-robin data sharding across ranks
        seq_global_counter = 0

        prompt_seed_text = "Long long time ago, "
        prompt_token_ids = hf_tokenizer.encode(prompt_seed_text)
        cached_prompt_tensor = torch.tensor(
            prompt_token_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training (DP)") if is_main_process else None

        try:
            while global_step < total_training_steps:
                try:
                    tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)

                    if file_idx == -1:
                        if is_main_process:
                            print("End of dataset - restarting")
                        data_position.next_epoch()
                        token_gen = streaming_token_generator(
                            data_files, hf_tokenizer, block_size, rank=rank,
                        )
                        seq_global_counter = 0
                        continue

                    # Round-robin: each rank only keeps sequences where
                    # seq_global_counter % world_size == rank
                    seq_global_counter += 1
                    if (seq_global_counter - 1) % world_size != rank:
                        # Update data position tracking even for skipped sequences
                        data_position.update_streaming_position(file_idx, position, chunk_idx,
                                                                data_files[file_idx] if file_idx < len(data_files) else None)
                        continue

                    token_buffer.append(tokens)
                    last_meta = (shard_name, file_idx, position, chunk_idx)
                    data_position.update_streaming_position(file_idx, position, chunk_idx,
                                                            data_files[file_idx] if file_idx < len(data_files) else None)

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

                    autocast_context = (
                        torch.amp.autocast("cuda", dtype=amp_dtype)
                        if torch.cuda.is_available()
                        else contextlib.nullcontext()
                    )

                    with autocast_context:
                        outputs = ddp_model(input_ids=x_local)
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

                        # LR ramp logic
                        if (
                            lr_ramp_tracker
                            and lr_ramp_tracker.get("steps_completed", 0)
                            < lr_ramp_tracker.get("total_steps", 0)
                        ):
                            completed = lr_ramp_tracker["steps_completed"]
                            total = max(1, lr_ramp_tracker["total_steps"])
                            ramp_scale = min((completed + 1) / total, 1.0)
                            ramp_lr = max(min_lr, current_lr * ramp_scale)
                            current_lr = scheduler.override_step(global_step, ramp_lr)
                            lr_ramp_tracker["steps_completed"] = completed + 1

                            if (
                                is_main_process
                                and lr_ramp_tracker["steps_completed"]
                                == lr_ramp_tracker["total_steps"]
                            ):
                                print("✓ LR ramp complete: reached target LR %.6e" % current_lr)

                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                            optimizer.step()

                        optimizer.zero_grad(set_to_none=True)

                        global_step += 1
                        micro_step = 0

                        if pbar is not None:
                            pbar.update(1)

                        if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                            current_total_tokens = total_tokens_processed + tokens_in_this_session
                            print(f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {current_total_tokens:,} | LR: {current_lr:.6e}")

                        if global_step % 500 == 0:
                            current_total_tokens = total_tokens_processed + tokens_in_this_session

                            # Text generation (only rank 0 samples, others wait)
                            ddp_model.eval()
                            if is_main_process:
                                with torch.no_grad():
                                    gen_model = ddp_model.module
                                    gen_ids = gen_model.generate(
                                        cached_prompt_tensor,
                                        max_length=cached_prompt_tensor.shape[1] + 100,
                                        do_sample=True,
                                        temperature=0.7,
                                        top_k=50,
                                        top_p=0.9,
                                    )
                                    generated_text = hf_tokenizer.decode(gen_ids[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")
                            ddp_model.train()

                            # Save checkpoint (rank 0 only)
                            dist.barrier()
                            _save_dp_checkpoint(
                                ddp_model, optimizer, scheduler,
                                global_step, current_total_tokens,
                                last_loss_value, data_position,
                                batch_size, grad_accum_steps,
                                amp_dtype, world_size, rank,
                                lr_ramp_tracker,
                            )
                            dist.barrier()
                            gc.collect()
                            torch.cuda.empty_cache()

                        # --- Wall-time checkpoint ---
                        if not wall_time_checkpoint_saved and time.monotonic() >= wall_time_deadline:
                            if global_step % 500 != 0:
                                current_total_tokens = total_tokens_processed + tokens_in_this_session
                                if is_main_process:
                                    elapsed = time.monotonic() - job_start_time
                                    print(f"\n⏰ Wall-time checkpoint triggered at step {global_step} "
                                          f"(elapsed: {elapsed/3600:.2f}h)")

                                dist.barrier()
                                _save_dp_checkpoint(
                                    ddp_model, optimizer, scheduler,
                                    global_step, current_total_tokens,
                                    last_loss_value, data_position,
                                    batch_size, grad_accum_steps,
                                    amp_dtype, world_size, rank,
                                    lr_ramp_tracker,
                                    save_dir="dp_pretrained",
                                )
                                dist.barrier()
                                gc.collect()
                                torch.cuda.empty_cache()
                            else:
                                if is_main_process:
                                    print(f"\n⏰ Wall-time deadline reached at step {global_step} "
                                          f"(regular checkpoint already saved)")

                            wall_time_checkpoint_saved = True

                except StopIteration:
                    if is_main_process:
                        print("StopIteration - restarting dataset")
                    data_position.next_epoch()
                    token_gen = streaming_token_generator(
                        data_files, hf_tokenizer, block_size, rank=rank,
                    )
                    seq_global_counter = 0
                    continue
        finally:
            if using_cuda and prefetch_stream is not None:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
            prefetched_batches.clear()
            prefetch_warmup_done = not using_cuda
            if pbar is not None:
                pbar.close()

    # ---- Final save / export -----------------------------------------------
    final_token_count = total_tokens_processed + tokens_in_this_session
    training_finished = (global_step >= total_training_steps)

    if training_finished:
        if is_main_process:
            print(f"\n===== TRAINING COMPLETE =====")
            print(f"Total tokens: {final_token_count:,}")
            print(f"Final step: {global_step}")

            # Export full model
            export_model = ArgonneModel(config)
            export_state = _state_dict_to_cpu(ddp_model.module.state_dict())
            export_model.load_state_dict(export_state, strict=True)

            output_dir = "Argonne2.5-DP"
            os.makedirs(output_dir, exist_ok=True)
            export_model.save_pretrained(output_dir, safe_serialization=False)
            hf_tokenizer.save_pretrained(output_dir)
            print(f"✓ Final model saved to {output_dir}")
    else:
        if is_main_process:
            print(f"\nTraining interrupted (wall-time or signal) at step {global_step}")
            print(f"  Tokens processed this session: {tokens_in_this_session:,}")
            print(f"  Total tokens: {final_token_count:,}")
            print(f"  Resume from the latest checkpoint in dp_pretrained/")

    if dist.is_initialized():
        dist.barrier()


# ---------- CLI -------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume Argonne pretraining with Data Parallelism (DDP)")
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    parser.add_argument(
        "--data-glob", type=str, default=default_data_glob,
        help="Glob pattern for parquet shards",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="Filesystem directory containing the pretrained tokenizer.",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Optional path to checkpoint file or directory. Use 'NONE' to start from scratch.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=DEFAULT_MAX_TRAINING_STEPS,
        help="Total number of training steps to run.",
    )
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument(
        "--initial-batch-size", type=int, default=32,
        help="Initial per-GPU batch size to try (reduced on OOM).",
    )
    parser.add_argument(
        "--min-batch-size", type=int, default=2,
        help="Minimum per-GPU batch size.",
    )
    parser.add_argument(
        "--batch-size-step", type=int, default=4,
        help="Amount to reduce batch size on OOM.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--min-learning-rate", type=float, default=1e-5,
        help="Minimum learning rate.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--rewarmup-steps", type=int, default=100,
        help="Number of re-warmup steps when resuming.",
    )
    parser.add_argument(
        "--no-streaming", action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--weight-decay", type=float, default=0.1,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Allow loading tokenizers that require custom code.",
    )
    parser.add_argument(
        "--force-from-scratch", action="store_true",
        help="Force training from scratch, ignoring any checkpoints.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing", action="store_true",
        help="Disable gradient checkpointing (requires more GPU memory).",
    )
    parser.add_argument(
        "--disable-compile", action="store_true",
        help="Disable torch.compile (enabled by default for DDP training).",
    )
    parser.add_argument(
        "--wall-time", type=int, default=43200,
        help="Job wall time in seconds (default: 43200 = 12 hours). "
             "Checkpoint saved ~5 min before deadline.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resume_training_dp(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        total_training_steps=args.total_steps,
        block_size=args.block_size,
        initial_batch_size=args.initial_batch_size,
        min_batch_size=args.min_batch_size,
        batch_size_step=args.batch_size_step,
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
        wall_time=args.wall_time,
        use_torch_compile=not args.disable_compile,
    )


if __name__ == "__main__":
    main()
