import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import re
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
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from data_processing import (
    collate_batch,
    chunk_tokens,
    load_tokenizer,
)
from model import ArgonneConfig, ArgonneModel
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    cast_state_dict_to_dtype,
    determine_document_boundary_tokens,
    load_streaming_shard,
    log_dataset_plan,
    safe_torch_load,
    safe_torch_save,
    resolve_data_files,
    validate_tokenizer_path,
)

# Import all tensor parallel components from training.py
from training import (
    DataPosition,
    init_tensor_parallel_group,
    TensorParallelModel,
    _ensure_gradient_dtype_matches_params,
    shard_attention_layer,
    shard_mlp_layer,
    shard_tensor_parallel_correctly,
    _gcd,
)

# Enable TF32 precision on Ampere/Hopper GPUs and prefer flash attention kernels
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
with contextlib.suppress(Exception):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

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
            full_state[key] = torch.cat(tensors, dim=0)
        elif key.endswith(COLUMN_PARALLEL_BIAS_SUFFIXES):
            full_state[key] = torch.cat(tensors, dim=0)
        elif key.endswith(ROW_PARALLEL_WEIGHT_SUFFIXES):
            # Skip ranks that do not hold this shard (bias-less linears)
            ordered = [value for value in values if isinstance(value, torch.Tensor)]
            full_state[key] = torch.cat(ordered, dim=1)
        else:
            # Parameters replicated on all ranks (embeddings, norms, lm_head, biases)
            full_state[key] = tensors[0]

    return full_state


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
        """Best-effort resolution of chat-style special tokens.

        Some tokenizers (notably chat-oriented ones) expose boundary symbols via
        ``additional_special_tokens`` but round-trip conversions through
        ``convert_ids_to_tokens`` may normalise whitespace, causing the shared
        resolver in ``training_utils`` to give up.  When that happens we fall
        back to direct ``convert_tokens_to_ids`` / ``encode`` probes here so the
        resume script can still honour ``--add-document-boundary-tokens``.
        """

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
                                f"Tokenizer worker exception: {future_error}. Skipping batch.",
                                flush=True,
                            )
                        token_batches = [[] for _ in batch_records]
                    finally:
                        pending_batches.popleft()

                    for (item_position, _), tokens, resume_mid_document in zip(
                        batch_records, token_batches, batch_resume_flags
                    ):
                        if not tokens:
                            continue

                        token_list = tokens

                        if document_tokens_enabled:
                            token_list = [bos_token_id, *token_list, eos_token_id]

                            if resume_mid_document and token_list:
                                token_list = token_list[1:]

                        if len(token_list) < 10:
                            continue

                        for chunk_idx, chunk in enumerate(chunk_tokens(token_list, block_size)):
                            if (
                                file_idx == initial_file_idx
                                and item_position == initial_position
                                and chunk_idx < resume_chunk_offset
                            ):
                                continue

                            processed_count += 1
                            yield chunk, file_idx, item_position, shard_name, chunk_idx

                        if (
                            resume_mid_document
                            and file_idx == initial_file_idx
                            and item_position == initial_position
                            and resume_chunk_offset > 0
                        ):
                            resume_chunk_offset = 0

                if is_main_process:
                    print(f"Finished processing {shard_name}, cleaning up resources...")

                _clear_pending()
                del dataset
                dataset = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                import time

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

    if is_main_process:
        print(f"Completed processing all available files. Processed {processed_count} samples.")

    return None, -1, -1, "", -1


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

    default_dir = os.path.join(os.getcwd(), "distill-pretrained")
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
    teacher_path: str,
    student_path: str = "Argonne2.0",
    checkpoint_path: Optional[str] = None,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 2048,
    batch_size: int = 2,
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
    add_document_tokens: bool = True,
    teacher_low_cpu_mem_usage: bool = False,
    teacher_device_map: str = "auto",
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)
    init_tensor_parallel_group(world_size, rank)
    is_main_process = (rank == 0)
    grad_accum_steps = max(1, int(gradient_accumulation_steps))
    
    if is_main_process:
        cleanup_old_checkpoints("distill-pretrained", keep=50, rank=rank)
    
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

    # Load tokenizer and base student weights
    validate_tokenizer_path(student_path)
    hf_tokenizer = load_tokenizer(student_path, trust_remote_code=trust_remote_code)
    
    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})
    hf_tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)
    
    vocab_size = len(hf_tokenizer)

    # Build config - prefer loading from the saved student checkpoint
    try:
        config = ArgonneConfig.from_pretrained(student_path)
        config.max_position_embeddings = block_size
        config.use_gradient_checkpointing = use_gradient_checkpointing
        config.pad_token_id = hf_tokenizer.pad_token_id
        config.bos_token_id = getattr(hf_tokenizer, "bos_token_id", None)
        config.eos_token_id = hf_tokenizer.eos_token_id
        config.vocab_size = vocab_size
    except Exception:
        # Fallback to static config
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
            use_gradient_checkpointing=use_gradient_checkpointing,
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

    teacher_device_map_option = teacher_device_map
    teacher_device_map = None
    teacher_low_cpu_mem_usage = bool(teacher_low_cpu_mem_usage)
    if torch.cuda.is_available():
        if teacher_device_map_option == "auto":
            teacher_device_map = "auto"
            teacher_low_cpu_mem_usage = True
        elif teacher_device_map_option and teacher_device_map_option != "none":
            teacher_device_map = teacher_device_map_option
            teacher_low_cpu_mem_usage = True
        else:
            teacher_device_map = None
    else:
        teacher_device_map = None

    if is_main_process:
        if teacher_device_map == "auto":
            target_device = "CUDA (device_map=auto across all GPUs)"
        elif teacher_device_map:
            target_device = str(teacher_device_map)
        else:
            target_device = (
                f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            )
        cpu_mode = "low CPU memory" if teacher_low_cpu_mem_usage else "full CPU prefetch"
        print(
            f"✓ Loading teacher model from {teacher_path} on {target_device} "
            f"({cpu_mode})"
        )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=amp_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=teacher_low_cpu_mem_usage,
        device_map=teacher_device_map,
    )
    if teacher_device_map is None and torch.cuda.is_available():
        teacher_model = teacher_model.to(torch.cuda.current_device())
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Create base model (keep parameters in FP32 for stable optimizer state)
    try:
        base_model = ArgonneModel.from_pretrained(student_path, torch_dtype=torch.float32)
        missing_keys = base_model.resize_token_embeddings(vocab_size, pad_to_multiple_of=None)
        if missing_keys is not None and is_main_process:
            print(f"✓ Resized token embeddings to match tokenizer (new size={vocab_size})")
    except Exception:
        base_model = ArgonneModel(config)

    total_params = sum(p.numel() for p in base_model.parameters())
    if is_main_process:
        print(f"✓ Model contains {total_params:,} parameters")

    checkpoint_tensor_shards: Optional[List[Dict[str, torch.Tensor]]] = None
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

        checkpoint_world_size = ckpt.get("world_size", 1)
        is_tensor_checkpoint = ckpt.get("tensor_parallel", False)

        raw_state_dict = ckpt.get("model_state_dict", {}) or {}
        raw_shard_list = ckpt.get("tensor_parallel_shards")

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
                
                model_state_cpu = merge_tensor_parallel_shards(replicated_shards)

        if shard_states and checkpoint_world_size == world_size:
            load_shards_after_wrap = True
            checkpoint_tensor_shards = shard_states
        else:
            if shard_states and checkpoint_world_size != world_size:
                if is_main_process:
                    print(
                        "Merging tensor-parallel shards from checkpoint to rebuild full weights "
                        "(checkpoint world size=%d, target world size=%d)..."
                        % (checkpoint_world_size, world_size)
                    )
                model_state_cpu = merge_tensor_parallel_shards(shard_states)

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
    model = TensorParallelModel(base_model, world_size, rank)

    if load_shards_after_wrap and checkpoint_tensor_shards:
        if len(checkpoint_tensor_shards) != world_size:
            raise RuntimeError(
                "tensor_parallel_shards length does not match current world size; cannot load shard weights."
            )
        shard_state = checkpoint_tensor_shards[rank]
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

    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Attempt to compile for additional throughput; fall back gracefully on failure
    compiled_model = None
    if torch.cuda.is_available():
        try:
            compiled_model = torch.compile(model, mode="max-autotune")
            model = compiled_model
            if is_main_process:
                print("✓ Enabled torch.compile for training")
        except Exception as compile_err:
            if is_main_process:
                print(f"⚠ torch.compile failed ({compile_err}); continuing without compilation")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=True
    )

    saved_lr_ramp_state = None
    optimizer_state = None
    scheduler_state = None
    if load_checkpoint and resolved_checkpoint:
        saved_lr_ramp_state = ckpt.get("lr_ramp_state")
        optimizer_state = ckpt.get("optimizer_state_dict")
        scheduler_state = ckpt.get("scheduler_state_dict")

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

            last_lr = getattr(scheduler, "last_lr", None)
            if last_lr is not None:
                try:
                    return float(last_lr)
                except Exception:
                    pass

            return float(lr)

        if is_main_process:
            target_lr = _preview_lr_value(global_step)
            print(
                "⚠ Using %d-step LR ramp from %.6e to %.6e after optimizer reset"
                % (rewarmup_steps, min_lr, target_lr)
            )
            if lr_ramp_tracker["steps_completed"] > 0:
                print(
                    "  Continuing ramp progress: %d/%d steps already applied"
                    % (
                        lr_ramp_tracker["steps_completed"],
                        lr_ramp_tracker["total_steps"],
                    )
                )

    optimizer_restored = False
    scheduler_restored = False
    if load_checkpoint and resolved_checkpoint and optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
            if is_main_process:
                print("✓ Restored optimizer state from checkpoint")
        except Exception as opt_err:
            if is_main_process:
                print(f"⚠ Failed to load optimizer state: {opt_err}")
                print("  Falling back to fresh optimizer (incompatible across parallelism schemes)")

    if optimizer_restored and scheduler_state:
        try:
            scheduler.load_state_dict(scheduler_state)
            scheduler_restored = True
            if is_main_process:
                print("✓ Restored scheduler state from checkpoint")
        except Exception as sched_err:
            if is_main_process:
                print(f"⚠ Failed to load scheduler state: {sched_err}")
                print("  Recomputing schedule with current settings")

    if not scheduler_restored:
        scheduler.step(global_step)
        if load_checkpoint and resolved_checkpoint and optimizer_restored and is_main_process:
            print(f"⚠ Scheduler state rebuilt; re-warming over {effective_warmup} steps")
    
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
    using_cuda = torch.cuda.is_available()
    if using_cuda:
        teacher_model.to(first_device)
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
    current_batch_size = max(1, int(batch_size))

    optimizer.zero_grad(set_to_none=True)

    # Training loop
    if use_streaming:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"RESUMING TRAINING WITH TENSOR PARALLELISM")
            print(f"{'='*70}")
            print(f"Step: {global_step} / {total_training_steps}")
            print(f"Learning rate: {lr:.2e}")
            print(f"Batch size: {current_batch_size}")
            print(f"World size: {world_size}")
            print(f"{'='*70}\n")

        # Initialize the streaming generator (optionally adding document boundary tokens)
        token_gen = streaming_token_generator(
            data_files,
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

        distill_temperature = 1.0
        distill_alpha = 0.5

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
                            data_files,
                            hf_tokenizer,
                            block_size,
                            rank=rank,
                            add_document_tokens=add_document_tokens,
                        )
                        continue

                    token_buffer.append(tokens)
                    last_meta = (shard_name, file_idx, position, chunk_idx)
                    data_position.update_streaming_position(file_idx, position, chunk_idx, data_files[file_idx])

                    if shard_name != active_shard:
                        active_shard = shard_name
                        if is_main_process:
                            print(f"Processing shard {file_idx + 1}/{len(data_files)}: {shard_name}")

                    while len(token_buffer) >= current_batch_size:
                        batch_tokens_list = token_buffer[:current_batch_size]
                        token_buffer = token_buffer[current_batch_size:]

                        x_tens, y_tens = collate_batch(batch_tokens_list, block_size)
                        if x_tens is None or last_meta is None:
                            continue

                        x_local: torch.Tensor
                        y_local: torch.Tensor

                        if using_cuda:
                            enqueue_batch(x_tens, y_tens)
                            if not prefetch_warmup_done:
                                if len(prefetched_batches) < max_prefetch_batches:
                                    token_buffer = batch_tokens_list + token_buffer
                                    break
                                prefetch_warmup_done = True
                            x_local, y_local = pop_prefetched()
                        else:
                            enqueue_batch(x_tens, y_tens)
                            x_local, y_local = pop_prefetched()

                        batch_tokens = x_local.numel()
                        tokens_in_this_session += batch_tokens

                        autocast_context = torch.amp.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available() else contextlib.nullcontext()

                        loss_tensor = None
                        try:
                            with autocast_context:
                                outputs = model(input_ids=x_local)
                                logits = outputs.logits
                            with torch.no_grad():
                                teacher_out = teacher_model(input_ids=x_local)
                                teacher_logits = teacher_out.logits.to(logits.dtype)

                            student_ce = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                y_local.view(-1),
                                ignore_index=-100,
                            )

                            student_log_probs = F.log_softmax(
                                logits / distill_temperature, dim=-1
                            )
                            teacher_probs = F.softmax(
                                teacher_logits / distill_temperature, dim=-1
                            )

                            distill_loss = F.kl_div(
                                student_log_probs,
                                teacher_probs,
                                reduction="batchmean",
                            ) * (distill_temperature ** 2)

                            loss_tensor = distill_alpha * student_ce + (1.0 - distill_alpha) * distill_loss

                            last_loss_value = float(loss_tensor.detach().cpu().item())
                        except RuntimeError as oom_err:
                            if "out of memory" in str(oom_err).lower():
                                prev_batch = current_batch_size
                                current_batch_size = max(1, current_batch_size - 12)
                                token_buffer = batch_tokens_list + token_buffer
                                if using_cuda:
                                    torch.cuda.empty_cache()
                                optimizer.zero_grad(set_to_none=True)
                                if is_main_process:
                                    print(
                                        f"⚠ CUDA OOM encountered at batch size {prev_batch}; reducing to {current_batch_size}"
                                    )
                                if current_batch_size < 1:
                                    raise
                                break
                            raise

                        if loss_tensor is None:
                            continue

                    loss_for_backward = loss_tensor / grad_accum_steps

                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

                    micro_step += 1

                    if micro_step >= grad_accum_steps:
                        current_lr = scheduler.step(global_step)

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
                                print(
                                    "✓ LR ramp complete: reached target LR %.6e" % current_lr
                                )

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

                        if pbar is not None:
                            pbar.update(1)

                        if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                            current_total_tokens = total_tokens_processed + tokens_in_this_session
                            print(f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {current_total_tokens:,} | LR: {current_lr:.6e}")

                        if global_step % 430 == 0:
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

                            if is_main_process:
                                generated_text = hf_tokenizer.decode(generated[0].tolist())
                                print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                                # SIMPLE: Just save rank 0's model state (sharded weights)
                                model_state = cast_state_dict_to_dtype(
                                    model.base_model.state_dict(), amp_dtype
                                )

                                checkpoint_state = {
                                    "global_step": global_step,
                                    "tokens_processed": current_total_tokens,
                                    "model_state_dict": model_state,
                                    "optimizer_state_dict": optimizer.state_dict(),
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
                                    "world_size": world_size,
                                    "rank": rank,
                                }
                                os.makedirs("distill-pretrained", exist_ok=True)
                                save_path = (
                                    f"distill-pretrained/streaming_checkpoint_step_{global_step}.pth"
                                )
                                safe_torch_save(checkpoint_state, save_path)
                                print(f"Checkpoint saved @ step {global_step} -> {save_path}")

                except StopIteration:
                    if is_main_process:
                        print("StopIteration - restarting dataset")
                    data_position.next_epoch()
                    # Restart with same generator type
                    token_gen = streaming_token_generator(
                        data_files,
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

    # Gather tensor-parallel shards and export a full model + tokenizer for final use
    local_state = cast_state_dict_to_dtype(_state_dict_to_cpu(model.base_model.state_dict()), torch.float32)

    merged_state: Optional[Dict[str, torch.Tensor]] = None
    if dist.is_initialized() and world_size > 1:
        gathered_states: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None
        if rank == 0:
            gathered_states = [None for _ in range(world_size)]

        dist.gather_object(local_state, gather_list=gathered_states, dst=0)

        if rank == 0:
            shard_list: List[Mapping[str, torch.Tensor]] = [
                shard for shard in gathered_states if shard is not None
            ]
            merged_state = merge_tensor_parallel_shards(shard_list)
    else:
        merged_state = local_state

    if is_main_process:
        print(f"\n===== TRAINING COMPLETE =====")
        print(f"Total tokens: {final_token_count:,}")
        print(f"Final step: {global_step}")

        if not merged_state:
            raise RuntimeError("Unable to merge tensor-parallel shards for final model export")

        export_model = ArgonneModel(config)
        export_model.load_state_dict(merged_state, strict=True)

        output_dir = "Argonne2.0-distilled"
        os.makedirs(output_dir, exist_ok=True)
        export_model.save_pretrained(output_dir)
        hf_tokenizer.save_pretrained(output_dir)

        print(f"✓ Final model saved to {output_dir}")

    if dist.is_initialized():
        dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill Argonne model with tensor parallelism")
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=default_data_glob,
        help="Glob pattern for parquet shards",
    )
    parser.add_argument(
        "--student-path",
        type=str,
        default="Argonne2.0",
        help="Filesystem directory containing the pretrained student model and tokenizer.",
    )
    parser.add_argument(
        "--teacher-path",
        type=str,
        default=os.path.join("..", "Qwen2.5-32B-Instruct"),
        help="Filesystem directory containing the teacher model checkpoint.",
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
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument("--initial-batch-size", type=int, default=64)
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
        "--teacher-low-cpu-mem-usage",
        action="store_true",
        help="Use low-CPU-memory loading for the teacher model (slower but lighter).",
    )
    parser.add_argument(
        "--teacher-device-map",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Device map strategy for the teacher model (requires low CPU memory mode)",
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
    # REMOVED: --compile-model argument
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
        teacher_path=args.teacher_path,
        student_path=args.student_path,
        checkpoint_path=args.checkpoint_path,
        total_training_steps=args.total_steps,
        block_size=args.context_window,
        batch_size=args.initial_batch_size,
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
        teacher_low_cpu_mem_usage=args.teacher_low_cpu_mem_usage,
        teacher_device_map=args.teacher_device_map,
    )


if __name__ == "__main__":
    main()