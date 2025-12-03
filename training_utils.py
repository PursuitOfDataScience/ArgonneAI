"""Utility helpers shared across ArgonneAI training scripts."""
from __future__ import annotations

import contextlib
import glob
import math
import os
import re
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset


# Shared constant to keep the default training horizon consistent across scripts.
DEFAULT_MAX_TRAINING_STEPS = 4_000_000


def _resolve_token_id(
    tokenizer,
    id_attribute: str,
    token_attribute: str,
    fallback_tokens: Optional[Iterable[str]] = None,
) -> Tuple[Optional[int], Optional[str]]:
    """Return a token id/string pair from tokenizer attributes or fallbacks."""

    token_id = getattr(tokenizer, id_attribute, None)
    token_str = getattr(tokenizer, token_attribute, None)

    if token_id is not None:
        if token_str is None:
            try:
                token_str = tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                token_str = None
        return token_id, token_str

    candidates: Iterable[str] = fallback_tokens or ()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            candidate_id = tokenizer.convert_tokens_to_ids(candidate)
        except Exception:
            continue

        if not isinstance(candidate_id, int) or candidate_id < 0:
            continue

        try:
            round_trip = tokenizer.convert_ids_to_tokens(candidate_id)
        except Exception:
            round_trip = None

        if round_trip == candidate:
            return candidate_id, candidate

    if token_str:
        try:
            candidate_id = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(candidate_id, int) and candidate_id >= 0:
                try:
                    round_trip = tokenizer.convert_ids_to_tokens(candidate_id)
                except Exception:
                    round_trip = None
                else:
                    if round_trip == token_str:
                        return candidate_id, token_str
        except Exception:
            pass

    return None, token_str


def determine_document_boundary_tokens(
    tokenizer,
    bos_fallback_tokens: Optional[Iterable[str]] = None,
    eos_fallback_tokens: Optional[Iterable[str]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """Determine BOS/EOS token ids (and strings) for document boundaries.

    Chat-focused tokenizers occasionally omit ``bos_token_id`` even though a
    suitable token (e.g. ``"<|im_start|>"``) exists in
    ``additional_special_tokens``.  This helper searches common fallbacks so
    document boundary injection can remain enabled without manual overrides.
    """

    default_bos_candidates = [
        getattr(tokenizer, "bos_token", None),
        "<|im_start|>",
        "<s>",
        "[CLS]",
    ]
    default_eos_candidates = [
        getattr(tokenizer, "eos_token", None),
        "<|im_end|>",
        "</s>",
        "[SEP]",
    ]

    additional_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    for token in additional_tokens:
        if token not in default_bos_candidates:
            default_bos_candidates.append(token)
        if token not in default_eos_candidates:
            default_eos_candidates.append(token)

    if bos_fallback_tokens:
        for token in bos_fallback_tokens:
            if token not in default_bos_candidates:
                default_bos_candidates.append(token)
    if eos_fallback_tokens:
        for token in eos_fallback_tokens:
            if token not in default_eos_candidates:
                default_eos_candidates.append(token)

    bos_token_id, bos_token_str = _resolve_token_id(
        tokenizer,
        "bos_token_id",
        "bos_token",
        default_bos_candidates,
    )
    eos_token_id, eos_token_str = _resolve_token_id(
        tokenizer,
        "eos_token_id",
        "eos_token",
        default_eos_candidates,
    )

    return bos_token_id, eos_token_id, bos_token_str, eos_token_str


def _natural_key(path: str) -> List[object]:
    """Return a key that enables natural sorting for file paths."""
    base = os.path.basename(path)
    parts = re.split(r"(\d+)", base)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part.lower())
    return key


def resolve_data_files(
    primary_pattern: str,
    *,
    fallback_patterns: Iterable[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """Expand glob patterns and return a naturally-sorted, de-duplicated list.

    The function tries the primary pattern first and then any optional fallback
    patterns. Matches from all patterns are combined, allowing data shards to be
    organised across multiple directories. The returned ``used_patterns`` list
    indicates which patterns produced at least one match, preserving discovery
    order.
    """

    patterns: List[str] = [primary_pattern]
    if fallback_patterns:
        patterns.extend(list(fallback_patterns))

    seen: set[str] = set()
    resolved: List[str] = []
    used_patterns: List[str] = []

    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            continue
        used_patterns.append(pattern)
        for match in matches:
            if match not in seen:
                resolved.append(match)
                seen.add(match)

    if not resolved:
        raise FileNotFoundError(
            f"No dataset shards matched the provided patterns: {patterns}"
        )

    resolved.sort(key=_natural_key)
    return resolved, used_patterns


def log_dataset_plan(files: Sequence[str]) -> None:
    """Print the order in which dataset shards will be consumed."""
    if not files:
        return

    try:
        common_root = os.path.commonpath(files)
    except ValueError:
        common_root = ""

    digits = len(str(len(files)))

    for index, path in enumerate(files, start=1):
        display_path = os.path.relpath(path, common_root) if common_root else path



def safe_torch_save(obj, path: str) -> str:
    """Persist ``obj`` to ``path`` with fallbacks for large checkpoints.

    Some network file systems used on large HPC clusters exhibit unreliable
    behaviour when PyTorch's default zip-based serialization writes very large
    archives (multi-gigabyte optimizer states).  The symptom can surface as
    runtime write failures *or* as truncated archives that only raise an error
    when reloading.

    To make checkpointing resilient, we first try the legacy (non-zip)
    serializer which streams data sequentially and avoids the problematic code
    path altogether.  If that fails with an unrelated error, we fall back to the
    default zip writer.  Every attempt uses a temporary file and ``os.replace``
    to keep the operation atomic, and any intermediate artefacts are cleaned up
    on failure.
    """

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    base = os.path.basename(path)
    retryable_signatures = (
        "PytorchStreamWriter failed writing file",
        "unexpected pos",
    )

    def _save(use_zipfile: bool) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{base}.", suffix=".tmp")
        os.close(fd)
        try:
            save_kwargs = {}
            if not use_zipfile:
                save_kwargs["_use_new_zipfile_serialization"] = False
            torch.save(obj, tmp_path, **save_kwargs)
            os.replace(tmp_path, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)

    try:
        _save(use_zipfile=False)
        return path
    except RuntimeError as err:
        message = str(err)
        if not any(signature in message for signature in retryable_signatures):
            raise
        _save(use_zipfile=True)
        return path


def safe_torch_load(path: str, *, map_location=None, **kwargs):
    """Load a checkpoint while automatically retrying without ``weights_only``.

    When ``safe_torch_save`` falls back to the legacy serializer the resulting
    archive is not a zip container.  ``torch.load(..., weights_only=True)``
    always expects the zip-based format and therefore raises
    ``failed finding central directory``.  To make resume scripts robust across
    both formats we first try the caller-provided arguments and, upon hitting
    this specific error class, retry without the ``weights_only`` flag so that
    PyTorch can transparently handle legacy pickled checkpoints.
    """

    legacy_signatures = (
        "PytorchStreamReader failed reading zip archive",
        "failed finding central directory",
    )

    try:
        return torch.load(path, map_location=map_location, **kwargs)
    except RuntimeError as err:
        message = str(err)
        if not any(signature in message for signature in legacy_signatures):
            raise

        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("weights_only", None)

        try:
            return torch.load(path, map_location=map_location, **retry_kwargs)
        except RuntimeError as retry_err:
            retry_message = str(retry_err)
            if any(signature in retry_message for signature in legacy_signatures):
                raise RuntimeError(
                    "Failed to load checkpoint "
                    f"'{path}'. The file appears to be truncated or was written "
                    "with an unsupported format. Please verify that the checkpoint "
                    "was fully written and retry from a valid file."
                ) from retry_err
            raise


def cast_state_dict_to_dtype(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Return a CPU-based copy of ``state_dict`` cast to ``dtype`` when safe."""

    target_is_fp16 = dtype in (torch.float16, torch.bfloat16)
    converted: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu")
            if target_is_fp16 and torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=dtype)
            converted[key] = tensor
        else:
            converted[key] = value

    return converted


def validate_tokenizer_path(path: str) -> str:
    """Ensure that ``path`` points to a tokenizer directory on disk."""

    if not os.path.isdir(path):
        raise FileNotFoundError(
            "Tokenizer path must be a directory exported from a pretrained model. "
            f"Received: {path}"
        )

    return path


class CosineWarmupScheduler:
    """Lightweight cosine decay scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive for the cosine scheduler")

        self.optimizer = optimizer
        self.base_lr = max(base_lr, 0.0)
        self.min_lr = min(max(min_lr, 0.0), self.base_lr)
        self.warmup_steps = max(warmup_steps, 0)
        self.max_steps = max_steps
        self._step = 0
        self.step(0)

    def _lr_for_step(self, step: int) -> float:
        step = max(0, step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_frac = step / max(1, self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * warmup_frac

        decay_progress = 0.0
        if self.max_steps > self.warmup_steps:
            decay_progress = (step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self, step: int) -> float:
        lr = self._lr_for_step(step)
        self._apply_lr(lr)
        self._step = step
        return lr

    def preview_lr(self, step: int) -> float:
        """Return the learning rate the scheduler would use for ``step``."""

        return self._lr_for_step(step)

    def override_step(self, step: int, lr: float) -> float:
        """Apply ``lr`` manually while keeping the internal step in sync."""

        self._apply_lr(lr)
        self._step = step
        return lr

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict) -> None:
        step = int(state_dict.get("step", 0))
        self.step(step)

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

def load_streaming_shard(file_path: str):
    """
    Load a single parquet shard with proper resource management.
    Loads data into memory and immediately closes file handles.
    
    Args:
        file_path: Path to the parquet or arrow file
        
    Returns:
        Dataset: HuggingFace Dataset object with data in memory
        
    Raises:
        RuntimeError: If file cannot be loaded
    """
    from datasets import Dataset
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    try:
        if file_path.endswith('.parquet'):
            # Read parquet file completely into memory
            table = pq.read_table(file_path)
            
            # Convert to Dataset (creates in-memory copy)
            dataset = Dataset(table)
            
            # Explicitly delete table to free file handle immediately
            del table
            
        elif file_path.endswith('.arrow'):
            # For Arrow files, use memory map but load into memory
            with pa.memory_map(file_path, 'r') as source:
                loaded_table = pa.ipc.open_file(source).read_all()
            
            dataset = Dataset(loaded_table)
            del loaded_table
            
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Validate dataset has required columns
        if 'text' not in dataset.column_names:
            raise ValueError(f"Dataset missing 'text' column. Found: {dataset.column_names}")
        
        return dataset
        
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {type(e).__name__}: {str(e)}")
