"""Utility helpers shared across ArgonneAI training scripts."""
from __future__ import annotations

import glob
import math
import os
import re
from typing import Iterable, List, Sequence, Tuple

import torch


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

    common_root = os.path.commonpath(files)
    print("=== Dataset shard order ===")
    print(f"Root directory: {common_root}")
    for idx, path in enumerate(files, start=1):
        print(f"  [{idx:03d}] {os.path.basename(path)}")
    print("===========================")


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

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict) -> None:
        step = int(state_dict.get("step", 0))
        self.step(step)

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
