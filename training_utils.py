"""Utility helpers shared across ArgonneAI training scripts."""
from __future__ import annotations

import glob
import os
import re
from typing import Iterable, List, Sequence, Tuple


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
