import json
import os
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

os.environ["HF_DATASETS_CACHE"] = "./.cache"


#####################################
# Tokenizer Utilities
#####################################


def _prepare_data_files(data_files: Iterable[str]) -> Tuple[List[str], str]:
    """Materialise ``data_files`` and infer the HF dataset builder to use."""

    file_list = list(data_files)
    if not file_list:
        raise ValueError("No dataset shards were provided.")

    extension = os.path.splitext(file_list[0])[1].lower()
    if extension == ".parquet":
        dataset_builder = "parquet"
    elif extension == ".arrow":
        dataset_builder = "arrow"
    else:
        raise ValueError(
            "Unsupported dataset shard extension. Expected '.parquet' or '.arrow' but "
            f"received '{extension}'."
        )

    return file_list, dataset_builder


def create_text_file_from_arrow(
    shard_files: Iterable[str], output_file: str = "all_text_for_tokenizer.txt"
) -> None:
    """Utility kept for backwards compatibility when a fallback BPE tokenizer is required."""

    files, dataset_builder = _prepare_data_files(shard_files)
    print(
        f"Creating a combined text file '{output_file}' from {dataset_builder} shards..."
    )
    with open(output_file, "w", encoding="utf-8") as wf:
        ds = load_dataset(dataset_builder, data_files=files, streaming=True)
        if "train" in ds:
            ds = ds["train"]
        for example in ds:
            text = example.get("text", "")
            wf.write(text.replace("\n", " ") + "\n")


def train_bpe_tokenizer(text_file: str, vocab_size: int = 12000) -> PreTrainedTokenizerFast:
    """Train a local ByteLevel BPE tokenizer. Only used as a fallback."""

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[text_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|start_of_text|>",
            "<pad>",
            "<|end_of_text|>",
            "<unk>",
            "<mask>",
        ],
    )

    os.makedirs("bpe_tokenizer", exist_ok=True)
    tokenizer.save_model("bpe_tokenizer")

    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())

    tokenizer_config = {
        "model_max_length": 2048,
        "bos_token": "<|start_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
    with open(os.path.join("bpe_tokenizer", "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join("bpe_tokenizer", "tokenizer.json"),
        bos_token="<|start_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    hf_tokenizer.save_pretrained("bpe_tokenizer")
    return hf_tokenizer


def load_bpe_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer", use_fast=True)


def load_tokenizer(tokenizer_name_or_path: str, trust_remote_code: bool = False):
    """Load an existing tokenizer from disk. Assumes offline availability."""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    return tokenizer


#####################################
# Dataset utilities
#####################################


def streaming_token_generator(
    data_files: Iterable[str],
    hf_tokenizer,
    block_size: int,
    min_length: int = 0,
) -> Iterable[List[int]]:
    files, dataset_builder = _prepare_data_files(data_files)
    dataset = load_dataset(dataset_builder, data_files=files, streaming=True)
    if "train" in dataset:
        dataset = dataset["train"]

    for example in dataset:
        text = example.get("text", "") if isinstance(example, dict) else ""
        token_ids = hf_tokenizer.encode(text, add_special_tokens=False)
        for chunk in chunk_tokens(token_ids, block_size):
            if len(chunk) > max(min_length, 0):
                yield chunk


def load_nonstream_data(
    data_files: Iterable[str],
    hf_tokenizer,
    block_size: int,
    num_proc: int = 8,
    min_length: int = 0,
):
    processed_dir = os.path.join("processed_data", f"tokenized_data_bs{block_size}")
    if os.path.exists(processed_dir):
        print(f"Loading cached dataset from '{processed_dir}'...")
        ds = load_from_disk(processed_dir)
        return ds["token_ids"]

    print("No cached dataset found. Processing in parallel...")
    files, dataset_builder = _prepare_data_files(data_files)
    ds_dict = load_dataset(dataset_builder, data_files=files, streaming=False)
    ds = ds_dict["train"] if "train" in ds_dict else ds_dict

    all_chunks: List[List[int]] = []
    for example in tqdm(ds, desc="Tokenizing", leave=False):
        text = example.get("text", "")
        token_ids = hf_tokenizer.encode(text, add_special_tokens=False)
        for chunk in chunk_tokens(token_ids, block_size):
            if len(chunk) >= max(block_size + 1, min_length):
                all_chunks.append(chunk)

    if not all_chunks:
        print("No sequences met the minimum length requirement after tokenization.")
        return []

    token_dataset = Dataset.from_dict({"token_ids": all_chunks})

    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    token_dataset.save_to_disk(processed_dir)
    print(f"Processed dataset saved to '{processed_dir}'.")
    return token_dataset["token_ids"]


def collate_batch(token_list_batch: Iterable[List[int]], block_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    x_list: List[List[int]] = []
    y_list: List[List[int]] = []
    for tokens in token_list_batch:
        if len(tokens) < block_size + 1:
            continue
        tokens = tokens[: block_size + 1]
        x_list.append(tokens[:-1])
        y_list.append(tokens[1:])

    if not x_list:
        return None, None

    x_tensor = torch.tensor(x_list, dtype=torch.long)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_tensor, y_tensor

def chunk_tokens(
    tokens: List[int],
    block_size: int,
    *,
    stride: Optional[int] = None,
) -> Iterator[List[int]]:
    """Yield non-overlapping chunks of tokens with length block_size + 1."""

    chunk_len = block_size + 1
    if chunk_len <= 1 or len(tokens) < chunk_len:
        return

    if stride is None:
        stride = block_size

    if stride <= 0:
        raise ValueError("stride must be positive")

    max_start = len(tokens) - chunk_len
    for start in range(0, max_start + 1, stride):
        chunk = tokens[start : start + chunk_len]
        if len(chunk) == chunk_len:
            yield chunk

