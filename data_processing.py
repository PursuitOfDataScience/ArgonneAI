import os
import json
import torch
import intel_extension_for_pytorch as ipex
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

os.environ["HF_DATASETS_CACHE"] = "./.cache"
# Set Intel memory configuration for data processing
os.environ["IPEX_XPU_ALLOC_CONF"] = "expandable_segments:True"

#####################################
# BPE Tokenizer Utilities
#####################################

def create_text_file_from_arrow(arrow_files, output_file="all_text_for_tokenizer.txt"):
    """
    Given a list of Arrow files, extract the 'text' column and write
    it to a single text file (one text example per line).
    """
    print(f"Creating a combined text file '{output_file}' from Arrow files...")
    with open(output_file, "w", encoding="utf-8") as wf:
        for arrow_path in tqdm(arrow_files):
            # Load the Arrow file in *streaming* mode to avoid large memory usage
            ds = load_dataset("arrow", data_files=[arrow_path], streaming=True)
            # If "train" split exists, use ds["train"], else ds is the dataset
            if "train" in ds:
                ds = ds["train"]
            for example in ds:
                text = example.get("text", "")
                # Write one line of text
                wf.write(text.replace("\n", " ") + "\n")

def train_bpe_tokenizer(text_file, vocab_size=12000):
    """
    Train a ByteLevel BPE tokenizer on a *plain-text file* and save it.
    """
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
            "<mask>"
        ]
    )

    os.makedirs("bpe_tokenizer", exist_ok=True)
    tokenizer.save_model("bpe_tokenizer")

    # Save the full tokenizer JSON representation
    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())

    # Create a tokenizer configuration
    tokenizer_config = {
        "model_max_length": 2048,
        "bos_token": "<|start_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    }
    with open(os.path.join("bpe_tokenizer", "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    # Create a Hugging Face PreTrainedTokenizerFast instance
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join("bpe_tokenizer", "tokenizer.json"),
        bos_token="<|start_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    hf_tokenizer.save_pretrained("bpe_tokenizer")
    return hf_tokenizer


def load_bpe_tokenizer():
    """Load a previously trained BPE tokenizer in Hugging Face format."""
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer", use_fast=True)
    return hf_tokenizer

#####################################
# STREAMING MODE
#####################################

def streaming_token_generator(data_files, hf_tokenizer):
    """
    Yields tokenized examples from a streaming dataset (no shuffle).
    data_files should be a list of Arrow files.
    """
    dataset = load_dataset("arrow", data_files=data_files, streaming=True)
    if "train" in dataset:
        dataset = dataset["train"]

    for example in dataset:
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) > 0:
            yield token_ids

#####################################
# NON-STREAMING: Full Pass
#####################################

def load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=8):
    """
    Loads the entire dataset in memory either from a cached processed directory
    or processes it in parallel if not yet cached.
    Returns a list of token ID sequences.
    """

    processed_dir = "processed_data/tokenized_data"
    if os.path.exists(processed_dir):
        print(f"Loading cached dataset from '{processed_dir}'...")
        ds = load_from_disk(processed_dir)
        tokenized_data = ds["token_ids"]
        return tokenized_data

    print("No cached dataset found. Processing in parallel...")

    ds_dict = load_dataset("arrow", data_files=data_files, streaming=False)
    if "train" in ds_dict:
        ds = ds_dict["train"]
    else:
        ds = ds_dict

    def tokenize_and_truncate(example):
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) < block_size + 1:
            return {"token_ids": None}
        token_ids = token_ids[:block_size+1]
        return {"token_ids": token_ids}

    ds = ds.map(
        tokenize_and_truncate,
        batched=False,
        num_proc=num_proc
    )
    ds = ds.filter(lambda ex: ex["token_ids"] is not None,
                   num_proc=num_proc)

    if "text" in ds.column_names:
        ds = ds.remove_columns(["text"])

    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    ds.save_to_disk(processed_dir)
    print(f"Processed dataset saved to '{processed_dir}'.")

    tokenized_data = ds["token_ids"]
    return tokenized_data

def collate_batch(token_list_batch, block_size):
    """
    Convert a list of token-ID lists into x,y Tensors for causal LM.
    We'll truncate if longer than block_size+1, skip if shorter.
    """
    x_list, y_list = [], []
    for tokens in token_list_batch:
        if len(tokens) < block_size + 1:
            continue
        tokens = tokens[:block_size+1]
        x_list.append(tokens[:-1])
        y_list.append(tokens[1:])

    if not x_list:
        return None, None

    x_tensor = torch.tensor(x_list, dtype=torch.long)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_tensor, y_tensor