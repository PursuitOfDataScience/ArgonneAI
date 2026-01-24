#!/usr/bin/env python3
"""
Push Argonne model to Hugging Face Hub with sharded safetensors.

Converts pytorch .bin files to sharded safetensors (~5GB per shard) and
cleans up old .bin files from the HF repo.

Usage:
    python push_model_to_hf.py --model-dir /path/to/model --repo-id user/model-name
"""

import argparse
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from huggingface_hub import HfApi, create_repo, upload_folder, CommitOperationDelete
import torch

MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB per shard


MODEL_CARD_TEMPLATE = '''---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- text-generation
- causal-lm
- transformer
- argonne
- pretrained
pipeline_tag: text-generation
---

# Argonne 2.0

A **4.9 billion parameter** decoder-only transformer language model trained from scratch.

## Model Architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | ~4.9B |
| **Layers** | 24 transformer blocks |
| **Hidden Size** | 4,080 |
| **Attention Heads** | 24 query / 8 key-value (GQA) |
| **Context Length** | 4,096 tokens |
| **Vocabulary Size** | 151,665 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License

Apache 2.0

## Citation

```bibtex
@misc{{argonne2,
  author = {{PursuitOfDataScience}},
  title = {{Argonne 2.0: A 4.9B Parameter Language Model}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## Links

- GitHub: [PursuitOfDataScience](https://github.com/PursuitOfDataScience)
- Hugging Face: [PursuitOfDataScience](https://huggingface.co/PursuitOfDataScience)
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Push Argonne model to HF Hub")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID (e.g., user/model-name)")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--dry-run", action="store_true", help="Prepare files but don't upload")
    parser.add_argument("--shard-size-gb", type=float, default=5.0, help="Max shard size in GB (default: 5)")
    parser.add_argument("--commit-message", type=str, default="Upload model with sharded safetensors")
    return parser.parse_args()


def convert_to_sharded_safetensors(model_path, temp_path, max_shard_bytes):
    """Convert pytorch weights to sharded safetensors format."""
    from safetensors.torch import save_file as safetensors_save
    
    # Load all weights
    bin_files = sorted(model_path.glob("pytorch_model*.bin"))
    safetensor_files = sorted(model_path.glob("model*.safetensors"))
    
    # If safetensors already exist, just copy them
    if safetensor_files:
        print("  Safetensors already exist, copying...")
        for sf in safetensor_files:
            shutil.copy2(sf, temp_path / sf.name)
            print(f"    Copied: {sf.name}")
        index_file = model_path / "model.safetensors.index.json"
        if index_file.is_file():
            shutil.copy2(index_file, temp_path / index_file.name)
        return
    
    if not bin_files:
        print("  Warning: No weight files found")
        return
    
    print(f"  Loading {len(bin_files)} pytorch files...")
    full_state_dict = {}
    for bf in bin_files:
        print(f"    Loading: {bf.name}")
        shard_dict = torch.load(bf, map_location="cpu", weights_only=False)
        full_state_dict.update(shard_dict)
    
    # Clone shared tensors to avoid safetensors error
    storage_to_keys = {}
    for key, tensor in full_state_dict.items():
        if isinstance(tensor, torch.Tensor):
            storage_id = tensor.data_ptr()
            if storage_id not in storage_to_keys:
                storage_to_keys[storage_id] = []
            storage_to_keys[storage_id].append(key)
    
    for storage_id, keys in storage_to_keys.items():
        if len(keys) > 1:
            print(f"    Found shared tensors: {keys}, cloning...")
            for key in keys[1:]:
                full_state_dict[key] = full_state_dict[key].clone()
    
    # Calculate tensor sizes and create shards
    tensor_sizes = {k: v.numel() * v.element_size() for k, v in full_state_dict.items()}
    total_size = sum(tensor_sizes.values())
    print(f"  Total model size: {total_size / 1e9:.2f} GB")
    
    # Sort tensors by size (largest first) for better packing
    sorted_keys = sorted(tensor_sizes.keys(), key=lambda k: tensor_sizes[k], reverse=True)
    
    # Create shards
    shards = []
    current_shard = {}
    current_size = 0
    
    for key in sorted_keys:
        tensor_size = tensor_sizes[key]
        if current_size + tensor_size > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = full_state_dict[key]
        current_size += tensor_size
    
    if current_shard:
        shards.append(current_shard)
    
    print(f"  Creating {len(shards)} shards...")
    
    # Save shards and create index
    weight_map = {}
    total_shards = len(shards)
    
    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        shard_path = temp_path / shard_name
        print(f"    Saving: {shard_name} ({sum(t.numel() * t.element_size() for t in shard.values()) / 1e9:.2f} GB, {len(shard)} tensors)")
        safetensors_save(shard, str(shard_path))
        for key in shard.keys():
            weight_map[key] = shard_name
    
    # Create index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    index_path = temp_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"    Created: model.safetensors.index.json")


def prepare_upload_folder(model_dir, repo_id, max_shard_bytes):
    """Prepare folder with safetensors and other files."""
    model_path = Path(model_dir)
    temp_dir = tempfile.mkdtemp(prefix="argonne_upload_")
    temp_path = Path(temp_dir)
    
    print(f"Preparing upload folder: {temp_dir}")
    
    # Files to skip
    skip_patterns = {"pytorch_model*.bin", "pytorch_model.bin.index.json", ".gitattributes"}
    
    def should_skip(name):
        for pattern in skip_patterns:
            if "*" in pattern:
                import fnmatch
                if fnmatch.fnmatch(name, pattern):
                    return True
            elif name == pattern:
                return True
        return False
    
    # Copy non-weight files
    for item in model_path.iterdir():
        if item.is_file():
            if should_skip(item.name):
                print(f"  Skipping: {item.name}")
                continue
            shutil.copy2(item, temp_path / item.name)
            print(f"  Copied: {item.name}")
        elif item.is_dir() and item.name not in ["__pycache__", ".git"]:
            shutil.copytree(item, temp_path / item.name)
            print(f"  Copied directory: {item.name}/")
    
    # Convert weights to sharded safetensors
    convert_to_sharded_safetensors(model_path, temp_path, max_shard_bytes)
    
    # Copy model.py for trust_remote_code
    model_py = Path(SCRIPT_DIR) / "model.py"
    if model_py.is_file():
        shutil.copy2(model_py, temp_path / "model.py")
        print(f"  Copied: model.py")
    
    # Create README
    readme = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
    (temp_path / "README.md").write_text(readme)
    print(f"  Created: README.md")
    
    return temp_dir


def clean_old_files(api, repo_id):
    """Delete old .bin files from HF repo."""
    print(f"\nCleaning old files from {repo_id}...")
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        to_delete = [f for f in repo_files if f.endswith(".bin") or f == "pytorch_model.bin.index.json" or f == ".gitattributes"]
        
        if to_delete:
            print(f"  Deleting {len(to_delete)} old files:")
            for f in to_delete:
                print(f"    - {f}")
            operations = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
            api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                            commit_message="Remove old pytorch .bin files")
            print("  Done!")
        else:
            print("  No old files to delete.")
    except Exception as e:
        print(f"  Note: {e}")


def push_to_hub(upload_folder, repo_id, private, commit_message, dry_run):
    """Upload to HF Hub."""
    if dry_run:
        print("\n" + "=" * 60 + "\nDRY RUN - Files prepared:\n" + "=" * 60)
        for item in sorted(Path(upload_folder).rglob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(upload_folder)} ({size_mb:.2f} MB)")
        print("\nRun without --dry-run to upload.")
        return
    
    api = HfApi()
    
    print(f"\nCreating/accessing repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"  Ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Clean old files first
    clean_old_files(api, repo_id)
    
    # Upload
    print(f"\nUploading to {repo_id}...")
    api.upload_folder(folder_path=upload_folder, repo_id=repo_id, repo_type="model", commit_message=commit_message)
    
    print("\n" + "=" * 60)
    print("SUCCESS! Model uploaded to Hugging Face Hub")
    print("=" * 60)
    print(f"URL: https://huggingface.co/{repo_id}")


def main():
    args = parse_args()
    max_shard_bytes = int(args.shard_size_gb * 1024 * 1024 * 1024)
    
    print("=" * 60)
    print("Argonne Model - Push to Hugging Face Hub")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Repo: {args.repo_id}")
    print(f"Shard size: {args.shard_size_gb} GB")
    print()
    
    upload_folder = prepare_upload_folder(args.model_dir, args.repo_id, max_shard_bytes)
    
    try:
        push_to_hub(upload_folder, args.repo_id, args.private, args.commit_message, args.dry_run)
    finally:
        if not args.dry_run:
            print(f"\nCleaning up: {upload_folder}")
            shutil.rmtree(upload_folder, ignore_errors=True)
        else:
            print(f"\nFiles preserved at: {upload_folder}")


if __name__ == "__main__":
    main()
