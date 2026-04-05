#!/usr/bin/env python3
"""Push an Argonne checkpoint to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import CommitOperationDelete, HfApi, create_repo


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024
DEFAULT_REPO_ID = "PursuitOfDataScience/Argonne2.5-base"
DEFAULT_MODEL_NAME = "Argonne 2.5-base"


MODEL_CARD_TEMPLATE = """---
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

# {model_name}

{model_name} is a decoder-only transformer language model trained on a mixture of FineWeb and FineWeb-Edu data.

## Model architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | {parameter_count:,} (~1.27B) |
| **Layers** | 28 transformer blocks |
| **Hidden size** | 1,792 |
| **Attention heads** | 14 query / 7 key-value (GQA) |
| **Head dimension** | 128 |
| **Feed-forward** | SwiGLU MLP, 4,864 intermediate dim |
| **Context length** | 1,024 tokens |
| **Vocabulary size** | 151,669 |
| **Normalization** | RMSNorm (ε = 1e-6) |
| **Position encoding** | RoPE (θ = 10,000) |

## Training details

| Item | Value |
|------|-------|
| **Total steps** | 425,975 |
| **Tokens processed** | ~76.05B |
| **Final train loss** | 2.6119 |
| **Sequence length** | 1,024 |
| **Batch size per GPU** | 20 |
| **Gradient accumulation** | 4 |
| **Effective batch** | 245,760 tokens |
| **Learning rate** | 3e-4 |
| **Min LR ratio** | 0.1 |
| **Warmup** | 0 steps |
| **Precision** | bf16 autocast |
| **torch.compile** | Enabled |
| **GPUs** | 3 (DDP) |

## Training data

- FineWeb
- FineWeb-Edu
- Final stage training shard: 55.2B tokens
- Cumulative training across the full run: 76.05B tokens

{loss_curve_section}

## Inference

```python
{inference_snippet}
```

## Usage notes

- Load with `trust_remote_code=True`.
- The custom `generate` method uses `max_length` rather than `max_new_tokens`.
- Switch to greedy decoding if you want deterministic output.

## Citation

```bibtex
@misc{{argonne25,
  author = {{PursuitOfDataScience}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Push Argonne weights to the Hugging Face Hub.")
    parser.add_argument("--model-dir", required=True, help="Path to the saved checkpoint directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Human-readable model name.")
    parser.add_argument("--plot-path", default=None, help="Optional loss-curve image to include in the repo.")
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare files without uploading.")
    parser.add_argument("--shard-size-gb", type=float, default=5.0, help="Maximum shard size in GB.")
    parser.add_argument("--commit-message", default="Upload Argonne 2.5-base", help="Upload commit message.")
    return parser.parse_args()


def count_tensors_in_safetensors(files):
    from safetensors import safe_open

    total = 0
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_tensor(key).shape
                tensor_elements = 1
                for dim in shape:
                    tensor_elements *= dim
                total += tensor_elements
    return total


def copy_safetensors_or_convert(model_path, temp_path, max_shard_bytes):
    from safetensors.torch import save_file as safetensors_save

    safetensor_files = sorted(
        path for path in model_path.glob("*.safetensors") if not path.name.endswith(".index.json")
    )
    bin_files = sorted(model_path.glob("pytorch_model*.bin"))

    if safetensor_files:
        for file_path in safetensor_files:
            shutil.copy2(file_path, temp_path / file_path.name)
        index_file = model_path / "model.safetensors.index.json"
        if index_file.is_file():
            shutil.copy2(index_file, temp_path / index_file.name)
        return count_tensors_in_safetensors(temp_path.glob("*.safetensors"))

    if not bin_files:
        raise SystemExit(f"No weight files found in {model_path}")

    full_state_dict = {}
    for file_path in bin_files:
        shard = torch.load(file_path, map_location="cpu", weights_only=False)
        full_state_dict.update(shard)

    storage_to_keys = {}
    for key, tensor in full_state_dict.items():
        if isinstance(tensor, torch.Tensor):
            storage_id = tensor.data_ptr()
            storage_to_keys.setdefault(storage_id, []).append(key)

    for keys in storage_to_keys.values():
        if len(keys) > 1:
            for key in keys[1:]:
                full_state_dict[key] = full_state_dict[key].clone()

    tensor_sizes = {
        key: value.numel() * value.element_size()
        for key, value in full_state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    sorted_keys = sorted(tensor_sizes, key=lambda key: tensor_sizes[key], reverse=True)

    shards = []
    current_shard = {}
    current_size = 0
    for key in sorted_keys:
        tensor_size = tensor_sizes[key]
        if current_shard and current_size + tensor_size > max_shard_bytes:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = full_state_dict[key]
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    total_params = sum(tensor.numel() for tensor in full_state_dict.values() if isinstance(tensor, torch.Tensor))
    total_shards = len(shards)
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        shard_name = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
        safetensors_save(shard, str(temp_path / shard_name))
        for key in shard:
            weight_map[key] = shard_name

    index_path = temp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_size": sum(tensor_sizes.values())}, "weight_map": weight_map}, indent=2)
        + "\n"
    )
    return total_params


def copy_plot(plot_path, temp_path):
    if plot_path is None:
        return None

    source = Path(plot_path).expanduser()
    if not source.is_file():
        raise SystemExit(f"Plot not found: {source}")

    repo_path = Path("plots") / source.name
    destination = temp_path / repo_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(repo_path).replace(os.sep, "/")


def build_inference_snippet(repo_id):
    return f"""from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

prompt = "Write a short paragraph about scientific computing at Argonne National Laboratory."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 128,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
"""


def write_model_card(temp_path, repo_id, model_name, parameter_count, plot_repo_path):
    inference_snippet = build_inference_snippet(repo_id)
    loss_curve_section = ""
    if plot_repo_path:
        loss_curve_section = f"## Loss curve\n\n![Training loss curve]({plot_repo_path})\n"

    readme = MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        repo_id=repo_id,
        parameter_count=parameter_count,
        inference_snippet=inference_snippet,
        loss_curve_section=loss_curve_section,
    )
    (temp_path / "README.md").write_text(readme)


def prepare_upload_folder(model_dir, repo_id, model_name, max_shard_bytes, plot_path):
    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory not found: {model_path}")

    temp_dir = tempfile.mkdtemp(prefix="argonne_upload_")
    temp_path = Path(temp_dir)

    skip_names = {"README.md", "model.py", ".gitattributes"}
    skip_prefixes = ("pytorch_model",)

    for item in model_path.iterdir():
        if item.name in skip_names or item.name.startswith(skip_prefixes):
            continue
        if item.name.endswith(".safetensors") or item.name.endswith(".index.json"):
            continue
        if item.is_file():
            shutil.copy2(item, temp_path / item.name)
        elif item.is_dir() and item.name not in {"__pycache__", ".git"}:
            shutil.copytree(item, temp_path / item.name)

    parameter_count = copy_safetensors_or_convert(model_path, temp_path, max_shard_bytes)

    model_py = SCRIPT_DIR / "model.py"
    if model_py.is_file():
        shutil.copy2(model_py, temp_path / "model.py")

    plot_repo_path = copy_plot(plot_path, temp_path)
    write_model_card(temp_path, repo_id, model_name, parameter_count, plot_repo_path)
    return temp_dir


def clean_old_files(api, repo_id):
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    to_delete = [path for path in repo_files if path.endswith(".bin") or path == ".gitattributes"]
    if not to_delete:
        return

    operations = [CommitOperationDelete(path_in_repo=path) for path in to_delete]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Remove legacy files",
    )


def push_to_hub(upload_folder, repo_id, private, commit_message, dry_run):
    if dry_run:
        print("Prepared files:")
        for item in sorted(Path(upload_folder).rglob("*")):
            if item.is_file():
                print(f"  {item.relative_to(upload_folder)}")
        return

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    clean_old_files(api, repo_id)
    api.upload_folder(
        folder_path=upload_folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def main():
    args = parse_args()
    max_shard_bytes = int(args.shard_size_gb * 1024 * 1024 * 1024)

    upload_folder = prepare_upload_folder(
        args.model_dir,
        args.repo_id,
        args.model_name,
        max_shard_bytes,
        args.plot_path,
    )

    try:
        push_to_hub(upload_folder, args.repo_id, args.private, args.commit_message, args.dry_run)
    finally:
        shutil.rmtree(upload_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
