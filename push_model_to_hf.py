#!/usr/bin/env python3
"""
Push Argonne model to Hugging Face Hub.

This script uploads a trained Argonne model to the Hugging Face Hub,
including model weights, tokenizer, config, and a comprehensive README.

Usage:
    python push_model_to_hf.py --model-dir /path/to/model --repo-id PursuitOfDataScience/Argonne-2.0

Requirements:
    pip install huggingface_hub transformers

You must be logged in to Hugging Face:
    huggingface-cli login
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add the ArgonneAI directory to path for model imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

# Import and register the Argonne model
try:
    from model import ArgonneConfig, ArgonneModel
    
    # Register the model with Transformers Auto classes
    try:
        AutoConfig.register("argonne2", ArgonneConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(ArgonneConfig, ArgonneModel)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(ArgonneConfig, ArgonneModel)
    except ValueError:
        pass
except ImportError as e:
    print(f"Warning: Could not import Argonne model: {e}")


# Model card template for Argonne 2.0
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

A **4.9 billion parameter** decoder-only transformer language model trained from scratch using tensor parallelism on a single DGX A100 node.

## Model Description

Argonne 2.0 is a large language model. It was pretrained on ~22 billion tokens from FineWeb (CC-MAIN-2025-26) using custom tensor parallelism implementation.

## Training Loss Curve

The model was trained on **~22 billion tokens** from FineWeb (CC-MAIN-2025-26), achieving a final loss of approximately **2.5–3.5** after 1.35 million training steps.

## Model Architecture

| Component | Specification |
|-----------|--------------|
| **Parameters** | 4,918,072,800 (~4.9B) |
| **Layers** | 24 transformer blocks |
| **Hidden Size** | 4,080 |
| **Attention Heads** | 24 query heads / 8 key-value heads (Grouped-Query Attention) |
| **Head Dimension** | 170 |
| **Feed-Forward** | SwiGLU MLP (~10,880 intermediate dim) |
| **Context Length** | 4,096 tokens |
| **Vocabulary Size** | 151,665 (Qwen2.5-3B-Instruct tokenizer) |
| **Normalization** | RMSNorm (ε = 1e-6) |
| **Position Encoding** | Rotary Position Embeddings (RoPE) |
| **Precision** | bfloat16 mixed precision |

### Key Architectural Features

- **Grouped-Query Attention (GQA)**: Uses 24 query heads with 8 key-value heads (3:1 ratio), reducing memory bandwidth requirements while maintaining model quality.
- **SwiGLU Activation**: Employs the SwiGLU activation function in the MLP layers for improved training dynamics.
- **Flash Attention 2**: Uses FlashAttention 2 kernels when available for higher throughput and lower memory use.
- **RoPE**: Rotary position embeddings enable better length generalization compared to absolute positional encodings.

## Training Details

### Hardware Configuration

- **Node**: 1× DGX A100 (8× NVIDIA A100 80GB GPUs)
- **Parallelism**: Tensor parallelism across 8 GPUs
- **Interconnect**: NVLink for high-bandwidth GPU-to-GPU communication

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Total Training Steps** | 1,347,890 |
| **Tokens Processed** | ~21.9 billion |
| **Micro-batch Size** | 2–4 per GPU |
| **Gradient Accumulation** | 4 steps |
| **Effective Batch Size** | ~64–128 sequences |
| **Learning Rate** | 1e-4 (peak) → 1e-5 (cosine decay) |
| **Warmup Steps** | 2,000 |
| **Weight Decay** | 0.1 |
| **Gradient Clipping** | 1.0 |
| **Optimizer** | AdamW (fused) |

### Training Data

The model was trained on **FineWeb (CC-MAIN-2025-26)** data:
- 250 Parquet shards streamed sequentially
- Documents tokenized with BOS/EOS boundary markers
- Aggressive filtering of low-quality content
- Chunked into 4,096-token sequences

## Usage

### Installation

```bash
pip install transformers torch
```

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "{repo_id}"

# Load the model with trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_length=256,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Sample Generation

**Prompt:** "The meaning of life is"

> The meaning of life is tamed in many ways. It is a state of mental and physical development. It is a state of deep emotional strength and confidence, and it is a state of physical and mental balance...

## Loss Progression

| Milestone | Steps | Tokens | Loss |
|-----------|-------|--------|------|
| Start | 0 | 0 | ~9.3 |
| 1K steps | 1,000 | 16M | ~6.5 |
| 10K steps | 10,000 | 164M | ~5.5 |
| 100K steps | 100,000 | 1.6B | ~4.0 |
| 500K steps | 500,000 | 8.2B | ~3.5 |
| 1M steps | 1,000,000 | 16.4B | ~3.0 |
| Final | 1,347,890 | 21.9B | ~2.5–3.5 |

## Limitations

- This is a base pretrained model and has not been instruction-tuned or aligned with RLHF.
- The model may generate incorrect, biased, or harmful content.
- Performance on specific tasks may vary; fine-tuning is recommended for production use.

## License

This model is released under the Apache 2.0 License.

## Citation

```bibtex
@misc{{argonne2,
  author = {{Yu, Youzhi}},
  title = {{Argonne 2.0: A 4.9B Parameter Language Model}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## Author

- GitHub: [PursuitOfDataScience](https://github.com/PursuitOfDataScience)
- Hugging Face: [PursuitOfDataScience](https://huggingface.co/PursuitOfDataScience)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push Argonne model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Push model to Hugging Face
    python push_model_to_hf.py --model-dir /path/to/model --repo-id PursuitOfDataScience/Argonne-2.0
    
    # Push with private repository
    python push_model_to_hf.py --model-dir /path/to/model --repo-id user/model-name --private
    
    # Dry run (prepare files but don't upload)
    python push_model_to_hf.py --model-dir /path/to/model --repo-id user/model-name --dry-run
        """
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        required=True,
        help="Path to the local model directory (containing config.json, model weights, tokenizer files)"
    )
    parser.add_argument(
        "--repo-id", 
        type=str, 
        required=True,
        help="Hugging Face repository ID (e.g., 'PursuitOfDataScience/Argonne-2.0')"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Prepare files but don't upload to Hugging Face"
    )
    parser.add_argument(
        "--commit-message", 
        type=str, 
        default="Upload Argonne 2.0 model",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--include-model-py",
        action="store_true",
        default=True,
        help="Include model.py for trust_remote_code support (default: True)"
    )
    parser.add_argument(
        "--no-include-model-py",
        action="store_false",
        dest="include_model_py",
        help="Do not include model.py"
    )
    return parser.parse_args()


def validate_model_dir(model_dir: str) -> None:
    """Validate that the model directory contains required files."""
    model_path = Path(model_dir)
    
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for required files
    required_files = ["config.json"]
    for fname in required_files:
        if not (model_path / fname).is_file():
            raise FileNotFoundError(f"Required file '{fname}' not found in {model_dir}")
    
    # Check for model weights (various possible formats)
    weight_patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "model-*.safetensors",  # Sharded
        "pytorch_model-*.bin",  # Sharded
    ]
    
    has_weights = False
    for pattern in weight_patterns:
        if list(model_path.glob(pattern)):
            has_weights = True
            break
    
    if not has_weights:
        print("Warning: No model weight files found. Expected one of:")
        for pattern in weight_patterns:
            print(f"  - {pattern}")
    
    # Check for tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
    has_tokenizer = any((model_path / f).is_file() for f in tokenizer_files)
    
    if not has_tokenizer:
        print("Warning: No tokenizer files found.")


def prepare_upload_folder(model_dir: str, repo_id: str, include_model_py: bool) -> str:
    """
    Prepare a temporary folder with all files to upload.
    
    Returns the path to the temporary folder.
    """
    model_path = Path(model_dir)
    temp_dir = tempfile.mkdtemp(prefix="argonne_upload_")
    temp_path = Path(temp_dir)
    
    print(f"Preparing upload folder: {temp_dir}")
    
    # Copy all files from model directory
    for item in model_path.iterdir():
        if item.is_file():
            shutil.copy2(item, temp_path / item.name)
            print(f"  Copied: {item.name}")
        elif item.is_dir() and item.name not in ["__pycache__", ".git"]:
            shutil.copytree(item, temp_path / item.name)
            print(f"  Copied directory: {item.name}/")
    
    # Include model.py for trust_remote_code support
    if include_model_py:
        model_py_path = Path(SCRIPT_DIR) / "model.py"
        if model_py_path.is_file():
            shutil.copy2(model_py_path, temp_path / "model.py")
            print(f"  Copied: model.py (for trust_remote_code)")
        else:
            print(f"  Warning: model.py not found at {model_py_path}")
    
    # Create README.md (model card)
    readme_content = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
    readme_path = temp_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  Created: README.md (model card)")
    
    return temp_dir


def push_to_hub(
    upload_folder: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload model",
    dry_run: bool = False,
) -> None:
    """Push the prepared folder to Hugging Face Hub."""
    
    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Files prepared but not uploaded")
        print("=" * 60)
        print(f"Upload folder: {upload_folder}")
        print(f"Repository: {repo_id}")
        print(f"Private: {private}")
        print("\nFiles to upload:")
        for item in sorted(Path(upload_folder).rglob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(upload_folder)} ({size_mb:.2f} MB)")
        print("\nTo actually upload, run without --dry-run")
        return
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    print(f"\nCreating/accessing repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"  Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Upload the folder
    print(f"\nUploading files to {repo_id}...")
    print("This may take a while for large models...")
    
    try:
        api.upload_folder(
            folder_path=upload_folder,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print("\n" + "=" * 60)
        print("SUCCESS! Model uploaded to Hugging Face Hub")
        print("=" * 60)
        print(f"Repository URL: https://huggingface.co/{repo_id}")
        print(f"\nTo use the model:")
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
""")
    except Exception as e:
        print(f"\nError uploading to Hugging Face: {e}")
        raise


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("Argonne Model - Push to Hugging Face Hub")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Repository ID: {args.repo_id}")
    print(f"Private: {args.private}")
    print(f"Include model.py: {args.include_model_py}")
    print()
    
    # Validate model directory
    print("Validating model directory...")
    validate_model_dir(args.model_dir)
    print("  Validation passed!")
    print()
    
    # Prepare upload folder
    upload_folder = prepare_upload_folder(
        args.model_dir,
        args.repo_id,
        args.include_model_py,
    )
    
    try:
        # Push to hub
        push_to_hub(
            upload_folder=upload_folder,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
        )
    finally:
        # Cleanup temporary folder (unless dry run, for inspection)
        if not args.dry_run:
            print(f"\nCleaning up temporary folder: {upload_folder}")
            shutil.rmtree(upload_folder, ignore_errors=True)
        else:
            print(f"\nTemporary folder preserved for inspection: {upload_folder}")


if __name__ == "__main__":
    main()
