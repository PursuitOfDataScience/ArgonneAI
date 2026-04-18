#!/usr/bin/env python3
"""Push an Argonne checkpoint to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import CommitOperationDelete, HfApi, create_repo


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SOURCE_CODE_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/tree/main"
MODEL_CODE_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py"
SFT_SCRIPT_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/sft.py"
DPO_SHELL_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/dpo.sh"
SFT_SHELL_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/sft.sh"
DPO_SCRIPT_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/dpo.py"
INFERENCE_SCRIPT_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/inference.py"
MIDTRAINING_SCRIPT_URL = "https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/midtraining.py"
BASE_MODEL_URL = "https://huggingface.co/PursuitOfDataScience/Argonne2.5-base"
CTX13568_BASE_MODEL_URL = "https://huggingface.co/PursuitOfDataScience/Argonne-2.5-ctx13568"
LONGMINO_DATASET_URL = "https://huggingface.co/datasets/allenai/dolma3_longmino_pool"
ULTRACHAT_DATASET_URL = "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"
CHATBOT_ARENA_DATASET_URL = "https://huggingface.co/datasets/KatoHF/chatbot_arena_binarized"
CHECKPOINT_DTYPE = "bfloat16"
DEFAULT_WEIGHT_SHARD_COUNT = 5
TOKENIZER_NOTE = "This model uses the Qwen3 tokenizer family via the Qwen2Tokenizer compatibility class."
DEFAULT_BASE_REPO_ID = "PursuitOfDataScience/Argonne2.5-base"
DEFAULT_INSTRUCT_REPO_ID = "PursuitOfDataScience/Argonne2.5-instruct"
DEFAULT_MIDTRAINING_REPO_ID = "PursuitOfDataScience/Argonne-2.5-ctx13568"
DEFAULT_BASE_MODEL_NAME = "Argonne 2.5-base"
DEFAULT_INSTRUCT_MODEL_NAME = "Argonne 2.5-instruct"
DEFAULT_MIDTRAINING_MODEL_NAME = "Argonne-2.5-ctx13568"
DEFAULT_CTX13568_INSTRUCT_REPO_ID = "PursuitOfDataScience/Argonne-2.5-ctx13568-instruct"
DEFAULT_CTX13568_INSTRUCT_MODEL_NAME = "Argonne-2.5-ctx13568-instruct"
INSTRUCT_RECOMMENDED_ROWS = [
    ("**Context length**", "1,024 tokens"),
    ("**Temperature**", "0.8"),
    ("**Top-p**", "0.9"),
    ("**Repetition penalty**", "1.3"),
    ("**No-repeat n-gram size**", "4"),
    ("**Seed**", "444"),
]
CTX13568_INSTRUCT_RECOMMENDED_ROWS = [
    ("**Context length**", "13,568 tokens"),
    ("**Temperature**", "0.8"),
    ("**Top-p**", "0.9"),
    ("**Repetition penalty**", "1.3"),
    ("**No-repeat n-gram size**", "4"),
    ("**Seed**", "444"),
    ("**Continuation length**", "200 new tokens"),
]

ARCHITECTURE_ROWS = [
    ("**Parameters**", "~1.27B"),
    ("**Layers**", "28 transformer blocks"),
    ("**Hidden size**", "1,792"),
    ("**Attention heads**", "14 query / 7 key-value (GQA)"),
    ("**Head dimension**", "128"),
    ("**Feed-forward**", "SwiGLU MLP, 4,864 intermediate dim"),
    ("**Context length**", "1,024 tokens"),
    ("**Vocabulary size**", "151,669"),
    ("**Normalization**", "RMSNorm (ε = 1e-6)"),
    ("**Position encoding**", "RoPE (θ = 10,000)"),
]


def markdown_table(rows, headers=("Item", "Value")):
    lines = [f"| {headers[0]} | {headers[1]} |", f"|{'-' * (len(headers[0]) + 2)}|{'-' * (len(headers[1]) + 2)}|"]
    for left, right in rows:
        lines.append(f"| {left} | {right} |")
    return "\n".join(lines)


def build_inference_snippet(repo_id, max_new_tokens=128, temperature=0.8, top_p=0.95, top_k=50, do_sample=True):
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
    max_length=input_ids.shape[1] + {max_new_tokens},
    temperature={temperature},
    top_p={top_p},
    top_k={top_k},
    do_sample={str(do_sample)},
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
"""


def build_instruct_inference_snippet(repo_id):
    return f"""from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompt = "Write a short paragraph about scientific computing at Argonne National Laboratory."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

seed = 444
torch.manual_seed(seed)
if device.startswith("cuda"):
    torch.cuda.manual_seed_all(seed)

output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 128,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
"""


def build_ctx13568_instruct_inference_snippet(repo_id):
    return f"""from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

messages = [
    {{"role": "user", "content": "Explain what a black hole is in a way a 10-year-old would understand."}}
]
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

seed = 444
torch.manual_seed(seed)
if device.startswith("cuda"):
    torch.cuda.manual_seed_all(seed)

output_ids = model.generate(
    input_ids,
    max_length=min(model.config.max_position_embeddings, input_ids.shape[1] + 200),
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
)
gen_ids = output_ids[0, input_ids.shape[1]:].tolist()
eos_id = tokenizer.eos_token_id
if eos_id in gen_ids:
    gen_ids = gen_ids[: gen_ids.index(eos_id)]

reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
print(reply)
"""


def build_base_model_card(repo_id, model_name, parameter_count, plot_repo_path, shard_count):
    architecture_rows = [("**Parameters**", f"{parameter_count:,} (~1.27B)")] + ARCHITECTURE_ROWS[1:]
    training_rows = [
        ("**Total steps**", "425,975"),
        ("**Tokens processed**", "~76.05B"),
        ("**Final train loss**", "2.6119"),
        ("**Sequence length**", "1,024"),
        ("**Batch size per GPU**", "20"),
        ("**Gradient accumulation**", "4"),
        ("**Effective batch**", "245,760 tokens"),
        ("**Learning rate**", "3e-4"),
        ("**Min LR ratio**", "0.1"),
        ("**Warmup**", "1,000 steps"),
        ("**Precision**", "bf16 autocast"),
        ("**Checkpoint dtype**", CHECKPOINT_DTYPE),
        ("**Weight format**", f"{shard_count} sharded safetensors"),
        ("**torch.compile**", "Enabled"),
        ("**GPUs**", "3x H2000s (DDP)"),
    ]
    source_section = f"""## Source code

The release was built from the GitHub main branch codebase: {SOURCE_CODE_URL}

Key scripts:
- `pretrain.py`
- `continue_pretrain.py`
- `inference.py`
- `push_model_to_hf.py`

"""
    loss_curve_section = f"""## Loss curve

![Training loss curve]({plot_repo_path})

""" if plot_repo_path else ""
    return f"""---
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

{markdown_table(architecture_rows, ("Component", "Specification"))}

## Training details

{markdown_table(training_rows)}

## Training data

- FineWeb
- FineWeb-Edu
- Final stage training shard: 55.2B tokens
- Cumulative training across the full run: 76.05B tokens

## Tokenizer

{TOKENIZER_NOTE}

{source_section}{loss_curve_section}
## Inference

```python
{build_inference_snippet(repo_id)}
```

## Usage notes

- Load with `trust_remote_code=True`.
- The custom `generate` method uses `max_length` rather than `max_new_tokens`.
- Weights are published as {shard_count} bf16 safetensor shards.
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


def build_instruct_model_card(repo_id, model_name, shard_count):
    base_model_link = f"[Argonne 2.5-base]({BASE_MODEL_URL})"
    ultrachat_link = f"[HuggingFaceH4/ultrachat_200k]({ULTRACHAT_DATASET_URL})"
    chatbot_arena_link = f"[KatoHF/chatbot_arena_binarized]({CHATBOT_ARENA_DATASET_URL})"
    recommended_rows = markdown_table(INSTRUCT_RECOMMENDED_ROWS)
    source_section = f"""## Source code

The release was built from the GitHub main branch codebase: {SOURCE_CODE_URL}

Key scripts:
- [`sft.py`]({SFT_SCRIPT_URL})
- [`dpo.py`]({DPO_SCRIPT_URL})
- [`inference.py`]({INFERENCE_SCRIPT_URL})

"""
    recommended_section = f"""## Recommended inference config

{recommended_rows}

These settings are the recommended defaults for inference.

"""
    inference_section = f"""## Inference

```python
{build_instruct_inference_snippet(repo_id)}
```

"""

    return f"""---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- text-generation
- causal-lm
- transformer
- argonne
- instruct
- pretrained
pipeline_tag: text-generation
---

# {model_name}

{model_name} starts from the {base_model_link} checkpoint and is tuned in two stages.

## Training pipeline

First, supervised fine-tuning (SFT) adapts the base checkpoint on {ultrachat_link} using the `train_sft` split. That stage used NVIDIA H100 NVL hardware with 1,024-token sequences, batch size 24, gradient accumulation 2, learning rate 2e-5, and 100 warmup steps.

Second, direct preference optimization (DPO) refines the SFT checkpoint on {chatbot_arena_link} with the `chat_refine_strict` recipe. That stage used NVIDIA H100 PCIe hardware with 1,024-token sequences, batch size 4, gradient accumulation 8, learning rate 5e-6, beta 0.2, and 10 warmup steps.

The published checkpoint is stored in {CHECKPOINT_DTYPE} and split across {shard_count} safetensor shards for easier loading.

## Training data

- Base checkpoint: {base_model_link}
- SFT data: {ultrachat_link} (`train_sft`)
- DPO data: {chatbot_arena_link} (`chat_refine_strict`)

## Tokenizer

{TOKENIZER_NOTE}

{source_section}
{recommended_section}
{inference_section}

## Usage notes

- Load with `trust_remote_code=True`.
- The custom `generate` method accepts `repetition_penalty` and `no_repeat_ngram_size`.
- The sweep-derived repetition controls are available in the repository's custom
  generation loop, not the checkpoint's built-in `generate` method.
- Weights are published as {shard_count} bf16 safetensor shards.
- The instruct checkpoint inherits the base tokenizer and chat template.

## Citation

```bibtex
@misc{{argonne25instruct,
  author = {{PursuitOfDataScience}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def build_ctx13568_instruct_model_card(repo_id, model_name, parameter_count, shard_count):
    base_model_link = f"[PursuitOfDataScience/Argonne-2.5-ctx13568]({CTX13568_BASE_MODEL_URL})"
    ultrachat_link = f"[HuggingFaceH4/ultrachat_200k]({ULTRACHAT_DATASET_URL})"
    chatbot_arena_link = f"[KatoHF/chatbot_arena_binarized]({CHATBOT_ARENA_DATASET_URL})"
    architecture_rows = [
        ("**Parameters**", f"{parameter_count:,} (~1.27B)"),
        ("**Layers**", "28 transformer blocks"),
        ("**Hidden size**", "1,792"),
        ("**Attention heads**", "14 query / 7 key-value (GQA)"),
        ("**Context length**", "13,568 tokens"),
        ("**Vocabulary size**", "151,669"),
        ("**Position encoding**", "RoPE (θ = 10,000)"),
    ]
    recommended_rows = markdown_table(CTX13568_INSTRUCT_RECOMMENDED_ROWS)
    source_section = f"""## Source code

The release was built from the GitHub main branch codebase: {SOURCE_CODE_URL}

Key scripts:
- [`sft.py`]({SFT_SCRIPT_URL})
- [`dpo.py`]({DPO_SCRIPT_URL})
- [`model.py`]({MODEL_CODE_URL})

"""
    recommended_section = f"""## Recommended inference config

{recommended_rows}

These settings mirror the sampled generation path used in `dpo.py` for quality checks.

"""
    inference_section = f"""## Inference

```python
{build_ctx13568_instruct_inference_snippet(repo_id)}
```

"""
    return f"""---
license: apache-2.0
language:
- en
library_name: transformers
base_model: PursuitOfDataScience/Argonne-2.5-ctx13568
tags:
- text-generation
- causal-lm
- transformer
- argonne
- instruct
- long-context
- sft
- dpo
pipeline_tag: text-generation
---

# {model_name}

{model_name} starts from the long-context checkpoint {base_model_link} and is tuned in two stages: long-context SFT followed by DPO.

## Model architecture

{markdown_table(architecture_rows, ("Component", "Specification"))}

## Finetuning pipeline

Stage 1 was supervised fine-tuning on {ultrachat_link} using the local `train_sft` export at `/project/rcc/youzhi/data/HuggingFaceH4_ultrachat_200k/train_sft`. That run used `max_seq_length=13568`, `batch_size=1`, `grad_accum=4`, `lr=2e-5`, `num_epochs=1`, and `warmup_steps=100`, and produced the intermediate checkpoint `final_model_sft_long`.

Stage 2 was DPO on {chatbot_arena_link} using the local export at `/project/rcc/youzhi/data/KatoHF_chatbot_arena_binarized` with the `chat_refine_strict` recipe. That run used `max_seq_length=13568`, `batch_size=4`, `grad_accum=8`, `lr=5e-6`, `num_epochs=5`, `warmup_steps=10`, `beta=0.2`, `score_mode=avg`, `label_smoothing=0.0`, and `chosen_sft_weight=0.1`. The published checkpoint corresponds to `/project/rcc/youzhi/llm.c/checkpoints/final_model_dpo_long`.

The released weights are stored in {CHECKPOINT_DTYPE} and published as {shard_count} safetensor shards.

## Training data

- Base checkpoint: {base_model_link}
- SFT data: {ultrachat_link} (`train_sft`)
- DPO data: {chatbot_arena_link} (`chat_refine_strict`)

## Tokenizer

{TOKENIZER_NOTE}

{source_section}
{recommended_section}
{inference_section}

## Usage notes

- Load with `trust_remote_code=True`.
- Use the chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)` for instruct prompting.
- The custom `generate` method uses `max_length`, so the example trims the continuation at `tokenizer.eos_token_id` after generation.
- Weights are published as {shard_count} bf16 safetensor shards.
- The instruct checkpoint inherits the tokenizer and chat template from the long-context base model.

## Citation

```bibtex
@misc{{argonne25ctx13568instruct,
  author = {{PursuitOfDataScience}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def build_midtraining_model_card(repo_id, model_name, parameter_count, plot_repo_path, shard_count):
    base_model_link = f"[PursuitOfDataScience/Argonne2.5-base]({BASE_MODEL_URL})"
    longmino_link = f"[allenai/dolma3_longmino_pool]({LONGMINO_DATASET_URL})"
    architecture_rows = [
        ("**Parameters**", f"{parameter_count:,}"),
        ("**Layers**", "28 transformer blocks"),
        ("**Hidden size**", "1,792"),
        ("**Attention heads**", "14 query / 7 key-value (GQA)"),
        ("**Context length**", "13,568 tokens"),
        ("**Vocabulary size**", "151,669"),
        ("**Position encoding**", "RoPE (θ = 10,000)"),
    ]
    training_rows = [
        ("**Start checkpoint**", base_model_link),
        ("**Long-context tokens trained**", "16.0B tokens"),
        ("**Final cumulative tokens**", "92,050,960,384"),
        ("**Batch size per GPU**", "4"),
        ("**Gradient accumulation**", "1"),
        ("**Effective batch**", "108,544 tokens"),
        ("**Precision**", "bf16 autocast"),
        ("**Checkpoint dtype**", CHECKPOINT_DTYPE),
        ("**Weight format**", f"{shard_count} sharded safetensors"),
    ]
    source_section = f"""## Source code

The release was built from the GitHub main branch codebase: {SOURCE_CODE_URL}

Key scripts:
- [`midtraining.py`]({MIDTRAINING_SCRIPT_URL})

"""
    loss_curve_section = f"""## Loss curve

![Midtraining loss curve]({plot_repo_path})

""" if plot_repo_path else ""
    return f"""---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- text-generation
- causal-lm
- transformer
- argonne
- long-context
- continued-pretraining
pipeline_tag: text-generation
---

# {model_name}

{model_name} is a long-context continuation of {base_model_link}.

## Model architecture

{markdown_table(architecture_rows, ("Component", "Specification"))}

## Training details

{markdown_table(training_rows)}

## Long-context data

- Dataset source: {longmino_link}
- Selected subset: `16k-32k` LongMino pool
- Kept documents: 4,595,978
- Kept tokens (Qwen tokenizer): 105,223,923,033
- This release used the long-context stage only (no additional short-context stage in this run).

## Tokenizer

{TOKENIZER_NOTE}

{source_section}{loss_curve_section}
## Inference

```python
{build_inference_snippet(repo_id, max_new_tokens=256, temperature=0.8, top_p=0.9, top_k=50, do_sample=True)}
```

## Usage notes

- Load with `trust_remote_code=True`.
- `max_position_embeddings` is 13,568.
- Weights are published as {shard_count} bf16 safetensor shards.

## References

- Base checkpoint: {base_model_link}
- Long-context dataset: {longmino_link}

## Citation

```bibtex
@misc{{argonne25ctx13568,
  author = {{PursuitOfDataScience}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def build_model_card(profile, repo_id, model_name, parameter_count, plot_repo_path, shard_count):
    if profile == "base":
        return build_base_model_card(repo_id, model_name, parameter_count, plot_repo_path, shard_count)
    if profile == "instruct":
        return build_instruct_model_card(repo_id, model_name, shard_count)
    if profile == "midtraining":
        return build_midtraining_model_card(repo_id, model_name, parameter_count, plot_repo_path, shard_count)
    if profile == "ctx13568_instruct":
        return build_ctx13568_instruct_model_card(repo_id, model_name, parameter_count, shard_count)
    raise ValueError(f"Unknown profile: {profile}")


def parse_args():
    parser = argparse.ArgumentParser(description="Push Argonne weights to the Hugging Face Hub.")
    parser.add_argument(
        "--profile",
        choices=["base", "instruct", "midtraining", "ctx13568_instruct"],
        default="base",
        help="Model card profile to publish.",
    )
    parser.add_argument("--model-dir", required=True, help="Path to the saved checkpoint directory.")
    parser.add_argument("--repo-id", default=None, help="Hugging Face repo id.")
    parser.add_argument("--model-name", default=None, help="Human-readable model name.")
    parser.add_argument("--plot-path", default=None, help="Optional loss-curve image to include in the repo.")
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare files without uploading.")
    parser.add_argument(
        "--shard-count",
        type=int,
        default=DEFAULT_WEIGHT_SHARD_COUNT,
        help="Number of safetensor shards to write.",
    )
    parser.add_argument("--commit-message", default=None, help="Upload commit message.")
    return parser.parse_args()


def normalize_tensor_dtype(tensor):
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.bfloat16)
    return tensor.contiguous()


def load_weight_tensors(model_path):
    from safetensors import safe_open

    safetensor_files = sorted(
        path for path in model_path.glob("*.safetensors") if not path.name.endswith(".index.json")
    )
    bin_files = sorted(model_path.glob("pytorch_model*.bin"))

    state_dict = {}
    if safetensor_files:
        for file_path in safetensor_files:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    state_dict[key] = normalize_tensor_dtype(handle.get_tensor(key))
    elif bin_files:
        for file_path in bin_files:
            shard = torch.load(file_path, map_location="cpu", weights_only=False)
            for key, value in shard.items():
                if isinstance(value, torch.Tensor):
                    state_dict[key] = normalize_tensor_dtype(value)
    else:
        raise SystemExit(f"No weight files found in {model_path}")

    storage_to_keys = {}
    for key, tensor in state_dict.items():
        storage_to_keys.setdefault(tensor.data_ptr(), []).append(key)

    for keys in storage_to_keys.values():
        if len(keys) > 1:
            for key in keys[1:]:
                state_dict[key] = state_dict[key].clone()

    return state_dict


def split_state_dict_into_shards(state_dict, shard_count, temp_path):
    from safetensors.torch import save_file as safetensors_save

    shard_count = max(1, shard_count)
    shard_entries = [OrderedDict() for _ in range(shard_count)]
    shard_sizes = [0 for _ in range(shard_count)]

    def tensor_size_bytes(tensor):
        return tensor.numel() * tensor.element_size()

    sorted_items = sorted(state_dict.items(), key=lambda item: (-tensor_size_bytes(item[1]), item[0]))
    for key, tensor in sorted_items:
        shard_index = min(range(shard_count), key=lambda idx: (shard_sizes[idx], idx))
        shard_entries[shard_index][key] = tensor
        shard_sizes[shard_index] += tensor_size_bytes(tensor)

    total_size = sum(shard_sizes)
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    weight_map = {}

    for index, shard in enumerate(shard_entries, start=1):
        shard_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        safetensors_save(shard, str(temp_path / shard_name))
        for key in shard:
            weight_map[key] = shard_name

    index_path = temp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n"
    )
    return total_params


def rewrite_config_dtype(temp_path, dtype_name):
    config_path = temp_path / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text())
    config["dtype"] = dtype_name
    config["torch_dtype"] = dtype_name
    config_path.write_text(json.dumps(config, indent=2) + "\n")


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


def write_model_card(temp_path, profile, repo_id, model_name, parameter_count, plot_repo_path, shard_count):
    readme = build_model_card(profile, repo_id, model_name, parameter_count, plot_repo_path, shard_count)
    (temp_path / "README.md").write_text(readme)


def prepare_upload_folder(model_dir, profile, repo_id, model_name, shard_count, plot_path):
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

    state_dict = load_weight_tensors(model_path)
    parameter_count = split_state_dict_into_shards(state_dict, shard_count, temp_path)
    rewrite_config_dtype(temp_path, CHECKPOINT_DTYPE)

    model_py = SCRIPT_DIR / "model.py"
    if model_py.is_file():
        shutil.copy2(model_py, temp_path / "model.py")

    plot_repo_path = copy_plot(plot_path, temp_path)
    write_model_card(temp_path, profile, repo_id, model_name, parameter_count, plot_repo_path, shard_count)
    return temp_dir


def clean_old_files(api, repo_id):
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    to_delete = [
        path
        for path in repo_files
        if path == ".gitattributes"
        or path == "model.safetensors"
        or path == "model.safetensors.index.json"
        or (path.startswith("model-") and path.endswith(".safetensors"))
        or (path.startswith("model-") and path.endswith(".index.json"))
        or (path.startswith("pytorch_model") and (path.endswith(".bin") or path.endswith(".index.json")))
    ]
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
    if args.repo_id is None:
        if args.profile == "base":
            args.repo_id = DEFAULT_BASE_REPO_ID
        elif args.profile == "instruct":
            args.repo_id = DEFAULT_INSTRUCT_REPO_ID
        elif args.profile == "ctx13568_instruct":
            args.repo_id = DEFAULT_CTX13568_INSTRUCT_REPO_ID
        else:
            args.repo_id = DEFAULT_MIDTRAINING_REPO_ID
    if args.model_name is None:
        if args.profile == "base":
            args.model_name = DEFAULT_BASE_MODEL_NAME
        elif args.profile == "instruct":
            args.model_name = DEFAULT_INSTRUCT_MODEL_NAME
        elif args.profile == "ctx13568_instruct":
            args.model_name = DEFAULT_CTX13568_INSTRUCT_MODEL_NAME
        else:
            args.model_name = DEFAULT_MIDTRAINING_MODEL_NAME
    if args.commit_message is None:
        args.commit_message = f"Upload {args.model_name}"

    upload_folder = prepare_upload_folder(
        args.model_dir,
        args.profile,
        args.repo_id,
        args.model_name,
        args.shard_count,
        args.plot_path,
    )

    try:
        push_to_hub(upload_folder, args.repo_id, args.private, args.commit_message, args.dry_run)
    finally:
        shutil.rmtree(upload_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
