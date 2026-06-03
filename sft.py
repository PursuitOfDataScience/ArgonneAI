#!/usr/bin/env python3
"""
SFT the latest Argonne checkpoint on UltraChat train_sft.

Requirements implemented:
- Loads the final pretrained model from
  /project/rcc/youzhi/llm.c/checkpoints/final_model.
- Loads data from disk:
  /project/rcc/youzhi/data/HuggingFaceH4_ultrachat_200k/train_sft
- Uses `prompt` + `messages` to build x/y:
    x = full context before the final assistant turn
    y = final assistant answer
- Loss is computed only on y (assistant answer tokens); x is masked with -100.
- Runs 6 fixed quality questions before SFT, then every 200 optimizer steps.
- Optional intermediate checkpointing:
    * --save_strategy=steps  -> save every --save_steps optimizer steps
    * --save_strategy=time   -> save every --save_seconds wall-clock seconds
    * --save_total_limit=N   -> keep at most N most recent checkpoints
    * --resume_from_checkpoint=auto|<path>  -> restore model/optimizer/scheduler
    * --exit_after_checkpoint_save          -> exit cleanly after the first save
      (used for SLURM slice auto-resubmit)
    * --slice_time_limit=N                   -> save and exit before N seconds
- Saves the final model to <output_dir> via save_pretrained.
- Auto-detects the end-of-turn token from the chat template and sets it
  as eos_token so the model learns to stop generating.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Avoid tokenizer thread oversubscription on cluster nodes.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Default hyperparameters (can be overridden by CLI args)
# -----------------------------------------------------------------------------
MAX_SEQ_LEN = 1024

# SFT hyperparameters
SEED = 42
EPOCHS = 1
BATCH_SIZE = 24
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-5
MIN_LR_RATIO = 0.1
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
GRAD_CLIP = 1.0
LOG_EVERY = 10
QUALITY_EVERY = 200
MAX_NEW_TOKENS_QUALITY = 160

QUALITY_QUESTIONS = [
    # 1. Basic greeting — can it respond like a chatbot at all?
    "Hey! How's it going?",
    # 2. Open-ended helpfulness — the bread and butter of SFT
    "I'm planning a weekend trip. Any tips for packing light?",
    # 3. Instruction following — can it explain something clearly?
    "Explain what a black hole is in a way a 10-year-old would understand.",
    # 4. Empathy / emotional support — common in chat data
    "I just failed an exam I studied really hard for. I feel terrible.",
    # 5. Multi-step reasoning lite — tests coherence
    "What are three fun things to do on a rainy day, and why?",
    # 6. Creative writing — poem generation
    "Write a short poem about the ocean at night.",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EOS detection from chat template
# ---------------------------------------------------------------------------

def detect_eos_from_template(tokenizer) -> int:
    """Find the end-of-turn token the chat template places after assistant
    content by rendering a minimal conversation and walking backwards past
    any trailing whitespace tokens."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]

    for i in range(len(ids) - 1, -1, -1):
        token_str = tokenizer.decode([ids[i]]).strip()
        if token_str:
            return ids[i]
    return tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalize_conversation(prompt: str, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    prompt = (prompt or "").strip()
    cleaned: List[Dict[str, str]] = []

    for msg in messages or []:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    # Use prompt explicitly (as requested), while avoiding duplicate first user turn.
    if prompt:
        if cleaned and cleaned[0]["role"] == "user":
            cleaned[0]["content"] = prompt
        else:
            cleaned.insert(0, {"role": "user", "content": prompt})

    return cleaned


def find_last_assistant_turn(conversation: List[Dict[str, str]]) -> Optional[int]:
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]["role"] == "assistant" and conversation[i]["content"].strip():
            return i
    return None


def extract_input_ids(tokenized: Any) -> List[int]:
    """
    Normalize tokenizer outputs from apply_chat_template into List[int].
    Handles list[int], BatchEncoding-like dicts, and tensor outputs.
    """
    ids: Any
    if isinstance(tokenized, dict):
        ids = tokenized.get("input_ids")
    elif hasattr(tokenized, "input_ids"):
        ids = getattr(tokenized, "input_ids")
    else:
        ids = tokenized

    if ids is None:
        raise ValueError("apply_chat_template did not return input_ids")

    if torch.is_tensor(ids):
        ids = ids.tolist()

    # Some tokenizers return a batched structure [[...]].
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]

    return [int(x) for x in ids]


def wrap_empty_think_answer(content: str) -> str:
    """Train normal SFT answers to follow the always-think response contract."""
    answer = (content or "").strip()
    if not answer:
        return answer
    if "<think>" in answer and "</think>" in answer:
        return answer
    return f"<think>\n\n</think>\n\n{answer}"


def build_training_example(
    prompt: str,
    messages: List[Dict[str, Any]],
    tokenizer,
    max_seq_len: int,
) -> Optional[Dict[str, List[int]]]:
    conversation = normalize_conversation(prompt, messages)
    target_idx = find_last_assistant_turn(conversation)

    if target_idx is None or target_idx <= 0:
        return None

    context_all = conversation[:target_idx]  # x (full history before final assistant)
    target_turn = {
        **conversation[target_idx],
        "content": wrap_empty_think_answer(conversation[target_idx]["content"]),
    }  # y (empty think block + final assistant answer)
    # Keep the most recent context possible (drop oldest turns first)
    # so examples fit model context length while preserving multi-turn recency.
    last_user_idx = None
    for i in range(target_idx - 1, -1, -1):
        if conversation[i]["role"] == "user":
            last_user_idx = i
            break

    for start in range(0, target_idx):
        context = context_all[start:]
        if not context:
            continue
        if context[0]["role"] not in {"system", "user"}:
            continue

        # Require at least one user turn in x so assistant answer is grounded in a question.
        if last_user_idx is not None and start > last_user_idx:
            continue

        full_conv = context + [target_turn]
        prefix_ids = tokenizer.apply_chat_template(
            context,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_conv,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Normalize to plain Python lists (robust across tokenizer return types).
        prefix_ids = extract_input_ids(prefix_ids)
        full_ids = extract_input_ids(full_ids)

        # Ensure y corresponds to the final assistant answer continuation.
        if len(prefix_ids) >= len(full_ids):
            continue
        if len(full_ids) > max_seq_len:
            continue

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        if all(v == -100 for v in labels):
            continue

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UltraChatLastAssistantDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.raw = load_from_disk(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = random.Random(SEED + 20260328)

        # Keep records that have at least one assistant message;
        # token-length filtering is done lazily in __getitem__.
        self.indices: List[int] = []
        for i in range(len(self.raw)):
            msgs = self.raw[i]["messages"]
            if any(str(m.get("role", "")).strip() == "assistant" for m in msgs):
                self.indices.append(i)

        if not self.indices:
            raise RuntimeError("No usable examples with assistant turns found.")

        # Precompute one guaranteed-valid fallback sample.
        self.fallback = None
        probe_positions = list(range(min(2000, len(self.indices))))
        if len(self.indices) > 2000:
            random_positions = self.rng.sample(
                range(len(self.indices)),
                k=min(20000, len(self.indices)),
            )
            probe_positions.extend(random_positions)

        seen = set()
        for pos in probe_positions:
            if pos in seen:
                continue
            seen.add(pos)
            raw_idx = self.indices[pos]
            ex = self.raw[raw_idx]
            built = build_training_example(
                ex["prompt"],
                ex["messages"],
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            if built is not None:
                self.fallback = built
                break

        if self.fallback is None:
            raise RuntimeError(
                f"Could not construct any valid sample at max_seq_len={self.max_seq_len}. "
                "Try increasing MAX_SEQ_LEN."
            )

        print(
            f"Dataset loaded: {len(self.raw):,} records, "
            f"{len(self.indices):,} with assistant turns."
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        # Try nearby samples if this one is too long or malformed after tokenization.
        base = idx % len(self.indices)
        for offset in range(24):
            raw_idx = self.indices[(base + offset) % len(self.indices)]
            ex = self.raw[raw_idx]
            built = build_training_example(
                ex["prompt"],
                ex["messages"],
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            if built is not None:
                return built

        # If a local window fails, try random positions to avoid pathological clusters.
        for _ in range(24):
            pos = self.rng.randrange(len(self.indices))
            raw_idx = self.indices[pos]
            ex = self.raw[raw_idx]
            built = build_training_example(
                ex["prompt"],
                ex["messages"],
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            if built is not None:
                return built
        return self.fallback


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            labels = f["labels"]
            pad = max_len - len(ids)

            # Right-padding is safer here because the model ignores attention_mask.
            batch_input_ids.append(ids + [self.pad_token_id] * pad)
            batch_attention_mask.append(mask + [0] * pad)
            batch_labels.append(labels + [-100] * pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(device: torch.device, argonne_root: str, model_path: str, max_seq_len: int):
    sys.path.insert(0, argonne_root)
    from model import ArgonneConfig, ArgonneModel

    print(f"Loading model and tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect the end-of-turn token from the chat template and set it
    # as eos so that generation stops at the right place.
    detected_eos_id = detect_eos_from_template(tokenizer)
    if detected_eos_id != tokenizer.eos_token_id:
        old_eos = repr(tokenizer.eos_token)
        tokenizer.eos_token_id = detected_eos_id
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(detected_eos_id)
        print(
            f"EOS updated: {old_eos} -> {repr(tokenizer.eos_token)} "
            f"(id={detected_eos_id}) [detected from chat template]"
        )
    else:
        print(f"EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")

    # Load config from json, construct model, load safetensors weights.
    import json
    from safetensors.torch import load_file

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = ArgonneConfig(**{k: v for k, v in config_dict.items() if not k.startswith("_")})
    config.max_position_embeddings = max_seq_len
    config.block_size = max_seq_len
    config.use_flash_attention = True
    config._keep_in_fp32_modules = []

    model = ArgonneModel(config)

    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(
        f"State dict applied. missing_keys={len(missing_keys)} "
        f"unexpected_keys={len(unexpected_keys)}"
    )
    model.tie_weights()

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

@torch.no_grad()
def answer_questions(
    model,
    tokenizer,
    device: torch.device,
    questions: List[str],
    tag: str,
    step: int,
) -> None:
    was_training = model.training
    model.eval()

    print("\n" + "=" * 90)
    print(f"[QUALITY] {tag} | step={step}")
    print("=" * 90)
    for i, question in enumerate(questions, start=1):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = extract_input_ids(prompt_ids)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        max_length = min(MAX_SEQ_LEN, input_ids.shape[1] + MAX_NEW_TOKENS_QUALITY)
        if max_length <= input_ids.shape[1]:
            reply = ""
        else:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=1.0,
                do_sample=False,
            )
            gen_ids = output_ids[0, input_ids.shape[1]:].tolist()
            # Stop at end-of-turn token if the model generated past it.
            eos_id = tokenizer.eos_token_id
            if eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]
            reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"\nQ{i}: {question}")
        print(f"A{i}: {reply}")
    print("\n" + "=" * 90 + "\n")

    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# Loss (manual shift — ArgonneModel has no internal shift)
# ---------------------------------------------------------------------------

def compute_loss(model, input_ids: torch.Tensor, labels: torch.Tensor):
    x = input_ids[:, :-1].contiguous()
    y = labels[:, 1:].contiguous()
    outputs = model(x, labels=y)
    return outputs.loss


# ---------------------------------------------------------------------------
# Checkpoint helpers (save / load / rotate)
# ---------------------------------------------------------------------------

_CKPT_STEP_RE = re.compile(r"checkpoint-(\d+)$")


def _ckpt_step(path: str) -> int:
    """Extract the step number from a checkpoint directory name; -1 if not parseable."""
    m = _CKPT_STEP_RE.search(os.path.basename(os.path.normpath(path)))
    return int(m.group(1)) if m else -1


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path of the highest-step checkpoint under output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    candidates: List[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full):
            continue
        step = _ckpt_step(name)
        if step >= 0:
            candidates.append((step, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_checkpoint(
    output_dir: str,
    global_step: int,
    tokens_seen: int,
    epoch: int,
    model,
    optimizer,
    scheduler,
    args,
    last_save_time: float,
) -> str:
    """Save model+optimizer+scheduler+metadata to ``output_dir/checkpoint-step-N``.

    Returns the path of the saved checkpoint.
    """
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the model state dict via safetensors for portability and the full
    # Argonne model via save_pretrained so the next launch can reload the
    # architecture (config) without re-deriving it.
    from safetensors.torch import save_file

    state_dict = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))

    # Config snapshot for self-describing checkpoints.
    if hasattr(model, "config"):
        try:
            model.config.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"  warn: could not save model.config: {e}")

    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(rng_state, os.path.join(ckpt_dir, "rng_state.pt"))

    metadata = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "epoch": epoch,
        "saved_at_unix": time.time(),
        "saved_at_monotonic": time.monotonic(),
        "wall_since_last_save_s": time.monotonic() - last_save_time if last_save_time > 0 else 0.0,
        "args": {
            k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
            for k, v in vars(args).items()
        },
    }
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[ckpt] saved checkpoint-{global_step} -> {ckpt_dir}")
    return ckpt_dir


def rotate_checkpoints(output_dir: str, save_total_limit: int) -> None:
    """Keep only the ``save_total_limit`` most recent checkpoint directories."""
    if save_total_limit <= 0:
        return
    candidates: List[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full):
            continue
        step = _ckpt_step(name)
        if step >= 0:
            candidates.append((step, full))
    candidates.sort(key=lambda x: x[0])
    while len(candidates) > save_total_limit:
        step, path = candidates.pop(0)
        try:
            shutil.rmtree(path)
            print(f"[ckpt] rotated out checkpoint-{step} (limit={save_total_limit})")
        except OSError as e:
            print(f"  warn: could not remove {path}: {e}")


def load_checkpoint(
    ckpt_path: str,
    model,
    optimizer,
    scheduler,
    device: torch.device,
) -> Dict[str, int]:
    """Load model+optimizer+scheduler+RNG from ``ckpt_path``.

    Returns a dict with ``global_step``, ``tokens_seen``, ``epoch``.
    """
    from safetensors.torch import load_file

    model_path = os.path.join(ckpt_path, "model.safetensors")
    opt_path = os.path.join(ckpt_path, "optimizer.pt")
    sched_path = os.path.join(ckpt_path, "scheduler.pt")
    rng_path = os.path.join(ckpt_path, "rng_state.pt")
    meta_path = os.path.join(ckpt_path, "metadata.json")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model.safetensors missing in {ckpt_path}")

    state_dict = load_file(model_path, device=str(device))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[ckpt] loaded model from {model_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    if os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
        # Move optimizer state to the right device after load.
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)
        print(f"[ckpt] loaded optimizer state from {opt_path}")

    if os.path.isfile(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=False))
        print(f"[ckpt] loaded scheduler state from {sched_path}")

    if os.path.isfile(rng_path):
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["torch_cuda"]])
        print(f"[ckpt] loaded RNG state from {rng_path}")

    metadata: Dict[str, int] = {"global_step": 0, "tokens_seen": 0, "epoch": 0}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        metadata["global_step"] = int(meta.get("global_step", 0))
        metadata["tokens_seen"] = int(meta.get("tokens_seen", 0))
        metadata["epoch"] = int(meta.get("epoch", 0))
        print(
            f"[ckpt] loaded metadata: step={metadata['global_step']} "
            f"tokens={metadata['tokens_seen']:,} epoch={metadata['epoch']}"
        )
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training on UltraChat")
    parser.add_argument("--argonne_root", type=str, required=True, help="Path to ArgonneAI directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for final model")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quality_every", type=int, default=200, help="Run quality samples every N optimizer steps")
    parser.add_argument("--max_new_tokens_quality", type=int, default=160, help="Max new tokens for quality samples")
    parser.add_argument("--run_before_quality", type=int, default=1, choices=[0, 1], help="Run quality samples before SFT")
    parser.add_argument("--run_after_quality", type=int, default=1, choices=[0, 1], help="Run quality samples after SFT")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, stop after this many optimizer steps")
    parser.add_argument("--skip_final_save", action="store_true", help="Skip final save_pretrained output")

    # Checkpointing + resume
    parser.add_argument("--save_strategy", type=str, default="none",
                        choices=["none", "steps", "time"],
                        help="When to save intermediate checkpoints: 'steps', 'time' (wall clock), or 'none'.")
    parser.add_argument("--save_steps", type=int, default=600,
                        help="Save every N optimizer steps when save_strategy=steps.")
    parser.add_argument("--save_seconds", type=int, default=0,
                        help="Save every N wall-clock seconds when save_strategy=time. 0 disables.")
    parser.add_argument("--save_total_limit", type=int, default=-1,
                        help="Keep at most N most recent checkpoint directories (-1 = keep all).")
    parser.add_argument("--resume_from_checkpoint", type=str, default="",
                        help="Path to a checkpoint directory to resume from. Use 'auto' to pick the latest under --output_dir.")
    parser.add_argument("--exit_after_checkpoint_save", action="store_true",
                        help="Exit cleanly after the first save (used with SLURM auto-resubmit).")
    parser.add_argument("--slice_time_limit", type=int, default=0,
                        help="If > 0, save and exit before this many seconds of wall time elapse (best-effort).")

    args = parser.parse_args()
    
    # Set hyperparameters from args
    global MAX_SEQ_LEN, QUALITY_EVERY, MAX_NEW_TOKENS_QUALITY
    SEED = args.seed
    EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum
    LEARNING_RATE = args.lr
    WARMUP_STEPS = args.warmup_steps
    MAX_SEQ_LEN = args.max_seq_length
    QUALITY_EVERY = args.quality_every
    MAX_NEW_TOKENS_QUALITY = args.max_new_tokens_quality
    
    seed_everything(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device = torch.device("cuda")
    print("=" * 90)
    print("Argonne checkpoint SFT on UltraChat train_sft")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Data: {args.data_path}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    print(f"Batch size: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM_STEPS}")
    print(f"LR: {LEARNING_RATE} | Warmup: {WARMUP_STEPS}")
    if args.save_strategy == "none":
        print("Intermediate checkpoint saving: disabled (save_strategy=none)")
    else:
        if args.save_strategy == "steps":
            print(f"Intermediate checkpoint saving: every {args.save_steps} steps (keep up to {args.save_total_limit if args.save_total_limit > 0 else 'all'})")
        else:
            print(f"Intermediate checkpoint saving: every {args.save_seconds}s wall clock (keep up to {args.save_total_limit if args.save_total_limit > 0 else 'all'})")
        print(f"Exit after first save: {'yes' if args.exit_after_checkpoint_save else 'no'}")
        print(f"Resume from checkpoint: {args.resume_from_checkpoint or '<none>'}")
    print(f"Final model save dir: {args.output_dir}")

    model, tokenizer = build_model_and_tokenizer(device, args.argonne_root, args.model_path, MAX_SEQ_LEN)

    # Debug probe to ensure runtime tokenizer behavior matches local validation.
    probe_ds = load_from_disk(args.data_path)
    probe_valid = 0
    for i in range(min(256, len(probe_ds))):
        ex = probe_ds[i]
        built = build_training_example(
            ex["prompt"],
            ex["messages"],
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )
        if built is not None:
            probe_valid += 1
    print(f"Sanity valid-in-first-{min(256, len(probe_ds))}: {probe_valid}")

    dataset = UltraChatLastAssistantDataset(args.data_path, tokenizer, MAX_SEQ_LEN)
    collator = CausalCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Keep deterministic/reliable with custom tokenization in __getitem__.
        pin_memory=True,
        collate_fn=collator,
    )

    steps_per_epoch = len(loader) // GRAD_ACCUM_STEPS
    total_steps = max(1, steps_per_epoch * EPOCHS)
    print(f"DataLoader batches/epoch: {len(loader):,}")
    print(f"Optimizer steps/epoch: {steps_per_epoch:,}")
    print(f"Total optimizer steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    min_lr = LEARNING_RATE * MIN_LR_RATIO
    min_lr_scale = min_lr / LEARNING_RATE

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_lr_scale, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint (if requested). This must happen after optimizer
    # and scheduler are constructed so we can restore their state in-place.
    resume_metadata = {"global_step": 0, "tokens_seen": 0, "epoch": 0}
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if resume_path == "auto":
            latest = find_latest_checkpoint(args.output_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"--resume_from_checkpoint=auto but no checkpoints found under {args.output_dir}"
                )
            resume_path = latest
            print(f"[resume] auto-selected latest checkpoint: {resume_path}")
        if not os.path.isdir(resume_path):
            raise FileNotFoundError(f"--resume_from_checkpoint path does not exist: {resume_path}")
        resume_metadata = load_checkpoint(resume_path, model, optimizer, scheduler, device)

    # Baseline quality before SFT.
    if args.run_before_quality == 1:
        answer_questions(model, tokenizer, device, QUALITY_QUESTIONS, tag="BEFORE_SFT", step=0)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Resume counters (if a checkpoint was loaded).
    global_step = int(resume_metadata.get("global_step", 0))
    tokens_seen = int(resume_metadata.get("tokens_seen", 0))
    resume_epoch = int(resume_metadata.get("epoch", 0))
    micro_step = global_step * GRAD_ACCUM_STEPS
    running_loss = 0.0

    if global_step > 0:
        print(
            f"[resume] starting from global_step={global_step} "
            f"tokens_seen={tokens_seen:,} (resumed epoch index {resume_epoch})"
        )

    # Tracking for time-based checkpointing + slice time limit.
    last_save_time = time.monotonic()
    job_start_time = last_save_time
    save_locked_out = False  # becomes True after the first save when --exit_after_checkpoint_save is set

    for epoch in range(resume_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch")

        for batch in pbar:
            # Slice time limit reached -> save and exit so SLURM can resubmit.
            if (
                args.slice_time_limit > 0
                and (time.monotonic() - job_start_time) >= args.slice_time_limit
            ):
                print(
                    f"[slice] wall time {args.slice_time_limit}s reached; "
                    f"saving checkpoint and exiting."
                )
                save_checkpoint(
                    args.output_dir,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    last_save_time=last_save_time,
                )
                rotate_checkpoints(args.output_dir, args.save_total_limit)
                pbar.close()
                print("Exiting cleanly due to --slice_time_limit.")
                return

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Count non-padding tokens in this micro-batch.
            tokens_seen += int((batch["attention_mask"].sum()).item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                loss = compute_loss(model, input_ids=input_ids, labels=labels)
                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaled_loss.backward()
            running_loss += float(loss.detach().item())
            micro_step += 1

            if micro_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % LOG_EVERY == 0:
                    avg_loss = running_loss / (LOG_EVERY * GRAD_ACCUM_STEPS)
                    running_loss = 0.0
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Step {global_step:>6} | "
                        f"loss {avg_loss:.4f} | "
                        f"tokens {tokens_seen:,} | "
                        f"lr {lr:.2e}"
                    )

                if QUALITY_EVERY > 0 and global_step % QUALITY_EVERY == 0:
                    answer_questions(
                        model,
                        tokenizer,
                        device,
                        QUALITY_QUESTIONS,
                        tag="DURING_SFT",
                        step=global_step,
                    )

                # --- Intermediate checkpoint save (steps or wall-clock) ---
                should_save = False
                if not save_locked_out and args.save_strategy == "steps" and args.save_steps > 0:
                    if global_step % args.save_steps == 0:
                        should_save = True
                elif not save_locked_out and args.save_strategy == "time" and args.save_seconds > 0:
                    if (time.monotonic() - last_save_time) >= args.save_seconds:
                        should_save = True

                if should_save:
                    save_checkpoint(
                        args.output_dir,
                        global_step=global_step,
                        tokens_seen=tokens_seen,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        last_save_time=last_save_time,
                    )
                    last_save_time = time.monotonic()
                    rotate_checkpoints(args.output_dir, args.save_total_limit)
                    if args.exit_after_checkpoint_save:
                        print("[exit] --exit_after_checkpoint_save set; exiting after first save.")
                        pbar.close()
                        return

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            pbar.set_postfix({"step": global_step, "loss": f"{loss.detach().item():.4f}"})
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    print("\nTraining finished.")
    print(f"Optimizer steps: {global_step:,}")
    print(f"Total tokens seen: {tokens_seen:,}")

    if args.run_after_quality == 1:
        answer_questions(model, tokenizer, device, QUALITY_QUESTIONS, tag="AFTER_SFT", step=global_step)

    if args.skip_final_save:
        print("Skipping final model/tokenizer save (--skip_final_save).")
        return

    # Save final model and tokenizer for downstream CoT SFT.
    # Use save_pretrained for HF-compatible format (config.json + safetensors).
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model and tokenizer to: {args.output_dir}")

    # Write a completion marker so wrapper scripts can detect end-of-training.
    completion_path = os.path.join(args.output_dir, ".sft_complete")
    try:
        with open(completion_path, "w") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "finished_at_unix": time.time(),
                },
                f,
            )
        print(f"Wrote completion marker: {completion_path}")
    except OSError as e:
        print(f"  warn: could not write completion marker: {e}")


if __name__ == "__main__":
    main()
