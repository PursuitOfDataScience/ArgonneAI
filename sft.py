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
- Does not save intermediate checkpoints; saves final model to
  /project/rcc/youzhi/llm.c/checkpoints/final_model_sft.
- Auto-detects the end-of-turn token from the chat template and sets it
  as eos_token so the model learns to stop generating.
"""

import argparse
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Avoid tokenizer thread oversubscription on cluster nodes.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def is_rank0():
    """Check if current process is rank 0 (for DDP)"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

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
    parser.add_argument("--save_strategy", type=str, default="no", choices=["steps", "no"], help="Checkpoint saving strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Limit the total amount of checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--exit_after_checkpoint_save", action="store_true", help="Exit after saving a checkpoint for iterative training")
    
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
    print("Intermediate checkpoint saving: disabled")
    print(f"Final model save dir: {args.output_dir}")

    model, tokenizer = build_model_and_tokenizer(device, args.argonne_root, args.model_path, MAX_SEQ_LEN)
    
    # Add checkpoint resuming support  
    start_global_step = 0
    start_scheduler_state = None
    start_optimizer_state = None
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        checkpoint_path = args.resume_from_checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        # Load model weights from checkpoint
        weights_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Resuming from checkpoint. missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
            model.tie_weights()  # Re-tie after loading partial state dict
        # Load tokenizer from checkpoint
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.bin")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location='cpu')
            start_global_step = training_state.get('global_step', 0)
            start_tokens_seen = training_state.get('tokens_seen', 0)
            start_scheduler_state = training_state.get('scheduler_state_dict', None)
            start_optimizer_state = training_state.get('optimizer_state_dict', None)
            print(f"Resuming from global_step: {start_global_step}, tokens_seen: {start_tokens_seen}")
    else:
        start_tokens_seen = 0

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

    # Deterministic shuffled indices so jobs train on sequential chunks of the shuffled dataset
    rng = torch.Generator().manual_seed(9786)  # Fixed seed for reproducible shuffle
    all_indices = list(torch.randperm(len(dataset), generator=rng).tolist())

    # Calculate chunk boundaries
    batches_per_step = GRAD_ACCUM_STEPS  # micro-batches per optimizer step
    samples_per_step = batches_per_step * BATCH_SIZE  # samples per optimizer step
    start_sample = start_global_step * samples_per_step  # samples consumed so far

    # Total steps for full epoch (used for scheduler)
    full_steps_per_epoch = len(dataset) // samples_per_step
    total_steps = max(1, full_steps_per_epoch * EPOCHS)

    steps_this_job = total_steps - start_global_step  # steps remaining
    chunksize = steps_this_job * samples_per_step  # samples for this job

    if chunksize > 0:
        chunk_indices = all_indices[start_sample:start_sample + chunksize]
    else:
        chunk_indices = all_indices[start_sample:]

    chunk_dataset = torch.utils.data.Subset(dataset, chunk_indices) if chunk_indices else dataset
    loader = DataLoader(
        chunk_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Already shuffled via indices
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    steps_per_epoch = len(loader) // GRAD_ACCUM_STEPS
    print(f"DataLoader batches/epoch: {len(loader):,}")
    print(f"Optimizer steps/epoch (this job): {steps_per_epoch:,}")
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

    # Restore scheduler and optimizer state when resuming
    if start_scheduler_state is not None:
        scheduler.load_state_dict(start_scheduler_state)
        print(f"Restored scheduler state (last_epoch={start_scheduler_state.get('last_epoch', 'N/A')})")
    elif start_global_step > 0:
        # No scheduler state in checkpoint — manually advance to global_step
        for _ in range(start_global_step):
            scheduler.step()
        print(f"No scheduler state found, advanced scheduler to step {start_global_step}")
    if start_optimizer_state is not None:
        optimizer.load_state_dict(start_optimizer_state)
        print(f"Restored optimizer state")

    # Baseline quality before SFT.
    if args.run_before_quality == 1:
        answer_questions(model, tokenizer, device, QUALITY_QUESTIONS, tag="BEFORE_SFT", step=0)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_global_step
    tokens_seen = start_tokens_seen
    running_loss = 0.0
    
    if start_global_step > 0:
        print(f"Resuming training from global_step={global_step}, tokens_seen={start_tokens_seen}")

    micro_step = 0  # Track micro steps (batches before optimizer step)
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch")

        for batch in pbar:
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

                # Handle checkpoint saving
                if args.save_strategy == "steps" and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if is_rank0:
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        print(f"Saving checkpoint to {checkpoint_dir}")
                        model.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        
                        # Save training state
                        state = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "tokens_seen": tokens_seen,
                            "scheduler_state_dict": scheduler.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                        torch.save(state, os.path.join(checkpoint_dir, "training_state.bin"))
                        
                        # Handle save total limit
                        if args.save_total_limit is not None:
                            checkpoints = []
                            for d in os.listdir(args.output_dir):
                                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d)):
                                    try:
                                        step = int(d.split("-")[1])
                                        checkpoints.append((step, d))
                                    except ValueError:
                                        pass
                            checkpoints.sort(key=lambda x: x[0])
                            if len(checkpoints) > args.save_total_limit:
                                for step, d in checkpoints[:-args.save_total_limit]:
                                    shutil.rmtree(os.path.join(args.output_dir, d))
                                    print(f"Removed old checkpoint: {d}")

                        # Exit after checkpoint save for iterative training
                        # (but don't exit if we've reached max_steps — let the final save happen)
                        if args.exit_after_checkpoint_save and (args.max_steps <= 0 or global_step < args.max_steps):
                            print(f"Checkpoint saved at step {global_step}. Exiting for iterative training.")
                            sys.exit(0)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            pbar.set_postfix({"step": global_step, "loss": f"{loss.detach().item():.4f}"})
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save final checkpoint when max_steps reached
    if args.max_steps > 0 and global_step >= args.max_steps and global_step > 0:
        if is_rank0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Saving final checkpoint to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            state = {
                "epoch": EPOCHS - 1,
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(checkpoint_dir, "training_state.bin"))
            if args.save_total_limit is not None:
                checkpoints = []
                for d in os.listdir(args.output_dir):
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d)):
                        try:
                            step = int(d.split("-")[1])
                            checkpoints.append((step, d))
                        except ValueError:
                            pass
                checkpoints.sort(key=lambda x: x[0])
                if len(checkpoints) > args.save_total_limit:
                    for step, d in checkpoints[:-args.save_total_limit]:
                        shutil.rmtree(os.path.join(args.output_dir, d))
                        print(f"Removed old checkpoint: {d}")

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


if __name__ == "__main__":
    main()
