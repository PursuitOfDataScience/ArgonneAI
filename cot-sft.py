#!/usr/bin/env python3
"""
SFT training script for custom HuggingFace CausalLM models.

Key features:
- Dynamically imports --model_def to register custom HF model types.
- Loads model/tokenizer from local or remote --model_path.
- Extends context by updating rope_theta + max_position_embeddings and
  rebuilding rotary_emb at EVERY layer from --model_def.
- Loads chat data from either:
  - datasets load_from_disk() directory, or
  - JSONL file with `messages` column.
- Data is expected to already contain <think>...</think> blocks in
  assistant turns.  The Qwen3 template's own think injection is
  disabled via enable_thinking=False to avoid double wrapping.
- Masks loss so only assistant tokens contribute to training.
- Manual input/label shift in ShiftedLossTrainer because ArgonneModel's
  forward() computes loss as cross_entropy(logits, labels) with NO
  internal shift — the caller must align inputs and targets.
- Auto-detects the end-of-turn token from the chat template and sets
  it as eos_token so generation stops correctly.
- Supports DDP/multi-GPU when launched with torchrun.
"""

import argparse
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"


QUALITY_QUESTIONS = [
    "Hey! How's it going?",
    "I'm planning a weekend trip. Any tips for packing light?",
    "Explain what a black hole is in a way a 10-year-old would understand.",
    "I just failed an exam I studied really hard for. I feel terrible.",
    "What are three fun things to do on a rainy day, and why?",
    "Write a short poem about the ocean at night.",
]
MAX_NEW_TOKENS_QUALITY = 512


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
        enable_thinking=False,
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
# Model definition loader
# ---------------------------------------------------------------------------

def import_model_definition(model_def_path: str):
    if not os.path.isfile(model_def_path):
        raise FileNotFoundError(f"--model_def not found: {model_def_path}")
    module_name = f"custom_model_def_{abs(hash(os.path.abspath(model_def_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, model_def_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model definition from: {model_def_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "RotaryEmbedding"):
        raise AttributeError(
            f"RotaryEmbedding class not found in model definition: {model_def_path}"
        )
    if hasattr(module, "ArgonneModel"):
        missing = list(getattr(module.ArgonneModel, "_keys_to_ignore_on_load_missing", []) or [])
        if r"lm_head\.weight" not in missing:
            missing.append(r"lm_head\.weight")
        module.ArgonneModel._keys_to_ignore_on_load_missing = missing
    return module


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(data_path: str) -> Dataset:
    if os.path.isdir(data_path):
        ds = load_from_disk(data_path)
        if isinstance(ds, DatasetDict) and "train" in ds:
            ds = ds["train"]
    elif os.path.isfile(data_path) and data_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=data_path, split="train")
    else:
        raise ValueError(
            f"--data_path must be a load_from_disk directory or a .jsonl file, got: {data_path}"
        )

    if "messages" not in ds.column_names:
        raise ValueError(f"'messages' column not found in dataset columns: {ds.column_names}")
    return ds


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize message dicts, dropping any with invalid roles or empty content."""
    cleaned: List[Dict[str, str]] = []
    for m in messages or []:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


# ---------------------------------------------------------------------------
# Assistant-only loss masking
# ---------------------------------------------------------------------------

def build_masked_example(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
) -> Optional[Dict[str, List[int]]]:
    """Tokenize a conversation and return input_ids / attention_mask / labels
    with labels set to -100 for every token that is NOT part of an assistant
    turn.  Only assistant responses contribute to the training loss.

    Strategy: tokenize the conversation as incremental prefixes.  For each
    message[i] we template messages[0..i] and compare token lengths to find
    where the new message's tokens start.  Assistant tokens keep their ids in
    labels; everything else is -100.
    """
    if not messages:
        return None

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        return_attention_mask=True,
    )
    input_ids = [int(x) for x in full_enc["input_ids"]]
    attention_mask = [int(x) for x in full_enc["attention_mask"]]

    if not input_ids:
        return None

    # Start with everything masked (no loss).
    labels = [-100] * len(input_ids)

    # Walk through messages and unmask assistant regions.
    prev_token_len = 0
    for i, msg in enumerate(messages):
        prefix_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix_enc = tokenizer(
            prefix_text, add_special_tokens=False, truncation=False,
        )
        cur_token_len = len(prefix_enc["input_ids"])

        if msg["role"] == "assistant":
            # Clamp to actual (possibly truncated) length.
            start = min(prev_token_len, len(input_ids))
            end = min(cur_token_len, len(input_ids))
            for j in range(start, end):
                labels[j] = input_ids[j]

        prev_token_len = cur_token_len

    # If there are zero assistant tokens after masking, skip this example.
    if all(l == -100 for l in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LazySFTDataset(TorchDataset):
    def __init__(self, ds: Dataset, tokenizer, max_seq_len: int):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = random.Random(20260330)
        self.indices = [
            i
            for i in range(len(ds))
            if isinstance(ds[i].get("messages"), list) and len(ds[i]["messages"]) > 0
        ]
        if not self.indices:
            raise RuntimeError("No usable examples with messages found.")

    def __len__(self) -> int:
        return len(self.indices)

    def _build(self, raw_idx: int) -> Optional[Dict[str, List[int]]]:
        example = self.ds[raw_idx]
        msgs = example.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return None
        conv = clean_messages(msgs)
        if not conv:
            return None
        return build_masked_example(conv, self.tokenizer, self.max_seq_len)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        base = idx % len(self.indices)
        for offset in range(24):
            built = self._build(self.indices[(base + offset) % len(self.indices)])
            if built is not None:
                return built
        for _ in range(24):
            built = self._build(self.indices[self.rng.randrange(len(self.indices))])
            if built is not None:
                return built
        raise RuntimeError("Failed to build a valid tokenized example.")


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
            batch_input_ids.append(ids + [self.pad_token_id] * pad)
            batch_attention_mask.append(mask + [0] * pad)
            batch_labels.append(labels + [-100] * pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Quality / eval helpers
# ---------------------------------------------------------------------------

def chat_to_input_ids(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    return [int(x) for x in encoded]


@torch.no_grad()
def answer_questions(model, tokenizer, questions: List[str], tag: str, step: int) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    model_to_eval = model.module if hasattr(model, "module") else model
    was_training = model_to_eval.training
    model_to_eval.eval()

    device = next(model_to_eval.parameters()).device
    print("\n" + "=" * 90)
    print(f"[QUALITY] {tag} | step={step}")
    print("=" * 90)

    think_ids = tokenizer.encode("<think>", add_special_tokens=False)

    for i, q in enumerate(questions, start=1):
        prompt_ids = chat_to_input_ids(
            tokenizer,
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
        )
        # Seed generation with <think> to match the CoT format from training.
        prompt_ids = prompt_ids + think_ids
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        max_length = min(
            model_to_eval.config.max_position_embeddings,
            input_ids.shape[1] + MAX_NEW_TOKENS_QUALITY,
        )
        out = model_to_eval.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=False,
            temperature=1.0,
        )
        gen_ids = out[0, input_ids.shape[1] :].tolist()
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"\nQ{i}: {q}")
        print(f"A{i}: {text}")
    print("\n" + "=" * 90 + "\n")

    if was_training:
        model_to_eval.train()


class QualityCallback(TrainerCallback):
    def __init__(self, tokenizer, every_steps: int = 200) -> None:
        self.tokenizer = tokenizer
        self.every_steps = every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                answer_questions(model, self.tokenizer, QUALITY_QUESTIONS, "DURING_SFT", state.global_step)
        return control


# ---------------------------------------------------------------------------
# Shifted-loss Trainer
# ---------------------------------------------------------------------------
# ArgonneModel.forward() does NOT shift internally — it computes:
#   loss = cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
# So the caller must provide:
#   x = input_ids[:, :-1]   (input tokens 0..N-2)
#   y = labels[:, 1:]        (target tokens 1..N-1)
# This way logits[i] (predicted from token i) is trained against token i+1.

class ShiftedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        x = input_ids[:, :-1].contiguous()
        y = labels[:, 1:].contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1].contiguous()

        outputs = model(
            input_ids=x,
            attention_mask=attention_mask,
            labels=y,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# RoPE replacement helper
# ---------------------------------------------------------------------------

def replace_rotary_embeddings(model, RotaryEmbedding, rope_theta: float, max_seq_len: int) -> None:
    """Replace the rotary embedding on the model *and* inside every
    transformer block's attention layer so that all layers use the new
    rope_theta and max_position_embeddings."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Top-level (used by ArgonneModel — shared RoPE called in forward()).
    if hasattr(model, "rotary_emb"):
        model.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
        )

    # Per-block replacement (safety net in case any layer holds its own).
    blocks = None
    for attr in ("blocks", "layers", "decoder", "h"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            break
    if blocks is None and hasattr(model, "model"):
        for attr in ("blocks", "layers", "decoder", "h"):
            if hasattr(model.model, attr):
                blocks = getattr(model.model, attr)
                break

    if blocks is not None:
        for blk in blocks:
            for sub in [blk] + [getattr(blk, a, None) for a in ("attn", "self_attn", "attention")]:
                if sub is not None and hasattr(sub, "rotary_emb"):
                    sub.rotary_emb = RotaryEmbedding(
                        head_dim,
                        max_position_embeddings=max_seq_len,
                        base=rope_theta,
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT for custom HuggingFace model")
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_def", required=True, help="Path to custom model .py")
    p.add_argument("--data_path", required=True, help="JSONL or load_from_disk path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--rope_theta", type=float, default=80000.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--num_epochs", type=float, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_local_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = get_local_device()

    model_module = import_model_definition(args.model_def)
    RotaryEmbedding = getattr(model_module, "RotaryEmbedding")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

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

    # Manual model construction + weight loading to ensure lm_head is
    # properly tied to embed_tokens.  AutoModelForCausalLM.from_pretrained
    # fails to re-establish the tie for this custom model.
    import json
    from safetensors.torch import load_file

    ArgonneConfig = getattr(model_module, "ArgonneConfig")
    ArgonneModel = getattr(model_module, "ArgonneModel")

    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = ArgonneConfig(**{k: v for k, v in config_dict.items() if not k.startswith("_")})

    model = ArgonneModel(config)

    weights_path = os.path.join(args.model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()

    # Cast to training precision before moving to GPU.
    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif args.precision == "fp16":
        model = model.to(dtype=torch.float16)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters ({args.precision})")
    print(f"Tied: embed_tokens == lm_head -> {model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}")

    # ---- Extend RoPE / context ----
    model.config.rope_theta = args.rope_theta
    model.config.max_position_embeddings = args.max_seq_length
    replace_rotary_embeddings(model, RotaryEmbedding, args.rope_theta, args.max_seq_length)

    # ---- Flash attention + gradient checkpointing ----
    model.config.use_flash_attention = True
    if hasattr(model, "blocks"):
        for blk in model.blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "use_flash_attention"):
                blk.attn.use_flash_attention = True
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # ---- Data ----
    ds = load_training_data(args.data_path)
    print(f"Loaded dataset with {len(ds):,} rows from {args.data_path}")

    train_dataset = LazySFTDataset(ds, tokenizer=tokenizer, max_seq_len=args.max_seq_length)
    print(f"Usable rows with messages: {len(train_dataset):,}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_rank0 = torch.distributed.get_rank() == 0
    else:
        is_rank0 = True
    if is_rank0:
        answer_questions(model, tokenizer, QUALITY_QUESTIONS, "BEFORE_SFT", 0)

    os.makedirs(args.output_dir, exist_ok=True)

    bf16 = args.precision == "bf16"
    fp16 = args.precision == "fp16"
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        bf16=bf16,
        fp16=fp16,
        save_strategy="no",
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )

    # ShiftedLossTrainer because ArgonneModel.forward() does NOT shift
    # labels internally — it computes cross_entropy(logits, labels) directly.
    trainer = ShiftedLossTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
        callbacks=[QualityCallback(tokenizer=tokenizer, every_steps=200)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved final model/tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()