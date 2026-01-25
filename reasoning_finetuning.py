#!/usr/bin/env python3
"""
Reasoning (Chain-of-Thought) finetuning script for Argonne model.

This script finetunes the Argonne base model on Chain-of-Thought reasoning data,
where the model learns to generate <think>...</think> reasoning traces before
producing the final answer.

Features:
- Streaming tokenization: tokenizes on-the-fly during training (no pre-tokenization)
- Auto batch size reduction on OOM
- Periodic generation to monitor reasoning quality

Usage:
    python reasoning_finetuning.py --model-dir /path/to/base/model --data-dir /path/to/cot/data

The data should be a HuggingFace dataset with a "messages" column containing
conversations with user questions and assistant responses that include
<think>...</think> reasoning.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import glob
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Hopper (H100) specific optimizations
if torch.cuda.is_available():
    # Enable flash attention scaling for Hopper
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # Prefer flash/mem-efficient over math

from datasets import load_from_disk
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer,
    PreTrainedTokenizerFast, PreTrainedTokenizerBase, TrainingArguments, Trainer,
)
from transformers.trainer_callback import TrainerCallback


# =============================================================================
# Register Argonne model with Transformers
# =============================================================================
def _register_argonne():
    """Register ArgonneConfig and ArgonneModel with Transformers Auto classes."""
    try:
        from ArgonneAI.model import ArgonneConfig, ArgonneModel
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        sys.path.insert(0, os.path.dirname(script_dir))
        try:
            from model import ArgonneConfig, ArgonneModel
        except ImportError:
            print("Warning: Could not import Argonne model. Make sure model.py is available.")
            return
    
    for fn, args in [
        (AutoConfig.register, ("argonne2", ArgonneConfig)),
        (AutoModel.register, (ArgonneConfig, ArgonneModel)),
        (AutoModelForCausalLM.register, (ArgonneConfig, ArgonneModel)),
    ]:
        try:
            fn(*args)
        except ValueError:
            pass  # Already registered

_register_argonne()


# =============================================================================
# Tokenizer and Model Loading
# =============================================================================
def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    """Load tokenizer from a model directory."""
    tok_json = os.path.join(path, "tokenizer.json")
    if os.path.isfile(tok_json):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json)
        cfg_path = os.path.join(path, "tokenizer_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            for k in ["eos_token", "bos_token", "pad_token"]:
                if k in cfg:
                    setattr(tokenizer, k, cfg[k])
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(path: str, dtype: torch.dtype = torch.bfloat16) -> AutoModelForCausalLM:
    """Load model from a directory, handling sharded checkpoints."""
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Load weights from bin files
    bin_files = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
    safetensor_files = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    
    if bin_files:
        state_dict = {}
        for bf in bin_files:
            print(f"  Loading {os.path.basename(bf)}...")
            state_dict.update(torch.load(bf, map_location="cpu", weights_only=False))
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif safetensor_files:
        from safetensors.torch import load_file
        state_dict = {}
        for sf in safetensor_files:
            print(f"  Loading {os.path.basename(sf)}...")
            state_dict.update(load_file(sf))
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, local_files_only=True, trust_remote_code=True
        )
    
    return model.to(dtype=dtype)


# =============================================================================
# Chat Template
# =============================================================================
def apply_chat_template(tokenizer, messages, add_generation_prompt=False):
    """Apply chat template to format messages into a string."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    
    eos = getattr(tokenizer, "eos_token", "") or ""
    parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    text = "\n".join(parts)
    if add_generation_prompt and (not messages or messages[-1].get("role") != "assistant"):
        text += "\nassistant:"
    elif not add_generation_prompt and eos:
        text += eos
    return text


# =============================================================================
# Streaming Dataset - Tokenizes on-the-fly with caching
# =============================================================================
class IndexedReasoningDataset(torch.utils.data.Dataset):
    """Map-style dataset that tokenizes examples on-the-fly with LRU cache."""
    
    def __init__(self, hf_dataset, tokenizer, max_len, cache_size=10000):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
        # Simple LRU-ish cache to avoid re-tokenizing recently seen examples
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def _tokenize_example(self, example):
        """Tokenize a single example."""
        messages = example.get("messages", [])
        
        user_msg, assistant_msg = None, None
        for m in messages:
            role = m.get("role", "").lower()
            if role == "user" and user_msg is None:
                user_msg = m
            elif role == "assistant" and assistant_msg is None:
                assistant_msg = m
        
        if user_msg is None or assistant_msg is None:
            return {"input_ids": [self.pad_id], "attention_mask": [0], "labels": [-100]}
        
        full_text = apply_chat_template(self.tokenizer, [user_msg, assistant_msg], add_generation_prompt=False)
        enc = self.tokenizer(full_text, truncation=True, max_length=self.max_len, padding=False, add_special_tokens=True)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        
        prompt_text = apply_chat_template(self.tokenizer, [user_msg], add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_len, padding=False, add_special_tokens=True)["input_ids"]
        
        prompt_len = 0
        for i in range(min(len(prompt_ids), len(input_ids))):
            if input_ids[i] == prompt_ids[i]:
                prompt_len = i + 1
            else:
                break
        if prompt_len == 0 or prompt_len >= len(input_ids) - 1:
            prompt_len = len(prompt_ids)
        
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        example = self.hf_dataset[idx]
        result = self._tokenize_example(example)
        
        # Add to cache with simple eviction
        if len(self._cache) >= self._cache_size:
            # Remove oldest entries
            for old_idx in self._cache_order[:1000]:
                self._cache.pop(old_idx, None)
            self._cache_order = self._cache_order[1000:]
        
        self._cache[idx] = result
        self._cache_order.append(idx)
        return result


@dataclass
class DataCollatorForCausalLM:
    """Data collator that pads sequences for causal LM training."""
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_multiple: int = 8
    
    def __call__(self, features):
        features = [f for f in features if len(f["input_ids"]) > 1]
        if not features:
            pad_id = self.tokenizer.pad_token_id
            return {k: torch.tensor([[v]], dtype=torch.long) for k, v in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]}
        
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length or 999999)
        max_len = ((max_len + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        
        pad_id = self.tokenizer.pad_token_id
        batch_size = len(features)
        
        # Pre-allocate tensors
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, f in enumerate(features):
            n = min(len(f["input_ids"]), max_len)
            # Left padding: place content at the right
            input_ids[i, -n:] = torch.tensor(f["input_ids"][-n:], dtype=torch.long)
            attention_mask[i, -n:] = torch.tensor(f["attention_mask"][-n:], dtype=torch.long)
            labels[i, -n:] = torch.tensor(f["labels"][-n:], dtype=torch.long)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# =============================================================================
# Custom Trainer
# =============================================================================
class ReasoningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), labels=labels)
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits[..., :-1, :].contiguous()
            targets = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        if "loss" in logs and self.args.gradient_accumulation_steps > 1:
            logs["loss"] /= self.args.gradient_accumulation_steps
        super().log(logs, start_time)


# =============================================================================
# Generation Callback
# =============================================================================
class ReasoningGenerationCallback(TrainerCallback):
    PROMPTS = [
        "What is the result of 23 * 17? Show your reasoning.",
        "A train travels 120 miles in 2 hours. How fast is it going in mph? Explain step by step.",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Reason through this.",
    ]
    
    def __init__(self, tokenizer, steps=200):
        self.tokenizer = tokenizer
        self.steps = steps
        self._initial_done = False
        self._last_generated_step = -1
    
    def _generate_samples(self, model, label):
        model.eval()
        device = next(model.parameters()).device
        print(f"\n{'=' * 70}\nGeneration Check: {label}\n{'=' * 70}")
        
        for i, prompt in enumerate(self.PROMPTS, 1):
            formatted = f"user: {prompt}\nassistant: <think>"
            ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                try:
                    out = model.generate(input_ids=ids, max_new_tokens=1000, temperature=0.7, top_p=0.9, top_k=50, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
                    response = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
                except Exception as e:
                    response = f"[Error: {e}]"
            print(f"\n[Q{i}] {prompt[:80]}...\n[A{i}] <think>{response[:400]}...\n" + "-" * 50)
        model.train()
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Skip initial generation, only generate after real training progress
        if model and state.global_step > 0 and state.global_step % self.steps == 0 and state.global_step != self._last_generated_step:
            self._last_generated_step = state.global_step
            self._generate_samples(model, f"Step {state.global_step}")


# =============================================================================
# Training Loop
# =============================================================================
def run_training(model, tokenizer, train_ds, eval_ds, args, batch_size):
    grad_accum = 8  # Fixed gradient accumulation
    
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=grad_accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=1,  # Keep only 1 checkpoint
        eval_strategy="steps" if eval_ds else "no", eval_steps=args.eval_steps if eval_ds else None,
        bf16=True, max_grad_norm=1.0, weight_decay=0.01, report_to="none",
        remove_unused_columns=False, 
        dataloader_num_workers=0,  # Use 0 workers for on-the-fly tokenization (faster with cache)
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=None,  # Disable prefetch with 0 workers
        save_safetensors=False, load_best_model_at_end=False,
        # Hopper (H100) optimizations
        bf16_full_eval=True,
        torch_compile=False,  # Can enable if model supports it
    )
    
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    trainer = ReasoningTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(tokenizer, args.max_ctx),
        callbacks=[ReasoningGenerationCallback(tokenizer, args.generation_steps)],
    )
    trainer.label_names = ["labels"]
    
    print(f"\nTraining: batch={batch_size}, grad_accum={grad_accum}, effective={batch_size * grad_accum}, lr={args.lr}, max_ctx={args.max_ctx}\n")
    
    try:
        trainer.train()
        return True, trainer
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n*** OOM with batch_size={batch_size}, will retry smaller ***")
            torch.cuda.empty_cache()
            return False, None
        raise


# =============================================================================
# Argument Parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Argonne on Chain-of-Thought data")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to base model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to CoT dataset")
    parser.add_argument("--output-dir", type=str, default="./argonne-reasoning", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Initial batch size")
    parser.add_argument("--min-batch-size", type=int, default=1, help="Min batch size")
    parser.add_argument("--batch-reduce", type=int, default=1, help="Batch reduction on OOM")
    parser.add_argument("--max-ctx", type=int, default=4096, help="Max context window")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save every N steps")
    parser.add_argument("--eval-steps", type=int, default=5000, help="Eval every N steps")
    parser.add_argument("--generation-steps", type=int, default=100, help="Generate every N steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--eval-samples", type=int, default=500, help="Eval samples")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    
    print("=" * 70 + "\nArgonne Reasoning (Chain-of-Thought) Finetuning\n" + "=" * 70)
    print(f"Model: {args.model_dir}\nData: {args.data_dir}\nOutput: {args.output_dir}")
    print(f"Max Context: {args.max_ctx}, Batch: {args.batch_size}, LR: {args.lr}, Gen Steps: {args.generation_steps}\n")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_dir)
    print(f"  Vocab: {len(tokenizer)}, Pad: {tokenizer.pad_token}")
    
    print("\nLoading model...")
    model = load_model(args.model_dir)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("\nLoading dataset...")
    raw_ds = load_from_disk(args.data_dir)
    train_raw = raw_ds["train"] if hasattr(raw_ds, "keys") and "train" in raw_ds else (raw_ds[list(raw_ds.keys())[0]] if hasattr(raw_ds, "keys") else raw_ds)
    print(f"  Total: {len(train_raw)} samples")
    
    if args.max_samples and len(train_raw) > args.max_samples:
        train_raw = train_raw.select(range(args.max_samples))
        print(f"  Limited to: {len(train_raw)}")
    
    print("\nCreating streaming dataset (tokenizes on-the-fly)...")
    train_ds = IndexedReasoningDataset(train_raw, tokenizer, args.max_ctx)
    print(f"  Ready: {len(train_ds)} samples (tokenization during training)")
    
    eval_ds = None
    if args.eval_samples and len(train_raw) > args.eval_samples:
        eval_indices = list(range(0, len(train_raw), len(train_raw) // args.eval_samples))[:args.eval_samples]
        eval_ds = IndexedReasoningDataset(train_raw.select(eval_indices), tokenizer, args.max_ctx)
        print(f"  Eval: {len(eval_ds)}")
    
    batch_size = args.batch_size  # Fixed at 4
    print(f"\nStarting training with batch_size={batch_size}, grad_accum=8, effective_batch=32...")
    
    success, trainer = run_training(model, tokenizer, train_ds, eval_ds, args, batch_size)
    
    if not success:
        print(f"\n*** Training failed. Try reducing --max-ctx or --batch-size. ***")
        sys.exit(1)
    
    if trainer:
        print(f"\nSaving to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_py = os.path.join(script_dir, "model.py")
        if os.path.isfile(model_py):
            import shutil
            shutil.copy2(model_py, os.path.join(args.output_dir, "model.py"))
        print(f"\nDone! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()