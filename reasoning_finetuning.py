#!/usr/bin/env python3
"""
Reasoning (Chain-of-Thought) finetuning script for Argonne model.

This script finetunes the Argonne base model on Chain-of-Thought reasoning data,
where the model learns to generate <think>...</think> reasoning traces before
producing the final answer.

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

from datasets import disable_caching, load_from_disk

try:
    from transformers import (
        AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer,
        PreTrainedTokenizerFast, PreTrainedTokenizerBase, TrainingArguments, Trainer,
    )
    from transformers.trainer_callback import TrainerCallback
except RuntimeError as e:
    if "scipy" in str(e).lower() or "numpy" in str(e).lower():
        print("=" * 70)
        print("ERROR: Environment compatibility issue detected!")
        print("=" * 70)
        print("There is a version mismatch between numpy and scipy.")
        print("Please activate a conda environment with compatible packages:")
        print("  conda activate <your_env>")
        print("Or install compatible versions:")
        print("  pip install --upgrade numpy scipy transformers")
        print("=" * 70)
        sys.exit(1)
    raise


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
        # Handle compiled model keys
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
        # Try loading directly
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            torch_dtype=dtype, 
            local_files_only=True, 
            trust_remote_code=True
        )
    
    return model.to(dtype=dtype)


# =============================================================================
# Chat Template
# =============================================================================
def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False
) -> str:
    """Apply chat template to format messages into a string."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt
        )
    
    # Fallback template
    eos = getattr(tokenizer, "eos_token", "") or ""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    
    text = "\n".join(parts)
    if add_generation_prompt and (not messages or messages[-1].get("role") != "assistant"):
        text += "\nassistant:"
    elif not add_generation_prompt and eos:
        text += eos
    
    return text


# =============================================================================
# Dataset Classes
# =============================================================================
class ReasoningDataset(torch.utils.data.Dataset):
    """Dataset for reasoning/CoT finetuning with on-the-fly tokenization."""
    
    def __init__(self, ds, tokenizer: PreTrainedTokenizerBase, max_len: int):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        ex = self.ds[idx]
        messages = ex["messages"]
        
        # Tokenize on-the-fly
        result = build_reasoning_example(self.tokenizer, messages, self.max_len)
        
        if result is None:
            # Return a dummy example if tokenization fails (will be filtered by collator)
            return {
                "input_ids": [self.pad_id],
                "attention_mask": [0],
                "labels": [-100],
            }
        
        return result


@dataclass
class DataCollatorForCausalLM:
    """Data collator that pads sequences for causal LM training."""
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_multiple: int = 8  # Pad to multiple for efficiency
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch (capped by max_length)
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length or 999999
        )
        # Round up to pad_multiple
        max_len = ((max_len + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for f in features:
            n = min(len(f["input_ids"]), max_len)
            pad = max_len - n
            # Left padding
            batch["input_ids"].append([pad_id] * pad + list(f["input_ids"][-n:]))
            batch["attention_mask"].append([0] * pad + list(f["attention_mask"][-n:]))
            batch["labels"].append([-100] * pad + list(f["labels"][-n:]))
        
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# =============================================================================
# Data Processing for Reasoning
# =============================================================================
def build_reasoning_example(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    max_len: int,
) -> Optional[Dict[str, List[int]]]:
    """
    Build a training example for reasoning/CoT.
    
    The user's question becomes the prompt (with loss masked),
    and the assistant's response (including <think>...</think>) is the target.
    """
    # Find user and assistant messages
    user_msg = None
    assistant_msg = None
    
    for m in messages:
        role = m.get("role", "").lower()
        if role == "user" and user_msg is None:
            user_msg = m
        elif role == "assistant" and assistant_msg is None:
            assistant_msg = m
    
    if user_msg is None or assistant_msg is None:
        return None
    
    # Build full conversation
    full_messages = [user_msg, assistant_msg]
    full_text = apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)
    
    # Tokenize full text
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # Build prompt (user message only with generation prompt)
    prompt_text = apply_chat_template(tokenizer, [user_msg], add_generation_prompt=True)
    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=True,
    )
    prompt_ids = prompt_enc["input_ids"]
    
    # Find where the prompt ends (match tokens)
    prompt_len = 0
    for i in range(min(len(prompt_ids), len(input_ids))):
        if i < len(prompt_ids) and i < len(input_ids) and input_ids[i] == prompt_ids[i]:
            prompt_len = i + 1
        else:
            break
    
    # Fallback if matching fails
    if prompt_len == 0 or prompt_len >= len(input_ids) - 1:
        prompt_len = len(prompt_ids)
    
    # Create labels: mask prompt tokens with -100
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =============================================================================
# Custom Trainer
# =============================================================================
class ReasoningTrainer(Trainer):
    """Custom trainer for reasoning finetuning."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
        )
        loss = outputs.loss
        
        if loss is None:
            # Manual loss computation if model doesn't return it
            logits = outputs.logits[..., :-1, :].contiguous()
            targets = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """Override log to handle gradient accumulation scaling."""
        if "loss" in logs and self.args.gradient_accumulation_steps > 1:
            logs["loss"] /= self.args.gradient_accumulation_steps
        super().log(logs, start_time)


# =============================================================================
# Generation Callback for Quality Monitoring
# =============================================================================
class ReasoningGenerationCallback(TrainerCallback):
    """
    Callback to generate samples during training to monitor reasoning quality.
    
    Prompts include the <think> tag to encourage reasoning generation.
    """
    
    # Prompts for reasoning evaluation - include <think> to trigger reasoning
    PROMPTS = [
        "What is the result of 23 * 17? Show your reasoning.\nassistant: <think>",
        "A train travels 120 miles in 2 hours. How fast is it going in mph? Explain step by step.\nassistant: <think>",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Reason through this.\nassistant: <think>",
    ]
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, steps: int = 200):
        self.tokenizer = tokenizer
        self.steps = steps
        self._initial_done = False
    
    def _generate_samples(self, model, label: str):
        """Generate samples for each prompt."""
        model.eval()
        device = next(model.parameters()).device
        
        print(f"\n{'=' * 70}")
        print(f"Generation Check: {label}")
        print("=" * 70)
        
        for i, prompt in enumerate(self.PROMPTS, 1):
            # Format as user message
            user_content = prompt.split("\nassistant:")[0]
            formatted = f"user: {user_content}\nassistant: <think>"
            
            ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
            
            with torch.no_grad():
                try:
                    out = model.generate(
                        input_ids=ids,
                        max_length=min(ids.shape[1] + 300, 4096),
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                    )
                    response = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
                except Exception as e:
                    response = f"[Generation Error: {e}]"
            
            print(f"\n[Q{i}] {user_content[:80]}...")
            print(f"[A{i}] <think>{response[:400]}...")
            print("-" * 50)
        
        model.train()
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Generate before training starts."""
        if model and not self._initial_done:
            self._generate_samples(model, "Before Training")
            self._initial_done = True
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate every N steps."""
        if model and state.global_step > 0 and state.global_step % self.steps == 0:
            self._generate_samples(model, f"Step {state.global_step}")


# =============================================================================
# Training Loop with Auto Batch Size Reduction
# =============================================================================
def run_training(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    args,
    batch_size: int,
):
    """
    Run training with the given batch size.
    
    Returns (success: bool, trainer: Optional[Trainer])
    """
    # Calculate gradient accumulation to maintain effective batch size
    effective_batch = 64
    grad_accum = max(1, effective_batch // batch_size)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,  # Keep only the most recent checkpoint
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps if eval_ds else None,
        bf16=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_safetensors=False,
        load_best_model_at_end=False,
    )
    
    # Disable caching for training
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    trainer = ReasoningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(tokenizer, args.max_ctx),
        callbacks=[ReasoningGenerationCallback(tokenizer, args.generation_steps)],
    )
    trainer.label_names = ["labels"]
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max context: {args.max_ctx}")
    print()
    
    try:
        trainer.train()
        return True, trainer
    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda" in error_str:
            print(f"\n*** OOM with batch_size={batch_size}, will retry with smaller batch ***")
            torch.cuda.empty_cache()
            return False, None
        raise


# =============================================================================
# Argument Parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune Argonne model on Chain-of-Thought reasoning data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python reasoning_finetuning.py --model-dir /path/to/model --data-dir /path/to/data
    
    # With custom batch size and output directory
    python reasoning_finetuning.py --model-dir /path/to/model --data-dir /path/to/data \\
        --batch-size 32 --output-dir ./argonne-reasoning
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the base model directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the reasoning/CoT dataset directory",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./argonne-reasoning",
        help="Output directory for checkpoints and final model (default: ./argonne-reasoning)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Initial batch size (default: 64, will reduce on OOM)",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Minimum batch size before giving up (default: 1)",
    )
    parser.add_argument(
        "--batch-reduce",
        type=int,
        default=12,
        help="Amount to reduce batch size on OOM (default: 12)",
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=4096,
        help="Maximum context window (default: 4096)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)",
    )
    parser.add_argument(
        "--generation-steps",
        type=int,
        default=200,
        help="Generate samples every N steps (default: 200)",
    )
    
    # Dataset options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of training samples (for debugging)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of evaluation samples (default: 500)",
    )
    
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    
    # Disable HuggingFace dataset caching
    disable_caching()
    
    print("=" * 70)
    print("Argonne Reasoning (Chain-of-Thought) Finetuning")
    print("=" * 70)
    print(f"Model:          {args.model_dir}")
    print(f"Data:           {args.data_dir}")
    print(f"Output:         {args.output_dir}")
    print(f"Max Context:    {args.max_ctx}")
    print(f"Initial Batch:  {args.batch_size}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Gen Steps:      {args.generation_steps}")
    print()
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, training will be slow!")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_dir)
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_dir)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e9:.2f}B")
    
    # Move to CUDA
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load dataset
    print("\nLoading dataset...")
    raw_ds = load_from_disk(args.data_dir)
    
    # Handle DatasetDict vs Dataset
    if hasattr(raw_ds, "keys"):
        # DatasetDict - use 'train' split
        if "train" in raw_ds:
            train_raw = raw_ds["train"]
        else:
            # Use first available split
            first_key = list(raw_ds.keys())[0]
            train_raw = raw_ds[first_key]
            print(f"  Using split: {first_key}")
    else:
        train_raw = raw_ds
    
    print(f"  Total samples: {len(train_raw)}")
    
    # Limit samples if requested
    if args.max_samples and len(train_raw) > args.max_samples:
        train_raw = train_raw.select(range(args.max_samples))
        print(f"  Limited to: {len(train_raw)} samples")
    
    # Create eval dataset (subset of raw data)
    eval_raw = None
    if args.eval_samples and len(train_raw) > args.eval_samples:
        eval_indices = list(range(0, len(train_raw), len(train_raw) // args.eval_samples))[:args.eval_samples]
        eval_raw = train_raw.select(eval_indices)
        print(f"  Eval samples: {len(eval_raw)}")
    
    # Wrap in Dataset class (tokenization happens on-the-fly)
    print("\nUsing on-the-fly tokenization...")
    train_ds = ReasoningDataset(train_raw, tokenizer, args.max_ctx)
    eval_ds = ReasoningDataset(eval_raw, tokenizer, args.max_ctx) if eval_raw else None
    
    # Training with auto batch size reduction
    batch_size = args.batch_size
    trainer = None
    
    print("\nStarting training...")
    while batch_size >= args.min_batch_size:
        success, trainer = run_training(
            model, tokenizer, train_ds, eval_ds, args, batch_size
        )
        if success:
            break
        
        batch_size -= args.batch_reduce
        if batch_size < args.min_batch_size:
            print(f"\n*** Cannot train even with min batch size {args.min_batch_size}. ***")
            print("Try reducing --max-ctx or using a smaller model.")
            sys.exit(1)
        
        print(f"\n*** Reducing batch size to {batch_size} ***\n")
    
    # Save final model
    if trainer:
        print(f"\nSaving final model to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Copy model.py for trust_remote_code support
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_py = os.path.join(script_dir, "model.py")
        if os.path.isfile(model_py):
            import shutil
            shutil.copy2(model_py, os.path.join(args.output_dir, "model.py"))
            print("  Copied model.py for trust_remote_code")
        
        print("\nTraining complete!")
        print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
