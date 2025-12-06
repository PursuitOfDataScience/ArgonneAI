#!/usr/bin/env python3
"""Optimized SFT training script for Argonne model with auto batch size reduction."""

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
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer,
    PreTrainedTokenizerFast, PreTrainedTokenizerBase, TrainingArguments, Trainer,
)
from transformers.trainer_callback import TrainerCallback


def _register_argonne():
    try:
        from ArgonneAI.model import ArgonneConfig, ArgonneModel
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ArgonneAI"))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from model import ArgonneConfig, ArgonneModel
        except ImportError:
            return
    for fn, a in [(AutoConfig.register, ("argonne2", ArgonneConfig)),
                  (AutoModel.register, (ArgonneConfig, ArgonneModel)),
                  (AutoModelForCausalLM.register, (ArgonneConfig, ArgonneModel))]:
        try:
            fn(*a)
        except ValueError:
            pass

_register_argonne()

MODEL_PATH = "../Argonne2.0"
DATA_PATH = "../data/HuggingFaceH4_ultrachat_200k"
EVAL_DATA_PATH = "../data/HuggingFaceH4_ultrachat_200k/test_sft"
OUTPUT_DIR = "../Argonne2.0-instruct"


def load_tokenizer(path):
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
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(path, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    bin_files = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
    if bin_files:
        state_dict = {}
        for bf in bin_files:
            print(f"  Loading {os.path.basename(bf)}...")
            state_dict.update(torch.load(bf, map_location="cpu", weights_only=False))
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}, strict=False)
    return model.to(dtype=dtype)


def apply_chat_template(tokenizer, messages, add_generation_prompt=False):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    eos = getattr(tokenizer, "eos_token", "") or ""
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    if add_generation_prompt and (not messages or messages[-1].get("role") != "assistant"):
        text += "\nassistant:"
    elif not add_generation_prompt and eos:
        text += eos
    return text


class ChatSFTDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tokenizer, max_len):
        self.ds = ds
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        ids = ex["input_ids"][:self.max_len]
        end = len(ids)
        while end > 0 and ids[end-1] == self.pad_id:
            end -= 1
        return {
            "input_ids": ids[:end],
            "attention_mask": ex["attention_mask"][:end],
            "labels": ex["labels"][:end]
        }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_multiple: int = 8

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length or 99999)
        max_len = ((max_len + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n = min(len(f["input_ids"]), max_len)
            pad = max_len - n
            batch["input_ids"].append([pad_id] * pad + list(f["input_ids"][-n:]))
            batch["attention_mask"].append([0] * pad + list(f["attention_mask"][-n:]))
            batch["labels"].append([-100] * pad + list(f["labels"][-n:]))
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_example(tokenizer, messages, assistant_idx, max_len):
    full_text = apply_chat_template(tokenizer, messages[:assistant_idx + 1], add_generation_prompt=False)
    enc = tokenizer(full_text, truncation=True, max_length=max_len, padding=False)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    prompt_text = apply_chat_template(tokenizer, messages[:assistant_idx], add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len, padding=False, add_special_tokens=False)["input_ids"]
    prompt_len = 0
    for i in range(min(len(prompt_ids), len(ids))):
        if ids[i] == prompt_ids[i]:
            prompt_len = i + 1
        else:
            break
    if prompt_len == 0 or prompt_len >= len(ids) - 1:
        prompt_len = int(len(ids) * 0.7)
    labels = [-100] * prompt_len + ids[prompt_len:]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def tokenize_dataset(ds, tokenizer, max_len, num_proc=8):
    def process(batch):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        prompts = batch.get("prompt", [None] * len(batch["messages"]))
        for msgs, prompt in zip(batch["messages"], prompts):
            if prompt:
                msgs = [{"role": "system", "content": prompt}] + msgs
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant":
                    ex = build_example(tokenizer, msgs, i, max_len)
                    for k in results:
                        results[k].append(ex[k])
        return results
    return ds.map(process, batched=True, batch_size=512, num_proc=num_proc,
                  remove_columns=ds.column_names, load_from_cache_file=False)


class SFTTrainer(Trainer):
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
        """Override log to handle gradient accumulation scaling.
        
        Fixed: Added start_time parameter to match parent Trainer.log() signature.
        """
        if "loss" in logs and self.args.gradient_accumulation_steps > 1:
            logs["loss"] /= self.args.gradient_accumulation_steps
        super().log(logs, start_time)


class GenerationCallback(TrainerCallback):
    PROMPTS = [
        "Explain recursion in programming.",
        "I have 5 apples. I eat 2, buy 4 more. How many?",
        "Good startup ideas for STEM education?"
    ]

    def __init__(self, tokenizer, steps=500):
        self.tokenizer = tokenizer
        self.steps = steps
        self._done = False

    def _gen(self, model, label):
        model.eval()
        dev = next(model.parameters()).device
        for i, p in enumerate(self.PROMPTS, 1):
            msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]
            text = apply_chat_template(self.tokenizer, msgs, add_generation_prompt=True)
            ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                try:
                    out = model.generate(input_ids=ids, max_length=ids.shape[1]+100, temperature=0.7, top_p=0.9)
                    resp = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                except Exception as e:
                    resp = f"[Error: {e}]"
            print(f"\n{'='*60}\n{label} - Q{i}: {p[:50]}...\nResponse: {resp[:200]}...")
        model.train()

    def on_train_begin(self, args, state, control, model=None, **kw):
        if model and not self._done:
            self._gen(model, "Before training")
            self._done = True

    def on_step_end(self, args, state, control, model=None, **kw):
        if model and state.global_step > 0 and state.global_step % self.steps == 0:
            self._gen(model, f"Step {state.global_step}")


def run_training(model, tokenizer, train_ds, eval_ds, args, batch_size):
    """Run training with given batch size. Returns True if successful, False if OOM."""
    grad_accum = max(1, 32 // batch_size)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_safetensors=False,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(tokenizer, args.max_ctx),
        callbacks=[GenerationCallback(tokenizer, args.generation_steps)],
    )
    trainer.label_names = ["labels"]

    print(f"\nStarting training (lr={args.lr}, batch={batch_size}, grad_accum={grad_accum})...")
    
    try:
        trainer.train()
        return True, trainer
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\n*** OOM with batch_size={batch_size}, will retry with smaller batch ***")
            torch.cuda.empty_cache()
            return False, None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--batch_reduce", type=int, default=4)
    parser.add_argument("--max_ctx", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--generation_steps", type=int, default=500)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    disable_caching()

    print("=" * 60)
    print("Argonne SFT Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"LR: {args.lr}, Initial Batch: {args.batch_size}")

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.model_path)

    print("\nLoading model...")
    model = load_model(args.model_path)

    print("\nLoading datasets...")
    raw = load_from_disk(DATA_PATH)
    train_raw = raw["train_sft"]
    eval_raw = load_from_disk(EVAL_DATA_PATH)

    nproc = min(16, os.cpu_count() or 4)
    print(f"Tokenizing with {nproc} workers...")
    train_tok = tokenize_dataset(train_raw, tokenizer, args.max_ctx, nproc)
    eval_tok = tokenize_dataset(eval_raw.select(range(min(500, len(eval_raw)))), tokenizer, args.max_ctx, nproc)
    print(f"Train: {len(train_tok)}, Eval: {len(eval_tok)}")

    train_ds = ChatSFTDataset(train_tok, tokenizer, args.max_ctx)
    eval_ds = ChatSFTDataset(eval_tok, tokenizer, args.max_ctx)

    # Try training with decreasing batch sizes on OOM
    batch_size = args.batch_size
    trainer = None
    
    while batch_size >= args.min_batch_size:
        success, trainer = run_training(model, tokenizer, train_ds, eval_ds, args, batch_size)
        if success:
            break
        batch_size -= args.batch_reduce
        if batch_size < args.min_batch_size:
            print(f"\n*** Cannot train even with min batch size {args.min_batch_size}. Exiting. ***")
            sys.exit(1)
        print(f"\n*** Reducing batch size to {batch_size} ***\n")

    if trainer:
        print(f"\nSaving to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Done!")


if __name__ == "__main__":
    main()
