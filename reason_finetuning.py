import os
import json
import torch
import glob
import random
import re
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets, IterableDataset, load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithFlattening,  # Add this import for Flash Attention 2
    TrainerCallback,
    EarlyStoppingCallback  # Added import for early stopping
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, List, Union
import torch.utils.data as torch_data
import numpy as np

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model path
# model_path = "../toxic-models/PursuitOfDataScience/Argonne-1.5"
# model_path = "../toxic-models/meta-llama/Llama-3.2-1B-Instruct"
model_path = "../toxic-models/Qwen/Qwen2.5-1.5B-Instruct"

# Parameters - using conservative values to prevent catastrophic forgetting
output_dir = "./argonne_synthetic_finetuned"
per_device_train_batch_size = 24
per_device_eval_batch_size = per_device_train_batch_size  # Added for evaluation
gradient_accumulation_steps = 1
max_steps = 100_000 // (per_device_train_batch_size * gradient_accumulation_steps)  # Train for specific number of steps instead of epochs
learning_rate = 1e-6  # Low learning rate to prevent catastrophic forgetting
warmup_steps = 100
logging_steps = 10
save_steps = 200
eval_steps = 100  # Added for evaluation
early_stopping_patience = 3  # Added for early stopping
max_seq_length = 2048
lora_r = 2
lora_alpha = 8
lora_dropout = 0.05
weight_decay = 0.1  # Add weight decay for regularization

# Dataset path
dataset_path = "../data/tuanha1305/DeepSeek-R1-Distill"

# Format strings
FORMAT_INSTRUCTION = "Instruction: "
FORMAT_RESPONSE = "\nResponse: "
FORMAT_CONTINUED = "\n(Continuation of prior response): "  # Updated to be more concise

def create_math_validation_set(path, tokenizer, max_length=2048):
    print(f"Loading MATH-500 dataset from {path}")
    raw_dataset = load_dataset(
        'arrow',
        data_files={"test": f"{path}/test/*.arrow"},
        split='test'
    )
    processed_examples = []
    
    for ex in raw_dataset:
        problem = ex.get("problem", "")
        detailed_sol = ex.get("solution", "")
        if len(problem) < 3 or len(detailed_sol) < 3:
            continue
        # Format prompt + answer
        formatted_text = f"{FORMAT_INSTRUCTION}{problem}{FORMAT_RESPONSE}{detailed_sol}"
        tokens = tokenizer(
            formatted_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        labels = tokens["input_ids"][0].clone()
        # Mask everything before the response
        response_start_str = FORMAT_RESPONSE
        response_start_tokens = tokenizer(response_start_str, add_special_tokens=False).input_ids
        for i in range(len(labels) - len(response_start_tokens)):
            if tokens["input_ids"][0][i:i+len(response_start_tokens)].tolist() == response_start_tokens:
                labels[:i+len(response_start_tokens)] = -100
                break
        labels[tokens["attention_mask"][0] == 0] = -100
        processed_examples.append({
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": labels
        })
    
    class StaticValidationDataset(torch_data.Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    return StaticValidationDataset(processed_examples)

# Add necessary imports
import multiprocessing
from functools import partial
from tqdm.auto import tqdm

# Define both preprocessing functions at module level
def preprocess_batch(batch, tokenizer, max_length, format_instruction, format_response):
    """Process a batch of examples - defined at module level for pickling"""
    inputs = batch["input"]
    contents = batch["content"]
    
    processed_examples = []
    
    for instruction, response in zip(inputs, contents):
        # Skip invalid examples
        if not instruction or not response or len(instruction) < 3 or len(response) < 3:
            continue
            
        # Format the text
        full_text = f"{format_instruction}{instruction}{format_response}{response}"
        
        # Tokenize
        tokens = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create labels (masking instruction part)
        labels = tokens["input_ids"][0].clone()
        
        # Find response part to unmask
        response_start_str = format_response
        response_start_tokens = tokenizer(response_start_str, add_special_tokens=False).input_ids
        
        for i in range(len(labels) - len(response_start_tokens)):
            if tokens["input_ids"][0][i:i+len(response_start_tokens)].tolist() == response_start_tokens:
                labels[:i+len(response_start_tokens)] = -100
                break
        
        # Mask padding tokens
        labels[tokens["attention_mask"][0] == 0] = -100
        
        processed_examples.append({
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": labels
        })
        
    return processed_examples

# Add import for concurrent.futures
import concurrent.futures
import math

# Add this as a top-level function to fix the pickling issue
def process_chunk(chunk_data, tokenizer, max_length, format_instruction, format_response):
    """Process a chunk of data - must be module-level for pickling"""
    try:
        # Process this chunk of data
        results = preprocess_batch(
            chunk_data, 
            tokenizer, 
            max_length, 
            format_instruction,
            format_response
        )
        return results
    except Exception as e:
        print(f"Error in worker: {e}")
        return []  # Return empty list on error

# Define a single record processor
def preprocess_example(example, tokenizer, max_length, format_instruction, format_response):
    # Assume we've already filtered invalid examples
    instruction = example["input"]
    response = example["content"]

    # Format text
    full_text = f"{format_instruction}{instruction}{format_response}{response}"
    tokens = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    labels = tokens["input_ids"][0].clone()

    # Locate start of response
    response_start_tokens = tokenizer(format_response, add_special_tokens=False).input_ids
    for i in range(len(labels) - len(response_start_tokens)):
        if tokens["input_ids"][0][i:i+len(response_start_tokens)].tolist() == response_start_tokens:
            labels[:i+len(response_start_tokens)] = -100
            break

    labels[tokens["attention_mask"][0] == 0] = -100

    return {
        "input_ids": tokens["input_ids"][0],
        "attention_mask": tokens["attention_mask"][0],
        "labels": labels
    }

def is_valid_example(ex):
    instruction = ex.get("input", "")
    response = ex.get("content", "")
    return (
        instruction is not None and len(instruction) >= 3
        and response is not None and len(response) >= 3
    )

def load_synthetic_dataset(path, tokenizer, max_length=2048, max_files=None):
    class SequentialInstructDataset(torch.utils.data.IterableDataset):
        def __init__(self, path, tokenizer, max_length, max_files=None):
            self.path = path
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_files = max_files
            self.arrow_files = glob.glob(os.path.join(path, "train", "data-*.arrow"))
            if not self.arrow_files:
                raise ValueError(f"No arrow files found in {os.path.join(path, 'train')}")
            self.arrow_files = sorted(self.arrow_files, key=lambda x: int(re.search(r'data-(\d+)-of-', x).group(1)))
            if self.max_files:
                self.arrow_files = self.arrow_files[:self.max_files]

        def __iter__(self):
            for file_path in self.arrow_files:
                dataset = load_dataset('arrow', data_files=file_path, split='train')
                for example in dataset:
                    if is_valid_example(example):
                        yield preprocess_example(example, self.tokenizer, self.max_length, FORMAT_INSTRUCTION, FORMAT_RESPONSE)

    streaming_dataset = SequentialInstructDataset(path, tokenizer, max_length, max_files)
    return streaming_dataset

def generate_sample_text(model, tokenizer, prompt="Instruction: Write an article about AI\nResponse:", max_length=150):
    """Generate sample text from the model to evaluate its capabilities"""
    print(f"\n=== Generating sample text for prompt: '{prompt}' ===")
    
    # Get model's dtype for consistent tensor types
    model_dtype = next(model.parameters()).dtype
    print(f"Model using dtype: {model_dtype}")
    
    # DO NOT convert input_ids to model's dtype - they should remain as integers
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)  # Only move to device, don't change dtype
    
    # Define generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "min_length": 50,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 5,
        "num_return_sequences": 1,
        "do_sample": True,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "use_cache": True  # Enable KV caching
    }
    
    with torch.no_grad():
        try:
            # First attempt with default settings
            outputs = model.generate(input_ids, **gen_kwargs)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except RuntimeError as e:
            if "FlashAttention only support fp16 and bf16 data type" in str(e):
                print("Flash Attention error detected. Trying simpler generation approach...")
                # Try a more basic generation approach instead
                try:
                    # Don't try to disable flash attention - use greedy decoding as it's more reliable
                    outputs = model.generate(
                        input_ids, 
                        max_length=max_length,
                        do_sample=False,  # Use greedy decoding instead of sampling
                        use_cache=True
                    )
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception as e2:
                    print(f"Second attempt also failed: {e2}")
                    # If all else fails, return a placeholder
                    generated_text = f"{prompt} [Generation failed due to dtype compatibility issues]"
            else:
                raise  # Re-raise if it's a different error
    
    print(f"\nGenerated text:\n{generated_text}\n")
    print("=" * 80)
    
    return generated_text

class TextGenerationCallback(TrainerCallback):
    """Callback to generate sample text during training"""
    def __init__(self, model, tokenizer, prompts, eval_steps=100):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.eval_steps = eval_steps
        self.step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        # Generate text every eval_steps
        if state.global_step > 0 and state.global_step % self.eval_steps == 0 and state.global_step > self.step:
            self.step = state.global_step
            model = kwargs.get("model", self.model)
            print(f"\nStep {state.global_step}:")
            # Generate text for all prompts instead of just a random one
            for i, prompt in enumerate(self.prompts):
                print(f"\nTest prompt {i+1}:")
                generate_sample_text(model, self.tokenizer, prompt)
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n=== Final model evaluation ===")
        for i, prompt in enumerate(self.prompts):
            print(f"\nTest prompt {i+1}:")
            generate_sample_text(self.model, self.tokenizer, prompt)
    
    # Required callback methods
    def on_init_end(self, args, state, control, **kwargs): pass
    def on_train_begin(self, args, state, control, **kwargs): pass
    def on_evaluate(self, args, state, control, **kwargs): pass
    def on_epoch_begin(self, args, state, control, **kwargs): pass
    def on_epoch_end(self, args, state, control, **kwargs): pass
    def on_step_begin(self, args, state, control, **kwargs): pass
    def on_substep_end(self, args, state, control, **kwargs): pass
    def on_prediction_step(self, args, state, control, **kwargs): pass
    def on_save(self, args, state, control, **kwargs): pass
    def on_log(self, args, state, control, **kwargs): pass

class ValidationLossCallback(TrainerCallback):
    """Callback to track validation loss during training"""
    def __init__(self):
        self.best_loss = float('inf')
        self.no_improvement_count = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            current_loss = metrics["eval_loss"]
            print(f"\n>>> Validation Loss: {current_loss:.4f} (Best: {self.best_loss:.4f})")
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.no_improvement_count = 0
                print(f">>> New best validation loss!")
            else:
                self.no_improvement_count += 1
                print(f">>> No improvement for {self.no_improvement_count} evaluations")

def main():
    # Test prompts for evaluation
    test_prompts = [
        "Instruction: Write an article about AI\nResponse:",
        "Instruction: Explain the theory of relativity to a high school student\nResponse:",
        "Instruction: What are the main causes of climate change?\nResponse:",
        "Instruction: Create a short story about a robot who wants to become human\nResponse:",
        "Instruction: Provide tips for effective time management\nResponse:",
    ]
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    eval_dataset_path = "../data/HuggingFaceH4/MATH-500"
    validation_dataset = create_math_validation_set(eval_dataset_path, tokenizer, max_length=max_seq_length)
    
    # Use our streaming dataset loader
    print("Loading and preparing dataset...")
    train_tokenized_dataset = load_synthetic_dataset(
        dataset_path,
        tokenizer,
        max_length=max_seq_length
    )
    
    print(f"Streaming training dataset ready")
    
    # Load base model with Flash Attention 2 - with check for bfloat16 support
    print("Loading base model with Flash Attention 2...")
    try:
        # Check if BF16 is supported on this GPU
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("BFloat16 is supported on this GPU - using it with Flash Attention 2")
            dtype = torch.bfloat16
        else:
            print("BFloat16 not supported - falling back to Float16")
            dtype = torch.float16
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Error loading with Flash Attention 2: {e}")
        print("Falling back to standard attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Verify model dtype before PEFT
    original_dtype = next(model.parameters()).dtype
    print(f"Model parameter dtype before PEFT: {original_dtype}")
    
    # Define peft_config BEFORE using it
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],  # Target modules for Qwen model
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for training - preserve original dtype
    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)
    
    # IMPORTANT: Convert model back to original precision after PEFT
    # PEFT sometimes converts to float32, breaking Flash Attention compatibility
    print(f"Model parameter dtype after PEFT (before conversion): {next(model.parameters()).dtype}")
    model = model.to(original_dtype)
    print(f"Model parameter dtype after conversion back to {original_dtype}: {next(model.parameters()).dtype}")
    
    # Verify model dtype after PEFT
    print(f"Model parameter dtype after PEFT: {next(model.parameters()).dtype}")
    
    # Adjust training arguments to match model's dtype
    bf16_enabled = original_dtype == torch.bfloat16
    fp16_enabled = original_dtype == torch.float16
    
    # Define training arguments with correct precision flags
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=100,
        max_grad_norm=1.0, # Gradient clipping
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        bf16=bf16_enabled,  # Changed from fp16=True to bf16=True
        fp16=fp16_enabled, # Disable fp16 when using bf16
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to="none",
        # Add evaluation settings
        evaluation_strategy="steps",  
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better
        # Keep original settings for performance
        dataloader_num_workers=0,  
        group_by_length=False,
        dataloader_drop_last=False,
        gradient_checkpointing=False,
        optim="adamw_torch"
    )
    
    # Use the standard DataCollatorForLanguageModeling instead
    # Flash Attention 2 works at the model level without needing a special collator
    print("Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create callbacks
    text_gen_callback = TextGenerationCallback(model, tokenizer, test_prompts, eval_steps=100)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.0000001
    )
    validation_loss_callback = ValidationLossCallback()
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=validation_dataset,  # Add validation dataset
        data_collator=data_collator,
        callbacks=[text_gen_callback, early_stopping_callback, validation_loss_callback],  # Add new callbacks
    )
    
    # Generate sample text before training for all prompts
    print("\nGenerating sample text before training:")
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest prompt {i+1}:")
        generate_sample_text(model, tokenizer, prompt)
    
    # Start training with step-based progress tracking
    print(f"Starting training for {max_steps} steps with early stopping...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed ({max_steps} steps) and model saved!")

if __name__ == "__main__":
    main()