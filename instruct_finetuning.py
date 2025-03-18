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
model_path = "../toxic-models/PursuitOfDataScience/Argonne-1.5"

# Parameters - using conservative values to prevent catastrophic forgetting
output_dir = "./argonne_synthetic_finetuned"
per_device_train_batch_size = 32
per_device_eval_batch_size = per_device_train_batch_size  # Added for evaluation
max_steps = 100_000 // per_device_train_batch_size  # Train for specific number of steps instead of epochs
gradient_accumulation_steps = 1
learning_rate = 5e-7  # Low learning rate to prevent catastrophic forgetting
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
validation_size = 2000  # Added for validation set size

# Dataset path
dataset_path = "../data/tuanha1305/DeepSeek-R1-Distill"

# Format strings
FORMAT_INSTRUCTION = "Instruction: "
FORMAT_RESPONSE = "\nResponse: "
FORMAT_CONTINUED = "\nResponse (continued): "

def create_validation_set(path, tokenizer, max_length=2048, validation_size=200):
    """Create a validation dataset using the same processing as the training dataset."""
    print(f"Creating validation set with {validation_size} examples")
    
    # Get one arrow file for validation
    arrow_files = glob.glob(os.path.join(path, "train", "data-*.arrow"))
    if not arrow_files:
        raise ValueError(f"No arrow files found in {path}")
        
    # Use a random file for validation data
    val_file = random.choice(arrow_files)
    print(f"Using {os.path.basename(val_file)} for validation data")
    
    # Load validation examples
    try:
        dataset = load_dataset('arrow', data_files=val_file, split='train')
        # Sample random examples
        val_indices = random.sample(range(len(dataset)), min(validation_size, len(dataset)))
        val_examples = [dataset[i] for i in val_indices]
        print(f"Loaded {len(val_examples)} examples for validation")
    except Exception as e:
        print(f"Error loading validation file: {e}")
        return None
    
    # Process validation examples using the same format as training
    processed_examples = []
    
    for example in val_examples:
        try:
            instruction = example.get("input", "")
            response = example.get("content", "")
            
            if not instruction or not response or len(instruction) < 3 or len(response) < 3:
                continue
            
            # Format exactly as in training
            formatted_text = f"{FORMAT_INSTRUCTION}{instruction}{FORMAT_RESPONSE}{response}"
            
            # Tokenize
            tokens = tokenizer(
                formatted_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Create labels with -100 for input tokens
            labels = tokens["input_ids"][0].clone()
            
            # Find position where response starts
            response_start_str = FORMAT_RESPONSE
            response_start_tokens = tokenizer(response_start_str, add_special_tokens=False).input_ids
            
            for i in range(len(labels) - len(response_start_tokens)):
                if tokens["input_ids"][0][i:i+len(response_start_tokens)].tolist() == response_start_tokens:
                    # Mask everything before the response (including FORMAT_RESPONSE)
                    response_start_pos = i + len(response_start_tokens)
                    labels[:response_start_pos] = -100
                    break
            
            # Mask padding tokens
            labels[tokens["attention_mask"][0] == 0] = -100
            
            processed_examples.append({
                "input_ids": tokens["input_ids"][0],
                "attention_mask": tokens["attention_mask"][0],
                "labels": labels
            })
        except Exception as e:
            print(f"Error processing validation example: {e}")
            continue
    
    print(f"Created validation set with {len(processed_examples)} examples")
    
    # Create a static dataset for validation
    class StaticValidationDataset(torch_data.Dataset):
        def __init__(self, examples):
            self.examples = examples
            
        def __len__(self):
            return len(self.examples)
            
        def __getitem__(self, idx):
            return self.examples[idx]
    
    return StaticValidationDataset(processed_examples)

def load_synthetic_dataset(path, tokenizer, max_length=2048, max_files=None):
    """Load and process the synthetic dataset from arrow files in a strictly sequential manner."""
    print(f"Loading synthetic dataset from {path}")
    
    # Get all arrow files in the train directory
    arrow_files = glob.glob(os.path.join(path, "train", "data-*.arrow"))
    if not arrow_files:
        raise ValueError(f"No arrow files found in {os.path.join(path, 'train')}")
    
    # Sort files numerically to ensure consistent order
    def extract_number(filename):
        match = re.search(r'data-(\d+)-of-', filename)
        if match:
            return int(match.group(1))
        return 0
    
    arrow_files = sorted(arrow_files, key=extract_number)
    print(f"Found {len(arrow_files)} arrow files, starting with {os.path.basename(arrow_files[0])}")
    
    if max_files and max_files < len(arrow_files):
        arrow_files = arrow_files[:max_files]
        print(f"Using first {max_files} files")
    
    # Create a PyTorch-compatible dataset that processes data sequentially
    class SequentialInstructDataset(torch_data.IterableDataset):
        def __init__(self, file_paths, tokenizer, max_length):
            super().__init__()
            self.file_paths = file_paths
            self.tokenizer = tokenizer
            self.max_length = max_length
            self._length = 1000000  # Arbitrary large number for batching
            
            # Cache for common instructions
            self._instr_cache = {}
            
            # Cached format token lengths
            self.format_instr_len = len(tokenizer(FORMAT_INSTRUCTION, add_special_tokens=False).input_ids)
            self.format_resp_len = len(tokenizer(FORMAT_RESPONSE, add_special_tokens=False).input_ids)
            self.format_cont_len = len(tokenizer(FORMAT_CONTINUED, add_special_tokens=False).input_ids)
            
        def __len__(self):
            return self._length
            
        def tokenize_qa_pair(self, instruction, response):
            """Tokenize and chunk an instruction-response pair if needed."""
            # Get instruction tokens (with caching for common instructions)
            if instruction in self._instr_cache:
                instr_tokens = self._instr_cache[instruction]
            else:
                instr_tokens = self.tokenizer(instruction, add_special_tokens=False).input_ids
                if len(instruction) < 1000:  # Only cache reasonably sized instructions
                    self._instr_cache[instruction] = instr_tokens
                    
            # Get response tokens
            resp_tokens = self.tokenizer(response, add_special_tokens=False).input_ids
            
            # Calculate available space in first chunk
            format_overhead = self.format_instr_len + self.format_resp_len
            cont_format_overhead = self.format_instr_len + self.format_cont_len
            
            # If instruction is very long, truncate it
            max_instr_len = self.max_length // 3
            if len(instr_tokens) > max_instr_len:
                instr_tokens = instr_tokens[:max_instr_len]
                instruction = self.tokenizer.decode(instr_tokens, skip_special_tokens=True)
            
            # Calculate available space for response in chunks
            first_chunk_avail = self.max_length - len(instr_tokens) - format_overhead - 1  # -1 for EOS
            cont_chunk_avail = self.max_length - len(instr_tokens) - cont_format_overhead - 1
            
            chunks = []
            
            # Check if the entire QA pair fits in one chunk
            if len(instr_tokens) + len(resp_tokens) + format_overhead + 1 <= self.max_length:
                # Simple case - everything fits in one chunk
                full_text = f"{FORMAT_INSTRUCTION}{instruction}{FORMAT_RESPONSE}{response}"
                tokens = self.tokenizer(full_text).input_ids
                
                # Add EOS and create example
                if len(tokens) < self.max_length:
                    tokens = tokens + [self.tokenizer.eos_token_id]
                else:
                    tokens = tokens[:self.max_length-1] + [self.tokenizer.eos_token_id]
                    
                # Add padding
                attention_mask = [1] * len(tokens)
                padding_length = self.max_length - len(tokens)
                if padding_length > 0:
                    tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                
                # IMPROVED MASKING: Create a more robust way to find response start position
                labels = [-100] * len(tokens)  # Start with all masked
                
                # Find position where response starts (after FORMAT_RESPONSE)
                prompt_end_text = FORMAT_RESPONSE
                prompt_end_tokens = self.tokenizer(prompt_end_text, add_special_tokens=False).input_ids
                
                # Look for the token sequence
                for i in range(len(tokens) - len(prompt_end_tokens)):
                    if tokens[i:i+len(prompt_end_tokens)] == prompt_end_tokens:
                        # Found the response start marker - unmask everything AFTER the response marker
                        response_start_pos = i + len(prompt_end_tokens)
                        # Copy the original tokens for all positions after response start
                        labels[response_start_pos:] = tokens[response_start_pos:].copy()
                        break
                
                # Make sure padding stays masked
                if padding_length > 0:
                    labels[-padding_length:] = [-100] * padding_length
                
                chunks.append({
                    "input_ids": tokens,
                    "labels": labels,
                    "attention_mask": attention_mask
                })
                
            else:
                # Complex case - need to chunk the response
                
                # First chunk with start of response
                first_resp_tokens = resp_tokens[:first_chunk_avail]
                first_resp_text = self.tokenizer.decode(first_resp_tokens, skip_special_tokens=True)
                first_chunk_text = f"{FORMAT_INSTRUCTION}{instruction}{FORMAT_RESPONSE}{first_resp_text}"
                first_chunk_tokens = self.tokenizer(first_chunk_text).input_ids
                
                # Add EOS and handle padding for first chunk
                if len(first_chunk_tokens) < self.max_length:
                    first_chunk_tokens = first_chunk_tokens + [self.tokenizer.eos_token_id]
                else:
                    first_chunk_tokens = first_chunk_tokens[:self.max_length-1] + [self.tokenizer.eos_token_id]
                    
                attention_mask = [1] * len(first_chunk_tokens)
                padding_length = self.max_length - len(first_chunk_tokens)
                if padding_length > 0:
                    first_chunk_tokens = first_chunk_tokens + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                
                # IMPROVED MASKING for first chunk
                first_chunk_labels = [-100] * len(first_chunk_tokens)  # Start with all masked
                
                # Find position where response starts
                prompt_end_text = FORMAT_RESPONSE
                prompt_end_tokens = self.tokenizer(prompt_end_text, add_special_tokens=False).input_ids
                
                for i in range(len(first_chunk_tokens) - len(prompt_end_tokens)):
                    if first_chunk_tokens[i:i+len(prompt_end_tokens)] == prompt_end_tokens:
                        # Found the response start marker
                        response_start_pos = i + len(prompt_end_tokens)
                        # Copy tokens for the response part only
                        first_chunk_labels[response_start_pos:] = first_chunk_tokens[response_start_pos:].copy()
                        break
                
                # Ensure padding stays masked
                if padding_length > 0:
                    first_chunk_labels[-padding_length:] = [-100] * padding_length
                
                chunks.append({
                    "input_ids": first_chunk_tokens,
                    "labels": first_chunk_labels,
                    "attention_mask": attention_mask
                })
                
                # Process remaining response in continuation chunks
                remaining_tokens = resp_tokens[first_chunk_avail:]
                
                for i in range(0, len(remaining_tokens), cont_chunk_avail):
                    cont_tokens = remaining_tokens[i:i+cont_chunk_avail]
                    cont_text = self.tokenizer.decode(cont_tokens, skip_special_tokens=True)
                    chunk_text = f"{FORMAT_INSTRUCTION}{instruction}{FORMAT_CONTINUED}{cont_text}"
                    chunk_tokens = self.tokenizer(chunk_text).input_ids
                    
                    # Add EOS and handle padding
                    if len(chunk_tokens) < self.max_length:
                        chunk_tokens = chunk_tokens + [self.tokenizer.eos_token_id]
                    else:
                        chunk_tokens = chunk_tokens[:self.max_length-1] + [self.tokenizer.eos_token_id]
                        
                    attention_mask = [1] * len(chunk_tokens)
                    padding_length = self.max_length - len(chunk_tokens)
                    if padding_length > 0:
                        chunk_tokens = chunk_tokens + [self.tokenizer.pad_token_id] * padding_length
                        attention_mask = attention_mask + [0] * padding_length
                    
                    # IMPROVED MASKING for continuation chunks
                    chunk_labels = [-100] * len(chunk_tokens)  # Start with all masked
                    
                    # Find position where continued response starts
                    cont_marker = FORMAT_CONTINUED
                    cont_marker_tokens = self.tokenizer(cont_marker, add_special_tokens=False).input_ids
                    
                    for j in range(len(chunk_tokens) - len(cont_marker_tokens)):
                        if chunk_tokens[j:j+len(cont_marker_tokens)] == cont_marker_tokens:
                            # Found the continued response marker
                            response_start_pos = j + len(cont_marker_tokens)
                            # Copy tokens for the response part only
                            chunk_labels[response_start_pos:] = chunk_tokens[response_start_pos:].copy()
                            break
                    
                    # Ensure padding stays masked
                    if padding_length > 0:
                        chunk_labels[-padding_length:] = [-100] * padding_length
                    
                    chunks.append({
                        "input_ids": chunk_tokens,
                        "labels": chunk_labels,
                        "attention_mask": attention_mask
                    })
            
            return chunks
            
        def __iter__(self):
            # Process files one at a time sequentially
            for file_idx, file_path in enumerate(self.file_paths):
                try:
                    print(f"Processing file {file_idx+1}/{len(self.file_paths)}: {os.path.basename(file_path)}")
                    
                    # Load dataset one file at a time
                    dataset = load_dataset('arrow', data_files=file_path, split='train')
                    print(f"Loaded file with {len(dataset)} examples")
                    
                    # Process examples in sequence
                    for example_idx, example in enumerate(dataset):
                        # Print progress occasionally
                        if example_idx > 0 and example_idx % 1000 == 0:
                            print(f"  Processed {example_idx}/{len(dataset)} examples from file {file_idx+1}")
                        
                        # Extract instruction and response using new column names
                        instruction = example.get("input", "")
                        response = example.get("content", "")
                        
                        # Skip invalid examples
                        if not instruction or not response or len(instruction) < 3 or len(response) < 3:
                            continue
                            
                        # Process the QA pair (chunk if needed)
                        chunks = self.tokenize_qa_pair(instruction, response)
                        
                        # Yield all chunks for this example
                        for chunk in chunks:
                            yield chunk
                            
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
    
    # Create streaming dataset that processes files sequentially
    streaming_dataset = SequentialInstructDataset(arrow_files, tokenizer, max_length)
    
    print("Dataset ready for sequential streaming with smart chunking")
    return streaming_dataset

def generate_sample_text(model, tokenizer, prompt="Instruction: Write an article about AI\nResponse:", max_length=150):
    """Generate sample text from the model to evaluate its capabilities"""
    print(f"\n=== Generating sample text for prompt: '{prompt}' ===")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
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
    }
    
    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    
    # Create validation dataset (new)
    print("Creating validation dataset...")
    validation_dataset = create_validation_set(
        dataset_path, 
        tokenizer, 
        max_length=max_seq_length,
        validation_size=validation_size
    )
    
    # Load dataset with sequential processing
    print("Loading and preparing dataset...")
    train_tokenized_dataset = load_synthetic_dataset(
        dataset_path, 
        tokenizer, 
        max_length=max_seq_length
    )
    
    print(f"Sequential training dataset ready with smart chunking")
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["value"],
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for training
    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)
    
    # Define training arguments
    print("Setting up training arguments...")
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
        fp16=True,
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
        dataloader_num_workers=0,  # Force sequential processing with no workers
        group_by_length=False,
        dataloader_drop_last=False,
        gradient_checkpointing=False,
        optim="adamw_torch"
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create callbacks
    text_gen_callback = TextGenerationCallback(model, tokenizer, test_prompts, eval_steps=100)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.01
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