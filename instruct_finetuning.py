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
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, List, Union
import torch.utils.data as torch_data  # Add this import for PyTorch DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid warnings

# Model path
model_path = "../toxic-models/PursuitOfDataScience/Argonne-1.5"

# Parameters - using conservative values to prevent catastrophic forgetting
output_dir = "./argonne_synthetic_finetuned"
max_steps = 5000  # Train for specific number of steps instead of epochs
per_device_train_batch_size = 16
gradient_accumulation_steps = 1
learning_rate = 1e-5  # Low learning rate to prevent catastrophic forgetting
warmup_steps = 100
logging_steps = 10
save_steps = 200
max_seq_length = 2048
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
weight_decay = 0.01  # Add weight decay for regularization

# Dataset path
dataset_path = "../data/PrimeIntellect/SYNTHETIC-1"

def load_synthetic_dataset(path, tokenizer, max_length=2048, max_files=None):
    """Load and process the synthetic dataset from arrow files.
    Process, format, and tokenize data in a streaming fashion."""
    print(f"Loading synthetic dataset from {path}")
    
    # Get all arrow files in the train directory
    arrow_files = glob.glob(os.path.join(path, "train", "data-*.arrow"))
    if not arrow_files:
        raise ValueError(f"No arrow files found in {os.path.join(path, 'train')}")
    
    # Sort files numerically
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
    
    # Create a PyTorch-compatible dataset that processes data on-the-fly
    class SyntheticInstructDataset(torch_data.IterableDataset):
        def __init__(self, file_paths, tokenizer, max_length):
            super().__init__()
            self.file_paths = file_paths
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.spillover_text = None
            self._length = 1000000  # Arbitrary large number for batching
            
        def __len__(self):
            return self._length  # Just a placeholder value for DataLoader
            
        def process_example(self, example):
            """Process a single example: extract instruction and response."""
            try:
                instruction = example.get("prompt", "")
                response = example.get("llm_response", "")
                
                if not instruction or not response:
                    return None
                
                # Format as instruction tuning format
                formatted_text = f"Instruction: {instruction}\nResponse: {response}"
                
                if len(formatted_text) < 20:  # Filter out very short examples
                    return None
                    
                return formatted_text
            except Exception as e:
                print(f"Error processing example: {e}")
                return None
                
        def tokenize_text(self, text):
            """Tokenize text with handling of examples longer than max_length."""
            if self.spillover_text:
                text = self.spillover_text + text
                self.spillover_text = None
                
            # Tokenize without truncation
            tokens = self.tokenizer(text, truncation=False, padding=False)
            input_ids = tokens["input_ids"]
            
            # If text is too long, split it
            if len(input_ids) > self.max_length - 1:  # -1 for EOS token
                chunk = input_ids[:self.max_length - 1]
                # Save remainder for next sample
                self.spillover_text = self.tokenizer.decode(input_ids[self.max_length - 1:], skip_special_tokens=True)
                
                # Pad to max_length
                padded_chunk = chunk + [self.tokenizer.eos_token_id]
                while len(padded_chunk) < self.max_length:
                    padded_chunk.append(self.tokenizer.pad_token_id)
                    
                attention_mask = [1] * len(chunk) + [1] + [0] * (self.max_length - len(chunk) - 1)
                return {
                    "input_ids": padded_chunk,
                    "labels": padded_chunk.copy(),
                    "attention_mask": attention_mask
                }
            else:
                # Pad normally
                padded_input_ids = input_ids + [self.tokenizer.eos_token_id]
                padding_length = self.max_length - len(padded_input_ids)
                padded_input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                attention_mask = [1] * (len(input_ids) + 1) + [0] * padding_length
                return {
                    "input_ids": padded_input_ids,
                    "labels": padded_input_ids.copy(),
                    "attention_mask": attention_mask
                }
        
        def __iter__(self):
            worker_info = torch_data.get_worker_info()
            files_to_process = self.file_paths
            
            # If using multiple workers, split files among workers
            if worker_info:
                per_worker = len(files_to_process) // worker_info.num_workers
                files_to_process = files_to_process[
                    worker_info.id * per_worker : (worker_info.id + 1) * per_worker
                ]
            
            for file_idx, file_path in enumerate(files_to_process):
                try:
                    print(f"Worker {getattr(worker_info, 'id', 0)} streaming file {file_idx}/{len(files_to_process)}: {os.path.basename(file_path)}")
                    dataset = load_dataset('arrow', data_files=file_path, split='train')
                    print(f"Loaded file with {len(dataset)} rows")
                    
                    for example in dataset:
                        processed_text = self.process_example(example)
                        if processed_text is not None:
                            tokenized = self.tokenize_text(processed_text)
                            if tokenized:
                                yield tokenized
                                
                except Exception as e:
                    print(f"Error streaming from {file_path}: {str(e)}")
            
            # Reset spillover text at the end of the dataset
            self.spillover_text = None
    
    # Create streaming dataset
    streaming_dataset = SyntheticInstructDataset(arrow_files, tokenizer, max_length)
    
    print("Dataset ready for streaming with processing and tokenization")
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
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1,
        "do_sample": True,
        "repetition_penalty": 1.2,
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
            prompt = random.choice(self.prompts)
            print(f"\nStep {state.global_step}:")
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
    
    # Load dataset with integrated processing
    print("Loading and preparing dataset...")
    train_tokenized_dataset = load_synthetic_dataset(
        dataset_path, 
        tokenizer, 
        max_length=max_seq_length
    )
    
    print(f"Training dataset ready with streaming processing and chunking support")
    
    # Load base model - simple version without any fancy options
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
        task_type="CAUSAL_LM",
        target_modules=["query", "key", "value"],
    )
    
    # Prepare model for training
    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)
    
    # Define training arguments - simple version without DDP settings
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,  # Set maximum number of steps instead of epochs
        num_train_epochs=100,  # Set to a large number that won't be reached
        per_device_train_batch_size=per_device_train_batch_size,
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
        evaluation_strategy="no",
        dataloader_num_workers=1,
        group_by_length=False,
        dataloader_drop_last=False
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create text generation callback
    text_gen_callback = TextGenerationCallback(model, tokenizer, test_prompts, eval_steps=100)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        data_collator=data_collator,
        callbacks=[text_gen_callback],
    )
    
    # Generate sample text before training
    print("\nGenerating sample text before training:")
    generate_sample_text(model, tokenizer, test_prompts[0])
    
    # Start training with step-based progress tracking
    print(f"Starting training for {max_steps} steps...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed ({max_steps} steps) and model saved!")

if __name__ == "__main__":
    main()
