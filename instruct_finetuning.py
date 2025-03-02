import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_from_disk
import numpy as np
import multiprocessing
from mp_pretrain import ArgonneModelParallel, ArgonneConfig
import argparse
import glob
import re
from datetime import datetime

def get_optimal_num_processes(requested_procs=None):
    """
    Determine optimal number of processes based on available CPU cores.
    
    Args:
        requested_procs: Number of requested processes (None for auto)
    
    Returns:
        Number of processes to use
    """
    max_procs = multiprocessing.cpu_count()
    if requested_procs is None:
        # Use 75% of available cores by default
        return max(1, int(max_procs * 0.75))
    else:
        # Ensure we don't exceed available cores
        return min(max_procs, requested_procs)

def prepare_oasst_data(data_dir, tokenizer, max_length=2048, num_proc=None):
    """
    Prepare and format the OpenAssistant data for instruction fine-tuning.
    
    Args:
        data_dir: Directory containing the OpenAssistant data
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        num_proc: Number of processes to use for data processing
    
    Returns:
        Processed dataset
    """
    # Determine optimal number of processes
    num_proc = get_optimal_num_processes(num_proc)
    print(f"Using {num_proc} processes for data processing")
    
    # Load OpenAssistant datasets
    print(f"Loading OpenAssistant data from {data_dir}")
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    # Print column names and first example
    print(f"Dataset columns: {train_dataset.column_names}")
    print("\nExample data point:")
    print(train_dataset[0])
    
    # Define instruction format
    def format_instruction(example):
        """Format messages as a single instruction-following text"""
        # Check if the dataset has the newer conversation format
        if "conversation" in example:
            conversation_data = example.get("conversation", [])
            if not conversation_data:
                return {"formatted_text": "", "length": 0}
                
            conversation = ""
            for msg in conversation_data:
                role = msg.get("role", "").lower()
                text = msg.get("text", "")
                
                if role == "user":
                    conversation += f"USER: {text}\n\n"
                elif role == "assistant":
                    conversation += f"ASSISTANT: {text}\n\n"
        # Fall back to the original format
        else:
            messages = example.get("messages", [])
            if not messages:
                return {"formatted_text": "", "length": 0}
                
            conversation = ""
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                
                if role == "human" or role == "prompter":
                    conversation += f"USER: {content}\n\n"
                elif role == "assistant" or role == "gpt":
                    conversation += f"ASSISTANT: {content}\n\n"
                
        # Add tokenized version
        tokens = tokenizer.encode(conversation, truncation=True, max_length=max_length)
        return {
            "formatted_text": conversation,
            "tokenized_text": tokens,
            "length": len(tokens)
        }
    
    # Process datasets (with multiprocessing)
    columns_to_remove = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_instruction, 
        remove_columns=columns_to_remove,
        num_proc=num_proc,
        desc="Processing training data"
    )
    valid_dataset = valid_dataset.map(
        format_instruction, 
        remove_columns=valid_dataset.column_names,
        num_proc=num_proc,
        desc="Processing validation data"
    )
    
    # Filter out empty examples or too short ones (at least 16 tokens)
    train_dataset = train_dataset.filter(lambda x: x["length"] > 16)
    valid_dataset = valid_dataset.filter(lambda x: x["length"] > 16)
    
    print(f"Processed train dataset size: {len(train_dataset)}")
    print(f"Processed validation dataset size: {len(valid_dataset)}")
    
    # Create final format
    def prepare_for_training(examples):
        """Convert tokenized examples to input_ids and labels"""
        input_ids = examples["tokenized_text"]
        return {"input_ids": input_ids, "labels": input_ids.copy()}
    
    train_dataset = train_dataset.map(
        prepare_for_training,
        num_proc=num_proc,
        desc="Preparing training tensors"
    )
    valid_dataset = valid_dataset.map(
        prepare_for_training,
        num_proc=num_proc,
        desc="Preparing validation tensors"
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    return {"train": train_dataset, "validation": valid_dataset}

def get_latest_checkpoint(output_dir):
    """
    Find the most recent checkpoint in the output directory.
    
    Args:
        output_dir: Directory containing checkpoints
        
    Returns:
        Path to the most recent checkpoint directory or None if no checkpoints found
    """
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "instruct-checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Extract step numbers from checkpoint directories
    checkpoint_steps = []
    for dir_path in checkpoint_dirs:
        match = re.search(r'instruct-checkpoint-(\d+)$', dir_path)
        if match:
            step = int(match.group(1))
            checkpoint_steps.append((step, dir_path))
    
    if not checkpoint_steps:
        return None
    
    # Sort by step number and get the latest
    checkpoint_steps.sort(reverse=True)
    return checkpoint_steps[0][1]

def finetune_model(model_path, data_dir, output_dir="Argonne_LLM_Finetuned", batch_size=8, num_proc=None, resume_from=None):
    """
    Fine-tune the pretrained model on instruction data with automatic batch size adjustment.
    
    Args:
        model_path: Path to the pretrained model
        data_dir: Directory containing the OpenAssistant data
        output_dir: Directory to save the fine-tuned model
        batch_size: Initial batch size for training
        num_proc: Number of processes to use for data processing
        resume_from: Path to checkpoint directory to resume from (or None to auto-detect)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer and model
    print(f"Loading tokenizer from {model_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    # Load datasets with multiprocessing
    dataset = prepare_oasst_data(data_dir, tokenizer, num_proc=num_proc)
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training parameters
    learning_rate = 2e-5
    num_epochs = 3
    warmup_steps = 100
    gradient_accumulation_steps = 1
    min_batch_size = 1
    
    # Check for checkpoints to resume from
    if resume_from is None:
        resume_from = get_latest_checkpoint(output_dir)
    
    # Try with decreasing batch size
    current_batch_size = batch_size
    
    while current_batch_size >= min_batch_size:
        try:
            print(f"Attempting fine-tuning with batch_size={current_batch_size}")
            
            # Set initial training state
            global_step = 0
            start_epoch = 0
            best_val_loss = float('inf')
            
            # Load config and create model
            if resume_from:
                print(f"Resuming from checkpoint: {resume_from}")
                config = ArgonneConfig.from_pretrained(resume_from)
                model = ArgonneModelParallel.from_pretrained(resume_from, config=config)
                
                # Load training state if available
                training_state_path = os.path.join(resume_from, "training_state.pt")
                if os.path.exists(training_state_path):
                    training_state = torch.load(training_state_path)
                    global_step = training_state.get("global_step", 0)
                    start_epoch = training_state.get("epoch", 0)
                    best_val_loss = training_state.get("best_val_loss", float('inf'))
                    print(f"Resuming from step {global_step}, epoch {start_epoch}")
            else:
                config = ArgonneConfig.from_pretrained(model_path)
                model = ArgonneModelParallel.from_pretrained(model_path, config=config)
            
            model.distribute_model()  # Distribute across available GPUs
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scaler = torch.amp.GradScaler("cuda")
            
            # Load optimizer state if resuming
            if resume_from:
                optimizer_path = os.path.join(resume_from, "optimizer.pt")
                if os.path.exists(optimizer_path):
                    optimizer.load_state_dict(torch.load(optimizer_path))
                    print("Optimizer state loaded")
                
                scaler_path = os.path.join(resume_from, "scaler.pt")
                if os.path.exists(scaler_path):
                    scaler.load_state_dict(torch.load(scaler_path))
                    print("Gradient scaler state loaded")
            
            # Setup dataloaders
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=current_batch_size, 
                shuffle=True,
                collate_fn=data_collator
            )
            
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, 
                batch_size=current_batch_size, 
                shuffle=False,
                collate_fn=data_collator
            )
            
            # Training loop
            for epoch in range(start_epoch, num_epochs):
                model.train()
                train_loss = 0.0
                train_steps = 0
                epoch_start_time = datetime.now()
                
                # Training
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
                for step, batch in enumerate(progress_bar):
                    optimizer.zero_grad()
                    
                    first_device = model.devices[0]
                    input_ids = batch["input_ids"].to(first_device)
                    labels = batch["labels"].to(first_device)
                    
                    with torch.amp.autocast("cuda"):
                        logits, loss = model(input_ids, labels)
                    
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        train_loss += loss.item()
                        train_steps += 1
                        progress_bar.set_postfix({"loss": loss.item()})
                        
                        # Save checkpoint every 1000 steps
                        if global_step % 1000 == 0:
                            checkpoint_dir = os.path.join(output_dir, f"instruct-checkpoint-{global_step}")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            model.save_pretrained(checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_dir)
                            
                            # Save training state
                            training_state = {
                                "global_step": global_step,
                                "epoch": epoch,
                                "best_val_loss": best_val_loss
                            }
                            torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
                            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                            torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))
                            print(f"Checkpoint saved at {checkpoint_dir}")
                
                avg_train_loss = train_loss / max(1, train_steps)
                epoch_duration = datetime.now() - epoch_start_time
                print(f"Epoch {epoch+1}/{num_epochs}, Global Step {global_step}, "
                      f"Average Training Loss: {avg_train_loss:.4f}, Duration: {epoch_duration}")
                
                # Save end-of-epoch checkpoint
                checkpoint_dir = os.path.join(output_dir, f"instruct-checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # Save training state
                training_state = {
                    "global_step": global_step,
                    "epoch": epoch + 1,  # Save as next epoch for resuming
                    "best_val_loss": best_val_loss
                }
                torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))
                print(f"End of epoch checkpoint saved at {checkpoint_dir}")
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in tqdm(valid_dataloader, desc="Validation"):
                        first_device = model.devices[0]
                        input_ids = batch["input_ids"].to(first_device)
                        labels = batch["labels"].to(first_device)
                        
                        with torch.amp.autocast("cuda"):
                            logits, loss = model(input_ids, labels)
                            
                        val_loss += loss.item()
                        val_steps += 1
                
                avg_val_loss = val_loss / max(1, val_steps)
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
                
                # Save if best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    # Save best model training state
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "best_val_loss": best_val_loss
                    }
                    torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
                    print(f"Best model saved to {output_dir}")
            
            # Training completed successfully
            print(f"Fine-tuning completed successfully with batch_size={current_batch_size}")
            break
            
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory with batch_size={current_batch_size}")
            # Clean up memory
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            
            # Reduce batch size
            current_batch_size = current_batch_size - 12
            
            if current_batch_size < min_batch_size:
                print("Batch size too small, can't proceed")
                break
                
            print(f"Retrying with reduced batch_size={current_batch_size}")
            import time
            time.sleep(5)  # Give time for memory cleanup
    
    print("Fine-tuning process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Argonne LLM on instruction data")
    parser.add_argument("--model_path", type=str, default="./Argonne_LLM", 
                        help="Path to the pretrained model")
    parser.add_argument("--data_dir", type=str, default="./instruct_data/ProcessedOpenAssistant",
                        help="Directory containing the OpenAssistant data")
    parser.add_argument("--output_dir", type=str, default="./Argonne_LLM_Finetuned",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Initial batch size for training")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of processes to use for data processing")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start fine-tuning with automatic batch size adjustment and checkpoint resuming
    finetune_model(
        args.model_path, 
        args.data_dir, 
        args.output_dir, 
        batch_size=args.batch_size, 
        num_proc=args.num_proc,
        resume_from=args.resume_from
    )