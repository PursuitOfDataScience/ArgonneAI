import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_from_disk
import numpy as np
from mp_pretrain import ArgonneModelParallel, ArgonneConfig

def prepare_oasst_data(data_dir, tokenizer, max_length=2048):
    """
    Prepare and format the OpenAssistant data for instruction fine-tuning.
    
    Args:
        data_dir: Directory containing the OpenAssistant data
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Processed dataset
    """
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
    
    # Process datasets
    train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(format_instruction, remove_columns=valid_dataset.column_names)
    
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
    
    train_dataset = train_dataset.map(prepare_for_training)
    valid_dataset = valid_dataset.map(prepare_for_training)
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    return {"train": train_dataset, "validation": valid_dataset}

def finetune_model(model_path, data_dir, output_dir="Argonne_LLM_Finetuned", batch_size=8):
    """
    Fine-tune the pretrained model on instruction data with automatic batch size adjustment.
    
    Args:
        model_path: Path to the pretrained model
        data_dir: Directory containing the OpenAssistant data
        output_dir: Directory to save the fine-tuned model
        batch_size: Initial batch size for training
    """
    # Load the tokenizer and model
    print(f"Loading tokenizer from {model_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    # Load datasets
    dataset = prepare_oasst_data(data_dir, tokenizer)
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
    
    # Try with decreasing batch size
    current_batch_size = batch_size
    
    while current_batch_size >= min_batch_size:
        try:
            print(f"Attempting fine-tuning with batch_size={current_batch_size}")
            
            # Load config and model
            config = ArgonneConfig.from_pretrained(model_path)
            model = ArgonneModelParallel.from_pretrained(model_path, config=config)
            model.distribute_model()  # Distribute across available GPUs
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scaler = torch.amp.GradScaler("cuda")
            
            # Training loop
            global_step = 0
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
            
            # Save best model
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                train_steps = 0
                
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
                        
                        # Save checkpoint occasionally
                        if global_step % 500 == 0:
                            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            model.save_pretrained(checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_dir)
                            print(f"Checkpoint saved at {checkpoint_dir}")
                
                avg_train_loss = train_loss / train_steps
                print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
                
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
                
                avg_val_loss = val_loss / val_steps
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
                
                # Save if best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
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
            current_batch_size = current_batch_size // 2
            
            if current_batch_size < min_batch_size:
                print("Batch size too small, can't proceed")
                break
                
            print(f"Retrying with reduced batch_size={current_batch_size}")
            import time
            time.sleep(5)  # Give time for memory cleanup
    
    print("Fine-tuning process completed")

if __name__ == "__main__":
    model_path = "./Argonne_LLM"
    data_dir = "./instruct_data/OpenAssistant/oasst1" 
    output_dir = "./Argonne_LLM_Finetuned"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start fine-tuning with automatic batch size adjustment
    finetune_model(model_path, data_dir, output_dir, batch_size=128)