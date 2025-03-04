import os
import torch
from tqdm import tqdm
from datasets import load_from_disk
import json
import re
import logging
from datetime import datetime
import argparse

from mp_pretrain import ArgonneModelParallel, ArgonneConfig, load_bpe_tokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("instruct_finetuning")

def fix_pipeline_state_dict(state_dict):
    """Convert pipeline-parallel state_dict keys to match single-GPU model structure."""
    new_state_dict = {}
    
    # Pattern for pipeline stage keys
    pipeline_pattern = r'pipeline_stages\.(\d+)\.(\d+)\.(.*)'
    
    # First pass to detect structure
    processed_blocks = {}
    for key in state_dict.keys():
        if not key.startswith('pipeline_stages.'):
            continue
            
        match = re.match(pipeline_pattern, key)
        if match:
            gpu_idx, block_in_gpu_idx = int(match.group(1)), int(match.group(2))
            processed_blocks.setdefault(gpu_idx, set()).add(block_in_gpu_idx)
    
    # Determine blocks_per_gpu (maximum block index + 1)
    blocks_per_gpu = 1
    if processed_blocks:
        blocks_per_gpu = max(max(indices) for indices in processed_blocks.values()) + 1
        logger.info(f"Detected {blocks_per_gpu} blocks per GPU in the model")
    
    # Second pass to convert keys
    for key, value in state_dict.items():
        if key.startswith('pipeline_stages.'):
            match = re.match(pipeline_pattern, key)
            if match:
                gpu_idx, block_in_gpu_idx, rest = int(match.group(1)), int(match.group(2)), match.group(3)
                # Calculate global block index
                global_block_idx = gpu_idx * blocks_per_gpu + block_in_gpu_idx
                new_key = f'blocks.{global_block_idx}.{rest}'
                new_state_dict[new_key] = value
        else:
            # Copy other weights unchanged
            new_state_dict[key] = value
    
    return new_state_dict

def load_model(model_path, device):
    """Load model for single GPU."""
    logger.info(f"Loading model from {model_path}")
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config and model
    config = ArgonneConfig(**config_dict)
    model = ArgonneModelParallel(config)
    
    # Load weights - try safetensors first, then PyTorch format
    try:
        from safetensors.torch import load_file
        model_path_st = os.path.join(model_path, "model.safetensors")
        if os.path.exists(model_path_st):
            logger.info(f"Loading weights from {model_path_st}")
            state_dict = load_file(model_path_st)
            
            # Check if weights use pipeline structure
            has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
            if has_pipeline:
                logger.info(f"Converting pipeline structure to blocks structure...")
                state_dict = fix_pipeline_state_dict(state_dict)
            
            # Load weights
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {len(missing)} keys")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)} keys")
        else:
            model_path_pt = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_path_pt):
                logger.info(f"Loading weights from {model_path_pt}")
                state_dict = torch.load(model_path_pt, map_location="cpu")
                
                # Check if weights use pipeline structure
                has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
                if has_pipeline:
                    logger.info(f"Converting pipeline structure to blocks structure...")
                    state_dict = fix_pipeline_state_dict(state_dict)
                
                # Load weights
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys: {len(missing)} keys")
                if unexpected:
                    logger.warning(f"Unexpected keys: {len(unexpected)} keys")
            else:
                raise FileNotFoundError(f"No model weights found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Move model to device
    model.to(device)
    
    # Add devices attribute needed by generate method
    if not hasattr(model, 'devices'):
        model.devices = [device]
    
    return model

def prepare_data(data_dir, tokenizer, max_length=2048, batch_size=8):
    """Load and prepare data for training."""
    logger.info(f"Loading and preparing data from {data_dir}")
    
    # Load datasets
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Dataset column names: {train_dataset.column_names}")
    
    # Process conversation format datasets
    if 'conversation' in train_dataset.column_names:
        logger.info("Found conversation format dataset. Converting to input_ids and labels.")
        
        def format_conversation(example):
            """Format a conversation into a single text string for training."""
            formatted_text = ""
            conversation = example.get("conversation", [])
            
            for turn in conversation:
                role = turn.get("role", "").lower()
                text = turn.get("text", "")
                
                if role == "user":
                    formatted_text += f"USER: {text}\n\n"
                elif role == "assistant":
                    formatted_text += f"ASSISTANT: {text}\n\n"
            
            # Tokenize the formatted text
            tokens = tokenizer.encode(
                formatted_text,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).squeeze(0)
            
            return {
                "formatted_text": formatted_text,
                "input_ids": tokens,
                "labels": tokens.clone()
            }
        
        # Process datasets
        train_dataset = train_dataset.map(
            format_conversation,
            desc="Processing conversations (train)"
        )
        
        valid_dataset = valid_dataset.map(
            format_conversation,
            desc="Processing conversations (validation)"
        )
        
        logger.info(f"Processed dataset columns: {train_dataset.column_names}")
    
    # Ensure required fields exist
    required_columns = ["input_ids", "labels"]
    
    # Set format for PyTorch (only include columns we need)
    columns_to_keep = [col for col in train_dataset.column_names if col in required_columns]
    
    logger.info(f"Setting dataset format with columns: {columns_to_keep}")
    
    if not columns_to_keep:
        logger.error(f"No required columns found in dataset. Available columns: {train_dataset.column_names}")
        logger.error("Will try to generate them from raw data.")
        
        # Last resort - try to convert from any text field
        def generate_tokens_from_text(example):
            """Try to find any text field and tokenize it."""
            text = None
            # Look for any field that might contain text
            for field in ['text', 'content', 'formatted_text', 'conversation']:
                if field in example and example[field]:
                    if field == 'conversation':
                        # Handle conversation object specially
                        conversation = example[field]
                        if isinstance(conversation, list):
                            text = ""
                            for turn in conversation:
                                if isinstance(turn, dict) and 'text' in turn:
                                    role = turn.get('role', 'unknown')
                                    text += f"{role.upper()}: {turn['text']}\n\n"
                    else:
                        text = example[field]
                    break
            
            if text:
                tokens = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).squeeze(0)
                
                return {
                    "input_ids": tokens,
                    "labels": tokens.clone()
                }
            else:
                # Empty example as fallback
                return {
                    "input_ids": torch.zeros((1,), dtype=torch.long),
                    "labels": torch.zeros((1,), dtype=torch.long)
                }
        
        train_dataset = train_dataset.map(generate_tokens_from_text)
        valid_dataset = valid_dataset.map(generate_tokens_from_text)
        
        columns_to_keep = ["input_ids", "labels"]
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=columns_to_keep)
    valid_dataset.set_format(type="torch", columns=columns_to_keep)
    
    # Custom collate function for dynamic batching
    def collate_fn(examples):
        # Defensive check to prevent KeyError
        if not examples or 'input_ids' not in examples[0]:
            logger.error(f"Missing 'input_ids' in batch examples. Keys available: {examples[0].keys() if examples else 'No examples'}")
            # Return empty batch with right structure as fallback
            empty_tensor = torch.zeros((1, 1), dtype=torch.long)
            return {
                "input_ids": empty_tensor,
                "labels": empty_tensor
            }
        
        # Extract input_ids and labels, with error checking
        input_ids = []
        labels = []
        
        for example in examples:
            if 'input_ids' in example:
                input_ids.append(example['input_ids'])
                # If labels are missing, use input_ids as labels
                if 'labels' in example:
                    labels.append(example['labels'])
                else:
                    labels.append(example['input_ids'].clone())
            else:
                # Skip this example
                continue
        
        if len(input_ids) == 0:
            logger.error("No valid examples in batch")
            # Return empty batch with right structure
            empty_tensor = torch.zeros((1, 1), dtype=torch.long)
            return {
                "input_ids": empty_tensor,
                "labels": empty_tensor
            }
        
        # Determine max length in this batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        
        for ids, lbl in zip(input_ids, labels):
            # Skip any empty tensors
            if ids.numel() == 0 or lbl.numel() == 0:
                continue
                
            padding_length = max_length - len(ids)
            padded_ids = torch.nn.functional.pad(
                ids, (0, padding_length), value=tokenizer.pad_token_id
            )
            padded_input_ids.append(padded_ids)
            
            # Use -100 as padding for labels to ignore in loss calculation
            padded_lbl = torch.nn.functional.pad(
                lbl, (0, padding_length), value=-100
            )
            padded_labels.append(padded_lbl)
        
        # Check again in case we filtered all examples
        if len(padded_input_ids) == 0:
            logger.error("No valid examples after padding")
            # Return empty batch with right structure
            empty_tensor = torch.zeros((1, 1), dtype=torch.long)
            return {
                "input_ids": empty_tensor,
                "labels": empty_tensor
            }
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }
    
    # Create data loaders for single GPU
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, valid_loader

def log_gpu_memory(device, prefix=""):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        
        logger.info(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, "
                   f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")

def train(args):
    """Main training function for single GPU."""
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Select GPU if specified, otherwise use default
    gpu_id = args.gpu_id if hasattr(args, 'gpu_id') and args.gpu_id is not None else 0
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (no GPU available)")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load tokenizer
        tokenizer = load_bpe_tokenizer()
        
        # Prepare data
        train_loader, valid_loader = prepare_data(
            args.data_dir, tokenizer, batch_size=args.batch_size
        )
        
        # Reset peak memory stats and clear cache
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            logger.info("Memory reset before model loading")
        
        # Load model
        model = load_model(args.model_path, device)
        log_gpu_memory(device, prefix="After model loading: ")
        
        # Set up optimizer - use parameter groups for more efficient training
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_params,
            lr=args.learning_rate,
            eps=1e-8
        )
        
        # Training loop
        logger.info(f"Starting training with batch_size={args.batch_size}, "
                   f"accumulation_steps={args.gradient_accumulation_steps}")
        
        best_val_loss = float('inf')
        
        for epoch in range(args.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            # Reset gradients at the start of the epoch
            optimizer.zero_grad()
            accumulated_steps = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            
            # Clear GPU cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        outputs, loss = model(input_ids, labels)
                        # Scale loss by accumulation steps if using gradient accumulation
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Track metrics - scale back for logging if needed
                    train_loss += loss.item() * (args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 1)
                    train_steps += 1
                    accumulated_steps += 1
                    
                    # Update weights after accumulation steps or at end of epoch
                    if (args.gradient_accumulation_steps == 1) or (accumulated_steps % args.gradient_accumulation_steps == 0) or (batch_idx == len(train_loader) - 1):
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        accumulated_steps = 0
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item() * (args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 1)})
                    
                    # Log memory usage every 50 steps
                    if batch_idx % 50 == 0:
                        log_gpu_memory(device, prefix=f"Epoch {epoch+1}, Step {batch_idx}: ")
                        
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"Out of memory on batch {batch_idx}: {str(e)}")
                    # Skip this batch and continue
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            # Calculate average training loss
            avg_train_loss = train_loss / max(1, train_steps)
            logger.info(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            # Progress bar for validation
            val_pbar = tqdm(valid_loader, desc="Validation")
            
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        # Move batch to device
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        
                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda'):
                            outputs, loss = model(input_ids, labels)
                        
                        # Track metrics
                        val_loss += loss.item()
                        val_steps += 1
                        
                        # Update progress bar
                        val_pbar.set_postfix({"loss": loss.item()})
                        
                    except torch.cuda.OutOfMemoryError:
                        logger.warning("Out of memory during validation, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(1, val_steps)
            logger.info(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save the model
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                logger.info(f"Model saved to {args.output_dir}")
            
            # Print memory usage at the end of each epoch
            log_gpu_memory(device, prefix=f"End of epoch {epoch+1}: ")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Argonne LLM on a single GPU")
    parser.add_argument("--model_path", type=str, default="./Argonne_LLM")
    parser.add_argument("--data_dir", type=str, default="./instruct_data/ProcessedOpenAssistant")
    parser.add_argument("--output_dir", type=str, default="./Argonne_LLM_Finetuned")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    # The local_rank argument is still included for backward compatibility but will be ignored
    parser.add_argument("--local_rank", type=int, default=-1, help="Ignored in single-GPU version")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()