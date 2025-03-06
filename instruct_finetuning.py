import os
import torch
from tqdm import tqdm
from datasets import load_from_disk
import json
import re
import logging
import random
from datetime import datetime
import argparse
import math

from mp_pretrain import ArgonneModelParallel, ArgonneConfig, load_bpe_tokenizer

# Set up minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("instruct_finetuning")

def fix_pipeline_state_dict(state_dict):
    """Convert pipeline-parallel state_dict keys to match single-GPU model structure."""
    new_state_dict = {}
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
            state_dict = load_file(model_path_st)
            
            # Check if weights use pipeline structure
            has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
            if has_pipeline:
                state_dict = fix_pipeline_state_dict(state_dict)
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
        else:
            model_path_pt = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_path_pt):
                state_dict = torch.load(model_path_pt, map_location="cpu")
                
                # Check if weights use pipeline structure
                has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
                if has_pipeline:
                    state_dict = fix_pipeline_state_dict(state_dict)
                
                # Load weights
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"No model weights found at {model_path}")
    except Exception as e:
        raise
    
    # Move model to device
    model.to(device)
    
    # Add devices attribute needed by generate method
    if not hasattr(model, 'devices'):
        model.devices = [device]
    
    return model

def prepare_data(data_dir, tokenizer, max_length=2048, batch_size=8):
    """Load and prepare data for training."""
    # Load datasets
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    
    # Process conversation format datasets
    if 'conversation' in train_dataset.column_names:
        def process_conversation(example, idx):
            """Process conversation with proper masking for instruction tuning."""
            conversation = example.get("conversation", [])
            
            # Build the formatted text with role prefixes
            full_text = ""
            user_segments = []  # Track positions of user segments to mask
            
            for turn_idx, turn in enumerate(conversation):
                role = turn.get("role", "").lower()  # Use lowercase for comparison
                text = turn.get("text", "")
                
                start_pos = len(full_text)
                
                if role.lower() == "user":
                    full_text += f"USER: {text}\n\n"
                    user_segments.append((start_pos, len(full_text)))
                elif role.lower() == "assistant":
                    full_text += f"ASSISTANT: {text}\n\n"
                # Ignore other roles
            
            # Tokenize the full conversation
            tokens = tokenizer.encode(full_text, truncation=True, max_length=max_length)
            input_ids = torch.tensor(tokens)
            
            # Create labels - start by copying input_ids
            labels = input_ids.clone()
            
            # Simple approach - first encode each user segment separately to find token positions
            char_to_token_map = []
            running_text = ""
            
            # Build char-to-token mapping
            for token_idx, token_id in enumerate(tokens):
                token_text = tokenizer.decode([token_id])
                char_to_token_map.extend([token_idx] * len(token_text))
                running_text += token_text
            
            # Apply masking to user segments
            for start_char, end_char in user_segments:
                if start_char >= len(char_to_token_map):
                    continue  # Skip if beyond our mapping
                
                # Find token indices
                start_token = char_to_token_map[start_char]
                end_token = char_to_token_map[min(end_char-1, len(char_to_token_map)-1)] + 1
                
                # Mask user tokens in labels
                labels[start_token:end_token] = -100
            
            return {
                "input_ids": input_ids,
                "labels": labels
            }
        
        # Process datasets
        train_dataset = train_dataset.map(
            process_conversation,
            with_indices=True,
            desc="Processing conversations (train)"
        )
        
        valid_dataset = valid_dataset.map(
            process_conversation,
            with_indices=True,
            desc="Processing conversations (validation)"
        )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Custom collate function for dynamic batching
    def collate_fn(examples):
        if not examples or 'input_ids' not in examples[0]:
            empty_tensor = torch.zeros((1, 1), dtype=torch.long)
            return {"input_ids": empty_tensor, "labels": empty_tensor}
        
        # Extract input_ids and labels
        input_ids = [ex['input_ids'] for ex in examples if 'input_ids' in ex]
        labels = [ex['labels'] if 'labels' in ex else ex['input_ids'].clone() for ex in examples if 'input_ids' in ex]
        
        if len(input_ids) == 0:
            empty_tensor = torch.zeros((1, 1), dtype=torch.long)
            return {"input_ids": empty_tensor, "labels": empty_tensor}
        
        # Determine max length in this batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        
        for ids, lbl in zip(input_ids, labels):
            if ids.numel() == 0 or lbl.numel() == 0:
                continue
                
            padding_length = max_length - len(ids)
            padded_ids = torch.nn.functional.pad(ids, (0, padding_length), value=tokenizer.pad_token_id)
            padded_input_ids.append(padded_ids)
            
            # Use -100 as padding for labels to ignore in loss calculation
            padded_lbl = torch.nn.functional.pad(lbl, (0, padding_length), value=-100)
            padded_labels.append(padded_lbl)
        
        if len(padded_input_ids) == 0:
            empty_tensor = torch.zeros((1, 1), dtype=torch.long)
            return {"input_ids": empty_tensor, "labels": empty_tensor}
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=True
    )
    
    return train_loader, valid_loader

def generate_sample_response(model, tokenizer, device, prompt="USER: Can you write a scientific essay about climate change?\n\nASSISTANT:"):
    """Generate a sample response from the model to track training progress."""
    model.eval()  # Set model to evaluation mode
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response
    try:
        with torch.no_grad():
            # Use minimal parameters that should be supported by most generate() implementations
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                # Removed all optional parameters that might not be supported
            )
        
        # Get only the generated text (not the prompt)
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Check if output is empty or contains only special characters/whitespace
        if not generated_text or not any(c.isalnum() for c in generated_text):
            return "[Model generated empty or non-text output]"
            
        return generated_text
    except Exception as e:
        return f"Error generating sample: {str(e)}"

def train(args):
    """Main training function for single GPU."""
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Select GPU if specified, otherwise use default
    gpu_id = args.gpu_id if hasattr(args, 'gpu_id') and args.gpu_id is not None else 0
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
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
            torch.cuda.empty_cache()
        
        # Load model
        model = load_model(args.model_path, device)
        
        # Set up optimizer with even lower learning rate
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
        
        # Add learning rate scheduler with warmup and decay
        total_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% of steps for warmup
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        logger.info(f"Starting training with lr={args.learning_rate}, batch={args.batch_size}, accumulation={args.gradient_accumulation_steps}")
        
        best_val_loss = float('inf')
        
        # Define a few different prompts to test generation capacity
        sample_prompts = [
            "USER: Can you write a scientific essay about climate change?\n\nASSISTANT:",
            "USER: Explain the basic principles of quantum mechanics\n\nASSISTANT:",
            "USER: What are some healthy breakfast recipes?\n\nASSISTANT:"
        ]
        
        # Define the assess_text_quality function BEFORE we use it
        def assess_text_quality(text, prompt=""):
            if not text or text == "[Model generated empty or non-text output]":
                return 0
            
            # Basic checks for degraded output
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
            special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, len(text))
            
            # Quick rejection for obviously degraded output
            if uppercase_ratio > 0.3 or special_char_ratio > 0.3 or text.count(':') > 5:
                return 0.01
                
            # Check repetitions and other issues
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            short_lines = sum(1 for line in lines if len(line) < 15)
            if short_lines > len(lines) / 2 and len(lines) > 3:
                return 0.01
            
            # Process text for analysis
            words = text.split()
            unique_words = set(words)
            
            # Check for repeated identical lines
            if len(lines) > 3:
                unique_lines = set(lines)
                if len(unique_lines) < len(lines) * 0.7:  # More than 30% repeated lines
                    return 0.01
                    
            # Topic relevance detection
            topic_keywords = {}
            
            # Extract query topic from prompt
            if "breakfast recipes" in prompt.lower():
                topic_keywords = {
                    "breakfast": 5, "recipes": 3, "food": 2, "meal": 2, "morning": 2,
                    "eat": 2, "healthy": 2, "nutritious": 3, "ingredients": 2,
                    "eggs": 3, "oatmeal": 3, "cereal": 3, "toast": 3, "yogurt": 3,
                    "porridge": 3, "smoothie": 3, "protein": 2, "quick": 1, "simple": 1
                }
            elif "climate change" in prompt.lower():
                topic_keywords = {
                    "climate": 4, "change": 3, "global": 2, "warming": 3, "environment": 2,
                    "emissions": 3, "carbon": 3, "greenhouse": 3, "temperature": 2, "earth": 2,
                    "atmosphere": 2, "pollution": 2, "fossil": 2, "fuels": 2, "renewable": 2,
                    "sustainable": 2, "impact": 1, "weather": 1, "science": 1, "research": 1
                }
            elif "quantum mechanics" in prompt.lower():
                topic_keywords = {
                    "quantum": 4, "mechanics": 3, "physics": 2, "particle": 3, "wave": 3,
                    "superposition": 3, "uncertainty": 3, "principle": 2, "state": 2, "energy": 2,
                    "atom": 2, "electron": 2, "photon": 2, "theory": 2, "measurement": 2,
                    "observer": 2, "experiment": 1, "science": 1, "nature": 1, "reality": 1
                }
                
            # If we have topic keywords, check relevance
            topic_score = 0
            if topic_keywords:
                # Count keyword occurrences in the text
                text_lower = text.lower()
                for keyword, weight in topic_keywords.items():
                    occurrences = text_lower.count(keyword)
                    topic_score += occurrences * weight
                    
                # Normalize by text length
                topic_score = topic_score / max(10, len(words)) * 5  # Scale factor to make it significant
                
            # Calculate base quality score
            sentences = re.split(r'[.!?]+', text)
            good_sentences = sum(1 for s in sentences if len(s.strip().split()) >= 5)
            
            vocab_score = sum(1 for w in unique_words if len(w) > 3 and w.islower())
            sentence_score = good_sentences * 2
            
            # Repetition penalties
            rep_penalty = 0
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    rep_penalty += 3
                elif words[i] == words[i+1]:
                    rep_penalty += 1
                    
            # Non-sense word penalties  
            nonsense_penalty = sum(2 for w in words if w.count('.') > 1 or (len(w) > 3 and w.isupper()))
            
            # Final score combining fluency and topic relevance
            fluency_score = (vocab_score + sentence_score - rep_penalty - nonsense_penalty) / max(10, len(words))
            
            # Combine scores, giving topic relevance significant weight
            final_score = (fluency_score * 0.6) + (topic_score * 0.4)
            
            # Debug information
            print(f"Fluency: {fluency_score:.2f}, Topic: {topic_score:.2f}, Final: {final_score:.2f}")
            
            return max(0.01, min(5.0, final_score))
            
        # Test model generation capability before training
        print("\n==== Pre-training generation test ====")
        test_output = generate_sample_response(model, tokenizer, device, sample_prompts[0])
        print(f"Prompt: {sample_prompts[0]}\nResponse: {test_output}\n")
        
        # NOW we can use the assess_text_quality function
        initial_quality = assess_text_quality(test_output, sample_prompts[0])
        print(f"Initial quality score: {initial_quality:.4f}")
        print("="*50)
        
        # Initialize global step counter and training tracker
        global_step = 0
        best_loss_checkpoint = None
        
        # Initialize tracking for quality-based early stopping
        # Use the initial quality as starting point rather than 0
        best_generation_quality = initial_quality
        generation_quality_not_improved = 0
        max_quality_decrease = args.early_stop_patience
        
        for epoch in range(args.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            # Reset gradients
            optimizer.zero_grad()
            accumulated_steps = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass
                    with torch.amp.autocast('cuda'):
                        outputs, loss = model(input_ids, labels)
                        # Scale loss by accumulation steps
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Track metrics
                    train_loss += loss.item() * (args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 1)
                    train_steps += 1
                    accumulated_steps += 1
                    global_step += 1
                    
                    # Update weights after accumulation steps or at end of epoch
                    if (args.gradient_accumulation_steps == 1) or \
                       (accumulated_steps % args.gradient_accumulation_steps == 0) or \
                       (batch_idx == len(train_loader) - 1):
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        accumulated_steps = 0
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item() * (args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 1)})
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Generate sample response every 100 steps
                    if global_step % 100 == 0:
                        # Temporarily set model to eval mode for generation
                        model.eval()
                        prompt_idx = (global_step // 100) % len(sample_prompts)
                        current_prompt = sample_prompts[prompt_idx]
                        sample_output = generate_sample_response(model, tokenizer, device, current_prompt)
                        
                        print(f"\n==== Step {global_step} Generation ====")
                        print(f"Prompt: {current_prompt}")
                        print(f"Response: {sample_output}")
                        
                        # Evaluate text quality
                        quality_score = assess_text_quality(sample_output, current_prompt)
                        print(f"Quality: {quality_score:.4f}")
                        
                        # Check for output degradation
                        is_degraded = (sample_output.strip() in ["", "::::::::", "[Model generated empty or non-text output]"] or
                                      sample_output.count(":") > len(sample_output) * 0.2)
                        
                        if is_degraded:
                            print("WARNING: Output degradation detected!")
                            
                            # Save checkpoint before further degradation
                            if best_loss_checkpoint is None:
                                best_loss_checkpoint = global_step - 100  # Last good checkpoint
                                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{best_loss_checkpoint}")
                                os.makedirs(checkpoint_path, exist_ok=True)
                                model.save_pretrained(checkpoint_path)
                                tokenizer.save_pretrained(checkpoint_path)
                                
                                # Reduce learning rate further if degrading
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = param_group['lr'] * 0.5
                                print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                        
                        # Update best quality tracking - FIX BUG #2
                        if quality_score > best_generation_quality:
                            print(f"New best quality score: {quality_score:.4f} > previous {best_generation_quality:.4f}")
                            best_generation_quality = quality_score
                            generation_quality_not_improved = 0  # Reset counter only when quality actually improves
                            
                            # Save a checkpoint for best generation quality
                            quality_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-quality-{global_step}")
                            os.makedirs(quality_checkpoint_path, exist_ok=True)
                            model.save_pretrained(quality_checkpoint_path)
                            tokenizer.save_pretrained(quality_checkpoint_path)
                            print(f"Saved checkpoint with best quality score: {best_generation_quality:.4f}")
                        else:
                            # Calculate percentage change
                            percent_change = ((quality_score / best_generation_quality) * 100) - 100
                            generation_quality_not_improved += 1
                            print(f"Quality not improved: {generation_quality_not_improved}/{args.early_stop_patience} "
                                 f"(current: {quality_score:.4f}, best: {best_generation_quality:.4f}, change: {percent_change:.1f}%)")
                            
                            # Don't reset counter even if we saved a checkpoint for other reasons
                        
                        # Early stopping based on generation quality
                        if generation_quality_not_improved >= args.early_stop_patience:
                            print(f"Quality not improved for {args.early_stop_patience} checks. Early stopping.")
                            print(f"Saving final model before stopping...")
                            
                            # Save the current model state with an early-stop suffix
                            early_stop_path = os.path.join(args.output_dir, f"checkpoint-early-stop-{global_step}")
                            os.makedirs(early_stop_path, exist_ok=True)
                            model.save_pretrained(early_stop_path)
                            tokenizer.save_pretrained(early_stop_path)
                            print(f"Saved early-stopping checkpoint to {early_stop_path}")
                            
                            print("Training stopped due to lack of quality improvement.")
                            return  # Always exit training loop early when quality doesn't improve
                        
                        # Save periodic checkpoints every 500 steps
                        if global_step % 500 == 0:
                            periodic_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                            os.makedirs(periodic_checkpoint_path, exist_ok=True)
                            model.save_pretrained(periodic_checkpoint_path)
                            tokenizer.save_pretrained(periodic_checkpoint_path)
                            
                        print("="*50)
                        # Set model back to training mode
                        model.train()
                        
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    continue
            
            # Calculate average training loss
            avg_train_loss = train_loss / max(1, train_steps)
            print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")
            
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
                        
                        # Forward pass
                        with torch.amp.autocast('cuda'):
                            outputs, loss = model(input_ids, labels)
                        
                        # Track metrics
                        val_loss += loss.item()
                        val_steps += 1
                        
                        # Update progress bar
                        val_pbar.set_postfix({"loss": loss.item()})
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(1, val_steps)
            print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
            
            # Generate sample at the end of each epoch
            print(f"\n==== End of Epoch {epoch+1} Generation ====")
            sample_output = generate_sample_response(model, tokenizer, device, sample_prompts[0])
            print(f"Prompt: {sample_prompts[0]}")
            print(f"Response: {sample_output}")
            print("="*50)
            
            # Save checkpoint if it's the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save the model
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Argonne LLM on a single GPU")
    parser.add_argument("--model_path", type=str, default="../Argonne-1.0")
    parser.add_argument("--data_dir", type=str, default="./instruct_data/ProcessedOpenAssistant")
    parser.add_argument("--output_dir", type=str, default="./Argonne_LLM_Finetuned")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-7)  # Even lower than before
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)  # Even stronger gradient clipping
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--early_stop_patience", type=int, default=15, 
                        help="Number of quality assessments with no improvement before stopping")
    parser.add_argument("--local_rank", type=int, default=-1, help="Ignored in single-GPU version")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()