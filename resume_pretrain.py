import os
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from data_processing import collate_batch, load_bpe_tokenizer, load_nonstream_data, streaming_token_generator
from model import ArgonneConfig, ArgonneModel


def resume_training(
    data_path="data/*.arrow",
    checkpoint_path=None,
    total_training_steps=160_000,
    block_size=2048,
    batch_size=24,
    lr=3e-5,
    use_streaming=False,   
    num_proc=8
):
    # 1) Load tokenizer
    hf_tokenizer = load_bpe_tokenizer()

    # 2) Build config & base model
    config = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=12,
        n_head=12,
        n_embd=1296,
        dropout=0.1
    )
    base_model = ArgonneModel(config)  # some pipeline parallel logic

    # 3) Create optimizer
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr)

    # 4) GradScaler for CUDA
    from torch.cuda.amp import GradScaler
    scaler = torch.amp.GradScaler("cuda")

    # 5) Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Resuming from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    base_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Get global step from checkpoint
    global_step = ckpt.get("global_step", 0)
    print(f"Loaded checkpoint at global_step={global_step}")

    model = base_model
    model.distribute_model()  # Make sure model is distributed across GPUs

    # Try to apply torch.compile() if available
    if hasattr(torch, 'compile'):
        try:
            print("Applying torch.compile() to optimize model execution...")
            model = torch.compile(model)
            print("Model successfully compiled!")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Continuing with uncompiled model.")

    first_device = model.devices[0]

    # 6) Decide streaming vs non-streaming
    if use_streaming:
        print(f"=== Resuming training from global step {global_step} in STREAMING mode ===")
        print(f"=== Will train until reaching {total_training_steps} steps ===")
        
        # Train until we reach the target number of steps
        token_gen = streaming_token_generator(data_path, hf_tokenizer)
        token_buffer = []

        # Use tqdm to track global step progress toward total_training_steps
        with tqdm(initial=global_step, total=total_training_steps, desc="Training") as pbar:
            while global_step < total_training_steps:
                try:
                    tokens = next(token_gen)
                    token_buffer.append(tokens)

                    if len(token_buffer) == batch_size:
                        x_tens, y_tens = collate_batch(token_buffer, block_size)
                        token_buffer.clear()
                        if x_tens is None:
                            continue

                        x_tens = x_tens.to(first_device)
                        y_tens = y_tens.to(first_device)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            logits, loss = model(x_tens, y_tens)
                            loss = loss.to(first_device)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1
                        pbar.update(1)

                        if global_step % 50 == 0:
                            print(f"Step {global_step} | Loss: {loss.item():.4f}")
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                            generated = model.generate(prompt_tensor, max_new_tokens=50)
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                        if global_step % 2000 == 0:
                            ckpt_dict = {
                                "global_step": global_step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item()
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            save_path = f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                            torch.save(ckpt_dict, save_path)
                            print(f"Checkpoint saved @ step {global_step} -> {save_path}")

                except StopIteration:
                    print("Reached end of dataset stream. Restarting data generator.")
                    token_gen = streaming_token_generator(data_path, hf_tokenizer)
                    continue

    else:
        print(f"=== Resuming training from global step {global_step} in NON-STREAMING mode ===")
        print(f"=== Will train until reaching {total_training_steps} steps ===")

        # 1) Load entire data in memory. Possibly parallel map with `num_proc`.
        tokenized_data = load_nonstream_data(data_path, hf_tokenizer, block_size, num_proc=num_proc)
        total_samples = len(tokenized_data)
        print(f"Total in-memory tokenized samples: {total_samples}")

        # Calculate how many full passes we need
        batches_per_epoch = total_samples // batch_size
        remaining_steps = total_training_steps - global_step
        
        # Use tqdm to track global step progress toward total_training_steps
        with tqdm(initial=global_step, total=total_training_steps, desc="Training") as pbar:
            while global_step < total_training_steps:
                # Shuffle data for each epoch
                indices = torch.randperm(total_samples)
                
                for idx in range(0, total_samples, batch_size):
                    if global_step >= total_training_steps:
                        break
                        
                    # Get batch indices
                    batch_indices = indices[idx:min(idx + batch_size, total_samples)]
                    batch_token_lists = [tokenized_data[i] for i in batch_indices]
                    
                    x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                    if x_tens is None:
                        continue

                    x_tens = x_tens.to(first_device)
                    y_tens = y_tens.to(first_device)

                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda"):
                        logits, loss = model(x_tens, y_tens)
                        loss = loss.to(first_device)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1
                    pbar.update(1)
                    
                    if global_step % 50 == 0:
                        print(f"Step {global_step} | Loss: {loss.item():.4f}")
                        prompt_str = "Long long time ago, "
                        token_ids = hf_tokenizer.encode(prompt_str)
                        prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                        generated = model.generate(prompt_tensor, max_new_tokens=50)
                        generated_text = hf_tokenizer.decode(generated[0].tolist())
                        print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                    if global_step % 2000 == 0:
                        ckpt_dict = {
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item()
                        }
                        os.makedirs("pretrained", exist_ok=True)
                        save_path = f"pretrained/non_streaming_checkpoint_step_{global_step}.pth"
                        torch.save(ckpt_dict, save_path)
                        print(f"Checkpoint saved @ step {global_step} -> {save_path}")

    # Perform final save at the end of training
    try:
        model = model.half()  # Convert to FP16
        model.save_pretrained("Argonne_LLM")
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print(f"Training completed at step {global_step}. Final model saved.")
    except Exception as e:
        print(f"Failed to save final model: {e}")


def main():
    resume_training(
        data_path="data/*.arrow",
        checkpoint_path="pretrained/streaming_checkpoint_step_9000.pth", # manually set
        total_training_steps=160_000,
        block_size=2048,
        batch_size=128,
        lr=3e-5,
        use_streaming=True, 
        num_proc=4
    )

if __name__ == "__main__":
    main()
