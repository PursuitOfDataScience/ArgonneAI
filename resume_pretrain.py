import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# No intel_extension_for_pytorch import needed for pure CUDA
# from mpi4py import MPI  # only if you need MPI-based distributed

# Import from your main script (mp_pretrain.py or similar) that has:
#  - ArgonneConfig, ArgonneModelParallel
#  - load_bpe_tokenizer, streaming_token_generator, collate_batch
#  - load_nonstream_data (if you do non-streaming parallel map)
from mp_pretrain import (
    ArgonneConfig,
    ArgonneModelParallel,
    load_bpe_tokenizer,
    streaming_token_generator,
    collate_batch,
    load_nonstream_data  # optional, if you do parallel map for non-streaming
)

def resume_training(
    data_path="data/*.arrow",
    checkpoint_path=None,
    epochs=3,
    steps_per_epoch=1000,
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
        n_layer=24,
        n_head=24,
        n_embd=1296,
        dropout=0.1
    )
    base_model = ArgonneModelParallel(config)  # some pipeline parallel logic

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

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"Loaded checkpoint with epoch={start_epoch}, global_step={global_step}")

    model = base_model  # rename for clarity

    # 6) Decide streaming vs non-streaming
    if use_streaming:
        print("=== Resuming in STREAMING mode (CUDA) ===")
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"=== Resume Epoch {epoch} (streaming) ===")
            token_gen = streaming_token_generator(data_path, hf_tokenizer)
            step_in_epoch = 0
            token_buffer = []

            while step_in_epoch < steps_per_epoch:
                try:
                    tokens = next(token_gen)
                    token_buffer.append(tokens)

                    if len(token_buffer) == batch_size:
                        x_tens, y_tens = collate_batch(token_buffer, block_size)
                        token_buffer.clear()
                        if x_tens is None:
                            continue

                        # We'll feed to model.devices[0], if your pipeline logic expects that
                        first_device = model.devices[0]
                        x_tens = x_tens.to(first_device)
                        y_tens = y_tens.to(first_device)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            logits, loss = model(x_tens, y_tens)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1
                        step_in_epoch += 1

                        if global_step % 100 == 0:
                            print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")

                        if global_step % 2000 == 0:
                            ckpt_dict = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item()
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            save_path = f"pretrained/checkpoint_step_{global_step}.pth"
                            torch.save(ckpt_dict, save_path)
                            print(f"Checkpoint saved @ step {global_step} -> {save_path}")

                except StopIteration:
                    print("Reached end of dataset (stream) early.")
                    break

    else:
        print("=== Resuming in NON-STREAMING (Full-Pass) mode (CUDA) ===")

        # 1) Load entire data in memory. Possibly parallel map with `num_proc`.
        tokenized_data = load_nonstream_data(data_path, hf_tokenizer, block_size, num_proc=num_proc)
        total_samples = len(tokenized_data)
        print(f"Total in-memory tokenized samples: {total_samples}")

        # 2) Full pass each epoch
        batches_per_epoch = total_samples // batch_size

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"=== Resume Epoch {epoch} (CUDA non-streaming) ===")

            for batch_idx in range(batches_per_epoch):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_token_lists = tokenized_data[start_idx:end_idx]

                x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                if x_tens is None:
                    continue

                first_device = model.devices[0]
                x_tens = x_tens.to(first_device)
                y_tens = y_tens.to(first_device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits, loss = model(x_tens, y_tens)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                if global_step % 100 == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")

                if global_step % 2000 == 0:
                    ckpt_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item()
                    }
                    os.makedirs("pretrained", exist_ok=True)
                    save_path = f"pretrained/checkpoint_step_{global_step}.pth"
                    torch.save(ckpt_dict, save_path)
                    print(f"Checkpoint saved @ step {global_step} -> {save_path}")

    # 7) Final save
    model.save_pretrained("Argonne_LLM_CUDA_Resumed")
    hf_tokenizer.save_pretrained("Argonne_LLM_CUDA_Resumed")
    print("Resumed training finished on CUDA; final model saved to Argonne_LLM_CUDA_Resumed.")

def main():
    resume_training(
        data_path="data/*.arrow",
        checkpoint_path="pretrained/checkpoint_step_2000.pth", # manually set
        epochs=5,
        steps_per_epoch=500,
        block_size=2048,
        batch_size=24,
        lr=3e-5,
        use_streaming=False, 
        num_proc=4
    )

if __name__ == "__main__":
    main()
