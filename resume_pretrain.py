import os
import torch
import torch.nn.functional as F
import torch.nn as nn

# Import functions from mp_pretrain.py
from mp_pretrain import (
    ArgonneConfig,
    ArgonneModelParallel,
    load_bpe_tokenizer,
    streaming_token_generator,
    collate_batch
)

def resume_training(
    data_path="data/*.arrow",
    checkpoint_path="pretrained/checkpoint_step_2000.pth", # needs to be manually updated
    epochs=1,
    steps_per_epoch=1000,
    block_size=2048,
    batch_size=24,
    lr=3e-5
):
    """
    Resume training from a specified checkpoint file (checkpoint_path).
    Continues training for `epochs` more epochs, with `steps_per_epoch` steps each.
    Uses the same streaming logic from mp_pretrain.
    """
    # 1) Load tokenizer
    hf_tokenizer = load_bpe_tokenizer()

    # 2) Build config & model
    config = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=24,
        n_head=24,
        n_embd=1296,
        dropout=0.1
    )
    model = ArgonneModelParallel(config)

    # 3) Create optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")

    # 4) Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Resuming from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"Loaded checkpoint with epoch={start_epoch}, global_step={global_step}")

    # 5) Resume training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"=== Resume Epoch {epoch} ===")
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

                    first_device = model.devices[0]
                    x_tens, y_tens = x_tens.to(first_device), y_tens.to(first_device)

                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type='cuda'):
                        logits, loss = model(x_tens, y_tens)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1
                    step_in_epoch += 1

                    if global_step % 100 == 0:
                        print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")

                    if global_step % 2000 == 0:
                        ckpt = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item()
                        }
                        os.makedirs("pretrained", exist_ok=True)
                        torch.save(ckpt, f"pretrained/checkpoint_step_{global_step}.pth")
                        print(f"Checkpoint saved @ step {global_step}")

            except StopIteration:
                print("Reached end of dataset early.")
                break

    # 6) Save final resumed model
    model.save_pretrained("Argonne_LLM")
    hf_tokenizer.save_pretrained("Argonne_LLM")
    print("Resumed training finished. Model saved successfully.")

def main():
    resume_training(
        data_path="data/*.arrow",
        checkpoint_path="pretrained/checkpoint_step_2000.pth",
        epochs=1,
        steps_per_epoch=500,
        block_size=2048,
        batch_size=24,
        lr=3e-5
    )

if __name__ == "__main__":
    main()
