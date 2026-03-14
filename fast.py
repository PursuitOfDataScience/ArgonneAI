"""
LLM.c style training for Argonne model using Qwen tokenizer and fineweb-edu data.
Uses DistributedDataParallel for multi-GPU training.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Paths
QWEN_TOKENIZER_PATH = "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base"
FINEWEB_DATA_PATH = "/project/rcc/youzhi/fineweb-binary-qwen3/train.bin"
CHECKPOINT_DIR = "/project/rcc/youzhi/llm.c/checkpoints"

# Model architecture
HIDDEN_SIZE = 2048
NUM_LAYERS = 16
NUM_HEADS = 16
NUM_KV_HEADS = 8  # GQA

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of LR")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size per GPU")
parser.add_argument("--total_batch_size", type=int, required=True, help="Total batch size in tokens")
parser.add_argument("--block_size", type=int, required=True, help="Sequence length")
parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")
parser.add_argument("--flash_attention", type=int, default=1, choices=[0, 1], help="Use flash attention")
parser.add_argument("--checkpoint_interval", type=int, default=1800, help="Checkpoint interval in seconds")
parser.add_argument("--run_id", type=int, default=0, help="Run ID for logging")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum epochs to train")
parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Use gradient checkpointing")
parser.add_argument("--torch_compile", type=int, default=0, choices=[0, 1], help="Use torch.compile for speedup")
parser.add_argument("--torch_compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint file")
args = parser.parse_args()

# Distributed setup
def setup_distributed():
    """Initialize distributed training. Works with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Single GPU fallback (launched with python3 instead of torchrun)
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

RANK, LOCAL_RANK, WORLD_SIZE = setup_distributed()
IS_MAIN = RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}"

# Compute gradient accumulation
TOKENS_PER_MICRO = args.batch_size * WORLD_SIZE * args.block_size
GRAD_ACCUM_STEPS = args.total_batch_size // TOKENS_PER_MICRO
assert GRAD_ACCUM_STEPS >= 1, (
    f"total_batch_size ({args.total_batch_size}) too small for "
    f"{WORLD_SIZE} GPU(s) x batch_size {args.batch_size} x block_size {args.block_size}"
)
ACTUAL_TOTAL_BATCH = GRAD_ACCUM_STEPS * TOKENS_PER_MICRO

# Data loading
def load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        magic = header[0]
        if magic != 20240801:
            raise ValueError(f"Unknown magic number: {magic}")
        tokens = np.memmap(filename, dtype=np.uint32, mode='r', offset=256*4)
    return tokens

class DataLoader:
    """Data loader that shards data across DDP ranks."""
    def __init__(self, filename, B, T, rank=0, world_size=1):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.tokens = load_data_shard(filename)
        self.current_position = rank * B * T  # each rank starts at different offset
        self.epoch = 0
        if rank == 0:
            print(f"DataLoader: {len(self.tokens):,} tokens")

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long).pin_memory()
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        # Advance by world_size * B * T so ranks don't overlap
        self.current_position += B * T * self.world_size
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = self.rank * B * T
            self.epoch += 1
            if self.rank == 0:
                print(f"\n*** Epoch {self.epoch} completed ***\n")
        return x, y

    def get_position(self):
        return self.current_position

    def set_position(self, position):
        self.current_position = position

# Import model
sys.path.insert(0, '/home/youzhi/ArgonneAI')
from model import ArgonneConfig, ArgonneModel
from transformers import AutoTokenizer


def generate_text(model, tokenizer, device, prompt="Long long time ago", max_new_tokens=100):
    """Generate text from a prompt."""
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_length = input_ids.shape[1] + max_new_tokens
        gen_model = model.module if hasattr(model, 'module') else model
        if hasattr(gen_model, '_orig_mod'):
            gen_model = gen_model._orig_mod
        output = gen_model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.8, top_p=0.95)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return generated_text


def get_base_model(model):
    """Unwrap DDP and torch.compile to get the underlying model."""
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, loss, data_position, checkpoint_dir):
    """Save model checkpoint. Only called from rank 0."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")

    base_model = get_base_model(model)

    checkpoint = {
        'global_step': global_step,
        'tokens_processed': tokens_processed,
        'loss': loss,
        'data_position': data_position,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def main():
    if IS_MAIN:
        print("=" * 60)
        print("LLM.c Style Training for Argonne Model (DDP)")
        print("=" * 60)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if IS_MAIN:
        print(f"Using device: {DEVICE}, World size: {WORLD_SIZE}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_PATH, trust_remote_code=True)
    VOCAB_SIZE = len(tokenizer)
    if IS_MAIN:
        print(f"Vocab size: {VOCAB_SIZE}, EOS token ID: {tokenizer.eos_token_id}")

    # Create model
    config = ArgonneConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=args.block_size,
        use_flash_attention=args.flash_attention == 1,
        tie_word_embeddings=False,
    )
    config._keep_in_fp32_modules = []
    model = ArgonneModel(config)
    model = model.to(DEVICE)

    if args.precision == "fp32":
        model = model.to(torch.float32)
    elif args.precision == "fp16":
        model = model.to(torch.float16)
    else:
        model = model.to(torch.bfloat16)

    # Gradient checkpointing (before DDP and compile)
    if args.gradient_checkpointing == 1 and args.torch_compile == 0:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if IS_MAIN:
                print("Gradient checkpointing enabled")

    # Wrap with DDP
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"Using {WORLD_SIZE} GPUs with DistributedDataParallel")

    # torch.compile
    if args.torch_compile == 1:
        if IS_MAIN:
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.torch_compile_mode)

    if IS_MAIN:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

    # Create data loader (each rank gets its own shard)
    train_loader = DataLoader(FINEWEB_DATA_PATH, args.batch_size, args.block_size, RANK, WORLD_SIZE)

    # Estimate steps for scheduler
    num_tokens = len(train_loader.tokens)
    estimated_steps = int((num_tokens * args.max_epochs) / ACTUAL_TOTAL_BATCH)
    if IS_MAIN:
        print(f"Training for {args.max_epochs} epoch(s) ~= {estimated_steps} steps ({num_tokens * args.max_epochs:,} tokens)")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    # Cosine scheduler with warmup
    min_lr = args.lr * args.min_lr_ratio
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, estimated_steps - args.warmup_steps)
            return max(min_lr / args.lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    resume_from = args.resume_from
    if not resume_from:
        import glob
        checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_step_*.pt"))
        if checkpoints:
            steps = [int(f.split("_step_")[-1].replace(".pt", "")) for f in checkpoints]
            latest_step = max(steps)
            resume_from = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{latest_step}.pt")

    if resume_from and os.path.exists(resume_from):
        if IS_MAIN:
            print(f"\n=== Resuming from checkpoint: {resume_from} ===")
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        base_model = get_base_model(model)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint['global_step']
        tokens_processed = checkpoint['tokens_processed']
        # Restore data position: reconstruct per-rank position
        data_position = checkpoint.get('data_position', 0)
        train_loader.set_position(data_position + RANK * args.batch_size * args.block_size)
        train_loader.epoch = tokens_processed // num_tokens
        for _ in range(global_step):
            scheduler.step()
        if IS_MAIN:
            print(f"Resumed from step {global_step}, tokens: {tokens_processed:,}, epoch: {train_loader.epoch}")
        is_resumed = True
    else:
        is_resumed = False

    # Training loop
    if IS_MAIN:
        print("\nStarting training...")
        print(f"GPUs: {WORLD_SIZE}, Batch size per GPU: {args.batch_size}")
        print(f"Sequence length: {args.block_size}")
        print(f"Total batch size: {ACTUAL_TOTAL_BATCH} tokens (requested: {args.total_batch_size})")
        print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
        print(f"Training for {args.max_epochs} epoch(s) (estimated ~{estimated_steps} steps)")
        print(f"LR: {args.lr}, Warmup: {args.warmup_steps}, Min LR Ratio: {args.min_lr_ratio}, Precision: {args.precision}, TorchCompile: {args.torch_compile}")
        print(f"Checkpoint interval: {args.checkpoint_interval} seconds")
        print("-" * 60)

    if not is_resumed:
        global_step = 0
        tokens_processed = 0
    last_checkpoint_time = time.time()
    training_start_time = time.time()
    train_losses = []

    pbar = None
    if IS_MAIN:
        pbar = tqdm(total=estimated_steps, desc="Training", unit="step")
        if is_resumed:
            pbar.update(global_step)

    model.train()

    while True:
        start_time = time.time()
        optimizer.zero_grad()

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            # DDP: disable gradient sync for accumulation steps, sync on last
            if WORLD_SIZE > 1 and micro_step < GRAD_ACCUM_STEPS - 1:
                with model.no_sync():
                    outputs = model(x, labels=y)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
                    loss.backward()
            else:
                outputs = model(x, labels=y)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

            tokens_processed += args.batch_size * args.block_size * WORLD_SIZE
            train_losses.append(loss.item() * GRAD_ACCUM_STEPS)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        if pbar:
            pbar.update(1)

        current_lr = optimizer.param_groups[0]['lr']

        if IS_MAIN and global_step % 10 == 0:
            loss_val = loss.item() * GRAD_ACCUM_STEPS
            perplexity = np.exp(loss_val)
            print(f"Step {global_step} | Loss: {loss_val:.4f} | PPL: {perplexity:.2f} | Tokens: {tokens_processed:,} | LR: {current_lr:.2e}")
            if pbar:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}", "tokens": f"{tokens_processed/1e6:.2f}M"})

        # Periodic checkpoint (rank 0 only)
        current_time = time.time()
        if current_time - last_checkpoint_time >= args.checkpoint_interval:
            if IS_MAIN:
                print("\n" + "=" * 60)
                print("Saving checkpoint...")
                data_position = train_loader.get_position()
                checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, loss.item() * GRAD_ACCUM_STEPS, data_position, CHECKPOINT_DIR)
                print(f"Checkpoint saved: {checkpoint_path}")

                print("\nGenerating sample text...")
                generated = generate_text(model, tokenizer, DEVICE, prompt="Long long time ago")
                print(f"Generated: {generated}")
                print("=" * 60 + "\n")

            # All ranks wait for checkpoint to finish
            if WORLD_SIZE > 1:
                dist.barrier()
            last_checkpoint_time = time.time()

        # Epoch completion check
        if train_loader.epoch >= args.max_epochs:
            if IS_MAIN:
                print(f"\nCompleted {args.max_epochs} epoch(s) at step {global_step}. Finalizing...")
            break

    if pbar:
        pbar.close()

    if IS_MAIN:
        print("-" * 60)
        elapsed_time = time.time() - training_start_time
        print(f"Training completed in {elapsed_time:.1f} seconds!")

    # Evaluate on validation (rank 0 only)
    if IS_MAIN:
        print("\nEvaluating on validation...")
        model.eval()
        val_losses = []
        val_tokens = min(1_000_000, len(train_loader.tokens) // 2)
        val_batches = val_tokens // (args.batch_size * args.block_size)

        with torch.no_grad():
            original_pos = train_loader.current_position
            train_loader.current_position = 0

            for _ in range(min(val_batches, 100)):
                x, y = train_loader.next_batch()
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                outputs = model(x, labels=y)
                val_losses.append(outputs.loss.item())

            train_loader.current_position = original_pos

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else 0
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save final training checkpoint
        print("\nSaving final checkpoint...")
        data_position = train_loader.get_position()
        checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, train_loss, data_position, CHECKPOINT_DIR)
        print(f"Final checkpoint saved: {checkpoint_path}")

        # Save complete model + tokenizer + config
        final_model_dir = os.path.join(CHECKPOINT_DIR, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        save_model = get_base_model(model)

        # Trim extra embedding rows if needed
        actual_vocab = len(tokenizer)
        embed = save_model.get_input_embeddings()
        if embed.weight.shape[0] > actual_vocab:
            print(f"Trimming embeddings from {embed.weight.shape[0]} to {actual_vocab}")
            embed.weight = nn.Parameter(embed.weight[:actual_vocab])
            lm_head = save_model.get_output_embeddings()
            if lm_head is not None:
                lm_head.weight = nn.Parameter(lm_head.weight[:actual_vocab])
            save_model.config.vocab_size = actual_vocab

        save_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        config.save_pretrained(final_model_dir)
        print(f"Final model + tokenizer + config saved to: {final_model_dir}")

        elapsed_time = time.time() - training_start_time
        print("\n" + "=" * 60)
        print(f"SUMMARY: train_loss={train_loss:.4f} val_loss={val_loss:.4f} tokens_per_sec={tokens_processed/elapsed_time:.2f} steps={global_step}")
        print("=" * 60)

    # Wait for rank 0 to finish saving
    if WORLD_SIZE > 1:
        dist.barrier()

    cleanup_distributed()

if __name__ == "__main__":
    main()