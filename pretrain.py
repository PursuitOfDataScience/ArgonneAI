"""
DDP training for Argonne model.
Supports pretraining, continued pretraining on new data (--reset_schedule),
and automatic checkpoint resume.
"""

import os
import sys
import glob
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Model architecture
HIDDEN_SIZE = 3072
NUM_LAYERS = 24
NUM_HEADS = 12
NUM_KV_HEADS = 4  # GQA
ENABLE_QK_NORM = True
ENABLE_V_NORM = True
Z_LOSS_WEIGHT = 0.0
ENABLE_SANDWICH_NORM = True
ROPE_THETA = 1000000.0
ENABLE_MTP = False
MTP_HORIZON = 1
MTP_LOSS_WEIGHT = 0.0
ENABLE_INTERLEAVED_LOCAL_ATTENTION = True
LOCAL_ATTENTION_WINDOW = 256
LOGIT_SOFTCAP = 15.0

# Parse arguments
parser = argparse.ArgumentParser()
# Paths
parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
parser.add_argument("--data_path", type=str, required=True, help="Path to training data (.bin)")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory for checkpoints")
# Training hyperparameters
# Production LR: 6e-4
# Inherited from exp_317 (2.88B arch, tuned at 24K effective batch in nextrun3 search).
# Production effective batch is ~1M tokens, ~41x larger. Batch-scaling rules suggest:
#   - Linear: 6e-4 * 41 = 2.5e-2  (too aggressive, would diverge)
#   - Sqrt:   6e-4 * 6.4 = 3.8e-3  (aggressive for cold start at this scale)
# Counter-balancing effect: 2.88B is 2.2x larger than the 1.3B llm.c baseline,
# which used LR=3e-4. Larger models typically want slightly lower LR at fixed batch.
# Net: 6e-4 is a conservative, exp_317-validated starting point that survives
# both adjustments. Verify at scale with a short probe (2-5B tokens) before
# committing to a full run. Safe range to explore: 4e-4 to 1e-3.
parser.add_argument("--lr", type=float, default=6.0e-4, help="Learning rate")
parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of LR")
parser.add_argument("--batch_size", type=int, default=19, help="Batch size per GPU")
parser.add_argument("--total_batch_size", type=int, default=1011712, help="Total batch size in tokens")
parser.add_argument("--block_size", type=int, default=1024, help="Sequence length")
parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--schedule", type=str, default="wsd", choices=["cosine", "wsd"], help="LR schedule")
parser.add_argument("--cooldown", type=int, default=4000, help="Cooldown steps at end of WSD schedule")
# Argonne-3.5 recipe: express the WSD cooldown as a fraction of the estimated run instead of a
# fixed step count. When >0 this OVERRIDES --cooldown and is recomputed from estimated_steps on
# every resume, so the terminal LR anneal lands correctly regardless of corpus size or how the
# run is sliced across wall-time-limited SLURM jobs. The launcher's old --cooldown 0 left the
# WSD schedule with NO decay phase at all (LR flat at peak forever); 0.15 restores the anneal.
parser.add_argument("--cooldown_frac", type=float, default=0.0, help="WSD cooldown as a fraction of estimated steps; if >0 overrides --cooldown. Argonne-3.5 recipe: 0.15.")
parser.add_argument("--grad_clip", type=float, default=0.4, help="Gradient clipping")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")
# Argonne-3.5 FP8: torchao float8 (tensorwise dynamic) on the Linear matmuls + lm_head. Recipe-search
# result: ~1.25x H200 throughput at neutral quality (needs torch_compile=1; master weights stay fp32).
parser.add_argument("--fp8", type=int, default=0, choices=[0, 1], help="Enable FP8 training via torchao float8 (requires torch_compile=1)")
parser.add_argument("--fp8_lm_head", type=int, default=1, choices=[0, 1], help="Also FP8 the (tied) lm_head — recipe default; tie is preserved")
parser.add_argument("--flash_attention", type=int, default=1, choices=[0, 1], help="Use flash attention")
parser.add_argument("--checkpoint_interval", type=int, default=1800, help="Checkpoint interval in seconds")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum epochs to train")
parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Use gradient checkpointing")
parser.add_argument("--torch_compile", type=int, default=1, choices=[0, 1], help="Use torch.compile for speedup")
parser.add_argument("--torch_compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint file")
parser.add_argument("--wall_time", type=int, default=0, help="Wall time in seconds. If > 0, save checkpoint 5 min before this limit. 0 = disabled.")
parser.add_argument("--reset_schedule", type=int, default=0, choices=[0, 1], help="Reset LR schedule, step counter, and data position when resuming. Use for continued pretraining on new data.")
parser.add_argument("--val_data_path", type=str, default=None, help="Optional path to held-out validation data (.bin)")
parser.add_argument("--final_model_dir", type=str, default=None, help="Optional directory for the final Hugging Face model export.")
parser.add_argument("--completion_marker", type=str, default=None, help="Optional marker file written only after max_epochs is completed and the final model export succeeds.")
parser.add_argument("--seed", type=int, default=444, help="Base random seed")
args = parser.parse_args()

# Distributed setup
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

RANK, LOCAL_RANK, WORLD_SIZE = setup_distributed()
IS_MAIN = RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}"

BASE_SEED = int(args.seed)
RUN_SEED = BASE_SEED + RANK
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
torch.manual_seed(RUN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RUN_SEED)

# Compute gradient accumulation
TOKENS_PER_MICRO = args.batch_size * WORLD_SIZE * args.block_size
GRAD_ACCUM_STEPS = args.total_batch_size // TOKENS_PER_MICRO
assert GRAD_ACCUM_STEPS >= 1, (
    f"total_batch_size ({args.total_batch_size}) too small for "
    f"{WORLD_SIZE} GPU(s) x batch_size {args.batch_size} x block_size {args.block_size}"
)
ACTUAL_TOTAL_BATCH = GRAD_ACCUM_STEPS * TOKENS_PER_MICRO

# Wall time: save 5 minutes before limit
WALL_TIME_SAVE = args.wall_time - 300 if args.wall_time > 0 else 0

# Autocast setup
if args.precision == "bf16":
    AUTOCAST_DTYPE = torch.bfloat16
    USE_AUTOCAST = True
elif args.precision == "fp16":
    AUTOCAST_DTYPE = torch.float16
    USE_AUTOCAST = True
else:
    AUTOCAST_DTYPE = None
    USE_AUTOCAST = False

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
    def __init__(self, filename, B, T, rank=0, world_size=1, start_token_offset=0):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.tokens = load_data_shard(filename)
        self.start_token_offset = int(start_token_offset)
        self.current_position = self.start_token_offset + rank * B * T
        self.epoch = 0
        if self.current_position + (B * T + 1) > len(self.tokens):
            raise ValueError(
                f"Start offset {self.start_token_offset} is too close to end of dataset "
                f"for batch_size={B}, block_size={T}, world_size={world_size}"
            )
        if rank == 0:
            print(f"DataLoader: {len(self.tokens):,} tokens (start_offset={self.start_token_offset:,})")

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long).pin_memory()
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.world_size
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = self.start_token_offset + self.rank * B * T
            self.epoch += 1
            if self.rank == 0:
                print(f"\n*** Epoch {self.epoch} completed ***\n")
        return x, y

    def get_position(self):
        return self.current_position

    def set_position(self, position):
        self.current_position = position

# Import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ArgonneConfig, ArgonneModel
from transformers import AutoTokenizer


def get_base_model(model):
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def apply_fp8_training(model, include_lm_head=True):
    """Argonne-3.5: swap eligible nn.Linear -> torchao Float8Linear (tensorwise DYNAMIC scaling) so the
    Linear matmuls run in FP8 on H200 tensor cores (~1.25x throughput, quality-neutral per the recipe
    search). Dynamic scaling adds NO persistent params/buffers, so checkpoints + state_dict stay 1:1
    with plain nn.Linear (resume/export work). Skips any Linear whose dims aren't /16 (scaled_mm req).
    Call BEFORE DDP/compile, with master weights in fp32. Returns (n_converted, n_skipped)."""
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig, Float8LinearRecipeName
    cfg = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.TENSORWISE)
    stat = {"c": 0, "s": 0}

    def flt(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if ("lm_head" in fqn) and not include_lm_head:
            stat["s"] += 1
            return False
        if (mod.in_features % 16) or (mod.out_features % 16):
            stat["s"] += 1
            return False
        stat["c"] += 1
        return True

    convert_to_float8_training(model, module_filter_fn=flt, config=cfg)
    return stat["c"], stat["s"]


def generate_text(model, tokenizer, device, prompt="Long long time ago", max_new_tokens=100):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_length = input_ids.shape[1] + max_new_tokens
        gen_model = get_base_model(model)
        with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
            output = gen_model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.8, top_p=0.95)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return generated_text


def save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, loss, data_position, checkpoint_dir):
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
    latest_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    latest_tmp_path = latest_path + ".tmp"
    try:
        if os.path.lexists(latest_tmp_path):
            os.remove(latest_tmp_path)
        os.symlink(os.path.basename(checkpoint_path), latest_tmp_path)
        os.replace(latest_tmp_path, latest_path)
    except OSError:
        pass
    return checkpoint_path


def get_latest_checkpoint_path(checkpoint_dir):
    latest_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    if os.path.exists(latest_path):
        return latest_path

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    if not checkpoints:
        return None

    steps = [int(f.split("_step_")[-1].replace(".pt", "")) for f in checkpoints]
    latest_step = max(steps)
    return os.path.join(checkpoint_dir, f"checkpoint_step_{latest_step}.pt")


def write_completion_marker(marker_path, global_step, tokens_processed, final_model_dir):
    marker_dir = os.path.dirname(marker_path)
    if marker_dir:
        os.makedirs(marker_dir, exist_ok=True)
    tmp_path = marker_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"completed_at_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.write(f"global_step={global_step}\n")
        f.write(f"tokens_processed={tokens_processed}\n")
        f.write(f"final_model_dir={final_model_dir}\n")
    os.replace(tmp_path, marker_path)


def main():
    if IS_MAIN:
        print("=" * 60)
        print("Argonne Model Training (DDP)")
        print("=" * 60)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Using device: {DEVICE}, World size: {WORLD_SIZE}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
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
        rope_theta=ROPE_THETA,
        use_flash_attention=args.flash_attention == 1,
        qk_norm=ENABLE_QK_NORM,
        v_norm=ENABLE_V_NORM,
        sandwich_norm=ENABLE_SANDWICH_NORM,
        z_loss_weight=Z_LOSS_WEIGHT,
        mtp_horizon=MTP_HORIZON if ENABLE_MTP else 1,
        mtp_loss_weight=MTP_LOSS_WEIGHT if ENABLE_MTP else 0.0,
        interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
        local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
        logit_softcap=LOGIT_SOFTCAP,
        tie_word_embeddings=True,
    )
    config._keep_in_fp32_modules = []
    model = ArgonneModel(config)
    model = model.to(DEVICE)
    # Model stays in fp32 — autocast handles bf16/fp16 for forward pass
    # This keeps optimizer states in fp32 for proper precision

    # Gradient checkpointing (before DDP and compile)
    if args.gradient_checkpointing == 1:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if IS_MAIN:
                print("Gradient checkpointing enabled")

    # Argonne-3.5 FP8 (torchao float8): convert Linear matmuls to FP8 BEFORE DDP/compile.
    if args.fp8 == 1:
        if args.torch_compile != 1 and IS_MAIN:
            print("WARNING: --fp8 without --torch_compile — FP8 scaling won't fuse; expect NO speedup.")
        embed_w = model.get_input_embeddings().weight            # capture the tied weight pre-conversion
        n_conv, n_skip = apply_fp8_training(model, include_lm_head=(args.fp8_lm_head == 1))
        # The tie must survive (torchao reuses the weight Parameter). Fail loudly if it silently broke.
        assert model.get_output_embeddings().weight is embed_w, (
            "FP8 conversion broke the embedding<->lm_head tie — do not train (would be a different model)")
        if IS_MAIN:
            print(f"FP8 training ON (torchao tensorwise, lm_head={args.fp8_lm_head}): "
                  f"converted {n_conv} Linear, skipped {n_skip}; embedding tie preserved.")

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
        print(f"Mixed precision: {'autocast ' + args.precision if USE_AUTOCAST else 'fp32 (no autocast)'}")

    # Create data loader
    train_loader = DataLoader(
        args.data_path,
        args.batch_size,
        args.block_size,
        RANK,
        WORLD_SIZE,
        start_token_offset=0,
    )
    val_loader = None
    if IS_MAIN and args.val_data_path:
        val_loader = DataLoader(
            args.val_data_path,
            args.batch_size,
            args.block_size,
            rank=0,
            world_size=1,
            start_token_offset=0,
        )

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

    # Scheduler with warmup (cosine or WSD)
    min_lr = args.lr * args.min_lr_ratio
    min_lr_scale = min_lr / args.lr

    # Argonne-3.5: resolve the WSD cooldown length. A fraction (--cooldown_frac) is preferred
    # because estimated_steps is recomputed identically on every resume, so cooldown_start below
    # is stable across the wall-time-sliced training and the anneal always finishes at the run's
    # true end. Falls back to the fixed --cooldown step count when the fraction is 0.
    cooldown_steps = int(args.cooldown_frac * estimated_steps) if args.cooldown_frac > 0 else args.cooldown

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)

        if args.schedule == "cosine":
            progress = (step - args.warmup_steps) / max(1, estimated_steps - args.warmup_steps)
            return max(min_lr_scale, 0.5 * (1.0 + np.cos(np.pi * progress)))

        if cooldown_steps <= 0:
            return 1.0

        cooldown_start = max(args.warmup_steps, estimated_steps - cooldown_steps)
        if step < cooldown_start:
            return 1.0

        cooldown_progress = min(1.0, (step - cooldown_start) / max(1, cooldown_steps))
        return 1.0 - cooldown_progress * (1.0 - min_lr_scale)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    resume_from = args.resume_from or get_latest_checkpoint_path(args.checkpoint_dir)

    if resume_from and os.path.exists(resume_from):
        if IS_MAIN:
            print(f"\n=== Resuming from checkpoint: {resume_from} ===")
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        base_model = get_base_model(model)
        base_model.load_state_dict(checkpoint['model_state_dict'])

        if args.reset_schedule == 1:
            if IS_MAIN:
                print("Reset schedule mode: fresh optimizer, scheduler, step counter, data position")
                print(f"Previous training: {checkpoint['tokens_processed']:,} tokens, step {checkpoint['global_step']}")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            global_step = 0
            tokens_processed = 0
            is_resumed = False
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            else:
                for _ in range(checkpoint['global_step']):
                    scheduler.step()
            global_step = checkpoint['global_step']
            tokens_processed = checkpoint['tokens_processed']
            data_position = checkpoint.get('data_position', 0)
            train_loader.set_position(data_position + RANK * args.batch_size * args.block_size)
            train_loader.epoch = tokens_processed // num_tokens
            if IS_MAIN:
                print(f"Resumed from step {global_step}, tokens: {tokens_processed:,}, epoch: {train_loader.epoch}, LR: {scheduler.get_last_lr()[0]:.2e}")
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
        print(f"Seed: base={BASE_SEED}, rank_seed={RUN_SEED}")
        print(
            f"QK norm: {ENABLE_QK_NORM} | v_norm: {ENABLE_V_NORM} | "
            f"sandwich_norm: {ENABLE_SANDWICH_NORM} | z_loss_weight: {Z_LOSS_WEIGHT} | "
            f"rope_theta: {ROPE_THETA} | mtp: {ENABLE_MTP} (horizon={MTP_HORIZON}, weight={MTP_LOSS_WEIGHT}) | "
            f"interleaved_local_attention: {ENABLE_INTERLEAVED_LOCAL_ATTENTION} (window={LOCAL_ATTENTION_WINDOW}) | "
            f"logit_softcap: {LOGIT_SOFTCAP}"
        )
        print(f"LR: {args.lr}, Warmup: {args.warmup_steps}, Min LR Ratio: {args.min_lr_ratio}, Precision: {args.precision}, TorchCompile: {args.torch_compile}")
        print(f"Schedule: {args.schedule}, Cooldown: {cooldown_steps} steps (frac={args.cooldown_frac}, ~{(cooldown_steps/max(1,estimated_steps)*100):.1f}% of run), Grad clip: {args.grad_clip}")
        print(f"Checkpoint interval: {args.checkpoint_interval} seconds")
        print(f"Validation data: {args.val_data_path if args.val_data_path else 'disabled (no held-out file provided)'}")
        if args.wall_time > 0:
            print(f"Wall time: {args.wall_time}s, will save checkpoint at {WALL_TIME_SAVE}s")
        if args.reset_schedule == 1:
            print("Mode: continued pretraining (fresh schedule)")
        print("-" * 60)

    if not is_resumed:
        global_step = 0
        tokens_processed = 0
    last_checkpoint_time = time.time()
    training_start_time = time.time()
    train_losses = []
    completed_max_epochs = train_loader.epoch >= args.max_epochs

    pbar = None
    if IS_MAIN:
        initial_steps = 0
        if is_resumed:
            # Use token-based progress so resumed runs remain accurate even if batch config changed.
            initial_steps = min(estimated_steps, int(tokens_processed // ACTUAL_TOTAL_BATCH))
        pbar = tqdm(total=estimated_steps, initial=initial_steps, desc="Training", unit="step", disable=False)

    model.train()

    if completed_max_epochs and IS_MAIN:
        print(f"\nCheckpoint is already at {train_loader.epoch} epoch(s); finalizing without more training.")

    while not completed_max_epochs:
        start_time = time.time()
        optimizer.zero_grad()
        step_loss_total = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if WORLD_SIZE > 1 and micro_step < GRAD_ACCUM_STEPS - 1:
                with model.no_sync():
                    with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                        outputs = model(x, labels=y)
                        micro_loss = outputs.loss
                        loss = micro_loss / GRAD_ACCUM_STEPS
                    loss.backward()
            else:
                with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                    outputs = model(x, labels=y)
                    micro_loss = outputs.loss
                    loss = micro_loss / GRAD_ACCUM_STEPS
                loss.backward()

            tokens_processed += args.batch_size * args.block_size * WORLD_SIZE
            step_loss_total += micro_loss.detach().float().item()
            train_losses.append(micro_loss.detach().float().item())

        step_loss = step_loss_total / GRAD_ACCUM_STEPS

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        if pbar:
            pbar.update(1)

        current_lr = optimizer.param_groups[0]['lr']

        if IS_MAIN and global_step % 50 == 0:
            perplexity = np.exp(step_loss)
            print(f"Step {global_step} | Loss: {step_loss:.4f} | PPL: {perplexity:.2f} | Tokens: {tokens_processed:,} | LR: {current_lr:.2e}")
            if pbar:
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{current_lr:.2e}", "tokens": f"{tokens_processed/1e6:.2f}M"})

        # Synchronized checkpoint decision
        should_checkpoint = torch.tensor([0], device=DEVICE)
        if IS_MAIN:
            current_time = time.time()
            if current_time - last_checkpoint_time >= args.checkpoint_interval:
                should_checkpoint[0] = 1
        if WORLD_SIZE > 1:
            dist.broadcast(should_checkpoint, src=0)

        if should_checkpoint[0] == 1:
            if IS_MAIN:
                print("\n" + "=" * 60)
                print("Saving checkpoint...")
                data_position = train_loader.get_position()
                checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, step_loss, data_position, args.checkpoint_dir)
                print(f"Checkpoint saved: {checkpoint_path}")

                print("\nGenerating sample text...")
                generated = generate_text(model, tokenizer, DEVICE, prompt="Long long time ago")
                print(f"Generated: {generated}")
                print("=" * 60 + "\n")

            if WORLD_SIZE > 1:
                dist.barrier()
            last_checkpoint_time = time.time()

        # Synchronized wall time check
        if WALL_TIME_SAVE > 0:
            should_wall_stop = torch.tensor([0], device=DEVICE)
            if IS_MAIN:
                elapsed = time.time() - training_start_time
                if elapsed >= WALL_TIME_SAVE:
                    should_wall_stop[0] = 1
            if WORLD_SIZE > 1:
                dist.broadcast(should_wall_stop, src=0)

            if should_wall_stop[0] == 1:
                if IS_MAIN:
                    print(f"\nApproaching wall time ({args.wall_time}s). Saving checkpoint and exiting...")
                    data_position = train_loader.get_position()
                    checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, step_loss, data_position, args.checkpoint_dir)
                    print(f"Wall time checkpoint saved: {checkpoint_path}")
                if WORLD_SIZE > 1:
                    dist.barrier()
                break

        # Synchronized epoch completion check
        should_stop = torch.tensor([0], device=DEVICE)
        if train_loader.epoch >= args.max_epochs:
            should_stop[0] = 1
        if WORLD_SIZE > 1:
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)

        if should_stop[0] == 1:
            if IS_MAIN:
                print(f"\nCompleted {args.max_epochs} epoch(s) at step {global_step}. Finalizing...")
            completed_max_epochs = True
            break

    if pbar:
        pbar.close()

    if IS_MAIN:
        print("-" * 60)
        elapsed_time = time.time() - training_start_time
        print(f"Training completed in {elapsed_time:.1f} seconds!")

    # Evaluate on validation (rank 0 only)
    if IS_MAIN:
        val_losses = []
        if val_loader is not None:
            print("\nEvaluating on validation...")
            model.eval()
            val_tokens = min(1_000_000, len(val_loader.tokens))
            val_batches = val_tokens // (args.batch_size * args.block_size)

            with torch.no_grad():
                original_pos = val_loader.current_position
                val_loader.current_position = 0

                for _ in range(min(val_batches, 100)):
                    x, y = val_loader.next_batch()
                    x = x.to(DEVICE, non_blocking=True)
                    y = y.to(DEVICE, non_blocking=True)

                    with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                        outputs = model(x, labels=y)
                    val_losses.append(outputs.loss.item())

                val_loader.current_position = original_pos
        else:
            print("\nValidation skipped: no held-out validation file was provided.")

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else float("nan")
        val_loss_str = f"{val_loss:.4f}" if val_losses else "n/a"
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss_str}")

        if completed_max_epochs:
            print("\nSaving final checkpoint...")
            data_position = train_loader.get_position()
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, train_loss, data_position, args.checkpoint_dir)
            print(f"Final checkpoint saved: {checkpoint_path}")

            final_model_dir = args.final_model_dir or os.path.join(args.checkpoint_dir, "final_model_complete")
            os.makedirs(final_model_dir, exist_ok=True)
            save_model = get_base_model(model)
            if args.fp8 == 1:
                # Export a clean nn.Linear model so the shipped HF checkpoint needs no torchao at
                # inference. Dynamic tensorwise FP8 keeps weights in high precision → state_dict is 1:1.
                clean = ArgonneModel(config)
                missing, unexpected = clean.load_state_dict(save_model.state_dict(), strict=False)
                assert not missing, (
                    f"FP8 export: clean model missing weights {missing[:8]} — refusing to export garbage")
                if hasattr(clean, "tie_weights"):
                    clean.tie_weights()
                if IS_MAIN:
                    print(f"FP8 export: rebuilt clean nn.Linear model (dropped {len(unexpected)} torchao keys)")
                save_model = clean.to(DEVICE)

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

            if args.completion_marker:
                write_completion_marker(args.completion_marker, global_step, tokens_processed, final_model_dir)
                print(f"Completion marker written to: {args.completion_marker}")
        else:
            print("\nFinal checkpoint/model export skipped because training stopped before completing max_epochs.")

        elapsed_time = time.time() - training_start_time
        print("\n" + "=" * 60)
        print(f"SUMMARY: train_loss={train_loss:.4f} val_loss={val_loss_str} tokens_per_sec={tokens_processed/elapsed_time:.2f} steps={global_step}")
        print("=" * 60)

    if WORLD_SIZE > 1:
        dist.barrier()

    cleanup_distributed()

if __name__ == "__main__":
    main()
