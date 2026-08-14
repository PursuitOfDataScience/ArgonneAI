"""
DDP training for Argonne model.
Supports pretraining, continued pretraining on new data (--reset_schedule),
and automatic checkpoint resume.
"""

import os
import re
import sys
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Autocast setup (keep model weights/optimizer states in fp32)
AUTOCAST_DTYPE = None
USE_AUTOCAST = False

# Model architecture -- argonne4.0: argonne3.5 arch SCALED TO ~1.04B (must MATCH
# pretrain.py's constants so the pretrain->continue->midtrain checkpoint resume shapes
# agree). head_dim is derived (1536//6 = 256).
HIDDEN_SIZE = 1536
NUM_LAYERS = 32
NUM_HEADS = 6
NUM_KV_HEADS = 2  # GQA
INTERMEDIATE_SIZE = 4096
ROPE_THETA = 1000000.0
ENABLE_QK_NORM = True
ENABLE_V_NORM = True
ENABLE_SANDWICH_NORM = True
Z_LOSS_WEIGHT = 0.0
ENABLE_INTERLEAVED_LOCAL_ATTENTION = True
LOCAL_ATTENTION_WINDOW = 256
LOGIT_SOFTCAP = 15.0

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
    def __init__(self, filename, B, T, rank=0, world_size=1):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.tokens = load_data_shard(filename)
        self.current_position = rank * B * T
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
    """Argonne-3.5 FP8 (torchao float8, tensorwise dynamic) on the Linear matmuls + tied lm_head.
    Convert BEFORE DDP/compile; fp32 master weights; no persistent buffers so ckpt/resume are 1:1 with
    nn.Linear. Skips Linears whose dims aren't /16. Returns (n_converted, n_skipped, lm_status)."""
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    try:
        from torchao.float8 import Float8LinearRecipeName            # torchao >= ~0.10
    except ImportError:
        from torchao.float8.config import Float8LinearRecipeName     # older torchao
    cfg = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.TENSORWISE)
    stat = {"c": 0, "s": 0, "lm": "absent"}

    def flt(mod, fqn):
        is_lm = "lm_head" in fqn
        if not isinstance(mod, nn.Linear):
            return False
        if is_lm and not include_lm_head:
            stat["s"] += 1; stat["lm"] = "skipped(flag_off)"; return False
        if (mod.in_features % 16) or (mod.out_features % 16):
            stat["s"] += 1
            if is_lm:
                stat["lm"] = f"SKIPPED(dim_not_/16: out={mod.out_features}) -> NO fp8 lm_head speedup!"
            return False
        stat["c"] += 1
        if is_lm:
            stat["lm"] = "converted"
        return True

    convert_to_float8_training(model, module_filter_fn=flt, config=cfg)
    return stat["c"], stat["s"], stat["lm"]


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


def _nonfinite_params(base_model, limit=5):
    """GATE 1 (pre-write): does the LIVE model contain NaN/Inf?

    This is the gate that makes latest-only retention safe. model.py deliberately turns a NaN
    loss into a zero-gradient no-op and keeps training, so a diverged run looks healthy from the
    loss curve alone -- and without this check the next save would write the corrupt weights and
    the prune would then delete the last GOOD checkpoint. GPU-side reduction, well under a second.
    """
    bad = []
    for name, prm in base_model.named_parameters():
        if not torch.isfinite(prm.detach()).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _verify_written_checkpoint(path, expect_step, expect_tensors):
    """GATE 2 (post-write, pre-commit): re-open the bytes we just wrote and prove they are a
    complete, loadable checkpoint. Returns None if good, else a reason string.

    Catches the truncated/partial write -- a slice killed mid-torch.save previously left a file
    that LOOKED like a checkpoint. mmap keeps this cheap; reading a sample of tensors forces real
    disk I/O so a file with a valid header but missing tail is still caught.
    """
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:  # unreadable, truncated, or not a zip at all
        return f"unreadable ({type(e).__name__}: {e})"
    for k in ("global_step", "tokens_processed", "model_state_dict", "optimizer_state_dict"):
        if k not in ck:
            return f"missing key {k!r}"
    if ck["global_step"] != expect_step:
        return f"global_step {ck['global_step']} != {expect_step}"
    sd = ck["model_state_dict"]
    if len(sd) != expect_tensors:
        return f"{len(sd)} tensors on disk, expected {expect_tensors}"
    names = list(sd)
    # Sample the ends, the biggest tensor, and 5 evenly spaced tensors through the stack. The
    # ends+largest alone collapse to just the two tied embedding tensors on this arch, which
    # would leave every transformer layer unread; spreading the sample forces real disk I/O
    # across the whole file so a valid-header/bad-tail write cannot slip through. The .tmp is
    # still in page cache at this point, so this costs well under a second.
    idx = {0, len(names) - 1}
    idx.update(round(i * (len(names) - 1) / 6) for i in range(1, 6))
    sample = {names[i] for i in idx}
    sample.add(max(names, key=lambda k: sd[k].numel()))
    for n in sorted(sample):
        try:
            if not torch.isfinite(sd[n].float()).all():
                return f"non-finite values in {n}"
        except Exception as e:
            return f"could not read {n} ({type(e).__name__}: {e})"
    return None


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    global_step,
    tokens_processed,
    loss,
    data_position,
    checkpoint_dir,
    dataset_epoch,
    dataset_base_global_step,
    dataset_base_tokens_processed,
    dataset_num_tokens,
    dataset_path,
):
    """Write a checkpoint only if it is provably good, then prune.

    Same two-gate contract as pretrain.py: gate the LIVE weights, write to .tmp, re-read and
    verify the bytes, and only then atomically commit and delete the previous checkpoint. On any
    failure this leaves the previous checkpoint untouched and returns None -- a run that cannot
    save is far better than a run whose only surviving checkpoint is corrupt.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    base_model = get_base_model(model)

    bad = _nonfinite_params(base_model)
    if bad:
        print(f"REFUSING TO SAVE at step {global_step}: non-finite parameters in {bad}"
              f"{' (and more)' if len(bad) >= 5 else ''}. The previous checkpoint is left intact.",
              flush=True)
        return None

    checkpoint = {
        'global_step': global_step,
        'tokens_processed': tokens_processed,
        'loss': loss,
        'data_position': data_position,
        'dataset_epoch': dataset_epoch,
        'dataset_base_global_step': dataset_base_global_step,
        'dataset_base_tokens_processed': dataset_base_tokens_processed,
        'dataset_num_tokens': dataset_num_tokens,
        'dataset_path': dataset_path,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    n_tensors = len(checkpoint['model_state_dict'])
    tmp_path = checkpoint_path + ".tmp"
    # write to .tmp so a mid-write kill can never leave a plausible-looking checkpoint_step_N.pt
    torch.save(checkpoint, tmp_path)

    why = _verify_written_checkpoint(tmp_path, global_step, n_tensors)
    if why is not None:
        print(f"REFUSING TO COMMIT checkpoint at step {global_step}: {why}. "
              f"Discarding {tmp_path}; the previous checkpoint is left intact.", flush=True)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None

    os.replace(tmp_path, checkpoint_path)   # atomic: the file appears complete or not at all

    latest_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    latest_tmp_path = latest_path + ".tmp"
    try:
        if os.path.lexists(latest_tmp_path):
            os.remove(latest_tmp_path)
        os.symlink(os.path.basename(checkpoint_path), latest_tmp_path)
        os.replace(latest_tmp_path, latest_path)
    except OSError:
        pass
    # only now is deleting the previous checkpoint safe
    print(f"[retention] checkpoint at step {global_step} verified; pruning older checkpoints",
          flush=True)
    prune_old_checkpoints(checkpoint_dir, keep_path=checkpoint_path)
    return checkpoint_path


def prune_old_checkpoints(checkpoint_dir, keep_path):
    """LATEST-ONLY retention (2026-08-05 owner directive, global CLAUDE.md).

    Keep `keep_path` (the checkpoint just written) and delete every other checkpoint_step_*.pt in
    THIS dir -- i.e. this run's own earlier checkpoints. At 12-35 GB apiece a non-rotating dir eats
    hundreds of GB, and one checkpoint is all a resume needs.

    HISTORY, because this spot previously said the opposite: a keep-last-3 prune added 2026-07-28
    was removed 2026-07-29 after it silently deleted 14 of 17 checkpoints and read as data loss.
    The owner reversed the policy on 2026-08-05: latest-only is now the default. What was actually
    wrong then was silence and aggressiveness, so this version PRINTS every deletion.

    Deliberately narrow, so it can only ever remove this run's superseded checkpoints:
      * only `checkpoint_step_<int>.pt` directly in `checkpoint_dir` (glob, not a walk)
      * never `keep_path`, never a symlink (so `checkpoint_last.pt` is untouched), never a `.tmp`
      * never the numerically-highest step present, even if `keep_path` somehow is not it
      * per-file try/except: a failed unlink must not abort training after a good save
    """
    try:
        keep = {os.path.realpath(keep_path)}
        found = []
        for p in glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt")):
            m = re.fullmatch(r"checkpoint_step_(\d+)\.pt", os.path.basename(p))
            if m and not os.path.islink(p):
                found.append((int(m.group(1)), p))
        if len(found) <= 1:
            return
        # Belt and braces: whatever has the highest step number also stays.
        keep.add(os.path.realpath(max(found)[1]))
        for _, p in sorted(found):
            if os.path.realpath(p) in keep:
                continue
            try:
                gib = os.path.getsize(p) / 2**30
                os.remove(p)
                print(f"[retention] removed superseded checkpoint {os.path.basename(p)} "
                      f"({gib:.1f} GiB freed); keeping {os.path.basename(keep_path)}", flush=True)
            except OSError as e:
                print(f"[retention] could not remove {p}: {e}", flush=True)
    except Exception as e:  # retention must never take down a run that just saved successfully
        print(f"[retention] prune skipped: {e}", flush=True)


def cleanup_stale_tmp_checkpoints(checkpoint_dir, min_age_s=3600):
    """Remove abandoned .tmp checkpoint writes left behind by a slice killed mid-save.

    save_checkpoint writes to checkpoint_step_<N>.pt.tmp and renames it into place only after it
    verifies. If SLURM kills the slice during that write (preemption, node failure, wall clock)
    the .tmp survives -- and nothing else ever removes it, because prune_old_checkpoints
    deliberately ignores .tmp so it can never delete an in-flight write. Each file is the full
    checkpoint (~25 GB at 2.06B: fp32 weights + AdamW moments) and the name carries the step, so
    they do not overwrite each other; a handful of preempted slices silently costs hundreds of GB
    across a multi-week chain.

    Only files older than min_age_s are removed, so this can never race a save that is genuinely
    in progress -- a 25 GB torch.save takes minutes, not an hour.
    """
    try:
        now = time.time()
        for p in glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt.tmp")):
            try:
                age = now - os.path.getmtime(p)
                if age < min_age_s:
                    print(f"[retention] leaving {os.path.basename(p)} alone "
                          f"({age/60:.1f} min old -- may be an active write)", flush=True)
                    continue
                gib = os.path.getsize(p) / 2**30
                os.remove(p)
                print(f"[retention] removed abandoned partial write {os.path.basename(p)} "
                      f"({gib:.1f} GiB freed, {age/3600:.1f} h old)", flush=True)
            except OSError as e:
                print(f"[retention] could not remove {p}: {e}", flush=True)
    except Exception as e:  # cleanup must never take down a slice before it even starts
        print(f"[retention] stale-tmp cleanup skipped: {e}", flush=True)


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


def write_progress_marker(marker_path, global_step, tokens_processed, dataset_path):
    marker_dir = os.path.dirname(marker_path)
    if marker_dir:
        os.makedirs(marker_dir, exist_ok=True)
    tmp_path = marker_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"started_at_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.write(f"global_step={global_step}\n")
        f.write(f"tokens_processed={tokens_processed}\n")
        f.write(f"dataset_path={dataset_path}\n")
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

    global AUTOCAST_DTYPE, USE_AUTOCAST
    if args.precision == "bf16":
        AUTOCAST_DTYPE = torch.bfloat16
        USE_AUTOCAST = True
    elif args.precision == "fp16":
        AUTOCAST_DTYPE = torch.float16
        USE_AUTOCAST = True
    else:
        AUTOCAST_DTYPE = None
        USE_AUTOCAST = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    VOCAB_SIZE = len(tokenizer)
    if args.fp8 == 1 and args.fp8_lm_head == 1 and (VOCAB_SIZE % 16 != 0):
        # Pad vocab to a mult of 128 so the tied lm_head GEMM is FP8-eligible (see pretrain.py). Export
        # trims back. Must match the pretrain-stage padded vocab so the resumed checkpoint shape agrees.
        padded = ((VOCAB_SIZE + 127) // 128) * 128
        if IS_MAIN:
            print(f"FP8 lm_head: padding vocab {VOCAB_SIZE} -> {padded} (mult of 128; export trims back)")
        VOCAB_SIZE = padded
    if IS_MAIN:
        print(f"Vocab size: {VOCAB_SIZE}, EOS token ID: {tokenizer.eos_token_id}")

    # Create model
    config = ArgonneConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=args.block_size,
        rope_theta=ROPE_THETA,
        use_flash_attention=args.flash_attention == 1,
        qk_norm=ENABLE_QK_NORM,
        v_norm=ENABLE_V_NORM,
        sandwich_norm=ENABLE_SANDWICH_NORM,
        z_loss_weight=Z_LOSS_WEIGHT,
        interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
        local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
        logit_softcap=LOGIT_SOFTCAP,
        loss_chunk_size=args.loss_chunk_size,
        tie_word_embeddings=True,
    )
    config._keep_in_fp32_modules = []
    model = ArgonneModel(config)
    model = model.to(DEVICE)

    # Keep model weights in fp32 — autocast handles bf16/fp16 for forward pass

    # Gradient checkpointing (before DDP and compile)
    if args.gradient_checkpointing == 1:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if IS_MAIN:
                print("Gradient checkpointing enabled")
        if args.checkpoint_stride > 1 and hasattr(model, 'checkpoint_stride'):
            model.checkpoint_stride = int(args.checkpoint_stride)
            if IS_MAIN:
                print(f"Selective activation checkpointing: stride={args.checkpoint_stride} "
                      f"(checkpoint all EXCEPT store every {args.checkpoint_stride}th layer)")

    # Argonne-3.5 FP8 (torchao float8): convert Linear matmuls to FP8 BEFORE DDP/compile.
    if args.fp8 == 1:
        if args.torch_compile != 1 and IS_MAIN:
            print("WARNING: --fp8 without --torch_compile — FP8 scaling won't fuse; expect NO speedup.")
        embed_w = model.get_input_embeddings().weight
        n_conv, n_skip, lm_status = apply_fp8_training(model, include_lm_head=(args.fp8_lm_head == 1))
        assert model.get_output_embeddings().weight is embed_w, (
            "FP8 conversion broke the embedding<->lm_head tie — do not train (would be a different model)")
        if args.fp8_lm_head == 1 and lm_status != "converted" and IS_MAIN:
            print(f"WARNING: --fp8_lm_head 1 requested but lm_head {lm_status} — expect ~1.18x not 1.25x.")
        if IS_MAIN:
            print(f"FP8 training ON (torchao tensorwise): converted {n_conv} Linear, skipped {n_skip}; "
                  f"lm_head={lm_status}; embedding tie preserved.")

    # Wrap with DDP
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], gradient_as_bucket_view=True)
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
    train_loader = DataLoader(args.data_path, args.batch_size, args.block_size, RANK, WORLD_SIZE)
    val_loader = None
    if IS_MAIN and args.val_data_path:
        val_loader = DataLoader(args.val_data_path, args.batch_size, args.block_size, rank=0, world_size=1)

    # Estimate steps for scheduler
    num_tokens = len(train_loader.tokens)
    estimated_steps = int((num_tokens * args.max_epochs) / ACTUAL_TOTAL_BATCH)
    dataset_base_global_step = 0
    dataset_base_tokens_processed = 0
    if IS_MAIN:
        print(f"Training for {args.max_epochs} epoch(s) ~= {estimated_steps} steps ({num_tokens * args.max_epochs:,} tokens)")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )

    # Scheduler with warmup (cosine or WSD)
    min_lr = args.lr * args.min_lr_ratio
    min_lr_scale = min_lr / args.lr

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)

        if args.schedule == "cosine":
            progress = (step - args.warmup_steps) / max(1, estimated_steps - args.warmup_steps)
            return max(min_lr_scale, 0.5 * (1.0 + np.cos(np.pi * progress)))

        if args.cooldown <= 0:
            return 1.0

        cooldown_start = max(args.warmup_steps, estimated_steps - args.cooldown)
        if step < cooldown_start:
            return 1.0

        cooldown_progress = min(1.0, (step - cooldown_start) / max(1, args.cooldown))
        return 1.0 - cooldown_progress * (1.0 - min_lr_scale)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    # A previous slice killed mid-save leaves a multi-GB .pt.tmp that nothing else collects.
    if IS_MAIN:
        cleanup_stale_tmp_checkpoints(args.checkpoint_dir)

    resume_from = args.resume_from or get_latest_checkpoint_path(args.checkpoint_dir)
    checkpoint = None
    initial_steps = 0

    if resume_from and os.path.exists(resume_from):
        if IS_MAIN:
            print(f"\n=== Resuming from checkpoint: {resume_from} ===")
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        base_model = get_base_model(model)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        global_step = checkpoint['global_step']
        tokens_processed = checkpoint['tokens_processed']
        data_position = checkpoint.get('data_position', 0)
        checkpoint_dataset_epoch = checkpoint.get('dataset_epoch')
        checkpoint_dataset_num_tokens = checkpoint.get('dataset_num_tokens')
        checkpoint_dataset_path = checkpoint.get('dataset_path')
        checkpoint_dataset_base_step = checkpoint.get('dataset_base_global_step')
        checkpoint_dataset_base_tokens = checkpoint.get('dataset_base_tokens_processed')

        if args.reset_schedule == 1:
            # NEW PHASE on new data (continued-pretrain / midtrain seed): keep MODEL weights only, and
            # start a FRESH optimizer + WSD schedule at --lr; restart the data cursor. Do NOT inherit the
            # seed's optimizer/scheduler -- otherwise the seed's (post-cooldown, ~min) LR carries over via
            # the scheduler and the new phase trains far too gently to LEARN the new data. Fresh AdamW
            # moments are also correct for a new-data phase. (2026-07-14: fixes the scheduler-carryover
            # caveat that made --lr a no-op; validated the seed loss starts low, LR now honors --lr.)
            if IS_MAIN:
                print(f"Reset-schedule (NEW PHASE): model weights seeded; FRESH optimizer + WSD @ lr={args.lr}; data cursor restarted")
                print(f"Seed had {checkpoint['tokens_processed']:,} tokens, step {checkpoint['global_step']}")
            train_loader.set_position(RANK * args.batch_size * args.block_size)
            train_loader.epoch = 0
            dataset_base_global_step = global_step
            dataset_base_tokens_processed = tokens_processed
            is_resumed = False
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            else:
                for _ in range(checkpoint['global_step']):
                    scheduler.step()
            train_loader.set_position(data_position + RANK * args.batch_size * args.block_size)
            metadata_matches = (
                checkpoint_dataset_base_step is not None
                and checkpoint_dataset_num_tokens == num_tokens
                and (checkpoint_dataset_path is None or checkpoint_dataset_path == args.data_path)
            )
            if metadata_matches:
                dataset_base_global_step = int(checkpoint_dataset_base_step)
                if checkpoint_dataset_base_tokens is not None:
                    dataset_base_tokens_processed = int(checkpoint_dataset_base_tokens)
                else:
                    dataset_base_tokens_processed = max(0, tokens_processed - data_position)
                train_loader.epoch = int(checkpoint_dataset_epoch) if checkpoint_dataset_epoch is not None else 0
            else:
                cursor_steps = int(max(0, data_position) // ACTUAL_TOTAL_BATCH)
                dataset_base_global_step = max(0, global_step - cursor_steps)
                dataset_base_tokens_processed = max(0, tokens_processed - data_position)
                train_loader.epoch = 0
                if IS_MAIN:
                    print("Legacy or dataset-mismatched checkpoint metadata; inferring dataset-local progress from the saved data cursor.")
            # Derive dataset-local progress from token deltas so resumes stay
            # correct even when world size or gradient accumulation changes.
            dataset_progress_tokens = max(0, tokens_processed - dataset_base_tokens_processed)
            dataset_progress_steps = int(dataset_progress_tokens // ACTUAL_TOTAL_BATCH)
            if IS_MAIN:
                print(
                    f"Resumed from step {global_step}, tokens: {tokens_processed:,}, "
                    f"dataset epoch: {train_loader.epoch}, dataset progress: {dataset_progress_steps}/{estimated_steps} step(s), "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            is_resumed = True
            initial_steps = min(estimated_steps, dataset_progress_steps)
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
        print(f"Dataset-local progress at launch: {initial_steps}/{estimated_steps} step(s), dataset epoch {train_loader.epoch}")
        print(f"LR: {args.lr}, Warmup: {args.warmup_steps}, Min LR Ratio: {args.min_lr_ratio}, Precision: {args.precision}, TorchCompile: {args.torch_compile}")
        print(f"Checkpoint interval: {args.checkpoint_interval} seconds")
        print(f"Validation data: {args.val_data_path if args.val_data_path else 'disabled (no held-out file provided)'}")
        if args.wall_time > 0:
            print(f"Wall time: {args.wall_time}s, will save checkpoint at {WALL_TIME_SAVE}s")
        if args.reset_schedule == 1:
            print("Mode: continued pretraining (restart data cursor, preserve optimizer/scheduler)")
        print("-" * 60)

    if not is_resumed:
        if checkpoint is None:
            global_step = 0
            tokens_processed = 0
    last_checkpoint_time = time.time()
    training_start_time = time.time()
    train_losses = []
    export_refused = False   # set on rank 0 if the final checkpoint fails its integrity gates
    completed_max_epochs = train_loader.epoch >= args.max_epochs

    pbar = None
    if IS_MAIN:
        pbar = tqdm(total=estimated_steps, initial=initial_steps, desc="Training", unit="step")

    model.train()

    if completed_max_epochs and IS_MAIN:
        print(f"\nCheckpoint is already at {train_loader.epoch} epoch(s); finalizing without more training.")

    while not completed_max_epochs:
        start_time = time.time()
        optimizer.zero_grad()
        # accumulate on the GPU: a float accumulator would force a device sync per micro-step,
        # stalling the pipeline GRAD_ACCUM_STEPS times per optimizer step for a number only read
        # once. Mean over micro-losses == mean over step-losses (fixed micro count), so the
        # logged trajectory is unchanged.
        step_loss_total = torch.zeros((), device=DEVICE, dtype=torch.float32)

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
            step_loss_total += micro_loss.detach().float()   # stays on GPU, no sync

        # the single sync per optimizer step
        step_loss = (step_loss_total / GRAD_ACCUM_STEPS).item()
        train_losses.append(step_loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        if pbar:
            pbar.update(1)

        current_lr = optimizer.param_groups[0]['lr']

        if IS_MAIN and global_step % 10 == 0:
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
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    tokens_processed,
                    step_loss,
                    data_position,
                    args.checkpoint_dir,
                    train_loader.epoch,
                    dataset_base_global_step,
                    dataset_base_tokens_processed,
                    num_tokens,
                    args.data_path,
                )
                if checkpoint_path is None:
                    # save_checkpoint already explained why. Do NOT advance the progress marker:
                    # it would tell the resubmit chain we got further than the last good checkpoint.
                    print("Checkpoint REFUSED at this step; continuing to train on the previous one.")
                else:
                    print(f"Checkpoint saved: {checkpoint_path}")
                    if args.started_marker:
                        write_progress_marker(args.started_marker, global_step, tokens_processed, args.data_path)
                        print(f"Continued-pretrain progress marker written to: {args.started_marker}")

                # Pure telemetry, and it runs on rank 0 while every other rank is already
                # waiting on the barrier below -- so an exception here would not just lose a
                # sample, it would strand the whole slice until SLURM killed it at the wall
                # clock, AFTER a good checkpoint had been written. Same treatment the post-save
                # validation eval already gets. This path had never executed before 2026-08-14.
                print("\nGenerating sample text...")
                try:
                    generated = generate_text(model, tokenizer, DEVICE, prompt="Long long time ago")
                    print(f"Generated: {generated}")
                except Exception as exc:  # noqa: BLE001 - never fail a slice on post-save sampling
                    print(f"WARNING: sample generation failed and was skipped "
                          f"({type(exc).__name__}: {exc})")
                    torch.cuda.empty_cache()
                print("=" * 60 + "\n")

            if WORLD_SIZE > 1:
                dist.barrier()
            last_checkpoint_time = time.time()

        # Synchronized wall-time / deadline check -- save ONE checkpoint, then exit cleanly (the
        # weekend.sh chain resubmits on that clean exit = save-THEN-submit). Primary trigger =
        # --save_deadline_epoch (absolute wall clock, startup-immune); WALL_TIME_SAVE (elapsed) = backup.
        if WALL_TIME_SAVE > 0 or args.save_deadline_epoch > 0:
            should_wall_stop = torch.tensor([0], device=DEVICE)
            if IS_MAIN:
                elapsed = time.time() - training_start_time
                if WALL_TIME_SAVE > 0 and elapsed >= WALL_TIME_SAVE:
                    should_wall_stop[0] = 1
                if args.save_deadline_epoch > 0 and time.time() >= args.save_deadline_epoch:
                    should_wall_stop[0] = 1
            if WORLD_SIZE > 1:
                dist.broadcast(should_wall_stop, src=0)

            if should_wall_stop[0] == 1:
                if IS_MAIN:
                    print(f"\nApproaching wall limit (deadline_epoch={args.save_deadline_epoch}, wall_time={args.wall_time}s). Saving checkpoint and exiting...")
                    data_position = train_loader.get_position()
                    checkpoint_path = save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        tokens_processed,
                        step_loss,
                        data_position,
                        args.checkpoint_dir,
                        train_loader.epoch,
                        dataset_base_global_step,
                        dataset_base_tokens_processed,
                        num_tokens,
                        args.data_path,
                    )
                    if checkpoint_path is None:
                        print("Wall-time checkpoint REFUSED; exiting on the previous good checkpoint.")
                    else:
                        print(f"Wall time checkpoint saved: {checkpoint_path}")
                        if args.started_marker:
                            write_progress_marker(args.started_marker, global_step, tokens_processed, args.data_path)
                            print(f"Continued-pretrain progress marker written to: {args.started_marker}")
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

            # This block runs AFTER the wall-clock checkpoint is already on disk, so it must never
            # be able to fail the slice: a non-zero exit makes the launcher refuse to auto-resubmit
            # and the whole chain stalls. That is exactly what happened to the three 08-03 phase-C
            # slices (52949059/52949900/52950396) -- each trained ~40 steps, saved cleanly, then
            # OOMed here and broke the chain. Free the training-time allocator pools first (the
            # eval forward is eager and uncompiled, so it needs headroom torch.compile did not),
            # and treat any eval failure as "no val number this slice", not as a run failure.
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    original_pos = val_loader.current_position
                    val_loader.current_position = 0

                    for _ in range(min(val_batches, 100)):
                        x, y = val_loader.next_batch()
                        x = x.to(DEVICE, non_blocking=True)
                        y = y.to(DEVICE, non_blocking=True)

                        with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                            # UNWRAPPED module, never the DDP wrapper. This block is rank-0-only, so a
                            # forward through DDP makes rank 0 enter collectives that ranks 1..N-1 never
                            # join -> NCCL desync -> SIGABRT. That killed slice 52938317 AFTER it had
                            # trained 600 steps and saved, so the run limped instead of chaining. Phase B
                            # never hit it because it ran with no --val_data_path.
                            outputs = (model.module if hasattr(model, "module") else model)(x, labels=y)
                        val_losses.append(outputs.loss.item())

                    val_loader.current_position = original_pos
            except Exception as exc:  # noqa: BLE001 - never fail a slice on post-save eval
                val_losses = []
                print(f"WARNING: validation eval failed and was skipped ({type(exc).__name__}: {exc})")
                print("         Checkpoint is already saved; continuing so the slice chain resumes.")
                torch.cuda.empty_cache()
        else:
            print("\nValidation skipped: no held-out validation file was provided.")

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else float("nan")
        val_loss_str = f"{val_loss:.4f}" if val_losses else "n/a"
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss_str}")

        if completed_max_epochs:
            print("\nSaving final checkpoint...")
            data_position = train_loader.get_position()
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                global_step,
                tokens_processed,
                train_loss,
                data_position,
                args.checkpoint_dir,
                train_loader.epoch,
                dataset_base_global_step,
                dataset_base_tokens_processed,
                num_tokens,
                args.data_path,
            )
            if checkpoint_path is None:
                # The gate refused these weights. Exporting them to final_model_dir would publish
                # exactly the corrupt model the gate exists to stop, so skip the export. Do NOT
                # raise here: this block is rank-0 only, and bailing out would strand the other
                # ranks on the barrier below until SLURM killed the job at the wall clock. Flag it
                # and exit nonzero after every rank has cleaned up.
                export_refused = True
                print("Final checkpoint REFUSED — skipping HF export. The last good checkpoint in "
                      f"{args.checkpoint_dir} is the deliverable; investigate before exporting.")
            else:
                print(f"Final checkpoint saved: {checkpoint_path}")

                final_model_dir = args.final_model_dir or os.path.join(args.checkpoint_dir, "final_model_complete")
                os.makedirs(final_model_dir, exist_ok=True)
                save_model = get_base_model(model)
                if args.fp8 == 1:
                    # Export a clean nn.Linear model (no torchao at inference). Dynamic tensorwise FP8 keeps
                    # weights in high precision -> state_dict loads 1:1; refuse to export if any weight missing.
                    clean = ArgonneModel(config)
                    missing, unexpected = clean.load_state_dict(save_model.state_dict(), strict=False)
                    assert not missing, (
                        f"FP8 export: clean model missing weights {missing[:8]} — refusing to export garbage")
                    if hasattr(clean, "tie_weights"):
                        clean.tie_weights()
                    if IS_MAIN:
                        print(f"FP8 export: rebuilt clean nn.Linear model ({len(missing)} missing / {len(unexpected)} extra keys)")
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
                if args.started_marker:
                    write_progress_marker(args.started_marker, global_step, tokens_processed, args.data_path)
                    print(f"Continued-pretrain progress marker written to: {args.started_marker}")
        else:
            print("\nFinal checkpoint/model export skipped because training stopped before completing max_epochs.")

        elapsed_time = time.time() - training_start_time
        print("\n" + "=" * 60)
        print(f"SUMMARY: train_loss={train_loss:.4f} val_loss={val_loss_str} tokens_per_sec={tokens_processed/elapsed_time:.2f} steps={global_step}")
        print("=" * 60)

    if WORLD_SIZE > 1:
        dist.barrier()

    cleanup_distributed()

    # Every rank has now cleaned up, so it is safe to fail the job. Nonzero tells the resubmit
    # chain NOT to treat a refused final checkpoint as a successful finish.
    if export_refused:
        sys.exit(1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (.bin)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory for checkpoints")
    # Training hyperparameters
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of LR")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size per GPU")
    parser.add_argument("--total_batch_size", type=int, required=True, help="Total batch size in tokens")
    parser.add_argument("--block_size", type=int, required=True, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--schedule", type=str, default="wsd", choices=["cosine", "wsd"], help="LR schedule")
    parser.add_argument("--cooldown", type=int, default=0, help="Cooldown steps at end of WSD schedule")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")
    # Argonne-3.5 FP8 (torchao float8, tensorwise) — same as pretrain.py. Requires --torch_compile 1.
    parser.add_argument("--fp8", type=int, default=0, choices=[0, 1], help="Enable FP8 training via torchao float8")
    parser.add_argument("--fp8_lm_head", type=int, default=1, choices=[0, 1], help="Also FP8 the (tied) lm_head")
    parser.add_argument("--loss_chunk_size", type=int, default=0, help="If >0, chunked cross-entropy over this many (batch*seq) rows/chunk -- frees the full-logit fp32 transient so batch can grow at long context. 0 = off.")
    parser.add_argument("--flash_attention", type=int, default=1, choices=[0, 1], help="Use flash attention")
    parser.add_argument("--checkpoint_interval", type=int, default=1800, help="Checkpoint interval in seconds")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum epochs to train")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Use gradient checkpointing")
    parser.add_argument("--checkpoint_stride", type=int, default=1, help="Selective activation checkpointing (ported from argonne4.0). 1=checkpoint ALL layers (default, prior behavior). >=2=checkpoint every layer EXCEPT store (un-checkpoint) every Sth layer (store ceil(n_layers/S), recompute the rest) -> smaller S stores MORE = more HBM + less recompute = faster (too-small S OOMs). Numerically identical; requires --gradient_checkpointing 1.")
    parser.add_argument("--torch_compile", type=int, default=0, choices=[0, 1], help="Use torch.compile for speedup")
    parser.add_argument("--torch_compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint file")
    parser.add_argument("--wall_time", type=int, default=0, help="Wall time in seconds. If > 0, save checkpoint 3 min before this limit. 0 = disabled.")
    parser.add_argument("--save_deadline_epoch", type=int, default=0, help="Absolute wall-clock deadline (unix epoch seconds): when time.time() reaches it, save ONE checkpoint and exit cleanly. SLURM-clock-relative (run_full_training.sh sets it = job_start + slice_seconds - lead), so it fires a FIXED time before the SLURM kill regardless of compile/startup drift. 0 = disabled.")
    parser.add_argument("--reset_schedule", type=int, default=0, choices=[0, 1], help="Restart the data position from the beginning of the current dataset when resuming, while preserving optimizer, scheduler, and cumulative step/token counters.")
    parser.add_argument("--val_data_path", type=str, default=None, help="Optional path to held-out validation data (.bin)")
    parser.add_argument("--final_model_dir", type=str, default=None, help="Optional directory for the final Hugging Face model export.")
    parser.add_argument("--completion_marker", type=str, default=None, help="Optional marker file written only after max_epochs is completed and the final model export succeeds.")
    parser.add_argument("--started_marker", type=str, default=None, help="Optional marker file written after the first continued-pretrain checkpoint is saved.")
    args = parser.parse_args()

    RANK, LOCAL_RANK, WORLD_SIZE = setup_distributed()
    IS_MAIN = RANK == 0
    DEVICE = f"cuda:{LOCAL_RANK}"

    TOKENS_PER_MICRO = args.batch_size * WORLD_SIZE * args.block_size
    GRAD_ACCUM_STEPS = args.total_batch_size // TOKENS_PER_MICRO
    assert GRAD_ACCUM_STEPS >= 1, (
        f"total_batch_size ({args.total_batch_size}) too small for "
        f"{WORLD_SIZE} GPU(s) x batch_size {args.batch_size} x block_size {args.block_size}"
    )
    ACTUAL_TOTAL_BATCH = GRAD_ACCUM_STEPS * TOKENS_PER_MICRO

    # argonne4.0: ~1.04B checkpoint (~11 GB) flushes fast. This training-ELAPSED margin is a BACKUP;
    # the PRIMARY save trigger is --save_deadline_epoch (absolute wall clock, startup-immune).
    WALL_TIME_SAVE = args.wall_time - 180 if args.wall_time > 0 else 0

    main()
