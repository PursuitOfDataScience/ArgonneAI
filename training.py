import argparse
import os
from collections import deque
from typing import Deque, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.state_dict_type import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW

from data_processing import collate_batch, load_tokenizer, streaming_token_generator
from model import ArgonneConfig, ArgonneModel, Block
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    log_dataset_plan,
    resolve_data_files,
    validate_tokenizer_path,
)


__all__ = [
    "DEFAULT_MAX_TRAINING_STEPS",
    "train_fsdp",
]


def cleanup_old_checkpoints(directory: str, keep: int = 3, rank: int = 0) -> None:
    """Keep only the most recent checkpoint files in a directory."""

    if rank != 0 or keep <= 0 or not os.path.isdir(directory):
        return

    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(directory):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue

        try:
            step_token = name.split("_")[1]
            step_str = os.path.splitext(step_token)[0]
            step = int(step_str)
        except Exception:
            continue

        candidates.append((step, path))

    if len(candidates) <= keep:
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, path in candidates[keep:]:
        try:
            os.remove(path)
        except OSError:
            continue


def init_distributed() -> Tuple[int, int, int]:
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def _build_token_iterator(
    data_files: List[str],
    tokenizer,
    block_size: int,
    *,
    rank: int,
    world_size: int,
    resume_cursor: int = 0,
) -> Iterator[Tuple[List[int], int]]:
    base_iter = streaming_token_generator(data_files, tokenizer, block_size, min_length=block_size)
    # Shard tokens across data parallel workers using modulo stride
    for global_idx, tokens in enumerate(base_iter):
        if global_idx < resume_cursor:
            continue
        if global_idx % world_size != rank:
            continue
        yield tokens, global_idx


def _prepare_model(cfg: ArgonneConfig, dtype: torch.dtype) -> FSDP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base_model = ArgonneModel(cfg)
    base_model = base_model.to(dtype=dtype, device=torch.cuda.current_device())

    auto_wrap_policy = transformer_auto_wrap_policy({Block})
    mp_cfg = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    return FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False),
        mixed_precision=mp_cfg,
        device_id=torch.cuda.current_device(),
    )


def _load_checkpoint(
    model: FSDP,
    optimizer: AdamW,
    scheduler: CosineWarmupScheduler,
    checkpoint_path: str,
) -> Tuple[int, int, int]:
    if not os.path.exists(checkpoint_path):
        return 0, 0, 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    return (
        int(checkpoint.get("global_step", 0)),
        int(checkpoint.get("tokens_processed", 0)),
        int(checkpoint.get("data_cursor", 0)),
    )


def _save_checkpoint(
    model: FSDP,
    optimizer: AdamW,
    scheduler: CosineWarmupScheduler,
    *,
    global_step: int,
    tokens_processed: int,
    data_cursor: int,
    checkpoint_dir: str,
    rank: int,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step:07d}.pt")

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
            "tokens_processed": tokens_processed,
            "data_cursor": data_cursor,
        }

    if rank == 0:
        torch.save(state, checkpoint_path)
        cleanup_old_checkpoints(checkpoint_dir, keep=5, rank=rank)
    dist.barrier()


def train_fsdp(
    *,
    data_glob: str,
    tokenizer_path: str,
    total_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 16384,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    checkpoint_dir: str = "fsdp_checkpoints",
    resume_from: Optional[str] = None,
    save_interval: int = 500,
    trust_remote_code: bool = False,
    require_resume: bool = False,
    entrypoint: str = "training.py",
) -> None:
    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    if require_resume and resume_from is None:
        raise ValueError(
            f"{entrypoint} requires --resume-from for continuation; start fresh training with training.py"
        )

    if resume_from is not None and not os.path.exists(resume_from):
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

    if is_main:
        print(f"Using FSDP with world size {world_size} (local_rank={local_rank})")
        print("Defaulting to bf16 where supported.")

    validate_tokenizer_path(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)
    vocab_size = len(tokenizer)

    data_files, used_patterns = resolve_data_files(data_glob)
    if is_main:
        print(f"Found {len(data_files)} shards via {used_patterns}")
        log_dataset_plan(data_files)

    cfg = ArgonneConfig(
        vocab_size=vocab_size,
        hidden_size=8192,
        max_position_embeddings=block_size,
        num_hidden_layers=24,
        num_attention_heads=64,
        num_key_value_heads=16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_flash_attention=True,
        use_gradient_checkpointing=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=tokenizer.eos_token_id,
    )

    amp_dtype = default_dtype()
    model = _prepare_model(cfg, amp_dtype)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )

    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        min_lr=min_lr,
    )
    current_lr = lr_scheduler.step(0)

    start_step = 0
    tokens_processed = 0
    data_cursor = 0
    if resume_from:
        start_step, tokens_processed, data_cursor = _load_checkpoint(
            model, optimizer, lr_scheduler, resume_from
        )
        current_lr = lr_scheduler.preview_lr(start_step)
        lr_scheduler.step(start_step)
        if is_main:
            print(
                f"Resumed from {resume_from} @ step {start_step} tokens {tokens_processed} cursor {data_cursor}"
            )

    # Initialise wandb on rank0
    if is_main:
        wandb.init(
            project="argonne-fsdp",
            config={"block_size": block_size, "amp_dtype": str(amp_dtype)},
            resume="allow",
        )

    token_iter = _build_token_iterator(
        data_files,
        tokenizer,
        block_size,
        rank=rank,
        world_size=world_size,
        resume_cursor=data_cursor,
    )

    loss_window: Deque[torch.Tensor] = deque(maxlen=50)

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    global_step = start_step
    micro_batches = start_step * grad_accum_steps
    samples_seen = data_cursor

    while global_step < total_steps:
        batch_tokens: List[List[int]] = []
        try:
            while len(batch_tokens) < micro_batch_size:
                tokens, cursor_index = next(token_iter)
                batch_tokens.append(tokens)
                samples_seen = cursor_index + 1
        except StopIteration:
            # restart dataset for continued training
            token_iter = _build_token_iterator(
                data_files,
                tokenizer,
                block_size,
                rank=rank,
                world_size=world_size,
                resume_cursor=0,
            )
            continue

        x_cpu, y_cpu = collate_batch(batch_tokens, block_size)
        if x_cpu is None or y_cpu is None:
            continue

        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            outputs = model(input_ids=x)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            loss = loss / grad_accum_steps

        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_window.append(loss.detach())
        tokens_processed += x.numel()

        micro_batches += 1
        micro_step = micro_batches % grad_accum_steps
        if micro_step == grad_accum_steps - 1:
            if grad_scaler.is_enabled():
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            current_lr = lr_scheduler.step(global_step)

            if is_main and len(loss_window) > 0:
                loss_tensor = torch.stack(list(loss_window)).mean().detach()
                wandb.log(
                    {
                        "train/loss": loss_tensor,
                        "train/lr": current_lr,
                        "train/tokens": tokens_processed,
                    },
                    step=global_step,
                )

            if global_step % save_interval == 0:
                _save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    global_step=global_step,
                    tokens_processed=tokens_processed,
                    data_cursor=samples_seen,
                    checkpoint_dir=checkpoint_dir,
                    rank=rank,
                )

    if is_main:
        wandb.finish()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh Argonne FSDP pretraining (bf16 by default)",
    )
    parser.add_argument("--data-glob", type=str, required=True, help="Glob pattern for dataset shards")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_MAX_TRAINING_STEPS)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=str, default="fsdp_checkpoints")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fsdp(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        total_steps=args.total_steps,
        block_size=args.block_size,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=None,
        save_interval=args.save_interval,
        trust_remote_code=args.trust_remote_code,
        require_resume=False,
        entrypoint="training.py",
    )
