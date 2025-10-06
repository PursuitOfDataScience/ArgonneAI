import argparse
import math
import os
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from data_processing import (
    collate_batch,
    load_nonstream_data,
    load_tokenizer,
    streaming_token_generator,
)
from model import ArgonneConfig, ArgonneModel, Block
from training_utils import (
    log_dataset_plan,
    resolve_data_files,
    validate_tokenizer_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argonne v2 FSDP pretraining entrypoint")
    parser.add_argument("--data-glob", type=str, required=True, help="Glob pattern pointing to pre-tokenization Arrow files")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Directory containing the pretrained tokenizer to load",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store checkpoints and logs")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=5120)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=100000.0)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming and load dataset into memory")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom tokenizer code")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--min-sample-length", type=int, default=256)
    return parser.parse_args()


def setup_distributed(rank: int, world_size: int) -> None:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_model(args: argparse.Namespace, tokenizer) -> ArgonneModel:
    vocab_size = len(tokenizer)
    config = ArgonneConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.sequence_length,
        hidden_size=args.model_dim,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        hidden_dropout=args.dropout,
        attention_dropout=args.dropout,
        rope_theta=args.rope_theta,
        use_gradient_checkpointing=args.gradient_checkpointing,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=tokenizer.eos_token_id,
    )
    model = ArgonneModel(config)
    if args.gradient_checkpointing:
        model.set_gradient_checkpointing(True)
    return model


def create_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


def create_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def shard_streaming_generator(
    data_files: Iterable[str],
    tokenizer,
    rank: int,
    world_size: int,
    min_length: int,
    block_size: int,
) -> Iterable[List[int]]:
    for sample_index, tokens in enumerate(
        streaming_token_generator(
            data_files, tokenizer, block_size, min_length=min_length
        )
    ):
        if sample_index % world_size != rank:
            continue
        if len(tokens) < block_size + 1:
            continue
        yield tokens


def data_iterator(
    data_files: List[str],
    tokenizer,
    rank: int,
    world_size: int,
    block_size: int,
    global_batch_size: int,
    streaming: bool,
    min_length: int,
):
    if streaming:
        generator = shard_streaming_generator(
            data_files, tokenizer, rank, world_size, min_length, block_size
        )
        batch: List[List[int]] = []
        for tokens in generator:
            batch.append(tokens)
            if len(batch) == global_batch_size:
                yield collate_batch(batch, block_size)
                batch = []
        if batch:
            yield collate_batch(batch, block_size)
    else:
        tokenized_data = load_nonstream_data(
            data_files,
            tokenizer,
            block_size,
            num_proc=max(1, world_size),
            min_length=min_length,
        )
        shard = tokenized_data[rank :: world_size]
        batch: List[List[int]] = []
        for tokens in shard:
            batch.append(tokens)
            if len(batch) == global_batch_size:
                yield collate_batch(batch, block_size)
                batch = []
        if batch:
            yield collate_batch(batch, block_size)


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    tokens_seen: int,
    output_dir: str,
    rank: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"step_{step:06d}.pt")
    full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
        state_dict = model.state_dict()
    if rank == 0:
        payload = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "tokens_seen": tokens_seen,
        }
        torch.save(payload, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    rank: int,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
        model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if rank == 0:
        print(f"Resumed from checkpoint {path} at step {checkpoint.get('step', 0)}")
    return checkpoint.get("step", 0), checkpoint.get("tokens_seen", 0)


def run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_distributed(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    fallback_patterns = [
        os.path.join("..", "data", "*.arrow"),
        os.path.join("data", "*.arrow"),
    ]
    data_files, used_patterns = resolve_data_files(
        args.data_glob, fallback_patterns=fallback_patterns
    )

    if rank == 0:
        print(f"Discovered {len(data_files)} data files")
        print("Data patterns contributing shards:")
        for pattern in used_patterns:
            print(f"  - {pattern}")
        log_dataset_plan(data_files)

    validate_tokenizer_path(args.tokenizer_path)
    tokenizer = load_tokenizer(args.tokenizer_path, trust_remote_code=args.trust_remote_code)
    model = build_model(args, tokenizer)
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={Block})
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as compile_error:  # pragma: no cover - runtime guard
            if rank == 0:
                print(f"torch.compile failed: {compile_error}. Continuing without compilation.")

    start_step = 0
    tokens_seen = 0
    if args.resume:
        start_step, tokens_seen = load_checkpoint(model, optimizer, scheduler, args.resume, rank)

    world_batch = args.global_batch_size
    assert world_batch % world_size == 0, "Global batch size must be divisible by world size"
    per_rank_batch = world_batch // world_size
    micro_batch = args.micro_batch_size
    if per_rank_batch % micro_batch != 0:
        raise ValueError("Per-rank batch size must be divisible by micro-batch size")
    grad_accum_steps = per_rank_batch // micro_batch

    generator = data_iterator(
        data_files,
        tokenizer,
        rank,
        world_size,
        args.sequence_length,
        per_rank_batch,
        streaming=not args.no_streaming,
        min_length=args.min_sample_length,
    )

    model.train()
    if rank == 0:
        print(
            f"Starting training for {args.max_steps} steps | seq_len={args.sequence_length} | "
            f"micro_batch={micro_batch} | grad_accum={grad_accum_steps}"
        )

    step = start_step
    while step < args.max_steps:
        batch = next(generator, None)
        if batch is None:
            generator = data_iterator(
                data_files,
                tokenizer,
                rank,
                world_size,
                args.sequence_length,
                per_rank_batch,
                streaming=not args.no_streaming,
                min_length=args.min_sample_length,
            )
            batch = next(generator, None)

        if batch is None:
            continue

        inputs, labels = batch
        if inputs is None or labels is None:
            continue

        total_loss = 0.0
        tokens_seen += int(inputs.numel() * world_size)

        for micro_idx in range(grad_accum_steps):
            start = micro_idx * micro_batch
            end = start + micro_batch
            micro_inputs = inputs[start:end].to(device, non_blocking=True)
            micro_labels = labels[start:end].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(micro_inputs, labels=micro_labels)
                loss = outputs.loss / grad_accum_steps

            total_loss += loss.detach().float()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if rank == 0 and step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"step={step:06d} | loss={total_loss.item():.4f} | lr={lr:.2e} | tokens={tokens_seen:,}"
            )

        if args.save_interval and step % args.save_interval == 0 and step > start_step:
            save_checkpoint(model, optimizer, scheduler, step, tokens_seen, args.output_dir, rank)

        step += 1

    save_checkpoint(model, optimizer, scheduler, step - 1, tokens_seen, args.output_dir, rank)
    cleanup_distributed()


def main() -> None:
    args = parse_args()
    world_size = dist.get_world_size() if dist.is_initialized() else torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs detected. A DGX node with 8 GPUs is required.")

    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
