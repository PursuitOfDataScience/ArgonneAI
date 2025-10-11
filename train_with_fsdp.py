import argparse
import contextlib
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from data_processing import collate_batch, load_nonstream_data, load_tokenizer
from model import ArgonneConfig, ArgonneModel, Block
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    log_dataset_plan,
    resolve_data_files,
    safe_torch_load,
    safe_torch_save,
    validate_tokenizer_path,
)
from resume_pretrain import (
    CHECKPOINT_PATTERN,
    DataPosition,
    cleanup_old_checkpoints,
    streaming_token_generator,
    update_training_stats,
)


# ---------------------------------------------------------------------------
# Distributed setup utilities
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, int]:
    """Initialise the default NCCL process group using environment variables."""

    if not dist.is_available():  # pragma: no cover - defensive guard
        raise RuntimeError("torch.distributed is required for FSDP training")

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():  # pragma: no cover - runtime guard
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model/optimizer helpers
# ---------------------------------------------------------------------------


def build_model(
    block_size: int,
    tokenizer,
    use_gradient_checkpointing: bool,
) -> ArgonneModel:
    config = ArgonneConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=block_size,
        hidden_size=4096,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=8,
        rope_theta=500000.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_flash_attention=True,
        tie_word_embeddings=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    model = ArgonneModel(config)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
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


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------


def _default_checkpoint_dir() -> str:
    return os.path.join(os.getcwd(), "pretrained")


def _resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    if path and os.path.isfile(path):
        return path

    search_dirs: List[str] = []
    if path and os.path.isdir(path):
        search_dirs.append(path)
    default_dir = _default_checkpoint_dir()
    if default_dir not in search_dirs:
        search_dirs.append(default_dir)

    candidates: List[Tuple[int, str]] = []
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            match = CHECKPOINT_PATTERN.search(name)
            if not match:
                continue
            step = int(match.group(1))
            candidate_path = os.path.join(directory, name)
            if os.path.isfile(candidate_path):
                candidates.append((step, candidate_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_path = candidates[0]
    return best_path


@dataclass
class CheckpointState:
    step: int
    tokens: int
    data_position: Optional[dict]
    optimizer_state: Optional[dict]
    scheduler_state: Optional[dict]


def save_checkpoint(
    step: int,
    tokens: int,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    data_state: Optional[dict],
    output_dir: str,
    rank: int,
    tag: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{tag}_step_{step:06d}.pth")

    full_state = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state):
        state_dict = model.state_dict()

    if rank == 0:
        payload = {
            "global_step": step,
            "tokens_processed": tokens,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "data_position": data_state,
            "format": "fsdp_full_state",
        }
        safe_torch_save(payload, checkpoint_path)
        print(f"[checkpoint] Saved state to {checkpoint_path}")


def load_initial_state(
    checkpoint_path: Optional[str],
    rank: int,
    block_size: int,
    tokenizer,
    use_gradient_checkpointing: bool,
) -> Tuple[ArgonneModel, CheckpointState]:
    model = build_model(block_size, tokenizer, use_gradient_checkpointing)
    state = CheckpointState(step=0, tokens=0, data_position=None, optimizer_state=None, scheduler_state=None)

    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        if rank == 0:
            print("No existing checkpoint found. Starting from scratch.")
        return model, state

    if rank == 0:
        print(f"Loading checkpoint from {resolved}")
    checkpoint = safe_torch_load(resolved, map_location="cpu")

    # Handle compiled checkpoints saved by resume_pretrain.py
    model_state = checkpoint.get("model_state_dict", {})
    if any(k.startswith("_orig_mod.") for k in model_state.keys()):
        converted = {}
        for key, value in model_state.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                converted[new_key] = value
        model_state = converted

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if rank == 0:
        if missing:
            print(f"[checkpoint] Missing keys when loading: {missing}")
        if unexpected:
            print(f"[checkpoint] Unexpected keys when loading: {unexpected}")

    state.step = int(checkpoint.get("global_step", 0))
    state.tokens = int(checkpoint.get("tokens_processed", 0))
    state.data_position = checkpoint.get("data_position")
    state.optimizer_state = checkpoint.get("optimizer_state_dict")
    state.scheduler_state = checkpoint.get("scheduler_state_dict")

    if rank == 0:
        print(
            f"Resumed weights at step {state.step} with {state.tokens:,} tokens. "
            "Optimizer state will be reinitialized for FSDP."
        )

    return model, state


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _collate_for_rank0(
    all_tokens: List[List[int]],
    block_size: int,
    batch_size: int,
    world_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = collate_batch(all_tokens, block_size)
    if tensors[0] is None:
        raise RuntimeError("collate_batch returned None")
    inputs, labels = tensors
    # Shape: (world_size * batch_size, block_size)
    total = world_size * batch_size
    if inputs.size(0) != total:
        raise RuntimeError(
            f"Expected {total} samples but received {inputs.size(0)} from collate_batch"
        )

    inputs = inputs.view(world_size, batch_size, block_size).to(device, non_blocking=True)
    labels = labels.view(world_size, batch_size, block_size).to(device, non_blocking=True)
    return inputs, labels


class StreamingBatchProvider:
    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        block_size: int,
        batch_size: int,
        world_size: int,
        data_position: DataPosition,
    ) -> None:
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_position = data_position
        self.generator = streaming_token_generator(
            data_files,
            tokenizer,
            block_size,
            data_position.current_file_idx,
            data_position.position_in_file,
            data_position.chunk_offset,
        )

    def _restart(self) -> None:
        self.data_position.next_epoch()
        self.generator = streaming_token_generator(
            self.data_files, self.tokenizer, self.block_size
        )

    def next_batch(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        required = self.batch_size * self.world_size
        buffer: List[List[int]] = []
        while len(buffer) < required:
            sample = next(self.generator, None)
            if sample is None:
                self._restart()
                continue
            tokens, file_idx, position, shard_name, chunk_idx = sample
            if file_idx == -1:
                self._restart()
                continue

            buffer.append(tokens)
            self.data_position.update_streaming_position(
                file_idx,
                position,
                chunk_idx,
                self.data_files[file_idx] if file_idx < len(self.data_files) else None,
            )

        return _collate_for_rank0(buffer, self.block_size, self.batch_size, self.world_size, device)


    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size


class NonStreamingBatchProvider:
    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        block_size: int,
        batch_size: int,
        world_size: int,
        data_position: DataPosition,
        num_proc: int,
    ) -> None:
        self.data_position = data_position
        self.block_size = block_size
        self.batch_size = batch_size
        self.world_size = world_size

        tokenized = load_nonstream_data(
            data_files,
            tokenizer,
            block_size,
            num_proc=num_proc,
        )
        self.dataset = tokenized
        self.total_samples = len(tokenized)
        if self.data_position.shuffled_indices is None:
            self.data_position.generate_shuffled_indices(self.total_samples)

    def next_batch(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        required = self.batch_size * self.world_size
        batch_tokens: List[List[int]] = []

        while len(batch_tokens) < required:
            if self.data_position.shuffled_indices is None or (
                self.data_position.current_position >= len(self.data_position.shuffled_indices)
            ):
                self.data_position.next_epoch(self.total_samples)
                if self.data_position.shuffled_indices is None:
                    self.data_position.generate_shuffled_indices(self.total_samples)

            remaining = len(self.data_position.shuffled_indices) - self.data_position.current_position
            if remaining <= 0:
                continue

            take = min(required - len(batch_tokens), remaining)
            indices = self.data_position.shuffled_indices[
                self.data_position.current_position : self.data_position.current_position + take
            ]
            for idx in indices:
                batch_tokens.append(self.dataset[idx])
            self.data_position.update_nonstreaming_position(
                self.data_position.current_position + take
            )

        return _collate_for_rank0(batch_tokens, self.block_size, self.batch_size, self.world_size, device)


    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    if value < multiple:
        return multiple
    return max(multiple, (value // multiple) * multiple)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSDP continuation training for Argonne")
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    parser.add_argument("--data-glob", type=str, default=default_data_glob)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to an existing checkpoint file or directory (defaults to pretrained/)",
    )
    parser.add_argument("--output-dir", type=str, default=_default_checkpoint_dir())
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_MAX_TRAINING_STEPS)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision (useful for debugging at the cost of much higher memory)",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile where available")
    parser.add_argument("--checkpoint-keep", type=int, default=3, help="Number of recent checkpoints to keep")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default to reduce activation memory)",
    )
    parser.add_argument(
        "--auto-batch-max",
        type=int,
        default=512,
        help="Start automatic OOM backoff from this per-rank batch size (<=0 disables)",
    )
    parser.add_argument(
        "--auto-batch-min",
        type=int,
        default=None,
        help="Minimum per-rank batch size auto backoff can reach (defaults to --batch-size)",
    )
    parser.add_argument(
        "--disable-auto-batch",
        action="store_true",
        help="Disable automatic per-rank batch size reduction on OOM",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if args.bf16 and args.fp16:
        raise ValueError("Cannot enable both bf16 and fp16 modes simultaneously")

    # Resolve datasets
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    fallback_patterns = [
        os.path.join("data", "CC-MAIN-2025-26", "*.parquet"),
        os.path.join("..", "data", "*.arrow"),
        os.path.join("data", "*.arrow"),
    ]
    if args.data_glob != default_data_glob:
        fallback_patterns.insert(0, default_data_glob)

    data_files, used_patterns = resolve_data_files(args.data_glob, fallback_patterns=fallback_patterns)
    if rank == 0:
        print(f"Discovered {len(data_files)} data shards")
        print("Data patterns contributing shards:")
        for pattern in used_patterns:
            print(f"  - {pattern}")
        log_dataset_plan(data_files)

    # Tokenizer
    validate_tokenizer_path(args.tokenizer_path)
    tokenizer = load_tokenizer(args.tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # ``encode`` is used on raw documents before we chunk to ``block_size``.
    # Some shards can easily exceed the tokenizer's baked-in ``model_max_length``
    # (often 8k), which only serves to emit noisy warnings even though we never
    # feed those full sequences into the model. Lift the limit to a very large
    # sentinel value so encoding huge documents stays warning-free while the
    # actual training batches remain bounded by ``block_size``.
    tokenizer.model_max_length = max(args.block_size + 1, 1_000_000_000)
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["model_max_length"] = tokenizer.model_max_length

    # Model + checkpoint
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    base_model, ckpt_state = load_initial_state(
        args.checkpoint_path,
        rank,
        args.block_size,
        tokenizer,
        use_gradient_checkpointing,
    )

    if rank == 0:
        if use_gradient_checkpointing:
            print("Enabled gradient checkpointing to reduce activation memory.")
        else:
            print("Gradient checkpointing disabled; activation memory usage will be higher.")

    if args.compile and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(base_model, mode="default")
        except Exception as exc:  # pragma: no cover - runtime guard
            if rank == 0:
                print(f"torch.compile failed ({exc}); continuing without compilation")

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    if args.no_mixed_precision and (args.bf16 or args.fp16):
        raise ValueError("Cannot combine explicit mixed precision flags with --no-mixed-precision")

    mp_policy = None
    amp_dtype = None
    if not args.no_mixed_precision and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device_index)
        supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()

        if args.bf16:
            if not supports_bf16:
                raise RuntimeError("Requested bf16 mixed precision but the current GPU does not support it")
            amp_dtype = torch.bfloat16
        elif args.fp16:
            amp_dtype = torch.float16
        else:
            if supports_bf16:
                amp_dtype = torch.bfloat16
                if rank == 0:
                    print("Auto-enabled bf16 mixed precision for improved memory usage")
            else:
                amp_dtype = torch.float16
                if rank == 0:
                    print("Auto-enabled fp16 mixed precision for improved memory usage")

        if amp_dtype == torch.bfloat16:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
        elif amp_dtype == torch.float16:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float16,
            )

    fsdp_model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        use_orig_params=True,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )

    optimizer = create_optimizer(fsdp_model, args.learning_rate, args.weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.total_steps,
        min_lr=args.min_learning_rate,
    )

    if ckpt_state.scheduler_state:
        scheduler.load_state_dict(ckpt_state.scheduler_state)
        scheduler.step(ckpt_state.scheduler_state.get("step", ckpt_state.step))
    if ckpt_state.optimizer_state:
        if rank == 0:
            print("Optimizer state from the original checkpoint is incompatible with FSDP and was not restored.")

    grad_accum = max(args.gradient_accumulation_steps, 1)
    if args.batch_size % grad_accum != 0:
        raise ValueError("Per-rank batch size must be divisible by gradient accumulation steps")

    base_batch_size = args.batch_size
    min_auto_batch = args.auto_batch_min if args.auto_batch_min and args.auto_batch_min > 0 else base_batch_size
    min_auto_batch = max(min_auto_batch, base_batch_size)
    min_auto_batch = _round_down_to_multiple(min_auto_batch, grad_accum)
    if min_auto_batch < base_batch_size:
        min_auto_batch = base_batch_size

    if args.disable_auto_batch or args.auto_batch_max <= 0:
        current_batch_size = base_batch_size
    else:
        start_candidate = max(args.auto_batch_max, base_batch_size)
        start_candidate = _round_down_to_multiple(start_candidate, grad_accum)
        if start_candidate < base_batch_size:
            start_candidate = base_batch_size
        current_batch_size = start_candidate

    current_micro_batch = current_batch_size // grad_accum
    if current_micro_batch <= 0:
        raise ValueError("Per-rank batch size must be at least gradient accumulation steps")

    if rank == 0:
        msg = (
            f"FSDP training using world_size={world_size} | per-rank batch target={current_batch_size}"
        )
        if not args.disable_auto_batch and args.auto_batch_max > 0:
            msg += f" (floor={min_auto_batch}, baseline={base_batch_size})"
        print(msg)

    data_position = DataPosition(streaming=not args.no_streaming)
    if ckpt_state.data_position:
        data_position.restore_state(ckpt_state.data_position)

    provider: Optional[object] = None
    if rank == 0:
        if args.no_streaming:
            provider = NonStreamingBatchProvider(
                data_files,
                tokenizer,
                args.block_size,
                current_batch_size,
                world_size,
                data_position,
                args.num_proc,
            )
        else:
            provider = StreamingBatchProvider(
                data_files,
                tokenizer,
                args.block_size,
                current_batch_size,
                world_size,
                data_position,
            )

        # Ensure rank0 provider reflects the dynamically chosen batch size
        provider.set_batch_size(current_batch_size)

    start_step = ckpt_state.step
    total_tokens_processed = ckpt_state.tokens

    fsdp_model.train()

    if args.no_mixed_precision or not torch.cuda.is_available():
        scaler = None
        amp_dtype = None
    else:
        scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    if rank == 0:
        print(
            "Starting/resuming training at step "
            f"{start_step} | target steps {args.total_steps} | micro_batch {current_micro_batch} | grad_accum {grad_accum}"
        )

    step = start_step
    tokens_this_session = 0
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None and torch.cuda.is_available()
        else contextlib.nullcontext()
    )

    sample_prompt = "Long long time ago, "
    sample_interval = 50
    sample_new_tokens = 100

    pending_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    while step < args.total_steps:
        if pending_batch is not None:
            inputs_all, labels_all = pending_batch
            pending_batch = None
        else:
            if rank == 0:
                assert provider is not None
                inputs_all, labels_all = provider.next_batch(device)
            else:
                shape = (world_size, current_batch_size, args.block_size)
                inputs_all = torch.empty(shape, dtype=torch.long, device=device)
                labels_all = torch.empty(shape, dtype=torch.long, device=device)

            dist.broadcast(inputs_all, src=0)
            dist.broadcast(labels_all, src=0)

        inputs_all = inputs_all[:, :current_batch_size, :]
        labels_all = labels_all[:, :current_batch_size, :]
        batch_tokens = inputs_all.numel()

        inputs_local = inputs_all[rank]
        labels_local = labels_all[rank]

        local_loss = torch.zeros(1, device=device)
        scheduler.step(step)
        optimizer.zero_grad(set_to_none=True)

        oom_tensor = torch.zeros(1, device=device)

        try:
            for micro_idx in range(grad_accum):
                start = micro_idx * current_micro_batch
                end = start + current_micro_batch
                micro_inputs = inputs_local[start:end]
                micro_labels = labels_local[start:end]

                with autocast_ctx:
                    outputs = fsdp_model(micro_inputs, labels=micro_labels)
                    loss = outputs.loss / grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                local_loss += loss.detach()

            torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        except RuntimeError as exc:
            if _is_oom_error(exc):
                oom_tensor.fill_(1)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
            else:
                raise

        loss_tensor = local_loss.clone()

        dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
        if oom_tensor.item():
            if args.disable_auto_batch or args.auto_batch_max <= 0:
                torch.cuda.empty_cache()
                if rank == 0:
                    print(
                        "OOM encountered with automatic batch scaling disabled. "
                        "Please lower --batch-size or adjust other memory-intensive settings."
                    )
                raise RuntimeError("OOM encountered with auto-batch disabled")

            new_candidate = _round_down_to_multiple(current_batch_size // 2, grad_accum)
            new_candidate = max(new_candidate, min_auto_batch)
            if new_candidate < min_auto_batch or new_candidate == current_batch_size:
                if rank == 0:
                    print(
                        "Encountered OOM at minimal batch size; unable to continue. "
                        "Consider reducing gradient accumulation or block size."
                    )
                raise RuntimeError("OOM encountered at minimal batch size")

            current_batch_size = new_candidate
            current_micro_batch = current_batch_size // grad_accum
            if rank == 0:
                print(
                    "OOM detected; reducing per-rank batch to "
                    f"{current_batch_size} (micro_batch={current_micro_batch})."
                )
            if provider is not None:
                provider.set_batch_size(current_batch_size)
            inputs_all = inputs_all[:, :current_batch_size, :].contiguous()
            labels_all = labels_all[:, :current_batch_size, :].contiguous()
            pending_batch = (inputs_all, labels_all)
            torch.cuda.empty_cache()
            dist.barrier()
            continue

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_avg = loss_tensor.item() / max(1, world_size)

        tokens_this_session += batch_tokens

        if rank == 0 and step % 10 == 0:
            total_tokens = total_tokens_processed + tokens_this_session
            print(
                f"step={step:06d} | loss={loss_avg:.4f} | tokens={total_tokens:,} | lr={scheduler.last_lr:.3e}"
            )

        if step % sample_interval == 0:
            fsdp_was_training = fsdp_model.training
            fsdp_model.eval()
            generated_tokens: Optional[List[int]] = None
            try:
                prompt_ids = tokenizer.encode(sample_prompt)
                if not prompt_ids:
                    prompt_ids = [getattr(tokenizer, "bos_token_id", 0)]

                prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                if prompt_tensor.size(1) > args.block_size:
                    prompt_tensor = prompt_tensor[:, : args.block_size]
                max_append = max(0, min(sample_new_tokens, args.block_size - prompt_tensor.size(1)))

                with torch.no_grad():
                    generated = prompt_tensor
                    for _ in range(max_append):
                        with (
                            torch.amp.autocast("cuda", dtype=amp_dtype)
                            if amp_dtype is not None and torch.cuda.is_available()
                            else contextlib.nullcontext()
                        ):
                            outputs = fsdp_model(generated)
                            logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=-1)

                    if rank == 0:
                        generated_tokens = generated[0].detach().cpu().tolist()
            except Exception as exc:  # pragma: no cover - generation is best-effort
                if rank == 0:
                    print(f"[generation] Failed to produce sample text: {exc}")
            finally:
                fsdp_model.train(fsdp_was_training)

            if rank == 0 and generated_tokens is not None:
                generated_text = tokenizer.decode(generated_tokens)
                print(f"\n--- Generated text at step {step} ---\n{generated_text}\n")

        if args.save_interval > 0 and step > start_step and step % args.save_interval == 0:
            total_tokens = total_tokens_processed + tokens_this_session
            data_state = data_position.get_state() if rank == 0 else None
            save_checkpoint(
                step,
                total_tokens,
                fsdp_model,
                optimizer,
                scheduler,
                data_state,
                args.output_dir,
                rank,
                "fsdp",
            )
            if rank == 0 and args.checkpoint_keep > 0:
                cleanup_old_checkpoints(args.output_dir, keep=args.checkpoint_keep)

        step += 1

    total_tokens = total_tokens_processed + tokens_this_session
    data_state = data_position.get_state() if rank == 0 else None
    save_checkpoint(
        step - 1,
        total_tokens,
        fsdp_model,
        optimizer,
        scheduler,
        data_state,
        args.output_dir,
        rank,
        "final",
    )

    if rank == 0:
        model_ref = fsdp_model.module if hasattr(fsdp_model, "module") else fsdp_model
        update_training_stats(
            tokens=total_tokens,
            batch_size=current_batch_size * world_size,
            steps=step - 1,
            model=model_ref,
            n_layer=model_ref.config.num_hidden_layers,
            n_head=model_ref.config.num_attention_heads,
            n_embd=model_ref.config.hidden_size,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.total_steps,
            final=True,
        )

    cleanup_distributed()


if __name__ == "__main__":
    train()
