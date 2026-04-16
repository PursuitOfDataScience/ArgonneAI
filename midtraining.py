"""
Full long-context midtraining entrypoint for Argonne.

Behavior:
- First launch seeds from an explicit training checkpoint that includes
  optimizer/scheduler/global counters.
- Later launches auto-resume from the latest checkpoint in the midtraining
  checkpoint directory.
- Preserves cumulative global step/tokens while also tracking
  midtraining-local tokens from zero.
- Supports FSDP sharding so the two H200s can be used as a memory-saving
  training stack instead of pure DDP replication.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    OptimStateKeyType,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from continue_pretrain import (
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    DataLoader,
    cleanup_distributed,
    setup_distributed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ArgonneConfig, ArgonneModel, Block


AUTOCAST_DTYPE = None
USE_AUTOCAST = False


class DocManifestDataLoader:
    def __init__(
        self,
        manifest_path,
        B,
        T,
        rank=0,
        world_size=1,
        cache_size=4,
        shuffle_docs=False,
        shuffle_seed=1337,
    ):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.manifest_path = os.path.abspath(manifest_path)
        self.epoch = 0
        self.shuffle_docs = bool(shuffle_docs)
        self.shuffle_seed = int(shuffle_seed)
        self._cache_size = max(1, cache_size)
        self._shard_cache = OrderedDict()

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        tokenized_dir = manifest["tokenized_dir"]
        files = [item for item in manifest["files"] if int(item["docs_kept"]) > 0]
        if not files:
            raise ValueError(f"No kept documents found in manifest: {self.manifest_path}")

        self.shards = []
        doc_offsets = [0]
        total_docs = 0
        for item in files:
            bin_path = os.path.join(tokenized_dir, item["bin_path"])
            lengths_path = os.path.join(tokenized_dir, item["lengths_path"])
            docs_kept = int(item["docs_kept"])
            self.shards.append(
                {
                    "bin_path": bin_path,
                    "lengths_path": lengths_path,
                    "docs_kept": docs_kept,
                    "source_relpath": item["source_relpath"],
                }
            )
            total_docs += docs_kept
            doc_offsets.append(total_docs)

        self.doc_offsets = np.asarray(doc_offsets, dtype=np.int64)
        self.total_docs = total_docs
        self.total_tokens = int(manifest["qwen_tokens_kept"])
        self.docs_per_global_step = self.B * self.world_size
        self.usable_docs = (self.total_docs // self.docs_per_global_step) * self.docs_per_global_step
        if self.usable_docs <= 0:
            raise ValueError(
                f"Doc-aware dataset is too small for B={self.B}, world_size={self.world_size}: "
                f"total_docs={self.total_docs}"
            )

        self.num_tokens = self.usable_docs * self.T
        self.current_position = self.rank * self.B
        self.doc_order = np.arange(self.usable_docs, dtype=np.int64)
        self._refresh_doc_order()

        if self.rank == 0:
            print(f"DocManifestDataLoader: {self.total_docs:,} docs, {self.total_tokens:,} raw kept tokens")
            print(
                f"DocManifestDataLoader effective epoch: {self.usable_docs:,} docs -> "
                f"{self.num_tokens:,} training tokens"
            )
            print(f"DocManifestDataLoader manifest: {self.manifest_path}")
            if self.shuffle_docs:
                print(f"DocManifestDataLoader doc shuffling: enabled (seed={self.shuffle_seed})")

    def _load_shard(self, shard_idx):
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached

        shard = self.shards[shard_idx]
        lengths = np.load(shard["lengths_path"], mmap_mode="r")
        offsets = np.zeros(len(lengths), dtype=np.uint64)
        if len(lengths) > 1:
            np.cumsum(lengths[:-1], dtype=np.uint64, out=offsets[1:])
        tokens = np.memmap(shard["bin_path"], dtype=np.uint32, mode="r")
        cached = (tokens, lengths, offsets)
        self._shard_cache[shard_idx] = cached
        if len(self._shard_cache) > self._cache_size:
            self._shard_cache.popitem(last=False)
        return cached

    def _locate_doc(self, global_doc_idx):
        shard_idx = int(np.searchsorted(self.doc_offsets, global_doc_idx, side="right") - 1)
        local_doc_idx = int(global_doc_idx - self.doc_offsets[shard_idx])
        return shard_idx, local_doc_idx

    def _span_start(self, global_doc_idx, doc_len):
        max_start = doc_len - (self.T + 1)
        if max_start <= 0:
            return 0
        mixed = (
            (int(global_doc_idx) + 1) * 0x9E3779B185EBCA87
            + (int(self.epoch) + 1) * 0xC2B2AE3D27D4EB4F
        ) & 0xFFFFFFFFFFFFFFFF
        return mixed % (max_start + 1)

    def _refresh_doc_order(self):
        if not self.shuffle_docs:
            return
        rng = np.random.default_rng(self.shuffle_seed + int(self.epoch))
        self.doc_order = rng.permutation(self.usable_docs).astype(np.int64)

    def _doc_window(self, global_doc_idx):
        shard_idx, local_doc_idx = self._locate_doc(global_doc_idx)
        tokens, lengths, offsets = self._load_shard(shard_idx)
        doc_len = int(lengths[local_doc_idx])
        start = self._span_start(global_doc_idx, doc_len)
        doc_offset = int(offsets[local_doc_idx])
        buf = tokens[doc_offset + start:doc_offset + start + self.T + 1]
        if len(buf) != self.T + 1:
            raise RuntimeError(
                f"Short doc window for global_doc_idx={global_doc_idx}: "
                f"doc_len={doc_len}, start={start}, got={len(buf)}"
            )
        return np.asarray(buf, dtype=np.int64)

    def next_batch(self):
        batch_docs = []
        for i in range(self.B):
            doc_idx = self.current_position + i
            if self.shuffle_docs:
                doc_idx = int(self.doc_order[doc_idx])
            batch_docs.append(self._doc_window(doc_idx))

        buf = torch.from_numpy(np.stack(batch_docs, axis=0))
        if torch.cuda.is_available():
            buf = buf.pin_memory()
        x = buf[:, :-1]
        y = buf[:, 1:]

        self.current_position += self.docs_per_global_step
        if self.current_position + self.B > self.usable_docs:
            self.current_position = self.rank * self.B
            self.epoch += 1
            self._refresh_doc_order()
            if self.rank == 0:
                print(f"\n*** Epoch {self.epoch} completed ***\n")
        return x, y

    def get_position(self):
        return int(self.current_position)

    def set_position(self, position):
        self.current_position = int(position)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._refresh_doc_order()

    def start_from_beginning(self):
        self.current_position = self.rank * self.B

    def resume_from_checkpoint_position(self, position):
        self.current_position = int(position) + self.rank * self.B

    def steps_from_position(self, position):
        return int(max(0, position) // self.docs_per_global_step)


def build_train_loader(
    data_path,
    batch_size,
    block_size,
    rank,
    world_size,
    doc_shuffle=0,
    doc_shuffle_seed=1337,
):
    if data_path.endswith(".json"):
        return DocManifestDataLoader(
            data_path,
            batch_size,
            block_size,
            rank,
            world_size,
            shuffle_docs=bool(doc_shuffle),
            shuffle_seed=int(doc_shuffle_seed),
        )
    return DataLoader(data_path, batch_size, block_size, rank, world_size)


def set_loader_epoch(loader, epoch):
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    else:
        loader.epoch = int(epoch)


def unwrap_compiled_model(model):
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def get_fsdp_wrapper(model):
    model = unwrap_compiled_model(model)
    if isinstance(model, FSDP):
        return model
    return None


def get_training_module(model):
    model = unwrap_compiled_model(model)
    if isinstance(model, DDP):
        return model.module
    if isinstance(model, FSDP):
        return model.module
    return model


def generate_text(
    model,
    tokenizer,
    device,
    prompt,
    max_new_tokens,
    sample_do_sample=1,
    sample_temperature=0.8,
    sample_top_p=0.9,
    sample_repetition_penalty=1.3,
    sample_no_repeat_ngram_size=4,
    sample_seed=444,
):
    model.eval()
    with torch.no_grad():
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_length = input_ids.shape[1] + max_new_tokens
        gen_model = get_training_module(model)
        do_sample = bool(sample_do_sample)
        top_p = sample_top_p if sample_top_p is not None and sample_top_p > 0 else None
        with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
            output = gen_model.generate(
                input_ids,
                max_length=max_length,
                do_sample=do_sample,
                temperature=sample_temperature if do_sample else 1.0,
                top_p=top_p if do_sample else None,
                repetition_penalty=sample_repetition_penalty,
                no_repeat_ngram_size=sample_no_repeat_ngram_size,
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return generated_text


def generate_text_from_state_dict(
    model_state_dict,
    config,
    tokenizer,
    device,
    prompt,
    max_new_tokens,
    sample_do_sample=1,
    sample_temperature=0.8,
    sample_top_p=0.9,
    sample_repetition_penalty=1.3,
    sample_no_repeat_ngram_size=4,
    sample_seed=444,
):
    torch.cuda.empty_cache()
    gen_config = ArgonneConfig.from_dict(config.to_dict())
    gen_model = ArgonneModel(gen_config)
    gen_model.load_state_dict(model_state_dict)
    if USE_AUTOCAST and AUTOCAST_DTYPE is not None:
        gen_model = gen_model.to(device=device, dtype=AUTOCAST_DTYPE)
    else:
        gen_model = gen_model.to(device)
    try:
        return generate_text(
            gen_model,
            tokenizer,
            device,
            prompt,
            max_new_tokens,
            sample_do_sample=sample_do_sample,
            sample_temperature=sample_temperature,
            sample_top_p=sample_top_p,
            sample_repetition_penalty=sample_repetition_penalty,
            sample_no_repeat_ngram_size=sample_no_repeat_ngram_size,
            sample_seed=sample_seed,
        )
    finally:
        del gen_model
        torch.cuda.empty_cache()


def get_latest_checkpoint_path(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    latest_step_path = None
    latest_step = None
    if checkpoints:
        steps_and_paths = [
            (int(f.split("_step_")[-1].replace(".pt", "")), f)
            for f in checkpoints
        ]
        latest_step, latest_step_path = max(steps_and_paths, key=lambda item: item[0])

    latest_link_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    if os.path.exists(latest_link_path):
        if latest_step is None:
            return latest_link_path
        link_target = os.path.realpath(latest_link_path)
        match = re.search(r"checkpoint_step_(\d+)\.pt$", link_target)
        if match and int(match.group(1)) >= latest_step:
            return latest_link_path
        return latest_step_path

    return latest_step_path


def is_fsdp_model(model):
    return get_fsdp_wrapper(model) is not None


def optimizer_state_uses_param_names(optim_state_dict):
    state = optim_state_dict.get("state", {})
    if not state:
        return False
    return all(isinstance(k, str) for k in state.keys())


def build_sharding_strategy(name):
    mapping = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }
    return mapping[name]


def build_fsdp_mixed_precision(args):
    if args.precision != "bf16" or args.fsdp_mixed_precision == 0:
        return None
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def wrap_model_for_training(model, args, device, world_size, is_main):
    distributed_strategy = "single"
    delay_compile_until_after_optimizer_restore = (
        args.torch_compile == 1 and world_size > 1 and args.distributed_strategy == "fsdp"
    )

    should_enable_gradient_checkpointing = (
        args.gradient_checkpointing == 1
        and (args.torch_compile == 0 or args.distributed_strategy == "fsdp")
    )

    if should_enable_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if is_main:
                print("Gradient checkpointing enabled")

    if world_size > 1 and args.distributed_strategy == "fsdp":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=build_sharding_strategy(args.fsdp_sharding_strategy),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision=build_fsdp_mixed_precision(args),
            device_id=torch.device(device),
            limit_all_gathers=True,
            use_orig_params=True,
            sync_module_states=False,
        )
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        distributed_strategy = f"fsdp:{args.fsdp_sharding_strategy}"
        if is_main:
            mp_label = "bf16" if args.fsdp_mixed_precision == 1 and args.precision == "bf16" else "off"
            print(f"Using FSDP sharding strategy: {args.fsdp_sharding_strategy}")
            print(f"FSDP mixed precision: {mp_label}")
    elif world_size > 1:
        model = DDP(model, device_ids=[torch.device(device).index])
        distributed_strategy = "ddp"
        if is_main:
            print(f"Using {world_size} GPUs with DistributedDataParallel")

    if args.torch_compile == 1 and not delay_compile_until_after_optimizer_restore:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.torch_compile_mode)

    return model, distributed_strategy, delay_compile_until_after_optimizer_restore


def format_memory_snapshot(memory_rows):
    gib = 1024 ** 3
    parts = []
    for rank_idx, row in enumerate(memory_rows):
        cur_alloc, cur_reserved, peak_alloc, peak_reserved = [value / gib for value in row]
        parts.append(
            f"r{rank_idx} cur={cur_alloc:.1f}/{cur_reserved:.1f}GiB "
            f"peak={peak_alloc:.1f}/{peak_reserved:.1f}GiB"
        )
    return "; ".join(parts)


def collect_memory_snapshot(device, world_size):
    stats = torch.tensor(
        [
            float(torch.cuda.memory_allocated(device)),
            float(torch.cuda.memory_reserved(device)),
            float(torch.cuda.max_memory_allocated(device)),
            float(torch.cuda.max_memory_reserved(device)),
        ],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1 and dist.is_initialized():
        gathered = [torch.zeros_like(stats) for _ in range(world_size)]
        dist.all_gather(gathered, stats)
        stacked = torch.stack(gathered, dim=0)
    else:
        stacked = stats.unsqueeze(0)
    return stacked.cpu().tolist()


def reset_peak_memory_stats(device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


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
    midtraining_base_global_step,
    midtraining_base_tokens_processed,
    distributed_strategy,
    fsdp_sharding_strategy,
    is_main,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")

    fsdp_model = get_fsdp_wrapper(model)
    if fsdp_model is not None:
        model_state_dict = fsdp_model.state_dict()
        optimizer_state_dict = FSDP.optim_state_dict(fsdp_model, optimizer)
    else:
        model_state_dict = get_training_module(model).state_dict()
        optimizer_state_dict = optimizer.state_dict()

    checkpoint = {
        "global_step": global_step,
        "tokens_processed": tokens_processed,
        "midtraining_tokens_processed": max(0, tokens_processed - midtraining_base_tokens_processed),
        "loss": loss,
        "data_position": data_position,
        "dataset_epoch": dataset_epoch,
        "dataset_base_global_step": dataset_base_global_step,
        "dataset_base_tokens_processed": dataset_base_tokens_processed,
        "dataset_num_tokens": dataset_num_tokens,
        "dataset_path": dataset_path,
        "midtraining_base_global_step": midtraining_base_global_step,
        "midtraining_base_tokens_processed": midtraining_base_tokens_processed,
        "distributed_strategy": distributed_strategy,
        "fsdp_sharding_strategy": fsdp_sharding_strategy,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    if is_main:
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
        return checkpoint_path, model_state_dict
    return None, None


def save_final_model_artifacts(model, config, tokenizer, final_model_dir, is_main):
    fsdp_model = get_fsdp_wrapper(model)
    if fsdp_model is not None:
        full_state_dict = fsdp_model.state_dict()
        if not is_main:
            return
        save_model = ArgonneModel(ArgonneConfig.from_dict(config.to_dict()))
        save_model.load_state_dict(full_state_dict)
    else:
        if not is_main:
            return
        save_model = get_training_module(model)

    os.makedirs(final_model_dir, exist_ok=True)

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


def build_parser():
    parser = argparse.ArgumentParser(description="Argonne full long-context midtraining")

    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (.bin flat stream or doc-manifest .json)",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory for midtraining checkpoints")
    parser.add_argument("--init_checkpoint_path", type=str, required=True, help="Seed checkpoint for the first midtraining launch")
    parser.add_argument(
        "--doc_shuffle",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1 and using doc-manifest input, shuffle document order each epoch",
    )
    parser.add_argument(
        "--doc_shuffle_seed",
        type=int,
        default=1337,
        help="Base seed for doc-manifest shuffling",
    )

    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of LR")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size per GPU")
    parser.add_argument("--total_batch_size", type=int, required=True, help="Total batch size in tokens")
    parser.add_argument("--block_size", type=int, required=True, help="Sequence length")
    parser.add_argument("--rope_theta", type=float, required=True, help="RoPE theta")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--schedule", type=str, default="wsd", choices=["cosine", "wsd"], help="LR schedule")
    parser.add_argument("--cooldown", type=int, default=0, help="Cooldown steps at end of WSD schedule")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")
    parser.add_argument("--flash_attention", type=int, default=1, choices=[0, 1], help="Use flash attention")
    parser.add_argument("--checkpoint_interval", type=int, default=1800, help="Checkpoint interval in seconds")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum epochs to train")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Use gradient checkpointing")
    parser.add_argument("--torch_compile", type=int, default=0, choices=[0, 1], help="Use torch.compile for speedup")
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument("--resume_from", type=str, default=None, help="Optional explicit resume checkpoint path")
    parser.add_argument(
        "--reset_optimizer_on_init",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, when loading init_checkpoint_path (non-resume), start fresh optimizer/scheduler at --lr",
    )
    parser.add_argument("--wall_time", type=int, default=0, help="Wall time in seconds. If > 0, save checkpoint 3 min before this limit.")
    parser.add_argument(
        "--target_midtraining_tokens",
        type=int,
        default=0,
        help="If > 0, stop once cumulative midtraining tokens reaches this value.",
    )
    parser.add_argument("--val_data_path", type=str, default=None, help="Optional path to held-out validation data (.bin)")
    parser.add_argument("--sample_prompt", type=str, default="Long long time ago", help="Prompt used for periodic generation")
    parser.add_argument("--sample_max_new_tokens", type=int, default=4096, help="Number of new tokens to generate at checkpoints")
    parser.add_argument("--sample_do_sample", type=int, default=1, choices=[0, 1], help="Use stochastic sampling for periodic generation")
    parser.add_argument("--sample_temperature", type=float, default=0.8, help="Sampling temperature for periodic generation")
    parser.add_argument(
        "--sample_top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p for periodic generation (set <=0 to disable)",
    )
    parser.add_argument(
        "--sample_repetition_penalty",
        type=float,
        default=1.3,
        help="Repetition penalty applied during periodic generation (1.0 disables)",
    )
    parser.add_argument(
        "--sample_no_repeat_ngram_size",
        type=int,
        default=4,
        help="Block repeated n-grams of this size during periodic generation (0 disables)",
    )
    parser.add_argument("--sample_seed", type=int, default=444, help="Random seed for periodic generation")
    parser.add_argument(
        "--final_model_dir_name",
        type=str,
        default="final_model_complete",
        help="Directory name under checkpoint_dir for the final saved HF model artifacts",
    )
    parser.add_argument("--distributed_strategy", type=str, default="ddp", choices=["ddp", "fsdp"], help="Distributed training strategy when world_size > 1")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="shard_grad_op", choices=["shard_grad_op", "full_shard"], help="FSDP sharding strategy")
    parser.add_argument("--fsdp_mixed_precision", type=int, default=0, choices=[0, 1], help="Enable bf16 FSDP param/reduce/buffer mixed precision when precision=bf16")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    tokens_per_micro = args.batch_size * world_size * args.block_size
    grad_accum_steps = args.total_batch_size // tokens_per_micro
    assert grad_accum_steps >= 1, (
        f"total_batch_size ({args.total_batch_size}) too small for "
        f"{world_size} GPU(s) x batch_size {args.batch_size} x block_size {args.block_size}"
    )
    actual_total_batch = grad_accum_steps * tokens_per_micro
    wall_time_save = args.wall_time - 180 if args.wall_time > 0 else 0

    if is_main:
        print("=" * 72)
        print("Argonne Full Long-Context Midtraining")
        print("=" * 72)
        print(f"Using device: {device}, World size: {world_size}")
        print(f"Checkpoint dir: {args.checkpoint_dir}")
        print(f"Seed checkpoint: {args.init_checkpoint_path}")
        print(f"Block size: {args.block_size}")
        print(f"RoPE theta: {args.rope_theta}")
        print(f"Distributed strategy request: {args.distributed_strategy}")
        print("=" * 72)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    vocab_size = len(tokenizer)
    if is_main:
        print(f"Vocab size: {vocab_size}, EOS token ID: {tokenizer.eos_token_id}")

    config = ArgonneConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=args.block_size,
        rope_theta=args.rope_theta,
        use_flash_attention=args.flash_attention == 1,
        tie_word_embeddings=True,
    )
    config.block_size = args.block_size
    config.rope_theta = args.rope_theta
    config._keep_in_fp32_modules = []

    model = ArgonneModel(config)
    model = model.to(device)

    train_loader = build_train_loader(
        args.data_path,
        args.batch_size,
        args.block_size,
        rank,
        world_size,
        doc_shuffle=args.doc_shuffle,
        doc_shuffle_seed=args.doc_shuffle_seed,
    )
    if hasattr(train_loader, "num_tokens"):
        num_tokens = int(train_loader.num_tokens)
    else:
        num_tokens = len(train_loader.tokens)
    estimated_steps = int((num_tokens * args.max_epochs) / actual_total_batch)

    dataset_base_global_step = 0
    dataset_base_tokens_processed = 0
    midtraining_base_global_step = 0
    midtraining_base_tokens_processed = 0
    initial_steps = 0
    is_resumed = False

    if is_main:
        print(f"Training for {args.max_epochs} epoch(s) ~= {estimated_steps} steps ({num_tokens * args.max_epochs:,} tokens)")

    resume_from = args.resume_from or get_latest_checkpoint_path(args.checkpoint_dir)
    checkpoint = None
    checkpoint_path_used = None
    checkpoint_optimizer_state = None

    if resume_from and os.path.exists(resume_from):
        checkpoint_path_used = resume_from
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_optimizer_state = checkpoint["optimizer_state_dict"]
        if args.distributed_strategy == "fsdp" and world_size > 1 and not optimizer_state_uses_param_names(checkpoint_optimizer_state):
            checkpoint_optimizer_state = FSDP.rekey_optim_state_dict(
                checkpoint_optimizer_state,
                OptimStateKeyType.PARAM_NAME,
                model,
            )
        if is_main:
            print(f"\n=== Resuming midtraining from checkpoint: {resume_from} ===")
    elif args.init_checkpoint_path and os.path.exists(args.init_checkpoint_path):
        checkpoint_path_used = args.init_checkpoint_path
        checkpoint = torch.load(args.init_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_optimizer_state = checkpoint["optimizer_state_dict"]
        if args.distributed_strategy == "fsdp" and world_size > 1 and not optimizer_state_uses_param_names(checkpoint_optimizer_state):
            checkpoint_optimizer_state = FSDP.rekey_optim_state_dict(
                checkpoint_optimizer_state,
                OptimStateKeyType.PARAM_NAME,
                model,
            )
        if is_main:
            print(f"\n=== Initializing midtraining from seed checkpoint: {args.init_checkpoint_path} ===")
    else:
        if is_main:
            print("\n=== No prior checkpoint found; starting midtraining from scratch ===")

    model, distributed_strategy, delay_compile_until_after_optimizer_restore = wrap_model_for_training(
        model, args, device, world_size, is_main
    )

    if is_main:
        total_params = sum(p.numel() for p in get_training_module(model).parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Mixed precision: {'autocast ' + args.precision if USE_AUTOCAST else 'fp32 (no autocast)'}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

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
    loaded_optimizer_state = False

    if checkpoint is not None:
        is_init_checkpoint = checkpoint_path_used != resume_from
        reset_optimizer_on_init = is_init_checkpoint and args.reset_optimizer_on_init == 1
        if reset_optimizer_on_init:
            if is_main:
                print(
                    "Skipping seed checkpoint optimizer/scheduler state and using fresh optimizer at requested --lr."
                )
        else:
            if is_fsdp_model(model):
                sharded_optimizer_state = FSDP.shard_full_optim_state_dict(
                    checkpoint_optimizer_state,
                    model,
                    optim=optimizer,
                )
                optimizer.load_state_dict(sharded_optimizer_state)
            else:
                optimizer.load_state_dict(checkpoint_optimizer_state)
            loaded_optimizer_state = True

            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            else:
                for _ in range(checkpoint["global_step"]):
                    scheduler.step()

        global_step = checkpoint["global_step"]
        tokens_processed = checkpoint["tokens_processed"]

        if checkpoint_path_used == resume_from:
            data_position = checkpoint.get("data_position", 0)
            checkpoint_dataset_epoch = checkpoint.get("dataset_epoch")
            checkpoint_dataset_num_tokens = checkpoint.get("dataset_num_tokens")
            checkpoint_dataset_path = checkpoint.get("dataset_path")
            checkpoint_dataset_base_step = checkpoint.get("dataset_base_global_step")
            checkpoint_dataset_base_tokens = checkpoint.get("dataset_base_tokens_processed")
            midtraining_base_global_step = int(checkpoint.get("midtraining_base_global_step", global_step))
            midtraining_base_tokens_processed = int(checkpoint.get("midtraining_base_tokens_processed", tokens_processed))

            if hasattr(train_loader, "resume_from_checkpoint_position"):
                train_loader.resume_from_checkpoint_position(data_position)
            else:
                train_loader.set_position(data_position + rank * args.batch_size * args.block_size)
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
                set_loader_epoch(
                    train_loader,
                    int(checkpoint_dataset_epoch) if checkpoint_dataset_epoch is not None else 0,
                )
            else:
                if hasattr(train_loader, "steps_from_position"):
                    cursor_steps = int(train_loader.steps_from_position(data_position))
                else:
                    cursor_steps = int(max(0, data_position) // actual_total_batch)
                dataset_base_global_step = max(0, global_step - cursor_steps)
                dataset_base_tokens_processed = max(0, tokens_processed - cursor_steps * actual_total_batch)
                set_loader_epoch(train_loader, 0)
                if is_main:
                    print("Legacy or dataset-mismatched checkpoint metadata; inferring dataset-local progress from the saved data cursor.")

            dataset_progress_tokens = max(0, tokens_processed - dataset_base_tokens_processed)
            dataset_progress_steps = int(dataset_progress_tokens // actual_total_batch)
            initial_steps = min(estimated_steps, dataset_progress_steps)
            is_resumed = True
            if is_main:
                midtraining_tokens_processed = max(0, tokens_processed - midtraining_base_tokens_processed)
                print(
                    f"Resumed from step {global_step}, cumulative tokens: {tokens_processed:,}, "
                    f"midtraining tokens: {midtraining_tokens_processed:,}, dataset epoch: {train_loader.epoch}, "
                    f"dataset progress: {dataset_progress_steps}/{estimated_steps} step(s), "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
        else:
            dataset_base_global_step = global_step
            dataset_base_tokens_processed = tokens_processed
            midtraining_base_global_step = global_step
            midtraining_base_tokens_processed = tokens_processed
            if hasattr(train_loader, "start_from_beginning"):
                train_loader.start_from_beginning()
            else:
                train_loader.set_position(rank * args.batch_size * args.block_size)
            set_loader_epoch(train_loader, 0)
            if is_main:
                print(
                    f"Seed checkpoint loaded at step {global_step}, cumulative tokens: {tokens_processed:,}. "
                    "Starting the long-context data cursor from the beginning."
                )
                optimizer_state_label = "Loaded" if loaded_optimizer_state else "Fresh"
                print(f"{optimizer_state_label} optimizer LR: {optimizer.param_groups[0]['lr']:.2e}")
    else:
        global_step = 0
        tokens_processed = 0

    launch_midtraining_tokens = max(0, tokens_processed - midtraining_base_tokens_processed)
    target_midtraining_tokens = int(args.target_midtraining_tokens) if args.target_midtraining_tokens > 0 else 0
    remaining_midtraining_tokens = 0
    target_steps = 0
    if target_midtraining_tokens > 0:
        remaining_midtraining_tokens = max(0, target_midtraining_tokens - launch_midtraining_tokens)
        target_steps = int((remaining_midtraining_tokens + actual_total_batch - 1) // actual_total_batch)

    if delay_compile_until_after_optimizer_restore:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.torch_compile_mode)

    last_checkpoint_time = time.time()
    training_start_time = time.time()
    train_losses = []

    if is_main:
        print("\nStarting training...")
        print(f"GPUs: {world_size}, Batch size per GPU: {args.batch_size}")
        print(f"Sequence length: {args.block_size}")
        print(f"Total batch size: {actual_total_batch} tokens (requested: {args.total_batch_size})")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Training for {args.max_epochs} epoch(s) (estimated ~{estimated_steps} steps)")
        if target_midtraining_tokens > 0:
            print(
                f"Midtraining token target: {target_midtraining_tokens:,} "
                f"(launch={launch_midtraining_tokens:,}, remaining={remaining_midtraining_tokens:,} ~= {target_steps} step(s))"
            )
        print(f"Distributed strategy in use: {distributed_strategy}")
        print(f"Cumulative step at launch: {global_step}")
        print(f"Cumulative tokens at launch: {tokens_processed:,}")
        print(f"Midtraining tokens at launch: {launch_midtraining_tokens:,}")
        print(f"Dataset-local progress at launch: {initial_steps}/{estimated_steps} step(s), dataset epoch {train_loader.epoch}")
        print(
            f"LR: {args.lr}, Warmup: {args.warmup_steps}, Min LR Ratio: {args.min_lr_ratio}, "
            f"Precision: {args.precision}, TorchCompile: {args.torch_compile}"
        )
        print(f"Checkpoint interval: {args.checkpoint_interval} seconds")
        print(f"Validation data: {args.val_data_path if args.val_data_path else 'disabled (no held-out file provided)'}")
        print(
            f"Sample prompt: {args.sample_prompt!r}, sample_max_new_tokens={args.sample_max_new_tokens}, "
            f"sample_do_sample={args.sample_do_sample}, sample_temperature={args.sample_temperature}, "
            f"sample_top_p={args.sample_top_p}, sample_repetition_penalty={args.sample_repetition_penalty}, "
            f"sample_no_repeat_ngram_size={args.sample_no_repeat_ngram_size}, sample_seed={args.sample_seed}"
        )
        if args.wall_time > 0:
            print(f"Wall time: {args.wall_time}s, will save checkpoint at {wall_time_save}s")
        print("-" * 60)

    pbar_total = estimated_steps
    pbar_initial = initial_steps
    pbar_desc = "Training"
    if target_midtraining_tokens > 0:
        pbar_total = target_steps
        pbar_initial = 0
        pbar_desc = "Training(target)"

    pbar = None
    if is_main:
        pbar = tqdm(total=pbar_total, initial=pbar_initial, desc=pbar_desc, unit="step")

    model.train()
    reset_peak_memory_stats(device)
    target_reached_at_launch = target_midtraining_tokens > 0 and launch_midtraining_tokens >= target_midtraining_tokens
    if target_reached_at_launch and is_main:
        print(
            f"Target midtraining tokens already reached at launch: "
            f"{launch_midtraining_tokens:,} >= {target_midtraining_tokens:,}. Finalizing without extra training."
        )

    while True:
        if target_reached_at_launch:
            break

        optimizer.zero_grad(set_to_none=True)
        step_loss_total = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            try:
                if world_size > 1 and micro_step < grad_accum_steps - 1:
                    with model.no_sync():
                        with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                            outputs = model(x, labels=y)
                            micro_loss = outputs.loss
                            loss = micro_loss / grad_accum_steps
                        loss.backward()
                else:
                    with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                        outputs = model(x, labels=y)
                        micro_loss = outputs.loss
                        loss = micro_loss / grad_accum_steps
                    loss.backward()
            except torch.OutOfMemoryError:
                local_stats = collect_memory_snapshot(device, world_size)
                print(
                    f"[rank{rank}] OOM during forward/backward at step {global_step}. "
                    f"Memory snapshot: {format_memory_snapshot(local_stats)}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

            tokens_processed += args.batch_size * args.block_size * world_size
            step_loss_total += micro_loss.detach().float().item()
            train_losses.append(micro_loss.detach().float().item())

        step_loss = step_loss_total / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if pbar:
            pbar.update(1)

        current_lr = optimizer.param_groups[0]["lr"]
        midtraining_tokens_processed = max(0, tokens_processed - midtraining_base_tokens_processed)

        if is_main and global_step % 10 == 0:
            perplexity = np.exp(step_loss)
            print(
                f"Step {global_step} | Loss: {step_loss:.4f} | PPL: {perplexity:.2f} | "
                f"Tokens: {tokens_processed:,} | Midtraining Tokens: {midtraining_tokens_processed:,} | "
                f"LR: {current_lr:.2e}"
            )
            if pbar:
                pbar.set_postfix(
                    {
                        "loss": f"{step_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "tok": f"{tokens_processed/1e6:.2f}M",
                        "midtok": f"{midtraining_tokens_processed/1e6:.2f}M",
                    }
                )

        should_token_target_stop = torch.tensor([0], device=device)
        if target_midtraining_tokens > 0 and midtraining_tokens_processed >= target_midtraining_tokens:
            should_token_target_stop[0] = 1
        if world_size > 1:
            dist.all_reduce(should_token_target_stop, op=dist.ReduceOp.MAX)

        if should_token_target_stop[0] == 1:
            if is_main:
                print(
                    f"\nReached midtraining token target at step {global_step}: "
                    f"{midtraining_tokens_processed:,}/{target_midtraining_tokens:,}. Finalizing..."
                )
            break

        should_checkpoint = torch.tensor([0], device=device)
        if is_main:
            current_time = time.time()
            if current_time - last_checkpoint_time >= args.checkpoint_interval:
                should_checkpoint[0] = 1
        if world_size > 1:
            dist.broadcast(should_checkpoint, src=0)

        if should_checkpoint[0] == 1:
            print("\n" + "=" * 60) if is_main else None
            print("Saving checkpoint...") if is_main else None
            data_position = train_loader.get_position()
            checkpoint_path, generation_state_dict = save_checkpoint(
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
                midtraining_base_global_step,
                midtraining_base_tokens_processed,
                distributed_strategy,
                args.fsdp_sharding_strategy,
                is_main,
            )
            if is_main:
                print(f"Checkpoint saved: {checkpoint_path}")

                print("\nGenerating long sample text...")
                try:
                    generated = generate_text_from_state_dict(
                        generation_state_dict,
                        config,
                        tokenizer,
                        device,
                        prompt=args.sample_prompt,
                        max_new_tokens=args.sample_max_new_tokens,
                        sample_do_sample=args.sample_do_sample,
                        sample_temperature=args.sample_temperature,
                        sample_top_p=args.sample_top_p,
                        sample_repetition_penalty=args.sample_repetition_penalty,
                        sample_no_repeat_ngram_size=args.sample_no_repeat_ngram_size,
                        sample_seed=args.sample_seed,
                    )
                    print(f"Generated ({args.sample_max_new_tokens} new tokens target): {generated}")
                except Exception as e:
                    print(f"Sample generation skipped after checkpoint due to error: {e}")
                print("=" * 60 + "\n")
                del generation_state_dict

            if world_size > 1:
                dist.barrier()
            last_checkpoint_time = time.time()

        if wall_time_save > 0:
            should_wall_stop = torch.tensor([0], device=device)
            if is_main:
                elapsed = time.time() - training_start_time
                if elapsed >= wall_time_save:
                    should_wall_stop[0] = 1
            if world_size > 1:
                dist.broadcast(should_wall_stop, src=0)

            if should_wall_stop[0] == 1:
                if is_main:
                    print(f"\nApproaching wall time ({args.wall_time}s). Saving checkpoint and exiting...")
                data_position = train_loader.get_position()
                checkpoint_path, _ = save_checkpoint(
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
                    midtraining_base_global_step,
                    midtraining_base_tokens_processed,
                    distributed_strategy,
                    args.fsdp_sharding_strategy,
                    is_main,
                )
                if is_main:
                    print(f"Wall time checkpoint saved: {checkpoint_path}")
                if world_size > 1:
                    dist.barrier()
                break

        should_stop = torch.tensor([0], device=device)
        if train_loader.epoch >= args.max_epochs:
            should_stop[0] = 1
        if world_size > 1:
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)

        if should_stop[0] == 1:
            if is_main:
                print(f"\nCompleted {args.max_epochs} epoch(s) at step {global_step}. Finalizing...")
            break

    if pbar:
        pbar.close()

    elapsed_time = time.time() - training_start_time
    if is_main:
        print("-" * 60)
        print(f"Training completed in {elapsed_time:.1f} seconds!")

    train_loss = np.mean(train_losses) if train_losses else 0.0
    val_loss_str = "n/a"
    if is_main:
        print("\nValidation skipped: no held-out validation file was provided." if not args.val_data_path else "\nValidation skipped in this FSDP-focused run.")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss_str}")

    if is_main:
        print("\nSaving final checkpoint...")
    data_position = train_loader.get_position()
    checkpoint_path, _ = save_checkpoint(
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
        midtraining_base_global_step,
        midtraining_base_tokens_processed,
        distributed_strategy,
        args.fsdp_sharding_strategy,
        is_main,
    )
    if is_main:
        print(f"Final checkpoint saved: {checkpoint_path}")

    final_model_dir = os.path.join(args.checkpoint_dir, args.final_model_dir_name)
    save_final_model_artifacts(model, config, tokenizer, final_model_dir, is_main)

    if is_main:
        midtraining_tokens_processed = max(0, tokens_processed - midtraining_base_tokens_processed)
        print("\n" + "=" * 60)
        print(
            f"SUMMARY: train_loss={train_loss:.4f} val_loss={val_loss_str} "
            f"tokens_per_sec={tokens_processed/elapsed_time:.2f} "
            f"midtraining_tokens={midtraining_tokens_processed:,} steps={global_step}"
        )
        print("=" * 60)

    if world_size > 1:
        dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
