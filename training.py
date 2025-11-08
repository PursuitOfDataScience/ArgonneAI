import argparse
import contextlib
import os
from typing import List, Optional, Tuple

# CRITICAL: Disable cudagraph capture before importing torch so compiled graphs
# never attempt to enclose collective operations used by tensor parallelism.
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
# Silence tokenizer parallelism warnings from Hugging Face tokenizers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import torch.nn.functional as F

from model import ArgonneModel
from training_utils import DEFAULT_MAX_TRAINING_STEPS

# Enable TF32 precision on Ampere/Hopper GPUs for faster matmuls.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def _ensure_gradient_dtype_matches_params(model: torch.nn.Module) -> None:
    """Cast gradients to match their parameter's dtype/device for fused optimizers."""
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if grad.dtype != param.dtype or grad.device != param.device:
            param.grad = grad.to(device=param.device, dtype=param.dtype)


def _gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


class DataPosition:
    def __init__(self, streaming: bool = True) -> None:
        """Track dataset position during training."""
        self.streaming = streaming

        # For streaming mode
        self.current_file_idx = 0
        self.position_in_file = 0
        self.chunk_offset = 0

        # For non-streaming mode
        self.shuffled_indices: Optional[List[int]] = None
        self.current_position = 0
        self.epoch = 0

        # Files processed tracking
        self.files_processed = set()

    def get_state(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "streaming": self.streaming,
            "current_file_idx": self.current_file_idx,
            "position_in_file": self.position_in_file,
            "chunk_offset": self.chunk_offset,
            "current_position": self.current_position,
            "epoch": self.epoch,
            "files_processed": sorted(self.files_processed),
        }

    def restore_state(self, state: Optional[dict]) -> None:
        """Restore position information from checkpoint data."""
        if not state:
            return
        self.streaming = state.get("streaming", self.streaming)
        self.current_file_idx = state.get("current_file_idx", 0)
        self.position_in_file = state.get("position_in_file", 0)
        self.chunk_offset = state.get("chunk_offset", state.get("chunk_index", 0))
        self.current_position = state.get("current_position", 0)
        self.epoch = state.get("epoch", state.get("global_step", 0))
        files = state.get("files_processed", [])
        self.files_processed = {os.path.basename(f) for f in files}

    def update_streaming_position(
        self,
        file_idx: int,
        position: int,
        chunk_offset: int = 0,
        file_path: Optional[str] = None,
    ) -> None:
        """Update streaming position information."""
        self.current_file_idx = file_idx
        self.position_in_file = position
        self.chunk_offset = chunk_offset
        if file_path:
            self.files_processed.add(os.path.basename(file_path))

    def update_nonstreaming_position(self, position: int) -> None:
        """Update non-streaming position."""
        self.current_position = position

    def generate_shuffled_indices(self, total_samples: int) -> List[int]:
        """Generate shuffled indices for non-streaming mode."""
        if self.shuffled_indices is None or len(self.shuffled_indices) != total_samples:
            self.shuffled_indices = torch.randperm(total_samples).tolist()
        return self.shuffled_indices[self.current_position:]

    def next_epoch(self, total_samples: Optional[int] = None) -> None:
        """Move to next epoch."""
        self.epoch += 1
        if self.streaming:
            self.current_file_idx = 0
            self.position_in_file = 0
            self.chunk_offset = 0
        else:
            self.current_position = 0
            if total_samples:
                self.shuffled_indices = torch.randperm(total_samples).tolist()


def init_tensor_parallel_group(world_size: int, rank: int) -> None:
    """Initialize distributed process group for tensor parallelism."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    if rank == 0:
        print(f"Initialized tensor parallel group: rank {rank}/{world_size}")


def shard_attention_layer(layer: torch.nn.Module, world_size: int, rank: int) -> None:
    """Shard attention Q, K, V, and output projection across tensor parallel dimension."""
    import torch.nn as nn

    # Store original values
    original_num_heads = layer.num_heads
    original_num_kv_heads = layer.num_kv_heads
    original_head_dim = layer.head_dim

    if rank == 0:
        print(
            "  Original: num_heads=%d, num_kv_heads=%d, head_dim=%d"
            % (original_num_heads, original_num_kv_heads, original_head_dim)
        )

    # Shard Q, K, V projections (column-parallel)
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        if hasattr(layer, proj_name):
            old_proj = getattr(layer, proj_name)
            out_features = old_proj.out_features
            in_features = old_proj.in_features

            chunk_size = out_features // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features

            new_proj = nn.Linear(
                in_features,
                end_idx - start_idx,
                bias=old_proj.bias is not None,
            )
            new_proj.weight.data = old_proj.weight.data[start_idx:end_idx].clone()
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data[start_idx:end_idx].clone()

            setattr(layer, proj_name, new_proj)

    # Output projection (row-parallel)
    if hasattr(layer, "o_proj"):
        old_proj = layer.o_proj
        in_features = old_proj.in_features
        out_features = old_proj.out_features

        chunk_size = in_features // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features

        new_proj = nn.Linear(
            end_idx - start_idx,
            out_features,
            bias=old_proj.bias is not None,
        )
        new_proj.weight.data = old_proj.weight.data[:, start_idx:end_idx].clone()

        if old_proj.bias is not None:
            if rank == 0:
                new_proj.bias.data = old_proj.bias.data.clone()
            else:
                new_proj.bias = None

        setattr(layer, "o_proj", new_proj)

    # Update layer attributes for sharded dimensions
    layer.num_heads = original_num_heads // world_size
    layer.head_dim = original_head_dim
    layer.num_kv_heads = original_num_kv_heads // world_size
    layer.num_key_value_groups = layer.num_heads // layer.num_kv_heads

    if rank == 0:
        print(
            "  Sharded: num_heads=%d, num_kv_heads=%d, head_dim=%d"
            % (layer.num_heads, layer.num_kv_heads, layer.head_dim)
        )
        print(
            "  Dims: Q=%d, K=%d, V=%d"
            % (
                layer.q_proj.out_features,
                layer.k_proj.out_features,
                layer.v_proj.out_features,
            )
        )
        print(f"  Groups: {layer.num_key_value_groups}")


def shard_mlp_layer(mlp: torch.nn.Module, world_size: int, rank: int) -> None:
    """Shard MLP layers across tensor parallel dimension."""
    import torch.nn as nn

    # SwiGLUMLP uses: gate_proj, up_proj (column-parallel), down_proj (row-parallel)
    for proj_name in ["gate_proj", "up_proj"]:
        if hasattr(mlp, proj_name):
            old_proj = getattr(mlp, proj_name)
            out_features = old_proj.out_features
            in_features = old_proj.in_features

            chunk_size = out_features // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features

            new_proj = nn.Linear(
                in_features,
                end_idx - start_idx,
                bias=old_proj.bias is not None,
            )
            new_proj.weight.data = old_proj.weight.data[start_idx:end_idx].clone()
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data[start_idx:end_idx].clone()

            setattr(mlp, proj_name, new_proj)

    # down_proj: row-parallel (split input)
    if hasattr(mlp, "down_proj"):
        old_proj = mlp.down_proj
        in_features = old_proj.in_features
        out_features = old_proj.out_features

        chunk_size = in_features // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features

        new_proj = nn.Linear(
            end_idx - start_idx,
            out_features,
            bias=old_proj.bias is not None,
        )
        new_proj.weight.data = old_proj.weight.data[:, start_idx:end_idx].clone()

        if old_proj.bias is not None:
            if rank == 0:
                new_proj.bias.data = old_proj.bias.data.clone()
            else:
                new_proj.bias = None

        setattr(mlp, "down_proj", new_proj)


def shard_tensor_parallel_correctly(
    model: ArgonneModel, world_size: int, rank: int
) -> None:
    """Properly shard the model for tensor parallelism."""
    if rank == 0:
        print(
            f"Sharding model for tensor parallelism (world_size={world_size}, rank={rank})"
        )

    # Iterate through blocks and shard their components
    for block in model.blocks:
        if hasattr(block, "attn"):
            shard_attention_layer(block.attn, world_size, rank)
        if hasattr(block, "mlp"):
            shard_mlp_layer(block.mlp, world_size, rank)

    if rank == 0:
        print(f"✓ Successfully sharded {len(model.blocks)} transformer blocks")


class TensorParallelModel(torch.nn.Module):
    """Wrapper for ArgonneModel that implements tensor parallelism."""

    def __init__(self, base_model: ArgonneModel, world_size: int, rank: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.gradient_checkpointing = False

        # Move model to device first
        self.base_model = self.base_model.to(self.device)

        # Then shard it
        shard_tensor_parallel_correctly(self.base_model, world_size, rank)

        if rank == 0:
            print(f"✓ Model ready for tensor parallel training on {world_size} GPUs")

    def _block_forward(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single block with tensor parallelism."""
        residual = hidden_states
        normed = block.input_norm(hidden_states)
        attn_output = block.attn(normed, position_embeddings, attention_mask)
        attn_output = attn_output.contiguous()

        if self.world_size > 1:
            dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)

        hidden_states = residual + attn_output

        residual = hidden_states
        normed = block.post_norm(hidden_states)
        mlp_output = block.mlp(normed)
        mlp_output = mlp_output.contiguous()

        if self.world_size > 1:
            dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM)

        hidden_states = residual + mlp_output
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass with correct tensor parallelism."""
        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        bsz, seq_len = input_ids.size()
        hidden_states = self.base_model.embed_tokens(input_ids)

        cos, sin = self.base_model.rotary_emb(hidden_states, seq_len)
        position_embeddings = (cos, sin)

        if self.gradient_checkpointing and self.training:
            for block in self.base_model.blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._block_forward,
                    block,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    use_reentrant=False,
                )
        else:
            for block in self.base_model.blocks:
                hidden_states = self._block_forward(
                    block,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                )

        hidden_states = self.base_model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        from transformers.modeling_outputs import CausalLMOutput

        return CausalLMOutput(logits=logits, loss=loss)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Distributed text generation that respects tensor parallel sharding."""
        was_training = self.training
        self.eval()

        try:
            if not torch.is_tensor(input_ids):
                raise TypeError("input_ids must be a torch.Tensor")

            input_ids = input_ids.to(self.device)

            if self.world_size > 1 and dist.is_initialized():
                prompt_length = torch.tensor(
                    [input_ids.shape[1]],
                    device=self.device,
                    dtype=torch.long,
                )
                dist.broadcast(prompt_length, src=0)

                if input_ids.shape[1] != int(prompt_length.item()):
                    new_prompt = torch.zeros(
                        input_ids.shape[0],
                        prompt_length.item(),
                        dtype=input_ids.dtype,
                        device=self.device,
                    )
                    if self.rank == 0:
                        new_prompt.copy_(input_ids[:, : prompt_length.item()])
                    dist.broadcast(new_prompt, src=0)
                    input_ids = new_prompt
                else:
                    dist.broadcast(input_ids, src=0)

            generated = input_ids
            use_autocast = self.device.type == "cuda"
            amp_dtype = None
            if use_autocast:
                weight = self.base_model.embed_tokens.weight
                if weight.is_floating_point():
                    amp_dtype = weight.dtype

            with torch.no_grad():
                while generated.shape[1] < max_length:
                    context_window = generated[:, -self.base_model.config.max_position_embeddings :]
                    autocast_context = (
                        torch.amp.autocast("cuda", dtype=amp_dtype)
                        if use_autocast and amp_dtype is not None
                        else contextlib.nullcontext()
                    )

                    with autocast_context:
                        outputs = self.forward(context_window)
                        logits = outputs.logits[:, -1, :]

                    logits = logits / temperature

                    if do_sample:
                        if top_k is not None:
                            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits = logits.masked_fill(
                                logits < top_values[:, [-1]],
                                float("-inf"),
                            )
                        if top_p is not None:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(
                                F.softmax(sorted_logits, dim=-1),
                                dim=-1,
                            )
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                ..., :-1
                            ].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                1,
                                sorted_indices,
                                sorted_indices_to_remove,
                            )
                            logits = logits.masked_fill(indices_to_remove, float("-inf"))

                        if self.rank == 0:
                            probs = F.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.empty(
                                (generated.size(0), 1),
                                dtype=torch.long,
                                device=self.device,
                            )
                    else:
                        if self.rank == 0:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        else:
                            next_token = torch.empty(
                                (generated.size(0), 1),
                                dtype=torch.long,
                                device=self.device,
                            )

                    if self.world_size > 1 and dist.is_initialized():
                        dist.broadcast(next_token, src=0)

                    generated = torch.cat([generated, next_token], dim=-1)

                    if generated.shape[1] >= max_length:
                        break

            return generated
        finally:
            if was_training:
                self.train()

    def state_dict(self, *args, **kwargs):
        """Get state dict from base model."""
        return self.base_model.state_dict(*args, **kwargs)

    def parameters(self):
        """Get parameters from base model."""
        return self.base_model.parameters()

    def gradient_checkpointing_enable(self) -> None:
        """Enable per-block gradient checkpointing."""
        self.gradient_checkpointing = True
        if self.rank == 0:
            print("✓ Gradient checkpointing enabled (per-block)")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Argonne model from scratch with Tensor Parallelism",
    )
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=default_data_glob,
        help="Glob pattern for parquet shards.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Filesystem directory containing the pretrained tokenizer to reuse.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional path to checkpoint. Ignored for the initial training run.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_MAX_TRAINING_STEPS,
        help="Total number of training steps to run.",
    )
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="Minimum learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--rewarmup-steps",
        type=int,
        default=100,
        help="Number of re-warmup steps when resuming (unused for scratch runs).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading tokenizers that require custom code.",
    )
    parser.add_argument(
        "--force-from-scratch",
        action="store_true",
        default=True,
        help="Always start from scratch. This flag is kept for CLI compatibility.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing to speed up training (requires more GPU memory).",
    )
    parser.add_argument(
        "--add-document-boundary-tokens",
        action="store_true",
        help=(
            "Prepend the tokenizer BOS token and append the EOS token to each document "
            "before chunking."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_path and args.checkpoint_path.upper() != "NONE":
        print(
            "⚠ training.py always starts from scratch and will ignore --checkpoint-path. "
            "Use resume_pretrain_tensor.py to continue from an existing checkpoint.",
            flush=True,
        )

    from resume_pretrain_tensor import resume_training

    resume_training(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        checkpoint_path=None,
        total_training_steps=args.total_steps,
        block_size=args.block_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_streaming=not args.no_streaming,
        num_proc=args.num_proc,
        trust_remote_code=args.trust_remote_code,
        force_from_scratch=True,
        rewarmup_steps=args.rewarmup_steps,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        add_document_tokens=args.add_document_boundary_tokens,
    )


if __name__ == "__main__":
    main()
