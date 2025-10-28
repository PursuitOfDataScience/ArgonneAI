import argparse
import contextlib
import json
import os
import time
import traceback
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from data_processing import (
    collate_batch,
    load_tokenizer,
    chunk_tokens,
)
from model import ArgonneConfig, ArgonneModel
from training_utils import (
    CosineWarmupScheduler,
    DEFAULT_MAX_TRAINING_STEPS,
    cast_state_dict_to_dtype,
    load_streaming_shard,
    log_dataset_plan,
    resolve_data_files,
    safe_torch_save,
    validate_tokenizer_path,
)

# To silence the warning about tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 precision on Ampere/Hopper GPUs
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
    """Calculate greatest common divisor"""
    while b:
        a, b = b, a % b
    return a


class DataPosition:
    def __init__(self, streaming=True):
        """Track dataset position during training"""
        self.streaming = streaming

        # For streaming mode
        self.current_file_idx = 0
        self.position_in_file = 0
        self.chunk_offset = 0
        
        # For non-streaming mode
        self.shuffled_indices = None
        self.current_position = 0
        self.epoch = 0
        
        # Files processed tracking
        self.files_processed = set()
        
    def get_state(self):
        """Returns state dict for checkpointing"""
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

    def update_streaming_position(self, file_idx, position, chunk_offset=0, file_path=None):
        """Update streaming position information"""
        self.current_file_idx = file_idx
        self.position_in_file = position
        self.chunk_offset = chunk_offset
        if file_path:
            self.files_processed.add(os.path.basename(file_path))
    
    def update_nonstreaming_position(self, position):
        """Update non-streaming position"""
        self.current_position = position

    def generate_shuffled_indices(self, total_samples):
        """Generate shuffled indices for non-streaming mode"""
        if self.shuffled_indices is None or len(self.shuffled_indices) != total_samples:
            self.shuffled_indices = torch.randperm(total_samples).tolist()
        return self.shuffled_indices[self.current_position:]
    
    def next_epoch(self, total_samples=None):
        """Move to next epoch"""
        self.epoch += 1
        if self.streaming:
            self.current_file_idx = 0
            self.position_in_file = 0
            self.chunk_offset = 0
        else:
            self.current_position = 0
            if total_samples:
                self.shuffled_indices = torch.randperm(total_samples).tolist()


def streaming_token_generator(
    data_files: List[str],
    tokenizer,
    block_size: int,
    start_file_idx: int = 0,
    start_position: int = 0,
    start_chunk_offset: int = 0,
    rank: int = 0,
    add_document_tokens: bool = False,
):
    """Generator with chunk-level resume support."""
    import gc

    file_idx = max(start_file_idx, 0)
    processed_count = 0
    is_main_process = (rank == 0)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    document_tokens_enabled = bool(add_document_tokens)

    if document_tokens_enabled:
        missing_tokens = []
        if bos_token_id is None:
            missing_tokens.append("BOS")
        if eos_token_id is None:
            missing_tokens.append("EOS")

        if missing_tokens:
            if is_main_process:
                print(
                    "⚠ Unable to add document boundary tokens: missing "
                    + ", ".join(missing_tokens)
                    + " token id(s) in tokenizer"
                )
            document_tokens_enabled = False
        elif is_main_process:
            print(
                f"✓ Adding BOS/EOS tokens to each document "
                f"(bos_id={bos_token_id}, eos_id={eos_token_id})"
            )
    initial_file_idx = file_idx
    initial_position = start_position
    initial_chunk_offset = start_chunk_offset
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    while file_idx < len(data_files):
        dataset = None
        
        try:
            file_path = data_files[file_idx]
            shard_name = os.path.basename(file_path)
            if is_main_process:
                print(f"Streaming from shard {file_idx + 1}/{len(data_files)}: {shard_name}")

            try:
                dataset = load_streaming_shard(file_path)
                if is_main_process:
                    print(f"Successfully loaded dataset with {len(dataset)} rows")
                consecutive_errors = 0
                
            except Exception as file_error:
                consecutive_errors += 1
                if is_main_process:
                    print(f"ERROR: Could not read file {file_path}: {file_error}")
                    print(f"Consecutive errors: {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    if is_main_process:
                        print(f"⚠ Too many consecutive errors. Forcing cleanup and restarting...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    consecutive_errors = 0
                
                file_idx += 1
                continue

            if file_idx == initial_file_idx:
                position = initial_position
                resume_chunk_offset = initial_chunk_offset
                if is_main_process and (position > 0 or resume_chunk_offset > 0):
                    print(f"  >>> RESUMING from position {position}, chunk offset {resume_chunk_offset}")
            else:
                position = 0
                resume_chunk_offset = 0

            while position < len(dataset):
                try:
                    item = dataset[position]
                    if "text" in item and item["text"] and isinstance(item["text"], str):
                        text = item["text"]
                        
                        # Data quality filter
                        if len(text) > 50:
                            alpha_count = sum(c.isalpha() or c.isspace() for c in text)
                            digit_count = sum(c.isdigit() for c in text)
                            
                            if digit_count / len(text) > 0.5 or alpha_count / len(text) < 0.3:
                                position += 1
                                continue
                        
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        resume_mid_document = (
                            document_tokens_enabled
                            and file_idx == initial_file_idx
                            and position == initial_position
                            and resume_chunk_offset > 0
                        )

                        if document_tokens_enabled:
                            tokens = [bos_token_id, *tokens, eos_token_id]

                            if resume_mid_document:
                                # We are resuming from the middle of a document that was
                                # previously trained without BOS/EOS tokens. Dropping the
                                # prepended BOS ensures the chunk boundaries align with the
                                # checkpoint's stored chunk_offset while still allowing the
                                # trailing EOS to be learned for the remaining portion.
                                tokens = tokens[1:]

                        if len(tokens) < 10:
                            position += 1
                            continue

                        for chunk_idx, chunk in enumerate(chunk_tokens(tokens, block_size)):
                            if file_idx == initial_file_idx and position == initial_position:
                                if chunk_idx < resume_chunk_offset:
                                    continue
                            
                            processed_count += 1
                            yield chunk, file_idx, position, shard_name, chunk_idx
                            
                except Exception as e:
                    if is_main_process:
                        print(f"Error processing item at position {position}: {e}")
                
                position += 1
            
            if is_main_process:
                print(f"Finished processing {shard_name}, cleaning up resources...")
            
            del dataset
            dataset = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import time
            time.sleep(0.1)
            
            file_idx += 1
            
        except Exception as e:
            if is_main_process:
                print(f"Error processing file {data_files[file_idx]}: {e}")
            
            if dataset is not None:
                del dataset
                dataset = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            file_idx += 1

    if is_main_process:
        print(f"Completed processing all available files. Processed {processed_count} samples.")
    
    return None, -1, -1, "", -1


def init_tensor_parallel_group(world_size: int, rank: int) -> None:
    """Initialize distributed process group for tensor parallelism"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
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
        print(f"  Original: num_heads={original_num_heads}, num_kv_heads={original_num_kv_heads}, head_dim={original_head_dim}")
    
    # Shard Q, K, V projections (column-parallel)
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        if hasattr(layer, proj_name):
            old_proj = getattr(layer, proj_name)
            out_features = old_proj.out_features
            in_features = old_proj.in_features
            
            chunk_size = out_features // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features
            
            new_proj = nn.Linear(in_features, end_idx - start_idx, bias=old_proj.bias is not None)
            new_proj.weight.data = old_proj.weight.data[start_idx:end_idx].clone()
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data[start_idx:end_idx].clone()
            
            setattr(layer, proj_name, new_proj)
    
    # Output projection (row-parallel)
    if hasattr(layer, 'o_proj'):
        old_proj = layer.o_proj
        in_features = old_proj.in_features
        out_features = old_proj.out_features
        
        chunk_size = in_features // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features
        
        new_proj = nn.Linear(end_idx - start_idx, out_features, bias=old_proj.bias is not None)
        new_proj.weight.data = old_proj.weight.data[:, start_idx:end_idx].clone()
        
        if old_proj.bias is not None:
            if rank == 0:
                new_proj.bias.data = old_proj.bias.data.clone()
            else:
                new_proj.bias = None
        
        setattr(layer, 'o_proj', new_proj)
    
    # Get actual sharded dimensions
    actual_q_out = layer.q_proj.out_features
    actual_k_out = layer.k_proj.out_features
    actual_v_out = layer.v_proj.out_features
    
    # Update layer attributes for sharded dimensions
    layer.num_heads = original_num_heads // world_size
    layer.head_dim = original_head_dim
    layer.num_kv_heads = original_num_kv_heads // world_size
    layer.num_key_value_groups = layer.num_heads // layer.num_kv_heads
    
    if rank == 0:
        print(f"  Sharded: num_heads={layer.num_heads}, num_kv_heads={layer.num_kv_heads}, head_dim={layer.head_dim}")
        print(f"  Dims: Q={actual_q_out}, K={actual_k_out}, V={actual_v_out}")
        print(f"  Groups: {layer.num_key_value_groups}")


def shard_mlp_layer(mlp: torch.nn.Module, world_size: int, rank: int) -> None:
    """Shard MLP layers across tensor parallel dimension."""
    import torch.nn as nn
    
    # SwiGLUMLP uses: gate_proj, up_proj (column-parallel), down_proj (row-parallel)
    for proj_name in ['gate_proj', 'up_proj']:
        if hasattr(mlp, proj_name):
            old_proj = getattr(mlp, proj_name)
            out_features = old_proj.out_features
            in_features = old_proj.in_features
            
            chunk_size = out_features // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else out_features
            
            new_proj = nn.Linear(in_features, end_idx - start_idx, bias=old_proj.bias is not None)
            new_proj.weight.data = old_proj.weight.data[start_idx:end_idx].clone()
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data[start_idx:end_idx].clone()
            
            setattr(mlp, proj_name, new_proj)
    
    # down_proj: row-parallel (split input)
    if hasattr(mlp, 'down_proj'):
        old_proj = mlp.down_proj
        in_features = old_proj.in_features
        out_features = old_proj.out_features
        
        chunk_size = in_features // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else in_features
        
        new_proj = nn.Linear(end_idx - start_idx, out_features, bias=old_proj.bias is not None)
        new_proj.weight.data = old_proj.weight.data[:, start_idx:end_idx].clone()
        
        if old_proj.bias is not None:
            if rank == 0:
                new_proj.bias.data = old_proj.bias.data.clone()
            else:
                new_proj.bias = None
        
        setattr(mlp, 'down_proj', new_proj)


def shard_tensor_parallel_correctly(model: ArgonneModel, world_size: int, rank: int) -> None:
    """Properly shard the model for tensor parallelism."""
    if rank == 0:
        print(f"Sharding model for tensor parallelism (world_size={world_size}, rank={rank})")
    
    # Iterate through blocks and shard their components
    for block_idx, block in enumerate(model.blocks):
        # Shard attention
        if hasattr(block, 'attn'):
            shard_attention_layer(block.attn, world_size, rank)
        
        # Shard MLP
        if hasattr(block, 'mlp'):
            shard_mlp_layer(block.mlp, world_size, rank)
    
    if rank == 0:
        print(f"✓ Successfully sharded {len(model.blocks)} transformer blocks")


class TensorParallelModel(torch.nn.Module):
    """Wrapper for ArgonneModel that implements tensor parallelism."""
    def __init__(self, base_model: ArgonneModel, world_size: int, rank: int):
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
    
    def _block_forward(self, block, hidden_states, position_embeddings, attention_mask=None):
        """Forward pass for a single block with tensor parallelism."""
        # Attention with residual
        residual = hidden_states
        normed = block.input_norm(hidden_states)
        attn_output = block.attn(normed, position_embeddings, attention_mask)
        
        # All-reduce for row-parallel output projection
        if self.world_size > 1:
            dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)
        
        hidden_states = residual + attn_output
        
        # MLP with residual
        residual = hidden_states
        normed = block.post_norm(hidden_states)
        mlp_output = block.mlp(normed)
        
        # All-reduce for row-parallel down_proj
        if self.world_size > 1:
            dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM)
        
        hidden_states = residual + mlp_output
        
        return hidden_states
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass with correct tensor parallelism."""
        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Embeddings (replicated on all ranks)
        b, t = input_ids.size()
        hidden_states = self.base_model.embed_tokens(input_ids)
        
        # Get rotary embeddings
        cos, sin = self.base_model.rotary_emb(hidden_states, t)
        position_embeddings = (cos, sin)
        
        # Process through blocks
        for block in self.base_model.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._block_forward, 
                    block, 
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = self._block_forward(block, hidden_states, position_embeddings, attention_mask)
        
        # Final layer norm and output head (replicated)
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
    
    def generate(self, input_ids, max_length=1024, temperature=1.0, top_k=None, top_p=None, do_sample=True):
        """Distributed text generation that respects tensor parallel sharding."""
        was_training = self.training
        self.eval()

        try:
            if not torch.is_tensor(input_ids):
                raise TypeError("input_ids must be a torch.Tensor")

            input_ids = input_ids.to(self.device)

            # Make sure all ranks start with the same prompt when using tensor parallelism
            if self.world_size > 1 and dist.is_initialized():
                # Broadcast prompt length first to avoid shape mismatches
                prompt_length = torch.tensor([input_ids.shape[1]], device=self.device, dtype=torch.long)
                dist.broadcast(prompt_length, src=0)

                if input_ids.shape[1] != int(prompt_length.item()):
                    # Resize tensor for non-src ranks and broadcast prompt contents
                    new_prompt = torch.zeros(input_ids.shape[0], prompt_length.item(), dtype=input_ids.dtype, device=self.device)
                    if self.rank == 0:
                        new_prompt.copy_(input_ids[:, :prompt_length.item()])
                    dist.broadcast(new_prompt, src=0)
                    input_ids = new_prompt
                else:
                    dist.broadcast(input_ids, src=0)

            generated = input_ids

            with torch.no_grad():
                while generated.shape[1] < max_length:
                    context_window = generated[:, -self.base_model.config.max_position_embeddings :]
                    outputs = self.forward(context_window)
                    logits = outputs.logits[:, -1, :] / temperature

                    next_token: torch.Tensor
                    if do_sample:
                        if top_k is not None:
                            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
                        if top_p is not None:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits = logits.masked_fill(indices_to_remove, float("-inf"))

                        if self.rank == 0:
                            probs = F.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.empty((generated.size(0), 1), dtype=torch.long, device=self.device)
                    else:
                        if self.rank == 0:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        else:
                            next_token = torch.empty((generated.size(0), 1), dtype=torch.long, device=self.device)

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
        """Get state dict from base model"""
        return self.base_model.state_dict(*args, **kwargs)
    
    def parameters(self):
        """Get parameters from base model"""
        return self.base_model.parameters()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage"""
        self.gradient_checkpointing = True
        if self.rank == 0:
            print("✓ Gradient checkpointing enabled")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False


def train_from_scratch_tensor_parallel(
    data_glob: str,
    tokenizer_path: str,
    total_training_steps: int = DEFAULT_MAX_TRAINING_STEPS,
    block_size: int = 4096,
    initial_batch_size: int = 512,
    min_batch_size: int = 4,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    trust_remote_code: bool = False,
):
    """
    Train model from scratch with tensor parallelism and automatic batch size tuning.
    """
    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)
    init_tensor_parallel_group(world_size, rank)
    is_main_process = (rank == 0)
    
    if is_main_process:
        print("="*70)
        print("STARTING FRESH TRAINING FROM SCRATCH (TENSOR PARALLEL)")
        print("="*70)
        print(f"World size: {world_size} GPUs")
        print(f"Rank: {rank}")
    
    # Resolve data files
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    fallback_patterns = [
        os.path.join("data", "CC-MAIN-2025-26", "*.parquet"),
        os.path.join("..", "data", "*.arrow"),
        os.path.join("data", "*.arrow"),
    ]
    if data_glob != default_data_glob:
        fallback_patterns.insert(0, default_data_glob)
    data_files, used_patterns = resolve_data_files(data_glob, fallback_patterns=fallback_patterns)
    
    if is_main_process:
        print(f"Found {len(data_files)} data files")
        log_dataset_plan(data_files)

    # Load tokenizer
    validate_tokenizer_path(tokenizer_path)
    hf_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": hf_tokenizer.eos_token})
    hf_tokenizer.model_max_length = max(block_size + 1, 1_000_000_000)
    vocab_size = len(hf_tokenizer)

    # Build config - MUST MATCH resume_pretrain_tensor.py exactly
    config = ArgonneConfig(
        vocab_size=vocab_size,
        hidden_size=4080,  # 4080/24 = 170 exactly
        max_position_embeddings=block_size,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=8,  # CRITICAL: 24/8=3 groups, and 8 divides evenly by world_size=8
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_flash_attention=True,
        use_gradient_checkpointing=False,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        eos_token_id=hf_tokenizer.eos_token_id,
    )

    # Determine dtype
    supports_bf16 = False
    amp_dtype = torch.float32
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
        supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    # Batch size auto-tuning variables
    batch_size = initial_batch_size
    largest_successful_batch = None
    smallest_failed_batch = None
    
    # Data position tracking
    data_position = DataPosition(streaming=True)
    
    # Training state
    global_step = 0
    tokens_processed = 0
    model = None
    optimizer = None
    scaler = None
    
    # Main training loop with batch size adjustment
    while True:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"ATTEMPTING TRAINING WITH BATCH_SIZE = {batch_size}")
            print(f"{'='*70}")
        
        try:
            # Create fresh model for each attempt
            base_model = ArgonneModel(config)
            # Keep master weights in FP32 so AdamW maintains full-precision optimizer state
            
            # Create tensor parallel wrapper
            model = TensorParallelModel(base_model, world_size, rank)
            model.gradient_checkpointing_enable()
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr,
                weight_decay=weight_decay, 
                fused=False
            )
            
            # Setup scheduler
            scheduler = CosineWarmupScheduler(
                optimizer, 
                base_lr=lr, 
                warmup_steps=warmup_steps, 
                max_steps=total_training_steps, 
                min_lr=min_lr
            )
            
            if is_main_process:
                print(f"✓ Model initialized with tensor parallelism")
                print(f"  - Learning rate: {lr:.2e}")
                print(f"  - Batch size: {batch_size}")
                print(f"  - World size: {world_size}")
            
            # Setup mixed precision
            use_grad_scaler = amp_dtype == torch.float16 and torch.cuda.is_available()
            scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

            if is_main_process:
                if supports_bf16:
                    print("✓ Using torch.bfloat16 autocast")
                elif amp_dtype == torch.float16:
                    print("✓ Using torch.float16 autocast with GradScaler")

            first_device = model.device
            last_loss_value: Optional[float] = None

            # Reset training state
            global_step = 0
            tokens_processed = 0
            data_position = DataPosition(streaming=True)

            if is_main_process:
                print(f"\n{'='*70}")
                print(f"STARTING TRAINING")
                print(f"{'='*70}")
                print(f"Target steps: {total_training_steps}")
                print(f"{'='*70}\n")

            token_gen = streaming_token_generator(data_files, hf_tokenizer, block_size, rank=rank)
            token_buffer: List[List[int]] = []
            active_shard: Optional[str] = None

            pbar = tqdm(initial=global_step, total=total_training_steps, desc="Training") if is_main_process else None
            
            try:
                while global_step < total_training_steps:
                    try:
                        tokens, file_idx, position, shard_name, chunk_idx = next(token_gen)

                        if file_idx == -1:
                            if is_main_process:
                                print("End of dataset - restarting")
                            data_position.next_epoch()
                            token_gen = streaming_token_generator(data_files, hf_tokenizer, block_size, rank=rank)
                            continue

                        token_buffer.append(tokens)
                        data_position.update_streaming_position(file_idx, position, chunk_idx, data_files[file_idx])

                        if shard_name != active_shard:
                            active_shard = shard_name
                            if is_main_process:
                                print(f"Processing shard {file_idx + 1}/{len(data_files)}: {shard_name}")

                        if len(token_buffer) < batch_size:
                            continue

                        x_tens, y_tens = collate_batch(token_buffer, block_size)
                        token_buffer.clear()
                        if x_tens is None:
                            continue

                        batch_tokens = x_tens.numel()
                        tokens_processed += batch_tokens
                        current_lr = scheduler.step(global_step)

                        x_local = x_tens.to(first_device)
                        y_local = y_tens.to(first_device)
                        optimizer.zero_grad(set_to_none=True)

                        autocast_context = torch.amp.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available() else contextlib.nullcontext()

                        with autocast_context:
                            outputs = model(input_ids=x_local)
                            logits = outputs.logits
                            loss_tensor = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                y_local.view(-1),
                                ignore_index=-100,
                            )

                        last_loss_value = float(loss_tensor.detach().cpu().item())

                        if scaler is not None:
                            scaler.scale(loss_tensor).backward()
                            scaler.unscale_(optimizer)
                            _ensure_gradient_dtype_matches_params(model)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss_tensor.backward()
                            _ensure_gradient_dtype_matches_params(model)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()

                        global_step += 1
                        if pbar is not None:
                            pbar.update(1)

                        if global_step % 50 == 0 and last_loss_value is not None and is_main_process:
                            print(f"Step {global_step} | Loss: {last_loss_value:.4f} | Tokens: {tokens_processed:,} | LR: {current_lr:.6e}")

                        # Save checkpoint
                        if global_step % 300 == 0:
                            if is_main_process:
                                print(f"\nSaving checkpoint at step {global_step}...")

                            # Generate sample text on all ranks so tensor parallel reductions stay in sync
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(first_device)
                            generated = model.generate(
                                prompt_tensor,
                                max_length=prompt_tensor.shape[1] + 100,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.9,
                            )

                            if is_main_process:
                                generated_text = hf_tokenizer.decode(generated[0].tolist())
                                print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                            if is_main_process:
                                # Save checkpoint with format compatible with resume script
                                model_state = cast_state_dict_to_dtype(model.base_model.state_dict(), amp_dtype)
                                checkpoint_state = {
                                    "global_step": global_step,
                                    "tokens_processed": tokens_processed,
                                    "model_state_dict": model_state,
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "loss": last_loss_value,
                                    "data_position": data_position.get_state(),
                                    "model_dtype": str(amp_dtype),
                                    "tensor_parallel": True,
                                    "world_size": world_size,
                                    "rank": rank,
                                    "batch_size": batch_size,
                                }
                                os.makedirs("pretrained", exist_ok=True)
                                save_path = f"pretrained/streaming_checkpoint_step_{global_step}.pth"
                                safe_torch_save(checkpoint_state, save_path)
                                print(f"✓ Checkpoint saved: {save_path}")
                                print(f"  (Compatible with resume_pretrain_tensor.py)\n")

                    except StopIteration:
                        if is_main_process:
                            print("StopIteration - restarting dataset")
                        data_position.next_epoch()
                        token_gen = streaming_token_generator(data_files, hf_tokenizer, block_size, rank=rank)
                        continue
            finally:
                if pbar is not None:
                    pbar.close()

            # Training completed successfully
            if largest_successful_batch is None or batch_size > largest_successful_batch:
                largest_successful_batch = batch_size
            
            if is_main_process:
                print(f"\n{'='*70}")
                print(f"TRAINING COMPLETE")
                print(f"{'='*70}")
                print(f"Total tokens: {tokens_processed:,}")
                print(f"Final step: {global_step}")
                print(f"Optimal batch size: {batch_size}")
                print(f"Checkpoints saved to: pretrained/")
                print(f"Resume with: torchrun --nproc_per_node={world_size} resume_pretrain_tensor.py --tokenizer-path {tokenizer_path}")
            
            break  # Exit the retry loop
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_message = str(e)
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_message.lower()
            
            if not is_oom:
                # Not an OOM error, re-raise
                raise e
            
            if is_main_process:
                print(f"\n{'='*70}")
                print("CUDA OUT OF MEMORY DETECTED")
                print(f"{'='*70}")
                print(f"Error: {error_message}")
                if hasattr(e, "__traceback__"):
                    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                    print("".join(tb_lines))
            
            # Cleanup
            model = None
            optimizer = None
            scaler = None
            torch.cuda.empty_cache()
            
            if dist.is_initialized():
                dist.barrier()
            
            # Update batch size search bounds
            if smallest_failed_batch is None or batch_size < smallest_failed_batch:
                smallest_failed_batch = batch_size
            
            # Calculate new batch size
            if largest_successful_batch is None:
                new_batch_size = batch_size // 2
            else:
                new_batch_size = (largest_successful_batch + smallest_failed_batch) // 2
            
            new_batch_size = max(new_batch_size, min_batch_size)
            
            if new_batch_size == batch_size:
                if batch_size <= min_batch_size:
                    if is_main_process:
                        print(f"\n{'='*70}")
                        print(f"FATAL: Already at minimum batch size ({min_batch_size})")
                        print(f"Cannot reduce further. Training failed.")
                        print(f"{'='*70}")
                    raise RuntimeError(f"Training failed even with minimum batch size {min_batch_size}")
                new_batch_size = max(batch_size - 1, min_batch_size)
            
            if is_main_process:
                print(f"\n{'='*70}")
                print(f"REDUCING BATCH SIZE")
                print(f"{'='*70}")
                print(f"Previous: {batch_size}")
                print(f"New: {new_batch_size}")
                print(f"Search bounds: success={largest_successful_batch}, failed={smallest_failed_batch}")
                print(f"Retrying in 5 seconds...")
                print(f"{'='*70}\n")
            
            batch_size = new_batch_size
            time.sleep(5)
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Train Argonne model from scratch with tensor parallelism")
    default_data_glob = os.path.join("..", "data", "CC-MAIN-2025-26", "*.parquet")
    parser.add_argument(
        "--data-glob",
        type=str,
        default=default_data_glob,
        help="Glob pattern for parquet shards",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Directory containing the pretrained tokenizer.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_MAX_TRAINING_STEPS,
        help="Total number of training steps.",
    )
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        default=4,
        help="Initial batch size to try (will be reduced on OOM).",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=4,
        help="Minimum batch size before giving up.",
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
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_from_scratch_tensor_parallel(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        total_training_steps=args.total_steps,
        block_size=args.block_size,
        initial_batch_size=args.initial_batch_size,
        min_batch_size=args.min_batch_size,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
