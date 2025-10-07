import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutput


class ArgonneConfig(PretrainedConfig):
    """Configuration for the Argonne v2 family of models."""

    model_type = "argonne2"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        sliding_window: Optional[int] = None,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        pad_token_id = pad_token_id if pad_token_id is not None else kwargs.pop("pad_token_id", None)
        bos_token_id = bos_token_id if bos_token_id is not None else kwargs.pop("bos_token_id", None)
        eos_token_id = eos_token_id if eos_token_id is not None else kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # Backwards compatibility with Argonne 1.x naming.
        if "n_layer" in kwargs:
            num_hidden_layers = kwargs["n_layer"]
        if "n_head" in kwargs:
            num_attention_heads = kwargs["n_head"]
        if "n_embd" in kwargs:
            hidden_size = kwargs["n_embd"]
        if "block_size" in kwargs:
            max_position_embeddings = kwargs["block_size"]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads // 2
        )
        if self.num_key_value_heads < 1:
            self.num_key_value_heads = 1
        if num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        if intermediate_size is None:
            width = int(8 * hidden_size / 3)
            self.intermediate_size = ((width + 255) // 256) * 256
        else:
            self.intermediate_size = intermediate_size

        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_flash_attention = use_flash_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias

        if self.pad_token_id is None and self.eos_token_id is not None:
            self.pad_token_id = self.eos_token_id

        # Backwards compatibility aliases
        self.n_embd = self.hidden_size
        self.n_layer = self.num_hidden_layers
        self.n_head = self.num_attention_heads
        self.block_size = self.max_position_embeddings


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x.to(orig_dtype))


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device or inv_freq.device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is None:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ArgonneConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj._is_residual = True

        self.attention_dropout = config.attention_dropout
        self.use_flash_attention = config.use_flash_attention

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return x
        bsz, num_kv, seqlen, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, num_kv, self.num_key_value_groups, seqlen, head_dim)
        return x.reshape(bsz, num_kv * self.num_key_value_groups, seqlen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        if hasattr(F, "scaled_dot_product_attention") and self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
        else:
            scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is None:
                causal_mask = torch.triu(
                    torch.ones(seqlen, seqlen, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))
            else:
                scores = scores + attention_mask
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ArgonneConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.down_proj._is_residual = True
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Transformer block with GQA attention and SwiGLU feed-forward."""

    def __init__(self, config: ArgonneConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GroupedQueryAttention(config)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ArgonneModel(PreTrainedModel):
    config_class = ArgonneConfig
    _no_split_modules = ["Block"]

    def __init__(self, config: ArgonneConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = config.use_gradient_checkpointing
        self.pipeline_partitions: Optional[List[Tuple[int, int, torch.device]]] = None
        self.devices: List[torch.device] = []
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embed_tokens = new_embeddings
        self.config.vocab_size = new_embeddings.num_embeddings
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings
        if isinstance(new_embeddings, nn.Linear):
            self.config.vocab_size = new_embeddings.out_features

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.config.hidden_size ** -0.5
            if hasattr(module, "_is_residual"):
                std = (2 * self.config.num_hidden_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.hidden_size ** -0.5)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:
        self.set_gradient_checkpointing(False)

    def distribute_model(self, device_ids: Optional[List[str]] = None) -> None:
        if device_ids is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus < 1:
                raise ValueError("No CUDA devices available for distribution.")
            device_ids = [f"cuda:{i}" for i in range(num_gpus)]

        if not device_ids:
            raise ValueError("device_ids must contain at least one device identifier.")

        self.devices = [torch.device(d) for d in device_ids]
        num_blocks = len(self.blocks)

        # Start with an even distribution but make sure the last stage doesn't
        # become a hotspot. It already hosts the final RMSNorm and LM head, so
        # we bias one additional transformer block toward the previous stage
        # whenever possible to ease the memory footprint on the final GPU.
        per_device_counts = [num_blocks // len(self.devices)] * len(self.devices)
        for i in range(num_blocks % len(self.devices)):
            per_device_counts[i] += 1

        if len(self.devices) > 1:
            last_idx = len(self.devices) - 1
            penultimate_idx = last_idx - 1
            if per_device_counts[last_idx] > 1:
                per_device_counts[last_idx] -= 1
                per_device_counts[penultimate_idx] += 1

        partitions: List[Tuple[int, int, torch.device]] = []
        start_idx = 0
        for device, block_count in zip(self.devices, per_device_counts):
            if block_count <= 0 or start_idx >= num_blocks:
                continue
            end_idx = min(start_idx + block_count, num_blocks)
            for block in self.blocks[start_idx:end_idx]:
                block.to(device)
            partitions.append((start_idx, end_idx, device))
            start_idx = end_idx

        if not partitions:
            partitions.append((0, num_blocks, self.devices[0]))

        self.pipeline_partitions = partitions

        first_device = self.devices[0]
        last_device = self.devices[-1]
        self.embed_tokens = self.embed_tokens.to(first_device)
        self.rotary_emb = self.rotary_emb.to(first_device)
        self.norm = self.norm.to(last_device)

        if self.config.tie_word_embeddings and len(self.devices) > 1:
            untied_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            untied_head.to(last_device)
            with torch.no_grad():
                untied_head.weight.copy_(self.embed_tokens.weight.to(last_device))
            self.lm_head = untied_head
            self.config.tie_word_embeddings = False
        else:
            self.lm_head = self.lm_head.to(last_device)

        print(f"Model distributed across {len(self.devices)} devices.")
        for idx, (start, end, device) in enumerate(self.pipeline_partitions):
            print(f"  Stage {idx}: layers {start}-{end - 1} on {device}")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutput:
        batch_size, seq_length = input_ids.shape

        if self.pipeline_partitions:
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.devices[0])

            hidden_states = self.embed_tokens(input_ids.to(self.devices[0]))
            cos, sin = self.rotary_emb(hidden_states, seq_length)

            for start, end, device in self.pipeline_partitions:
                if hidden_states.device != device:
                    hidden_states = hidden_states.to(device)
                rotary = (cos.to(device), sin.to(device))
                attn_mask = attention_mask.to(device) if attention_mask is not None else None

                for layer in self.blocks[start:end]:
                    if self.gradient_checkpointing and self.training:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            rotary,
                            attn_mask,
                            use_reentrant=False,
                        )
                    else:
                        hidden_states = layer(hidden_states, rotary, attn_mask)

            hidden_states = hidden_states.to(self.devices[-1])
        else:
            device = self.embed_tokens.weight.device
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            hidden_states = self.embed_tokens(input_ids.to(device))
            cos, sin = self.rotary_emb(hidden_states, seq_length)
            rotary = (cos, sin)

            for layer in self.blocks:
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        rotary,
                        attention_mask,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = layer(hidden_states, rotary, attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if shift_labels.device != shift_logits.device:
                shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        device = self.devices[0] if self.pipeline_partitions else self.embed_tokens.weight.device
        input_ids = input_ids.to(device)
        while input_ids.shape[1] < max_length:
            chunk = input_ids[:, -self.config.max_position_embeddings :]
            outputs = self.forward(chunk)
            logits = outputs.logits[:, -1, :] / temperature

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
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if input_ids.shape[1] >= max_length:
                break
        return input_ids.to(device)


AutoConfig.register("argonne2", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModel)

# Backwards compatibility exports
CausalSelfAttention = GroupedQueryAttention
MLP = SwiGLUMLP
