import math
from typing import Optional, Tuple

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


try:
    from flash_attn.flash_attn_interface import flash_attn_func

    _flash_attn_available = True
except ImportError:
    flash_attn_func = None
    _flash_attn_available = False


class ArgonneConfig(PretrainedConfig):
    """Configuration for the Argonne v2.5 family of models."""

    model_type = "argonne2"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
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
        # Clamp values to prevent overflow in pow(2)
        x = torch.clamp(x, min=-65504.0, max=65504.0)
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
        self.sliding_window = config.sliding_window

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

        use_flash_attn_2 = (
            _flash_attn_available
            and self.use_flash_attention
            and attention_mask is None
            and query.dtype in (torch.float16, torch.bfloat16)
            and self.head_dim % 4 == 0
        )
        use_scaled_dot = (
            hasattr(F, "scaled_dot_product_attention")
            and self.use_flash_attention
            and query.dtype in (torch.float16, torch.bfloat16)
            and self.head_dim % 4 == 0
        )

        attn_output = None
        if use_flash_attn_2:
            try:
                flash_dropout = self.attention_dropout if self.training else 0.0
                window = (
                    (self.sliding_window, self.sliding_window)
                    if self.sliding_window is not None
                    else (-1, -1)
                )
                q = query.transpose(1, 2).contiguous()
                k = key.transpose(1, 2).contiguous()
                v = value.transpose(1, 2).contiguous()
                attn_output = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=flash_dropout,
                    softmax_scale=None,
                    causal=True,
                    window_size=window,
                ).transpose(1, 2)
            except RuntimeError:
                attn_output = None

        if attn_output is None and use_scaled_dot:
            try:
                # Use is_causal=True when no attention_mask (faster Flash Attention path)
                # When attention_mask is provided, we need to combine it with causal masking
                if attention_mask is None:
                    attn_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=None,
                        dropout_p=self.attention_dropout if self.training else 0.0,
                        is_causal=True,
                    )
                else:
                    # With attention_mask: need to pass it explicitly (slower but correct)
                    # attention_mask should be 4D: (bsz, 1, seq, seq) or broadcastable
                    attn_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=self.attention_dropout if self.training else 0.0,
                        is_causal=False,  # Mask already includes causal component
                    )
            except RuntimeError:
                # Fallback to math attention when kernels are unavailable
                attn_output = None

        if attn_output is None:
            scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            # Apply causal mask - use large negative instead of -inf for numerical stability
            causal_mask = torch.triu(
                torch.ones(seqlen, seqlen, dtype=torch.bool, device=hidden_states.device),
                diagonal=1,
            )
            mask_value = -65504.0  # Large negative instead of -inf
            scores = scores.masked_fill(causal_mask, mask_value)
            # Apply attention_mask if provided
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.num_heads * self.head_dim)
        )
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
        # Clamp intermediate values to prevent overflow
        gate = self.gate_proj(x)
        gate = torch.clamp(gate, min=-65504.0, max=65504.0)
        up = self.up_proj(x)
        up = torch.clamp(up, min=-65504.0, max=65504.0)
        return self.dropout(self.down_proj(F.silu(gate) * up))


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
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

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

    def tie_weights(self, **kwargs) -> None:
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,  # Accept extra args from newer transformers (e.g., num_items_in_batch)
    ) -> CausalLMOutput:
        bsz, seq_length = input_ids.shape

        device = self.embed_tokens.weight.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            if attention_mask.device != device:
                attention_mask = attention_mask.to(device)

            if attention_mask.dim() == 2:
                # Convert a (B, L) validity mask into an additive (B, 1, L, L)
                # block that preserves causal direction and valid-token masking.
                valid = attention_mask.to(torch.bool)
                pair = valid.unsqueeze(1) & valid.unsqueeze(2)
                causal = torch.tril(
                    torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
                ).unsqueeze(0).unsqueeze(0)
                allow = pair.unsqueeze(1) & causal
                additive = torch.zeros(
                    (bsz, 1, seq_length, seq_length),
                    dtype=hidden_states.dtype,
                    device=device,
                )
                additive.masked_fill_(~allow, -65504.0)
                attention_mask = additive
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() != 4:
                raise ValueError(
                    f"attention_mask must have 2, 3, or 4 dims, got shape={tuple(attention_mask.shape)}"
                )

            if attention_mask.dtype == torch.bool:
                additive = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
                additive.masked_fill_(~attention_mask, -65504.0)
                attention_mask = additive
            elif attention_mask.dtype != hidden_states.dtype:
                attention_mask = attention_mask.to(hidden_states.dtype)

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
        
        # Check for NaN in logits and handle gracefully
        if torch.isnan(logits).any():
            # Replace NaN with zeros to prevent cascading failures
            logits = torch.nan_to_num(logits, nan=0.0, posinf=65504.0, neginf=-65504.0)

        loss = None
        if labels is not None:
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # Handle NaN loss
            if torch.isnan(loss):
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)

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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        self.eval()
        device = self.embed_tokens.weight.device
        input_ids = input_ids.to(device)
        while input_ids.shape[1] < max_length:
            chunk = input_ids[:, -self.config.max_position_embeddings :]
            outputs = self.forward(chunk)
            logits = outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for b in range(input_ids.shape[0]):
                    seen_tokens = torch.unique(input_ids[b])
                    seen_logits = logits[b, seen_tokens]
                    adjusted = torch.where(
                        seen_logits < 0,
                        seen_logits * repetition_penalty,
                        seen_logits / repetition_penalty,
                    )
                    logits[b, seen_tokens] = adjusted

            if no_repeat_ngram_size > 0 and input_ids.shape[1] + 1 >= no_repeat_ngram_size:
                n = int(no_repeat_ngram_size)
                for b in range(input_ids.shape[0]):
                    seq = input_ids[b].tolist()
                    ngrams = {}
                    for i in range(len(seq) - n + 1):
                        prefix = tuple(seq[i : i + n - 1]) if n > 1 else tuple()
                        next_token = seq[i + n - 1]
                        if prefix not in ngrams:
                            ngrams[prefix] = set()
                        ngrams[prefix].add(next_token)
                    current_prefix = tuple(seq[-(n - 1) :]) if n > 1 else tuple()
                    banned = ngrams.get(current_prefix, set())
                    if banned:
                        logits[b, list(banned)] = float("-inf")

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

            input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
            if input_ids.shape[1] >= max_length:
                break
        return input_ids.to(device)


AutoConfig.register("argonne2", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModel)

# Backwards compatibility exports
CausalSelfAttention = GroupedQueryAttention
MLP = SwiGLUMLP
