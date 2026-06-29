import math
import importlib.util
import os
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
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast


flash_attn_func = None
_flash_attn_available = False
if importlib.util.find_spec("flash_attn") is not None:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        _flash_attn_available = True
    except ImportError:
        _flash_attn_available = False

# One-time startup log of which attention kernel is in effect, because the
# sliding-window local attention is only implemented on the flash-attn path
# and silently degrades to full attention on the SDPA/math fallbacks.
_attention_path_logged = False


class ArgonneConfig(PretrainedConfig):
    """Configuration for the Argonne v3.0 family of models."""

    model_type = "argonne2"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 3072,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = 4,
        intermediate_size: Optional[int] = 8192,
        max_position_embeddings: int = 1024,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        sliding_window: Optional[int] = None,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False,
        qk_norm: bool = True,
        v_norm: bool = True,
        sandwich_norm: bool = True,
        z_loss_weight: float = 0.0,
        mtp_horizon: int = 1,
        mtp_loss_weight: float = 0.0,
        interleaved_local_attention: bool = True,
        local_attention_window: Optional[int] = 256,
        logit_softcap: float = 15.0,
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
        self.qk_norm = qk_norm
        self.v_norm = v_norm
        self.sandwich_norm = sandwich_norm
        self.z_loss_weight = z_loss_weight
        self.mtp_horizon = max(1, int(mtp_horizon))
        self.mtp_loss_weight = float(mtp_loss_weight)
        self.interleaved_local_attention = bool(interleaved_local_attention)
        self.local_attention_window = (
            int(local_attention_window) if local_attention_window is not None and int(local_attention_window) > 0 else None
        )
        self.logit_softcap = float(logit_softcap)
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

        # Always compute inv_freq on CPU to avoid meta-device issues during from_pretrained
        # (low_cpu_mem_usage=True initializes model on meta device, which would give uninitialized values)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, torch.device("cpu"), torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Always store cache in float32 to avoid precision loss when model dtype is bfloat16
        # (large freqs lose precision in bfloat16, corrupting cos/sin values)
        self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )

    def init_buffers(self) -> None:
        """Re-initialize inv_freq, cos_cached, sin_cached. Call after from_pretrained
        to fix meta-device corruption (low_cpu_mem_usage=True destroys buffer values)."""
        device = self.inv_freq.device
        if device.type == "meta":
            device = torch.device("cpu")
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu") / self.dim)
        )
        self.inv_freq = inv_freq.to(device)
        self._set_cos_sin_cache(
            self.max_position_embeddings,
            device,
            torch.float32,
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
        self.qk_norm = config.qk_norm
        self.v_norm = config.v_norm

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
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.v_norm:
            self.v_norm_layer = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, seqlen, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.v_norm:
            value = self.v_norm_layer(value)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # ---- KV cache (inference only; training keeps past_kv=None/use_cache=False) ----
        # Cache pre-repeat (num_kv_heads), post-rotary K/V. The SDPA/math path
        # ignores the sliding window (full causal), so storing all keys and having
        # the new query attend to everything reproduces the no-cache result.
        is_decode = past_kv is not None
        if is_decode:
            past_k, past_v = past_kv
            key = torch.cat([past_k, key], dim=2)
            value = torch.cat([past_v, value], dim=2)
        new_kv = (key, value) if use_cache else None
        kv_len = key.shape[2]

        # Additive mask used only when decoding with a cache: query at absolute
        # position (kv_len - seqlen + i) attends to keys j <= that position.
        decode_mask = None
        if is_decode:
            q_pos = torch.arange(kv_len - seqlen, kv_len, device=hidden_states.device)
            k_pos = torch.arange(kv_len, device=hidden_states.device)
            allowed = k_pos[None, :] <= q_pos[:, None]
            decode_mask = torch.zeros(seqlen, kv_len, dtype=query.dtype, device=hidden_states.device)
            decode_mask = decode_mask.masked_fill(~allowed, -65504.0)[None, None]

        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        use_flash_attn_2 = (
            not is_decode
            and _flash_attn_available
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
                if decode_mask is not None:
                    attn_output = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=decode_mask,
                        dropout_p=0.0, is_causal=False,
                    )
                elif attention_mask is None:
                    # Prefill / training: fast causal path.
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
            mask_value = -65504.0  # Large negative instead of -inf
            if decode_mask is not None:
                scores = scores + decode_mask
            else:
                # Apply causal mask - use large negative instead of -inf for stability
                causal_mask = torch.triu(
                    torch.ones(seqlen, kv_len, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask, mask_value)
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
        out = self.o_proj(attn_output)
        if use_cache:
            return out, new_kv
        return out


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
        self.sandwich_norm = config.sandwich_norm
        if self.sandwich_norm:
            self.attn_out_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_out_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_out = self.attn(
            hidden_states, position_embeddings, attention_mask,
            past_kv=past_kv, use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            hidden_states, new_kv = attn_out
        else:
            hidden_states = attn_out
        if self.sandwich_norm:
            hidden_states = self.attn_out_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.sandwich_norm:
            hidden_states = self.mlp_out_norm(hidden_states)
        hidden_states = residual + hidden_states
        if use_cache:
            return hidden_states, new_kv

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
        if config.interleaved_local_attention and config.local_attention_window is not None:
            for idx, block in enumerate(self.blocks):
                block.attn.sliding_window = config.local_attention_window if (idx % 2 == 1) else None
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
        self._nan_loss_count = 0
        self.post_init()

        global _attention_path_logged
        rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
        if not _attention_path_logged and rank == "0":
            _attention_path_logged = True
            window = (
                config.local_attention_window if config.interleaved_local_attention else None
            )
            if _flash_attn_available and config.use_flash_attention:
                note = (
                    f"flash-attn-2; sliding window {window} active on odd layers"
                    if window is not None
                    else "flash-attn-2; full attention"
                )
            else:
                reason = (
                    "flash-attn unavailable"
                    if config.use_flash_attention
                    else "use_flash_attention=False"
                )
                note = f"SDPA/math ({reason}); full attention"
                if window is not None:
                    note += (
                        f" — local_attention_window={window} is configured but"
                        " IGNORED on this path"
                    )
            print(f"[ArgonneModel] attention path: {note}", flush=True)

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

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Wrap PreTrainedModel.from_pretrained to self-heal after loading.

        from_pretrained may materialize the model via the meta device
        (low_cpu_mem_usage), which fills the rotary embedding's non-persistent
        buffers with uninitialized memory and can drop the
        embed_tokens/lm_head tie. Callers previously had to remember to call
        init_buffers() and tie_weights() by hand; do it here so every loader
        gets a working model.
        """
        model = super().from_pretrained(*args, **kwargs)
        for module in model.modules():
            if isinstance(module, RotaryEmbedding):
                module.init_buffers()
        model.tie_weights()
        return model

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
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,  # Accept extra args from newer transformers (e.g., num_items_in_batch)
    ) -> CausalLMOutput:
        _, seq_length = input_ids.shape

        device = self.embed_tokens.weight.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        hidden_states = self.embed_tokens(input_ids)

        # The training path does not use attention masks.
        attention_mask = None

        # RoPE positions are offset by the cached length so incremental decode
        # uses the correct absolute positions.
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        cos_full, sin_full = self.rotary_emb(hidden_states, past_len + seq_length)
        rotary = (cos_full[past_len:past_len + seq_length],
                  sin_full[past_len:past_len + seq_length])

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    rotary,
                    attention_mask,
                    use_reentrant=False,
                )
            elif use_cache:
                past_kv = past_key_values[i] if past_key_values else None
                hidden_states, layer_kv = layer(
                    hidden_states, rotary, attention_mask,
                    past_kv=past_kv, use_cache=True,
                )
                new_cache.append(layer_kv)
            else:
                hidden_states = layer(hidden_states, rotary, attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if self.config.logit_softcap > 0:
            cap = self.config.logit_softcap
            logits = torch.tanh(logits / cap) * cap
        
        # Sanitize logits only at inference. During training the full-tensor
        # scan forces a GPU sync every step over [batch, seq, vocab]; a NaN
        # would surface in the loss check below anyway.
        if not self.training and torch.isnan(logits).any():
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
            if self.training and self.config.mtp_horizon > 1:
                mtp_terms = []
                max_horizon = min(self.config.mtp_horizon, labels.shape[1])
                for horizon in range(2, max_horizon + 1):
                    shift = horizon - 1
                    shifted_logits = logits[:, :-shift, :]
                    shifted_labels = labels[:, shift:]
                    if shifted_logits.numel() == 0:
                        continue
                    mtp_loss = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.size(-1)),
                        shifted_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    mtp_terms.append(mtp_loss / horizon)
                if mtp_terms:
                    loss = loss + (self.config.mtp_loss_weight * torch.stack(mtp_terms).mean()).to(loss.dtype)
            if self.config.z_loss_weight > 0:
                z = torch.logsumexp(logits.float(), dim=-1)
                loss = loss + (self.config.z_loss_weight * z.pow(2).mean()).to(loss.dtype)

            # A NaN loss becomes a zero loss that is still connected to every
            # parameter — but through the parameters directly, NOT through the
            # network's (possibly NaN) activations. Backward then delivers an
            # exact zero gradient to each parameter: DDP/FSDP gradient hooks
            # all fire (no collective desync/hang) and the optimizer step is a
            # true no-op. Warn (bounded) instead of hiding it.
            if torch.isnan(loss):
                self._nan_loss_count += 1
                if self._nan_loss_count <= 5 or self._nan_loss_count % 100 == 0:
                    print(
                        f"WARNING: NaN loss detected (occurrence {self._nan_loss_count}); "
                        "zeroing this step's loss.",
                        flush=True,
                    )
                zero = torch.zeros((), device=loss.device, dtype=torch.float32)
                for p in self.parameters():
                    if p.requires_grad:
                        zero = zero + torch.nan_to_num(p).sum().float() * 0.0
                loss = zero.to(loss.dtype)

        if use_cache:
            return CausalLMOutputWithPast(logits=logits, loss=loss, past_key_values=new_cache)
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
        ctx = self.config.max_position_embeddings
        # KV-cache decode: prefill the prompt once, then feed a single token per
        # step reusing past_key_values. This yields identical logits to the
        # recompute-the-whole-prefix path (gated by verify_cache.py) but turns the
        # per-step cost from O(seq_len) down to O(1). When the running sequence
        # would exceed the context window we rebuild the cache from the truncated
        # window, matching the original chunk = input_ids[:, -ctx:] behavior.
        past = None
        outputs = None
        while input_ids.shape[1] < max_length:
            if past is None or past[0][0].shape[2] >= ctx:
                chunk = input_ids[:, -ctx:]
                outputs = self.forward(chunk, use_cache=True)
                past = outputs.past_key_values
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

            next_token = next_token.to(input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if input_ids.shape[1] >= max_length:
                break
            # Advance the cache by one token unless we must rebuild it next loop
            # (cache full); the top-of-loop guard re-prefills in that case.
            if past[0][0].shape[2] < ctx:
                outputs = self.forward(next_token, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
        return input_ids.to(device)


AutoConfig.register("argonne2", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModel)

# Backwards compatibility exports
CausalSelfAttention = GroupedQueryAttention
MLP = SwiGLUMLP
