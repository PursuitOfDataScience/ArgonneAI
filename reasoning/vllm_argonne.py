"""vLLM 0.11.2 custom-model port of ArgonneModel (argonne2) — §22 fast-inference engine.

Reproduces the deployed model EXACTLY (see reasoning/thinking_training.md §0/§16):
  * GQA 12 query / 4 KV heads, head_dim=256, RoPE θ=1e6 (NeoX).
  * qk-norm (RMSNorm on q AND k, per head) — like Qwen3.
  * v-norm (RMSNorm on the value, per head) — NOVEL to Argonne; applied before attention.
  * sandwich norm (input/post-attn/pre-mlp/post-mlp) — like Gemma2's 4-norm layer.
  * final logit softcap tanh(x/15)*15 (NO attention-logit softcap).
  * tied embeddings (no lm_head in checkpoint).
  * FULL CAUSAL on every layer — the config's local_attention_window=256 is IGNORED at
    runtime in the reference (§16), so we do NOT use sliding window (would diverge).
  * NO embedding scaling (unlike Gemma2).

Register + config (do this BEFORE LLM(...), in the driver / launcher):
    import sys; sys.path.insert(0, "/home/youzhi/ArgonneAI"); sys.path.insert(0, ".../reasoning")
    from transformers import AutoConfig
    from model import ArgonneConfig            # root model.py
    AutoConfig.register("argonne2", ArgonneConfig)
    from vllm import ModelRegistry
    ModelRegistry.register_model("ArgonneModel", "vllm_argonne:ArgonneForCausalLM")
`register()` at the bottom does exactly this — call it, or use it as a vllm.general_plugins entrypoint.
"""
from collections.abc import Iterable

import torch
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors


class ArgonneAttention(nn.Module):
    def __init__(self, *, config, cache_config, quant_config, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads  # 256
        self.q_size = self.total_num_heads * self.head_dim
        self.kv_size = self.total_num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size, self.head_dim, self.total_num_heads,
            self.total_num_kv_heads, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, self.hidden_size,
            bias=config.attention_bias, quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        # Argonne per-head norms (RMSNorm over head_dim). weight*x form (not Gemma's 1+w).
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else None
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.v_norm else None

        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta, is_neox_style=True)

        # FULL causal — no sliding window (§16: config window ignored at runtime),
        # no attention-logit softcap (Argonne only softcaps the FINAL logits).
        self.attn = Attention(
            self.total_num_heads, self.head_dim, self.scaling,
            num_kv_heads=self.total_num_kv_heads, cache_config=cache_config,
            quant_config=quant_config, prefix=f"{prefix}.attn")

    def _per_head_norm(self, x, norm):
        by_head = x.view(*x.shape[:-1], x.shape[-1] // self.head_dim, self.head_dim)
        by_head = norm(by_head)
        return by_head.view(x.shape)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm is not None:
            q = self._per_head_norm(q, self.q_norm)
            k = self._per_head_norm(k, self.k_norm)
        if self.v_norm is not None:
            v = self._per_head_norm(v, self.v_norm)  # value is NOT roped
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v)
        out, _ = self.o_proj(attn_out)
        return out


class ArgonneMLP(nn.Module):
    def __init__(self, *, config, quant_config, prefix=""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=config.mlp_bias,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()  # silu(gate)*up

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class ArgonneDecoderLayer(nn.Module):
    """Gemma2-style sandwich norm (mathematically identical to Argonne's Block):
    input_layernorm=input_norm, post_attention_layernorm=attn_out_norm,
    pre_feedforward_layernorm=post_norm, post_feedforward_layernorm=mlp_out_norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.self_attn = ArgonneAttention(
            config=config, cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.self_attn")
        self.mlp = ArgonneMLP(config=config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)  # sandwich (1-arg)
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # sandwich (1-arg)
        return hidden_states, residual


@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0, "positions": -1,
    "intermediate_tensors": 0, "inputs_embeds": 0,
})
class ArgonneInnerModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ArgonneDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class ArgonneForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = ArgonneInnerModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # tied embeddings -> lm_head reuses the input embedding weights
        self.lm_head = self.model.embed_tokens
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=getattr(config, "logit_softcap", None) or None)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    # ---- weight loading: map Argonne checkpoint names -> vLLM module names ----
    _RENAME = [
        (".attn.v_norm_layer.", ".self_attn.v_norm."),   # BEFORE the generic .attn. rule
        (".attn.", ".self_attn."),
        (".input_norm.", ".input_layernorm."),
        (".attn_out_norm.", ".post_attention_layernorm."),
        (".post_norm.", ".pre_feedforward_layernorm."),
        (".mlp_out_norm.", ".post_feedforward_layernorm."),
    ]
    _STACKED = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            if name.startswith("lm_head."):
                continue  # tied — reuses embed_tokens
            if name == "embed_tokens.weight":
                name = "model.embed_tokens.weight"
            elif name == "norm.weight":
                name = "model.norm.weight"
            elif name.startswith("blocks."):
                name = "model.layers." + name[len("blocks."):]
                for a, b in self._RENAME:
                    name = name.replace(a, b)
            matched = False
            for pname, sname, sid in self._STACKED:
                if sname in name:
                    tgt = name.replace(sname, pname)
                    if tgt not in params:
                        continue
                    p = params[tgt]
                    p.weight_loader(p, w, sid)
                    loaded.add(tgt)
                    matched = True
                    break
            if matched:
                continue
            if name in params:
                p = params[name]
                getattr(p, "weight_loader", default_weight_loader)(p, w)
                loaded.add(name)
        return loaded


def _shim_tokenizer_for_vllm():
    """transformers 5.x removed `all_special_tokens_extended`, which vLLM 0.11.2's
    get_cached_tokenizer still reads. Reconstruct it (the AddedToken objects for the
    special tokens, from added_tokens_decoder) so vLLM can build the tokenizer."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    def _aste(self):
        dec = getattr(self, "added_tokens_decoder", {}) or {}
        out = [dec[i] for i in self.all_special_ids if i in dec]
        return out if out else list(self.all_special_tokens)

    PreTrainedTokenizerBase.all_special_tokens_extended = property(_aste)


def register():
    """Register the config + model with transformers and vLLM. Call before LLM(...)."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    rdir = str(Path(__file__).resolve().parent)
    for p in (rdir, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    _shim_tokenizer_for_vllm()
    from transformers import AutoConfig
    from model import ArgonneConfig  # root model.py
    try:
        AutoConfig.register("argonne2", ArgonneConfig)
    except ValueError:
        pass  # already registered
    from vllm import ModelRegistry
    ModelRegistry.register_model("ArgonneModel", "vllm_argonne:ArgonneForCausalLM")


if __name__ == "__main__":
    register()
    print("registered ArgonneModel (argonne2) with vLLM")
