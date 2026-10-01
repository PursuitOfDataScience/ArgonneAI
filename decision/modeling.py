"""DecisionModel = the argonne-4.5-base-ctx13568 trunk + a pointer head over the options.

The trunk runs once over the sequence (right-padded batches are exact: attention is causal, so a real token
never sees the padding after it). The head never touches the 151k-vocab LM head: it projects the hidden state
at the answer token (query) and at each option's closing newline (keys) into a small space and scores
option k by q . k_k / sqrt(r). A softmax over the options that were given is the answer distribution, so the
output cannot leave the schema and the probabilities come straight from the model's internal representations.
"""
import json
import math
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from model import ArgonneConfig, ArgonneModel  # noqa: E402  (the base's own architecture code)

HEAD_FILE = "decision_head.pt"
CONFIG_FILE = "decision_config.json"


class DecisionHead(nn.Module):
    def __init__(self, hidden: int, rank: int = 256):
        super().__init__()
        self.rank = rank
        self.q = nn.Linear(hidden, rank, bias=False)
        self.k = nn.Linear(hidden, rank, bias=False)
        nn.init.normal_(self.q.weight, std=0.02)
        nn.init.normal_(self.k.weight, std=0.02)

    def forward(self, h_ans, h_opt, opt_mask):
        """h_ans (B, d), h_opt (B, K, d), opt_mask (B, K) bool -> logits (B, K) float32, -inf where masked.
        Always fp32, outside autocast: these few flops decide every probability the model reports."""
        with torch.autocast(device_type=h_ans.device.type, enabled=False):
            q = F.linear(h_ans.float(), self.q.weight.float())
            k = F.linear(h_opt.float(), self.k.weight.float())
            logits = torch.einsum("br,bkr->bk", q, k) / math.sqrt(self.rank)
        return logits.masked_fill(~opt_mask, float("-inf"))


class DecisionModel(nn.Module):
    def __init__(self, trunk: ArgonneModel, head: DecisionHead):
        super().__init__()
        self.trunk = trunk
        self.head = head
        self.grad_ckpt = False

    # ---- trunk ----------------------------------------------------------------------------------------------
    def hidden(self, input_ids, past_key_values=None, use_cache=False):
        """Final-norm hidden states (B, T, d); optionally with a KV cache so several question blocks can share
        one pass over the state."""
        m = self.trunk
        x = m.embed_tokens(input_ids)
        t = input_ids.shape[1]
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        cos, sin = m.rotary_emb(x, past_len + t)
        rotary = (cos[past_len:past_len + t], sin[past_len:past_len + t])
        new_cache = [] if use_cache else None
        for i, layer in enumerate(m.blocks):
            if use_cache:
                x, kv = layer(x, rotary, None, past_kv=past_key_values[i] if past_key_values else None,
                              use_cache=True)
                new_cache.append(kv)
            elif self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, rotary, None, use_reentrant=False)
            else:
                x = layer(x, rotary, None)
        x = m.norm(x)
        return (x, new_cache) if use_cache else x

    # ---- decisions ------------------------------------------------------------------------------------------
    def forward(self, input_ids, opt_pos, opt_mask, ans_pos):
        """input_ids (B, T); opt_pos (B, K) long, 0 where padded; opt_mask (B, K) bool; ans_pos (B,) ->
        option logits (B, K)."""
        h = self.hidden(input_ids)
        b = torch.arange(h.shape[0], device=h.device)
        h_ans = h[b, ans_pos]                                   # (B, d)
        h_opt = h[b[:, None], opt_pos]                          # (B, K, d)
        return self.head(h_ans, h_opt, opt_mask)

    # ---- persistence ----------------------------------------------------------------------------------------
    def save(self, out_dir, extra_config=None, trunk_dtype=torch.bfloat16, tokenizer_dir=None):
        """Write the trunk as a normal HF dir (loadable by the base's model.py) plus the head and its config.
        Writes into a temp dir and renames, so a crash never leaves a half-written model."""
        tmp = out_dir.rstrip("/") + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        sd = {k: v.detach().to("cpu", trunk_dtype) for k, v in self.trunk.state_dict().items()
              if not (k == "lm_head.weight" and self.trunk.config.tie_word_embeddings)}   # tied: re-tied on load
        self.trunk.save_pretrained(tmp, state_dict=sd, safe_serialization=True)
        shutil.copy(os.path.join(REPO, "model.py"), os.path.join(tmp, "model.py"))
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if tokenizer_dir and os.path.exists(os.path.join(tokenizer_dir, name)):
                shutil.copy(os.path.join(tokenizer_dir, name), os.path.join(tmp, name))
        torch.save({k: v.detach().cpu().float() for k, v in self.head.state_dict().items()},
                   os.path.join(tmp, HEAD_FILE))
        cfg = {"format": "argonne-decision/1", "head_rank": self.head.rank,
               "max_len": getattr(self.trunk.config, "max_position_embeddings", None), "temperatures": {}}
        cfg.update(extra_config or {})
        with open(os.path.join(tmp, CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)

    @classmethod
    def from_base(cls, base_dir, rank=256, dtype=torch.float32):
        trunk = _load_trunk(base_dir, dtype)
        return cls(trunk, DecisionHead(trunk.config.hidden_size, rank))

    @classmethod
    def load(cls, model_dir, dtype=torch.bfloat16):
        with open(os.path.join(model_dir, CONFIG_FILE)) as f:
            cfg = json.load(f)
        trunk = _load_trunk(model_dir, dtype)
        head = DecisionHead(trunk.config.hidden_size, cfg["head_rank"])
        head.load_state_dict(torch.load(os.path.join(model_dir, HEAD_FILE), map_location="cpu"))
        model = cls(trunk, head)
        model.decision_config = cfg
        return model


def local_dir(ref):
    """A local directory for a model reference: the path itself, or a Hugging Face Hub snapshot of that repo id."""
    if os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download
    return snapshot_download(ref)


def _load_trunk(path, dtype):
    """transformers >= 4.56 renamed torch_dtype to dtype; older versions only take torch_dtype.
    A saved decision model drops the tied lm_head (the decision head never reads it), and transformers would report
    it as MISSING and newly initialised on every load; its loading report is silenced here and the tie restored."""
    from transformers.utils import logging as hf_logging
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        try:
            trunk = ArgonneModel.from_pretrained(path, dtype=dtype)
        except TypeError:
            trunk = ArgonneModel.from_pretrained(path, torch_dtype=dtype)
    finally:
        hf_logging.set_verbosity(level)
    if getattr(trunk.config, "tie_word_embeddings", False) and hasattr(trunk, "tie_weights"):
        trunk.tie_weights()
    return trunk


def decision_loss(logits, labels=None, soft=None):
    """Mean over examples of -sum_k t_k log p_k; t = one-hot(labels) or a soft target (rows sum to 1).
    Log loss is a strictly proper scoring rule, so minimising it rewards calibrated probabilities."""
    logp = F.log_softmax(logits, dim=-1)
    if soft is not None:
        t = soft.to(logp.dtype)
        return -(t * logp.masked_fill(t == 0, 0.0)).sum(-1).mean()
    return F.nll_loss(logp, labels)
