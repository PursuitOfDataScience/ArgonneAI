#!/usr/bin/env python3
"""Verify KV-cache decoding matches the no-cache forward on the real model,
and measure the speedup. Gate before using the cache for STaR."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArgonneConfig, ArgonneModel  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/project/rcc/youzhi/models/instruct/think_mix2_ckpts"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
eos = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
model = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True,
                                             dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()

enc = tok.apply_chat_template([{"role": "user", "content": "What is 12 + 30?"}],
                              tokenize=True, add_generation_prompt=True,
                              enable_thinking=True, return_tensors="pt")
ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
L = ids.shape[1]

with torch.inference_mode():
    full = model(ids).logits  # [1, L, V] no cache
    # Incremental: feed ONE token at a time, building the cache from scratch.
    past, inc = None, []
    for t in range(L):
        out = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        inc.append(out.logits[:, -1])
    inc = torch.stack(inc, dim=1)
    # Also test the real usage: prefill the whole prompt once, then check the
    # last-position logits match.
    out_pf = model(ids, use_cache=True)
    pf_last_diff = (full[:, -1].float() - out_pf.logits[:, -1].float()).abs().max().item()
    print(f"prefill-once last-token diff: {pf_last_diff:.3e}")
maxdiff = (full.float() - inc.float()).abs().max().item()
# argmax agreement (what actually matters for greedy/sampling)
agree = (full.argmax(-1) == inc.argmax(-1)).float().mean().item()
print(f"VERIFY: max|full-cached|={maxdiff:.3e}  argmax_agree={agree*100:.1f}%  (L={L})")

# Speed: 200 tokens, cached vs no-cache (greedy, batch 4)
batch = ids.repeat(4, 1)
ctx = model.config.max_position_embeddings
for label, use_cache in [("cached", True), ("no-cache", False)]:
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        cur, past, step = batch, None, batch
        for _ in range(200):
            if use_cache:
                out = model(step, past_key_values=past, use_cache=True); past = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1, keepdim=True); step = nxt
            else:
                out = model(cur[:, -ctx:]); nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], -1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  {label:9s}: 200 steps x batch4 in {dt:.1f}s  ({4*200/dt:.0f} tok/s)")
