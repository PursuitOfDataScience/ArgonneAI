---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- text-generation
- causal-lm
- transformer
- argonne
- pretrained
- base-model
pipeline_tag: text-generation
---

# Argonne 4.5 Base

A 2.06B-parameter decoder-only base model, pretrained from scratch on 110B tokens in a single
continuous run. This is a **base** model: it has had no instruction tuning, no chat template and no
alignment, so it completes text rather than following instructions.

Every number below is read off the released artifact or the trainer's own logs; nothing is carried
over from an earlier model in the family.

## Model

| | |
|---|---|
| **Parameters** | 2,063,639,552 (2.06B), of which 388,272,640 (18.8%) are the tied embedding |
| **Layers** | 24 |
| **Model dimension** | 2,560 |
| **Attention** | 10 query heads / 2 key-value heads (grouped-query, 5:1) |
| **Feed-forward** | SwiGLU, intermediate 7,040 |
| **Normalisation** | RMSNorm (eps 1e-6), QK-norm, V-norm, sandwich norms |
| **Positions** | RoPE, theta 1e6 |
| **Logit soft-cap** | 15.0 |
| **Context length** | 1,024 tokens |
| **Vocabulary** | 151,669 (Qwen3 tokenizer) |
| **Output head** | tied to the input embedding |
| **Published dtype** | bfloat16 |

⚠️ **The context length is 1,024 tokens and it is a hard limit**, not a default: `block_size` and
`max_position_embeddings` are both 1,024 and the model was trained only at that length. A base model
trained at 1,024 does not extrapolate: see the family's long-context work for what extension costs.

Features present in the code but **inactive in this model**, so that the config is not misread:
sliding/local attention (`interleaved_local_attention: false`, `local_attention_window: null`),
document masking (`doc_mask: false`), multi-token prediction (`mtp_loss_weight: 0.0`) and z-loss
(`z_loss_weight: 0.0`).

## Training

| | |
|---|---|
| **Tokens** | 110,000,259,072 (~110B), one pass |
| **Optimizer steps** | 203,451 |
| **Tokens per step** | 540,672 |
| **Optimizer** | AdamW |
| **Sequence length** | 1,024 |
| **Data mixture** | 50% educational web / 30% math / 20% code |
| **Final training loss** | **1.80** (mean of the last 20 logged readings; sd 0.14, range 1.51-2.07) |
| **Hardware** | 8x NVIDIA A100-40GB (one node), ALCF Sophia, with gap-filling slices on ALCF Polaris |
| **Measured throughput** | 56,798 tokens/s on the 8-GPU node |
| **Completed** | 2026-09-13 |

The loss figure is a **mean over the last 20 logged readings, with its spread**, because single
readings at this interval swing by ~0.3 (1.51 to 2.07 over the final 20). A single terminal value
would be a noise sample, not a result.

Training computed in bfloat16 with fp32 optimizer master weights. The release is cast to bfloat16:
every inference load performs that cast anyway, so it is bit-identical for this artifact's purpose and
halves the download. The fp32 masters live in the resumable checkpoint, not here.

## Usage

The architecture is custom, so `trust_remote_code=True` is required: `model.py` ships with the
weights and `auto_map` in `config.json` points at it.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PursuitOfDataScience/argonne-4.5-base"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype="bfloat16", device_map="auto"
)

prompt = "The Treaty of Westphalia, signed in 1648,"
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

Keep inputs within 1,024 tokens. For throughput, serve it with vLLM rather than a per-token
`generate()` loop.

## Limitations

- **Base model.** No instruction tuning, no chat template, no safety alignment. It continues text.
- **1,024-token context.** Longer inputs are not supported and the model does not extrapolate.
- **English-centric**, and the mixture is weighted towards educational web text, mathematics and code;
  behaviour outside those domains is correspondingly weaker.
- **No evaluation numbers are quoted here on purpose.** The benchmark results for this model are not
  part of this artifact's verified record, and a base model's scores are easy to misreport; treat any
  figure not in this card as unverified.
- Standard caveats for a web-pretrained model apply: it can reproduce biases and factual errors
  present in its training data, and it has no notion of whether an output is true.
