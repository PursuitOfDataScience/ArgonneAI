# Argonne 3

This repository contains the full training and release pipeline for the Argonne causal LM family.

## Current model architecture (Argonne 3)

The active architecture in `argonne3.0` is defined by `model.py`, `pretrain.py`, and `continue_pretrain.py`:

| Component | Current setting |
|---|---|
| Parameters | 2,882,982,912 (~2.88B) |
| Layers | 24 transformer blocks |
| Hidden size | 3,072 |
| Attention heads | 12 query / 4 key-value (GQA) |
| Head dimension | 256 |
| Feed-forward | SwiGLU MLP, 8,192 intermediate dim |
| Normalization | RMSNorm + QK norm + V norm + sandwich norms |
| Attention pattern | Interleaved local/global attention |
| Local attention window | 256 |
| Logit stabilization | Final logit softcap = 15.0 |
| RoPE theta (pretrain/continue) | 1,000,000 |
| Base context length | 1,024 tokens (default pretrain/continue block size) |
| Long-context stage default | 13,568 tokens (`midtraining.sh`) |
| Tokenizer/vocab | From `--tokenizer_path` (config default vocab size is 151,936) |
| Multi-token prediction (MTP) | Configurable horizon & loss weight (disabled by default) |
| Z-loss | Configurable weight (0.0 by default) |

## Architecture comparison: Argonne 3 vs. Argonne 2.5

Argonne 2.5 lives on the `llm.c` branch and was released as `PursuitOfDataScience/Argonne2.5-base` (~1.27B params).

| Component | Argonne 3 (`argonne3.0`) | Argonne 2.5 (`llm.c`) |
|---|---|---|
| Parameters | ~2.88B | ~1.27B |
| Layers | 24 | 28 |
| Hidden size | 3,072 | 1,792 |
| Attention heads | 12 query / 4 key-value (GQA) | 14 query / 7 key-value (GQA) |
| Head dimension | 256 | 128 |
| Feed-forward intermediate dim | 8,192 | 4,864 |
| Normalization | RMSNorm + QK norm + V norm + sandwich norms | RMSNorm only |
| Attention pattern | Interleaved local/global attention | Full global attention |
| Local attention window | 256 | N/A |
| Logit softcap | 15.0 | None |
| RoPE theta | 1,000,000 | 10,000 |
| MTP support | Yes (configurable) | No |
| Z-loss support | Yes (configurable) | No |
| Base context length | 1,024 tokens | 1,024 tokens |
| Long-context stage | 13,568 tokens (dedicated `midtraining.py`) | Supported via continued pretraining |
| Default LR (pretrain) | 3.0e-4 | 3.0e-4 |
| Gradient clipping | 1.0 | 1.0 |
| Effective batch size | ~233K tokens | ~246K tokens |
| torch.compile | Enabled | Enabled |

## Training pipeline

End-to-end stages and launch scripts:

1. Data preprocessing: `preprocess_data.py` + `preprocess_job.sh`
2. Pretraining: `pretrain.py` + `run_full_training.sh`
3. Continued pretraining: `continue_pretrain.py` + `continue.sh`
4. Long-context midtraining: `midtraining.py` + `midtraining.sh`
5. SFT: `sft.py` + `sft.sh`
6. DPO: `dpo.py` + `dpo.sh`
7. Reasoning / CoT SFT, STaR, GRPO, evals: see `reasoning/` (start with `reasoning/thinking_training.md`)
8. Publishing: `push_model_to_hf.py`

## Inference example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "PursuitOfDataScience/Argonne2.5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

prompt = "Write a short paragraph about scientific computing at Argonne National Laboratory."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 128,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
