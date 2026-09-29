---
license: apache-2.0
language:
- en
library_name: transformers
base_model: PursuitOfDataScience/argonne-4.5-base-ctx13568
tags:
- text-generation
- causal-lm
- transformer
- argonne
- reasoning
- chain-of-thought
- math
datasets:
- HuggingFaceH4/ultrachat_200k
- argilla/dpo-mix-7k
- AI-MO/NuminaMath-CoT
- openai/gsm8k
pipeline_tag: text-generation
---

# Argonne 4.5-think

**A 2.06B-parameter reasoning model trained from scratch.** It writes a short `<think>…</think>`
trace, then a `\boxed{}` answer. Built on
[argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568).

It is the reasoning model of the 4.5 line: SFT, DPO and chain-of-thought SFT, then on-policy
distillation from [Argonne-3.5-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.5-think),
repair passes and one round of RL from verifiable rewards (RLVR-DPO). Post-training took the
four-pool arithmetic gate below from **38.40 to 60.43**. That is level with
[Argonne-4.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-4.0-think) (1.04B, +0.90,
p 0.32) and 3.80 below the 2.88B teacher. What it adds over 4.0-think is general capability: **+3.08
on the 8-task lm-eval mean** (51.04 against 47.96), with hellaswag +9.24 and boolq +12.11.

## Evaluation

Greedy decoding unless a column says otherwise, paired on identical items, exact McNemar on the
paired outcomes. `n` = 1000 (ASDiv, SVAMP), 500 (GSM-Plus, MAWPS). **Pooled** = correct answers over
all 3,000 items, the same convention as the Argonne-4.0-think card.

### Against the other Argonne reasoning models

Each row paired on the same items, all three models evaluated in the same way on the same hardware.

| pool | Argonne-4.0-think (1.04B) | **4.5-think (2.06B)** | delta | p |
|---|---:|---:|---:|---|
| ASDiv | 70.80 | **69.40** | −1.40 | 0.35 |
| SVAMP | 60.40 | **62.30** | +1.90 | 0.28 |
| GSM-Plus | 36.00 | **39.60** | +3.60 | 0.13 |
| MAWPS | 58.80 | **59.60** | +0.80 | 0.72 |
| **four-pool pooled** | **59.53** | **60.43** | **+0.90** | **0.32** |

| pool | Argonne-3.5-think (2.88B, the teacher) | **4.5-think (2.06B)** | delta | p |
|---|---:|---:|---:|---|
| ASDiv | 73.20 | **69.40** | −3.80 | 0.01 |
| SVAMP | 68.00 | **62.30** | −5.70 | 4.8e-4 |
| GSM-Plus | 42.20 | **39.60** | −2.60 | 0.32 |
| MAWPS | 60.80 | **59.60** | −1.20 | 0.53 |
| **four-pool pooled** | **64.23** | **60.43** | **−3.80** | **1.7e-5** |

**On arithmetic word problems it is level with the 1.04B 4.0-think**: no pool differs significantly
(every p ≥ 0.13), and neither does any decoding mode (self-consistency@8 67.83 vs 67.03, pass@8
79.97 vs 80.27, all p ≥ 0.3). It stays 3.80 below the teacher it was distilled from. Twice the parameters did not buy a better math
reasoner here; its base lost most of its step-by-step math in the long-context stage (gsm8k 20.62 to
4.78) and post-training had to win it back.

### What each post-training stage bought

Four-pool pooled, measured on each stage's own checkpoint; p is against the row above.

| stage | greedy | p | self-consistency@8 | pass@8 | no `</think>` % | no answer % |
|---|---:|---|---:|---:|---:|---:|
| 3: CoT-SFT (the start) | 38.40 | | 49.33 | 67.83 | 3.6 | 2.3 |
| 4: + 5 rounds of on-policy distillation | 53.67 | 1.6e-59 | 62.57 | 75.47 | 17.4 | 3.2 |
| 5: + 2 repair passes | 56.73 | 3.4e-5 | 64.47 | 77.20 | 13.0 | 0.0 |
| 6: + one more distillation round and repair | 56.53 | 0.83 | 65.60 | 78.50 | 14.1 | 0.8 |
| 7: + the same on unseen problems | 57.47 | 0.25 | 66.80 | 79.70 | 11.9 | 2.6 |
| 8: + RLVR-DPO and repair (**this model**) | **60.43** | **2.9e-5** | **67.83** | **79.97** | **11.2** | **0.0** |

Start to finish: **+22.03** pooled (p 1.8e-113), and every pool up by 20 to 24 points. Stages are
described under Training. Distillation did most of the work; the repair passes and the RLVR-DPO
stage are the next largest steps, about +3 each.

### Test-time compute, measured on these weights

| pool | greedy | self-consistency@8 | budget-forcing | budget-extend | pass@8 |
|---|---:|---:|---:|---:|---:|
| ASDiv | 69.40 | 77.10 | 71.00 | 71.90 | 87.30 |
| SVAMP | 62.30 | 72.30 | 63.30 | 64.00 | 86.40 |
| GSM-Plus | 39.60 | 47.20 | 38.40 | 39.40 | 63.60 |
| MAWPS | 59.60 | 61.00 | 59.60 | 59.80 | 68.80 |
| **pooled** | **60.43** | **67.83** | **61.10** | **61.83** | **79.97** |

- **Self-consistency is the cheapest gain:** 8 samples at temperature 0.8 with a majority vote add
  **+7.40** pooled.
- **Longer thinking barely helps.** Budget-forcing (cap the trace at 256 tokens, then force the
  close and read the answer) adds +0.67; budget-extend (from that cap, append "Wait, let me
  double-check that." and think 160 more tokens, twice) adds +1.40.
- **pass@8 of 79.97 against greedy 60.43:** the model often reaches the answer in some sample, and
  picking it is what is missing.
- Greedy outcomes over the 3,000 items: 60.43% correct, 28.33% a wrong answer, 11.23% never close
  `</think>` within 512 new tokens, 0.00% close it without an answer.

### Which pools, and why not the usual ones

- **GSM8K is excluded.** It is contaminated for Argonne reasoning models: the CoT-SFT mix saw about
  94% of its test set. GSM-Plus is perturbed GSM8K *test*, and the mix's GSM8K tier was audited to
  be train-split only (4,338 of 4,338 rows). The distillation stages sample GSM8K **train** problems.
- **MATH-500 is excluded.** 17 of its 319 items have a near-duplicate in the CoT-SFT mix.
- **The unseen training problems were decontaminated first.** NuminaMath-CoT contains four of the
  five evaluation pools verbatim; every problem within Jaccard 0.70 of an evaluation item (5,842
  rows) was removed before any sampling ([`reasoning/decontam_pool.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/decontam_pool.py)).

### General capability (lm-eval)

The base card's tasks, few-shot counts and metric rule (`acc_norm` for multiple choice, `acc` for
winogrande and mmlu), scored with no chat template so the columns compare. All three columns were
run with the same harness.

| task | 4.5-base-ctx13568 | Argonne-4.0-think | **4.5-think** |
|---|---:|---:|---:|
| arc_challenge | 41.72 | 33.19 | 36.77 |
| arc_easy | 62.04 | 50.34 | 54.42 |
| hellaswag | 55.83 | 42.70 | 51.94 |
| piqa | 70.51 | 64.58 | 68.77 |
| sciq | 85.40 | 80.50 | 81.60 |
| openbookqa | 38.00 | 31.60 | 34.00 |
| winogrande *(acc)* | 57.85 | 54.62 | 55.33 |
| mmlu *(acc)* | 26.03 | 26.19 | 25.47 |
| **8-task mean** | **54.67** | **47.96** | **51.04** |
| truthfulqa_mc2 | 40.93 | 45.75 | 45.08 |
| boolq | 67.61 | 55.02 | 67.13 |

Against 4.0-think it is ahead on seven of the eight tasks (mmlu is level with chance for both) and by
**+3.08** on the mean. Against its own base, post-training costs **3.63**, most on arc_easy (−7.62)
and arc_challenge (−4.95), and gains 4.15 on truthfulqa_mc2. gsm8k is left out on purpose: 23.65
against the base's 4.78, but for this model it is not a clean measure (see above).

### Instruction following (IFEval)

541 prompts with verifiable formatting constraints, chat template on, greedy, up to 1,280 new tokens,
the think block stripped before scoring (a trace that never closes is scored whole). Same harness for
every row; the full table, with public models, is on the
[argonne-4.5-instruct](https://huggingface.co/PursuitOfDataScience/argonne-4.5-instruct) card.

| model | prompt strict | instruction strict |
|---|---:|---:|
| **Argonne-4.5-think** | **11.83** | **20.62** |
| Argonne-4.0-think | 14.60 | 24.22 |
| argonne-4.5-instruct | 11.83 | 20.62 |

Level with 4.0-think within noise (paired McNemar p 0.11). 95 of the 541 replies never close
`</think>` within the cap, so a sixth of them are scored on an unfinished trace.

### Long context

Position-bucketed loss on held-out arXiv (40 windows of 24,576 tokens from proof-pile-2's **test**
split), nats per token, lower is better.

| token position | 4.5-base-ctx13568 | **4.5-think** |
|---|---:|---:|
| 0 to 1,024 | 1.642 | 1.916 |
| 1,024 to 2,048 | 1.354 | 1.593 |
| 2,048 to 4,096 | 1.210 | 1.420 |
| 4,096 to 8,192 | 0.979 | 1.166 |
| 8,192 to 13,568 | 0.854 | 1.029 |
| 13,568 to 20,480 *(past the window)* | 0.832 | 1.034 |

Loss still falls with position all the way to 13,568, so the model still uses its context. The
curve sits 0.17 to 0.27 nats above the base's, and the gap is largest in the first bucket, so this
is post-training moving the model away from arXiv prose rather than a loss of context. Reasoning
over long inputs was not evaluated.

## Training

| stage | data | detail |
|---|---|---|
| base | [argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568) | 145.21B tokens, context 13,568 |
| 1: SFT | [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 207,865 conversations, 1 epoch (12,991 steps, 247M tokens), length 4,096, effective batch 16, LR 2e-5 |
| 2: DPO | [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) | 6,750 pairs, 1 epoch, length 4,096, effective batch 8, LR 1e-6, β 0.03 |
| 3: CoT-SFT | Argonne-3.5-think's short-trace mix, 28,428 rows | 1 epoch, length 1,024, effective batch 12, LR 1e-5 |
| 4: on-policy distillation, 5 rounds | the student's **own** samples on 12,000 problems per round (4,000 each from GSM8K train, MATH train levels 1-3, MATH train levels 4-5), 8 per problem | per-token reverse KL to **[Argonne-3.5-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.5-think)**; about 35k rows, 1 epoch, LR 1e-5 per round |
| 5: repair, 2 passes | the model's own verified-correct samples (13,910, then 15,395) | plain cross-entropy, LR 3e-6, 1 epoch |
| 6: distillation + repair | same problem pools | one more round (35,140 rows), then repair (14,521 samples) |
| 7: distillation + repair on unseen problems | 12,000 problems from a decontaminated 233,675-problem NuminaMath-CoT pool | 35,999 rows, then repair (8,912 samples) |
| 8: RLVR-DPO + repair | 7,743 preference pairs from samples on unseen problems (the student's own on 12,000, pooled with an earlier checkpoint's on 36,000) | DPO, β 0.4, LR 5e-7, 1 epoch, reference = a frozen copy of the student; then repair (10,982 samples) |

Every stage from 4 on runs at length 1,024. The teacher is the released Argonne-3.5-think, bit for
bit, run in bf16.

**Distillation at the student's own states.** Stage 4 samples from the student, then matches the
teacher's full next-token distribution on those samples. It bought +15.27 pooled over five rounds,
then stopped: a sixth round did not beat the fifth, and a weight soup of rounds 4 to 6 was worse.

**The repair pass buys termination, not knowledge.** Plain cross-entropy on the model's own correct
samples at LR 3e-6. Each distillation round leaves more traces that never close; the repair brings
them back (no answer 3.2% to 0.0% in stage 5).

**RLVR-DPO needs length-matched negatives.** Pairs are a verified-correct sample (checked step by
step, degenerate ones dropped) against a wrong one, only on problems where the model's most common
answer is wrong. Taking the shortest wrong samples as negatives taught the model to write longer
and cost termination. Pairing each correct sample with wrong ones of similar length is worth +1.90
pooled on its own (p 0.004), and the matched version is the last row above.

**Tested and not kept**, each paired against the model it would replace: a third repair pass, a
second RLVR cycle, an NLL term on the chosen trace, β 0.2, pairs from every problem instead of the
mode-wrong ones, and spending the stage-7 rows on 3× more problems at one sample each. None beat
this checkpoint.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PursuitOfDataScience/Argonne-4.5-think"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
).cuda()

messages = [{"role": "user", "content": "Tom has 5 boxes with 12 apples each. He gives away 17 apples. How many apples does he have left?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()

out = model.generate(ids, max_length=ids.shape[1] + 512, do_sample=False)
print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

For throughput, serve it with vLLM or SGLang instead of `.generate()`.

## Usage notes

- Load with `trust_remote_code=True`; `config.json` carries an `auto_map`, so no manual setup.
- The custom `generate` takes `max_length` (total length), not `max_new_tokens`.
- `eos_token_id` is **151645** (`<|im_end|>`). Checked on the staged release files: four
  chat-templated prompts with **no** `eos_token_id` argument all stopped on their own, in 55 to 246
  tokens.
- `lm_head.weight` is reported missing on load. Expected: the embeddings are tied.
- **Traces are short by design**, 235 to 279 thinking tokens on average under greedy decoding,
  depending on the pool.
- Attention is full causal on every layer; `interleaved_local_attention` is `false`, matching the base.

## Limitations

- **The measured domain is arithmetic word problems.** Code, tool calling and open-ended chat are not
  characterized beyond the lm-eval table.
- **Selection is the binding constraint:** pass@8 79.97 against greedy 60.43. Use self-consistency
  if you can afford the samples.
- 11.23% of greedy answers never close the thinking block within 512 tokens.
- It still makes plain rate errors (in one smoke-test prompt, "3 pencils for $2, how much for 12?"
  came back as $24), and its traces can carry boilerplate from its training traces, such as
  "There's no policy violation".
- Post-training cost general knowledge (−3.63 on the 8-task lm-eval mean against its base).
- Fine-tuning ran at 1,024 to 4,096 tokens; behaviour on long inputs was not tested.
- No safety alignment beyond what UltraChat and the preference data provide.

## Source code

Everything is on the GitHub `main` branch:
[PursuitOfDataScience/ArgonneAI](https://github.com/PursuitOfDataScience/ArgonneAI/tree/main).

| file | role |
|---|---|
| [`model.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py) | the `argonne2` architecture, identical to the copy in this repo |
| [`sft.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/sft.py) | stage 1 |
| [`dpo.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/dpo.py) | stage 2 |
| [`reasoning/cot-sft.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/cot-sft.py) | stage 3 |
| [`reasoning/opd_train.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/opd_train.py) | stages 4-8: distillation, repair and RLVR-DPO |
| [`reasoning/rft_generate.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/rft_generate.py) | the on-policy samples each stage trains on |
| [`reasoning/decontam_pool.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/decontam_pool.py), [`reasoning/pool_novelty.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/pool_novelty.py) | the unseen-problem pool: decontamination and novelty filters |
| [`reasoning/effort_gate.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/effort_gate.py) | the four-pool paired evaluation |
| [`reasoning/run_lmeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_lmeval_vllm.py) | lm-eval through vLLM |
| [`reasoning/run_ifeval_vllm.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/run_ifeval_vllm.py) | IFEval with the chat template, through vLLM |
| [`reasoning/exp_longctx_learning.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/exp_longctx_learning.py) | the long-context probe |
| [`reasoning/release_smoke.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/reasoning/release_smoke.py) | the termination check run on these release files |

## Citation

```bibtex
@misc{argonne45think,
  title  = {Argonne 4.5-think},
  author = {Youzhi Yu},
  year   = {2026},
  url    = {https://huggingface.co/PursuitOfDataScience/Argonne-4.5-think}
}
```
