---
license: apache-2.0
language:
- en
library_name: pytorch
base_model: PursuitOfDataScience/argonne-4.5-base-ctx13568
tags:
- argonne
- decision
- classification
- zero-shot-classification
- calibration
pipeline_tag: zero-shot-classification
---

# Argonne 4.5-decision

**Typed decisions in one forward pass: pick one of up to 255 options, rate on your own scale, or give the
probability that a statement is true, with calibrated probabilities.** It is
[argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568) (2.06B
parameters, 13,568-token context) with a small decision head. It never generates text: every answer is a
probability over the options you gave it, so it cannot answer outside your schema.

Measured plainly: its probabilities are well calibrated (expected calibration error 0.022 on multiple choice
and 0.024 on ratings, against the 0.082 and 0.325 that Jev publishes), a call takes 59 to 118 ms on one A100,
and it is a 2B model: zero-shot on Banking77 it scores 62.5%, against Jev's 81.1%.

![How a decision call works: text and questions in, one pass through the trunk, the decision head scores each option, calibrated typed JSON out](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision/resolve/main/assets/how.png)

## Question types

- **Choice**: 2 to 255 options. Returns the chosen option, a confidence and a probability per option.
- **Score**: a legend `{level: description}`. Returns the expected level, a confidence and a probability per level.
- **Noul**: a statement. Returns the probability that it is true.

Every question about one text shares a single pass over that text, so ask them in one call. `confidence` is
1 minus the normalised entropy: 1.0 means all probability on one option.

## Evaluation

### Calibration

Expected calibration error on held-out test questions (71,208 Choice, 11,732 Score, 31,260 Noul), after one
temperature per question type fitted on separate validation questions.

![Calibration error by question type for argonne-4.5-decision against Jev's published numbers](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision/resolve/main/assets/calibration.png)

<details>
<summary>Show the numbers</summary>

| question type | ECE before temperature | **ECE after** | ECE after, on sets never trained on | Jev (published) |
|---|---:|---:|---:|---:|
| Choice | 0.087 | **0.022** | 0.049 | 0.082 |
| Score | 0.026 | **0.024** | | 0.325 |
| Noul | 0.016 | **0.010** | 0.160 | |

</details>

On yes/no statements unlike anything it has seen, probabilities are less reliable (ECE 0.160 on the
never-trained sets below), mostly because they bunch toward the middle. For a new yes/no task, choose the
threshold on a few labeled examples rather than trusting 0.5.

### Zero-shot intent classification

Banking77's 3,076 test questions, all 77 intent names as the options, never trained on.

![Banking77 zero-shot accuracy: argonne-4.5-decision against Jev and Qwen as Jev reports them](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision/resolve/main/assets/banking77.png)

<details>
<summary>Show the numbers</summary>

| model | top-1 | top-3 | top-5 |
|---|---:|---:|---:|
| **argonne-4.5-decision** | **62.5** | 80.7 | 86.4 |
| Jev (published) | 81.1 | | |
| Qwen (as reported by Jev) | 76.4 | | |

</details>

The misses are mostly near neighbours ("order physical card" against "get physical card") and intents whose
names describe their questions poorly. Most of the gap is model size: a 35B general instruction model answers
74.8% of the same items in our harness.

### Decision sets it never trained on

Each set brings a label list, a written rule or a yes/no statement the model has not seen. A "written rule"
question spells the decision out in the statement: "yes when the text is A, B or C; no for D and E".

![Accuracy on eleven decision sets the model never trained on](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision/resolve/main/assets/heldout.png)

<details>
<summary>Show the numbers</summary>

| set | questions | accuracy |
|---|---:|---:|
| Banking77 intents, as a written rule | 798 | 83.7 |
| 20 Newsgroups, as a written rule | 461 | 80.5 |
| Finance news topics, as a written rule | 1,012 | 69.9 |
| Spam email (yes / no) | 2,000 | 65.5 |
| Banking77 intents, 77 options | 3,076 | 62.5 |
| 20 Newsgroups, 20 options | 2,000 | 61.2 |
| Finance news sentiment, 3 options | 2,388 | 56.5 |
| Unsafe chat request (yes / no) | 2,741 | 56.4 |
| MMLU, 4 options | 4,000 | 45.5 |
| Finance news topics, 20 options | 4,117 | 40.3 |
| TruthfulQA MC1 | 817 | 31.7 |

</details>

It follows a rule written into the question well, including on label sets it has never seen. A brand-new
yes/no criterion with no rule attached is harder (56 to 66%), and knowledge-heavy multiple choice (MMLU,
TruthfulQA) stays at the level of its base.

### Latency

Median of repeated calls on one NVIDIA A100, batch of one.

![Latency per call on one A100 against Jev's published range](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision/resolve/main/assets/latency.png)

<details>
<summary>Show the numbers</summary>

| call | input tokens | median |
|---|---:|---:|
| support ticket, 3 questions | 156 | 117.5 ms |
| one 77-way intent question | 661 | 59.0 ms |
| 6,576-token document, 2 questions | 6,576 | 301.2 ms |

</details>

## Training

- **Stage 1**: from argonne-4.5-base-ctx13568, one epoch over 1.09M typed decision questions (multiple choice,
  classification, rating, entailment, tagging against a written definition, rules written into the question),
  log loss over the given options, LR 1e-5 for the trunk and 3e-4 for the head, 3% warmup, cosine to 10%, up to
  65,536 tokens per optimizer step.
- **Stage 2**: one more epoch over 210,000 questions that mix further decision tasks with a replay of stage 1,
  LR 5e-6 (head 1e-4), 5% warmup.
- **Calibration**: one temperature per question type, fitted on validation questions by log loss.

Trained on 4 NVIDIA A100 40GB GPUs with a sharded optimizer and gradient checkpointing, inputs up to 13,568
tokens. The head scores each option at the token that ends it, conditioned on the token that ends the question.

## Inference

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("PursuitOfDataScience/argonne-4.5-decision")
sys.path.insert(0, path)
from decision.api import ArgonneDecision, Choice, Score, Noul

m = ArgonneDecision(path)                      # a GPU when one is available, else the CPU
ticket = ("Hi, I ordered a laptop two weeks ago (order #88213) and I was charged twice on my credit card. "
          "I have called three times and nobody has fixed it. This is ridiculous. I want the duplicate charge "
          "refunded today or I will cancel my account.")
print(m.system_one(state=ticket, questions={
    "dept":   Choice("Which team should handle this ticket?", ["billing", "technical support", "shipping", "other"]),
    "refund": Noul("The customer is asking for a refund."),
    "anger":  Score("How angry is the customer?", {0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}),
}))
```

```json
{"model": "argonne-4.5-decision",
 "answers": {"dept":   {"type": "choice", "choice": "billing", "confidence": 0.9846,
                        "probabilities": {"billing": 0.9972, "technical support": 0.0013, "shipping": 0.0, "other": 0.0015}},
             "refund": {"type": "noul", "noul": 0.9991},
             "anger":  {"type": "score", "score": 1.9912, "confidence": 0.9542,
                        "legend": {"0": "Calm", "1": "Frustrated but civil", "2": "Very angry"},
                        "probabilities": {"0": 0.0002, "1": 0.0084, "2": 0.9914}}},
 "usage": {"input_tokens": 156, "output_tokens": 150}}
```

As a service: `python decision/api.py --model <path> --serve 8008`, then `POST /v1/system_one` with
`{"state": ..., "questions": {...}}`.

## Usage notes

- Needs `torch`, `transformers` and `safetensors`. `model.py` in this repo is the base's `argonne2`
  architecture; `decision/` holds the head and the API. The language-model output layer is not included: the
  decision head never uses it.
- When none of your options may fit, add "None of the above". With the correct intent removed from Banking77's
  list and that option offered, it chose it in 500 of 500 tests.
- Texts longer than 13,568 tokens are cut to their beginning and end.
- On a CPU a short call takes about 2 seconds.

## Limitations

- A 2B model: fine-grained zero-shot classification over many similar labels is well below Jev (Banking77
  62.5% against 81.1%).
- Yes/no probabilities on unfamiliar tasks bunch toward the middle; calibrate a threshold before relying on 0.5.
- English only. It cannot explain an answer, only score the options it was given.

## Source code

Everything is on the GitHub `main` branch:
[PursuitOfDataScience/ArgonneAI](https://github.com/PursuitOfDataScience/ArgonneAI/tree/main).

- [`decision/api.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/api.py): `system_one`, the HTTP server and the CLI
- [`decision/modeling.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/modeling.py): the decision head and loading
- [`decision/schema.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/schema.py): how questions and options are encoded
- [`decision/train.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/train.py): training (any number of GPUs, resumable)
- [`decision/evaluate.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/evaluate.py) and [`decision/calibrate.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/calibrate.py): accuracy, calibration error and the per-type temperatures
- [`decision/bench.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/decision/bench.py): latency and the out-of-category test
- [`model.py`](https://github.com/PursuitOfDataScience/ArgonneAI/blob/main/model.py): the `argonne2` architecture, identical to the copy in this repo

## Citation

```bibtex
@misc{argonne45decision,
  title  = {Argonne 4.5-decision},
  author = {Youzhi Yu},
  year   = {2026},
  url    = {https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision}
}
```
