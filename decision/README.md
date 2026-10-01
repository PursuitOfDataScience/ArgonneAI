<div align="center">

# Argonne Decision

**Typed decisions in one forward pass: pick an option, rate on your scale, or say how likely a statement is true.**

</div>

[argonne-4.5-decision](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision) is
[argonne-4.5-base-ctx13568](https://huggingface.co/PursuitOfDataScience/argonne-4.5-base-ctx13568) (2.06B, 13,568-token
context) with a small decision head. It never generates text: every answer is a calibrated probability over the
options you gave, so it cannot answer outside your schema.

```python
import sys
from huggingface_hub import snapshot_download
path = snapshot_download("PursuitOfDataScience/argonne-4.5-decision"); sys.path.insert(0, path)
from decision.api import ArgonneDecision, Choice, Score, Noul

m = ArgonneDecision(path)
m.system_one(state="I was charged twice for order #88213 and want the duplicate refunded today.", questions={
    "dept":   Choice("Which team should handle this ticket?", ["billing", "technical support", "shipping", "other"]),
    "refund": Noul("The customer is asking for a refund."),
    "anger":  Score("How angry is the customer?", {0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}),
})
```
```text
dept -> billing (0.863)    refund -> 0.998    anger -> 1.47 of 2 (between frustrated and very angry)
```

## 🧭 Question types

| Type | You give | You get |
|---|---|---|
| `Choice` | 2 to 255 options | the option, `confidence`, a probability per option |
| `Score` | a legend `{level: description}` | the expected level, `confidence`, a probability per level |
| `Noul` | a statement, rules and definitions welcome | the probability it is true |

## 📏 How good is it

| | argonne-4.5-decision | Jev |
|---|---|---|
| calibration error, Choice / Score | 0.022 / 0.024 | 0.082 / 0.325 |
| Banking77, never trained on | 62.5% | 81.1% |
| latency, 3 questions / 77-way choice (A100) | 118 / 59 ms | 150-250 ms |

Comparisons with open zero-shot classifiers and LLMs on the same held-out sets: [model card](https://huggingface.co/PursuitOfDataScience/argonne-4.5-decision).

## 🚀 Run it

1. `git clone https://github.com/PursuitOfDataScience/ArgonneAI && cd ArgonneAI`
2. `pip install torch transformers safetensors huggingface_hub`
3. `python decision/api.py --model "$(python -c 'from huggingface_hub import snapshot_download as s; print(s("PursuitOfDataScience/argonne-4.5-decision"))')" --serve 8008`
4. `curl -s localhost:8008/v1/system_one -d '{"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}}'`

## 🏋️ Train your own

| Step | Command |
|---|---|
| records | one JSON per line: `type` (choice / score / noul), `state`, `instructions`, `options`, `label`, optional `soft` |
| pack | `python decision/pack.py --built DIR --out PACKED` |
| train | `torchrun --nproc_per_node=4 decision/train.py --data PACKED --out RUN` (resumable, any GPU count) |
| evaluate + calibrate | `decision/evaluate.py`, then `decision/calibrate.py` |
| tests (CPU) | `python decision/test_decision.py && python decision/test_train.py` |

## ⚠️ Worth knowing

- Ask every question about one text in one call: they share a single pass over it.
- If none of your options may fit, add "None of the above": with the right label removed it picks that every time.
- On a brand-new yes/no task, set the threshold on a few labeled examples; unfamiliar statements score near the middle.
- Texts longer than 13,568 tokens are clipped to their head and tail.
