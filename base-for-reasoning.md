# base-for-reasoning.md — what would actually break the reasoning ceiling

**Audience:** whoever trains the next Argonne base (argonne-3.5 and beyond).
**Author's context:** written 2026-07-14 after the Argonne-3.0-**think** campaign proved, across 6+ independent
downstream methods (CoT-SFT, teacher-distillation, self-distillation, STaR, GRPO, weight-soups), that the
3.0-think reasoning ceiling is a **base-capability wall**, not a downstream-tuning problem. `pass@32 ≈ 78%` and
greedy ≈ 25% on grade-school word problems did **not** move for any post-training technique. This document is
about the only thing that *can* move it: the base's pretraining regime.

---

## 0. TL;DR (the verdict)

The argonne-3.5 **optimization recipe is good** (FP8, LR 6e-4, WSD warmup 8k + cooldown 0.15, grad_clip 0.4,
qk-norm — a ~148σ loss win over the 3.0 recipe, well-searched). **The data regime is the limiter**, and as
currently configured 3.5 will most likely **reproduce the 3.0-think reasoning ceiling**, just at lower loss.

Three first-order problems, in order of impact:

1. **Token budget far too small.** ~64B tokens for a 2.88B model ≈ **22 tokens/param** — Chinchilla
   *compute-optimal*, but **~250–400× below** what makes modern small models *capable* (Qwen2.5 ~6000:1,
   SmolLM2/Llama-3.2 ~6000–9000:1). Compute-optimal ≠ capability-optimal.
2. **No code in the pretrain mix, and too little math.** FineWeb+FineMath 85/15 ⇒ ~9.6B math tokens, **0 code**.
   Code↔reasoning transfer is one of the most robust results in the literature; 3.0 already showed "base can't
   code." A code-free math-light mix caps reasoning.
3. **The search optimized CE-per-GPU-hour, not capability.** A better loss/hour recipe reaches the *same*
   undertrained ceiling faster. Right objective for a recipe experiment; wrong one if the goal is a reasoner.

**The single cheapest high-leverage fix available *right now*:** the pretrain **cooldown has not started yet**
(current run is at step ~106.5k / 241.6k; cooldown begins ~step 205k = last 36,241 steps ≈ 9.6B tokens). Point
that cooldown — plus the planned ~6B-token mid-train stage — at a **math/code/reasoning-heavy anneal** instead
of more FineWeb. That is a *data-mix change, not more compute*, and annealing on high-quality reasoning data is
where models like MiniCPM/OLMo/Llama-3 get a large fraction of their downstream capability.

---

## 1. The core diagnosis: it's tokens (then mix, then objective) — not the recipe

### 1.1 The tokens/param gap
| model | params | train tokens | tokens/param |
|---|---:|---:|---:|
| **argonne-3.5 (this run)** | 2.88B | ~64B | **~22** |
| Chinchilla compute-optimal | 2.88B | ~58B | 20 |
| SmolLM2-1.7B | 1.7B | 11T | ~6,500 |
| Llama-3.2-1B | 1.24B | 9T | ~7,300 |
| Qwen2.5-3B | 3.1B | ~18T | ~5,800 |

Capability in small models comes from **massively over-training past Chinchilla** — you spend extra pretraining
compute because the model is used for inference forever. At 22:1 you get an efficient loss-minimizer with a
**modest, compute-optimal capability ceiling** — which is exactly what 3.0-think exhibited (greedy ~25% on
grade-school math; Qwen2.5-3B gets ~80%+ on GSM8K, and the delta is almost entirely tokens).

### 1.2 Reasoning capability tracks reasoning tokens *seen*
~9.6B FineMath tokens is not enough to build robust multi-step arithmetic. The dominant 3.0-think failure modes
(arithmetic-fact errors, non-termination) are undertraining signatures, not architecture limits.

### 1.3 Why the recipe can't fix it
The FP8/LR/WSD/grad-clip package lowers loss-per-GPU-hour. That gets you to the 22:1 ceiling *cheaper*. It does
not raise the ceiling. Keep the recipe; change the data regime.

---

## 2. The levers, ranked by impact

### 2.1 Token budget — the biggest lever (and the hard compute constraint)
Measured throughput on this run: **~4B tokens/day on 3×H200** (28B tokens in ~7 days, FP8, 2.88B, ctx1024).

| target tokens | tokens/param @2.88B | wall-clock @3×H200 (2.88B) |
|---:|---:|---:|
| 64B (current) | 22 | ~16 days |
| 300B | 104 | ~75 days |
| 1T | 347 | ~250 days |

1T tokens at 2.88B on 3 GPUs is ~8 months — **infeasible without more GPUs**. This makes model-size choice a
first-class lever (§3): throughput scales ~inversely with params, so a **1.3B model runs ~2× the tok/s** and can
be over-trained to a far better tokens/param ratio in the same wall-clock.

**Note on repetition:** up to ~4 epochs of a high-quality corpus is nearly as good as fresh tokens (Muennighoff
et al., "Scaling Data-Constrained LMs"). If unique high-quality tokens are the bottleneck, multi-epoch on a
curated math/code/reasoning corpus is a valid way to raise tokens/param.

### 2.2 Data MIX — nearly as important, and cheap to change
Current: FineWeb 85 / FineMath 15, **no code, no synthetic reasoning**. Proposed reasoning-oriented mix
(all datasets already on disk under `/project/rcc/youzhi/data`):

| tier | share | datasets available |
|---|---:|---|
| General web (quality-filtered) | 40–50% | `fineweb-edu`, `fineweb` |
| **Math** | 20–25% | `finemath`, `nvidia_OpenMathReasoning(_curated)` |
| **Code** ← the missing reasoning multiplier | 15–20% | `nick007x_github-code-2025`, `datasets--nvidia--Nemotron-Competitive-Programming-v1` |
| **Synthetic step-by-step reasoning / instruction** | 10–15% | `a-m-team_AM-DeepSeek-R1-Distilled-1.4M`, `open-r1_Mixture-of-Thoughts`, `PursuitOfDataScience_0.5M-thinking`, `datasets--nvidia--Nemotron-SFT-Instruction-Following-Chat-v2` |
| High-quality QA / textbook (optional) | ~5% | (curate) |

The ingredients for a genuinely strong reasoning mix already exist locally — they are simply not in this run's
64B FW+FM mix. Adding **code** is the highest-value single change to the mix.

### 2.3 Curriculum / staging — the cheapest high-leverage change
Use a 3-phase schedule and put the best data where it matters most:
1. **Pretrain (broad):** the mixed corpus above, most of the tokens.
2. **Mid-train (upweight):** the ~6B-token mid-train stage already in `run_full_training.sh`
   (`MIDTRAIN_TARGET_TOKENS=6e9`) → make it math+code+reasoning-dense.
3. **Cooldown / anneal (highest quality):** the WSD cooldown (already 15% ≈ **9.6B tokens**) is the ideal place
   to inject the *cleanest* reasoning + instruction + tool-use data. WSD's LR decay + high-quality anneal data
   is where MiniCPM/OLMo/Llama-3 realize a large slice of downstream capability. **This run's cooldown hasn't
   started yet — it can still be repurposed for a reasoning anneal without restarting.**

Combined, the mid-train + cooldown give ~15.6B tokens of anneal budget — a data-mix decision, not new compute.

### 2.4 Objective / gating — optimize for capability, not CE/GPU-hour
- Terminal metric = **downstream held-out reasoning**, not next-token CE. Use the honest suite from the think
  work: `reasoning/clean_eval.py` (clean SVAMP/ASDiv/MAWPS, GSM-Plus; **never GSM8K — contaminated**), plus
  GSM8K-raw-5shot as the coarse go/no-go already defined in the kickoff plan.
- Eval **intermediate** checkpoints (there's already `reasoning/probe_pretrain_ckpt.py`) and let capability, not
  loss, decide the cooldown timing and data mix.

### 2.5 Keep the good 3.5 recipe (retain all of this)
- FP8 (torchao tensorwise + lm_head) — ~1.25× end-to-end, validated, 0 NaN.
- LR 6e-4 with WSD warmup 8k + **cooldown_frac 0.15** (3.0 had *no* decay phase); grad_clip 0.4; **qk-norm is
  essential at this LR**. Re-sweep LR if the model size or token budget changes materially.
- Context: consider **raising ctx to 2048–4096 for the math/reasoning stage** — multi-step traces need room, and
  the arch-search explicitly flagged ctx as an *unmeasured capability lever* (ctx1024 was chosen for loss/hour).
- Known infra caveat: this env has **flash-attn-4, not 2**, so the sliding window is silently ignored (full
  attention). Fine at ctx1024; re-verify if you extend context or rely on the window.

---

## 3. Concrete recommended plans (given ~3 GPUs)

**Option A — best capability/compute (recommended): smaller model, over-trained, rich mix.**
- ~**1.3–1.7B** params (same arch family), **~400–600B tokens** on the §2.2 mix ⇒ **~300–450 tokens/param**.
- @~8–9B tok/day (smaller model → higher tok/s) ⇒ **~6–10 weeks** on 3×H200.
- Reasoning-heavy mid-train + cooldown (§2.3), eval-gated (§2.4).
- Rationale: 1.3B @ 400:1 will out-reason 2.88B @ 22:1 by a wide margin, for a realistic wall-clock. This is the
  SmolLM/Qwen-0.5B playbook.

**Option B — keep 2.88B, extend + re-mix.**
- Extend to **~300–500B tokens** (5–8× current), add code + reasoning to the mix, reasoning anneal.
- ~**2.5–4 months** on 3×H200 (the cost of staying at 2.88B).

**Option C — cheapest, do-now, partial: fix the anneal on the current run.**
- Leave params/tokens as-is, but repurpose the **not-yet-started cooldown (~9.6B) + mid-train (~6B)** for a
  math/code/reasoning/tool anneal (§2.3). Won't fully break the ceiling (still 22:1 pretrain), but is the
  highest-leverage change achievable without more compute, and directly testable via `clean_eval.py`.

**If more GPUs become available:** scale Option A/B tokens linearly; the token budget is the binding constraint,
not the recipe.

---

## 4. Tool use — bake it into the base, don't bolt it on downstream

The 3.0-think experience: tool-calling added at **CoT-SFT** (`build_mix_v7.py tool_calc` tier) was learned
perfectly (`think_v7`: 100% valid `<tool_call>`) but **washed out of the conservative weight-soup** (shipped v4
emits ~0% tool calls), and only paid off with a serving-side **tool-execution loop** (`reasoning/tool_decode.py`:
53%→100% on single-op arithmetic). Lesson for the base:
- Put `<tool_call>`/`<tool_response>` formatted traces (calculator/python) into the **pretrain/mid-train mix**
  (`Nemotron-SFT-Agentic-v2`, synthesized `tool_calc`-style traces) so tool-use is a *base* capability, not a
  fragile downstream layer that a soup can dilute below its activation threshold.
- Still ship the **tool-execution loop** at serving — even a strong base benefits from real arithmetic offload.
- Also carry over the **external-verifier reranker** (`reasoning/ext_verify.py`): on 3.0-think it lifted the
  deployable metric ~41→~70 (+28pt) by cashing the pass@K ceiling. A stronger base raises that ceiling, so the
  same recipe pays off more.

---

## 5. What NOT to repeat (anti-patterns from 3.0 / 3.5-as-configured)
- ❌ Don't make **CE-per-GPU-hour** the terminal objective — it optimizes toward the ceiling, not past it.
- ❌ Don't ship a **code-free, math-light** pretrain mix.
- ❌ Don't stop at **Chinchilla-optimal** tokens for a model meant to reason.
- ❌ Don't rely on **downstream SFT/soup** to add capability or tool-use — 6+ nulls prove it only *redistributes*
  a fixed capability (greedy ↔ self-consistency trade) and can *dilute* learned behaviors (tool-calling).
- ❌ Don't gate only on loss/PPL — a model at PPL 14 can still be a strong-for-its-class reasoner or a weak one
  depending entirely on the mix and token count.

---

## 6. Expected payoff & how to measure
- **Target:** clear grade-school math where 3.0/3.5 stall (~25% greedy) — a code+math+reasoning base at
  300–500+ tokens/param should reach **GSM8K-raw 50–70%+** and clean SVAMP/ASDiv well above the ~25–30% greedy /
  ~78% pass@32 that 3.0-think plateaued at. That would genuinely raise `pass@32` (the fixed ceiling that no
  downstream method could move) — the definition of "breaking the ceiling."
- **Measure with the honest judges** (never GSM8K as held-out): `reasoning/clean_eval.py` (SVAMP/ASDiv/MAWPS +
  GSM-Plus, Wilson CIs), `reasoning/probe_pretrain_ckpt.py` for intermediate checkpoints, and the 4-quadrant
  general guardrail so world-knowledge isn't sacrificed for math.

**Bottom line:** the recipe is not the problem — **compute (tokens) and mix are.** Spend the extra pretraining
compute (ideally on a smaller, over-trained model with code + math + reasoning + tool data and a reasoning
anneal), gate on capability, and the reasoning ceiling that blocked every downstream effort on 3.0-think will
finally move.
