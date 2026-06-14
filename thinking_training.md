# Training a Reasoning Model from Scratch — Argonne 3.0

A walkthrough of every stage we went through to turn a freshly-initialized
2.88B transformer into a chain-of-thought ("thinking") model, written for
learning. Each section says **what** we did, **why**, and **what we learned** —
because most of the real lessons came from the things that *didn't* work.

> TL;DR of the journey: pretrain → SFT → DPO → CoT-SFT gave us a model that
> *formats* reasoning but can't reliably *reason*. We traced the failure to
> weak pretraining numeracy (not model size), fixed format/fact errors with
> calibrated CoT data, then hit a wall where supervised methods saturate. The
> frontier now is **reinforcement learning with verifiable rewards (RLVR)**.

---

## 0. The model architecture (what we're training)

Argonne 3.0 is a ~2.88B parameter decoder-only transformer. The choices that
matter for the rest of this doc:

| Component | Value | Why it matters |
|---|---|---|
| Hidden size / layers | 3072 / 24 | Mid-size; small enough to train on 1 GPU, big enough to chat |
| Attention heads | 12 query / 4 KV (**GQA**) | Grouped-query attention → smaller KV cache, faster inference |
| Vocab | 151,669 | Qwen-style tokenizer |
| Positional encoding | **RoPE**, θ=1e6 | Rotary embeddings; high θ lets us extend context later |
| Context length | 1024 → 13568 → 4096 | Grown across stages (see below) |
| Embeddings | **Tied** (lm_head = embed_tokens) | Saves params; causes the benign "lm_head MISSING" load warning |
| Norms | sandwich_norm, qk_norm, v_norm | Stability tricks for training |
| Attention | Interleaved local window 256 | *Ignored at runtime* on the SDPA/math path (no flash-attn on our nodes) |
| logit_softcap | 15.0 | Caps logits to prevent blow-up |

Key practical fact: **the model has no attention-mask / padding support** — its
forward pass forces `attention_mask=None` (pure causal). This constraint shaped
every inference and RL decision later (batching, padding direction).

---

## 1. Pretraining — teach it language (`pretrain.py`)

**What:** Train the randomly-initialized model on a large general text corpus
with the standard next-token-prediction objective (cross-entropy on every
token). Short context (1024) for throughput.

**Key knobs we got right (and had to fix):**
- Learning rate **3e-4** (production), not the 6e-4 in the argparse default.
- Real effective batch size and grad clipping (production values, not defaults).
- `torch.compile` on for speed.

**Why:** This is where the model learns grammar, facts, and — crucially —
*arithmetic patterns*. Everything downstream inherits the quality of this stage.
If pretraining never saw enough math, no amount of fine-tuning fully fixes it
(this becomes the central plot point in §6).

**Related:** `midtraining.py` / `continue_pretrain.py` continue/extend
pretraining (e.g., context-length extension via RoPE) before instruction tuning.

---

## 2. Supervised Fine-Tuning (SFT) — teach it to follow instructions (`sft.py`)

**What:** Fine-tune the base model on (prompt, response) chat pairs, training
only on the assistant tokens. Context extended to 13568. Output → `sft_ckpts`.

**Why:** The base model just continues text; it doesn't answer questions. SFT
teaches the chat format (`<|im_start|>user … <|im_start|>assistant …`) and the
behavior of *responding* rather than *continuing*.

**What we learned:** After SFT the model is a fluent chatbot for single-step
factual questions (capital of France, who wrote Romeo & Juliet ≈ 7/10) — but it
**cannot do arithmetic** ("100 ÷ 4 = 0.5", "17 − 5 is a popular online
platform"). The instruction-following worked; the reasoning did not. First hint
that the problem is upstream.

---

## 3. DPO — align preferences (`dpo.py`)

**What:** Direct Preference Optimization on (prompt, chosen, rejected) triples.
DPO nudges the model to prefer "chosen" responses over "rejected" ones directly,
without training a separate reward model. Output → `dpo_ckpts`.

**Why:** Polishes tone, helpfulness, and format consistency. It's a preference
method, not a capability method — it doesn't teach new skills like arithmetic.

**What we learned:** DPO maintained the chatbot quality but did **not** improve
math (same arithmetic failures as SFT). Confirms: preference alignment ≠
reasoning ability.

---

## 4. CoT SFT — teach it to "think" (`cot-sft.py`)

**What:** Fine-tune on reasoning traces where the assistant turn contains
`<think> … </think>` followed by the answer. The chat template (Qwen3-style)
parses the `<think>` span into a separate reasoning field. Context 4096. Started
from the DPO checkpoint. Output → `think_ckpts`.

**Why:** This is the step that *creates the reasoning model*. By imitating
long worked-solutions, the model learns to produce a chain of thought before
answering — the format that, in well-trained models, unlocks multi-step problems.

**What we learned (the painful part):** The first CoT model was a regression in
disguise.
- It learned the *format* (`<think>…</think>`) but the long traces **injected
  arithmetic errors and anchored on them** — e.g. it would write "7×6=42"
  mid-trace then conclude "answer: 7".
- It fell into **enumeration/repetition loops** ("Sam is taller than Bob but Sam
  is shorter than Bob…" forever).
- Heavy contamination from a narrow training set (OpenR1-Math + codeforces) made
  it dump **codeforces-style JSON / Python** for plain questions.
- Net effect: thinking mode scored *worse* than no-think mode on basics
  (0/10 math-CoT). **The mandated long trace itself was the pathology.**

This kicked off a diagnosis phase.

---

## 5. Diagnosis — is it the decoder, the data, or the model size?

Before throwing more compute at it, we built honest evals (`eval_think.py`,
`eval_numeracy.py`) and ruled out causes one by one. **This is the most
important methodological lesson in the whole project.**

**Decoder bugs first (cheapest to fix).** Early "gibberish" was partly an
*inference* artifact, not a training failure:
- `config.json` had `eos_token_id=null`, so generation never stopped on
  `<|im_end|>` — it ran to max length and rambled. Fix: set eos explicitly.
- A `from_pretrained` buffer bug + **prompt-inclusive n-gram bans** in the eval
  decoder produced garbage. Fix: a self-healing `from_pretrained` (re-ties
  lm_head, rebuilds RoPE buffers) and a clean decode loop (stops on eos,
  penalizes only *generated* tokens, no prompt-inclusive bans).
- Logged training loss looked alarming but was **inflated ~4× by grad-accum
  scaling** — a reporting artifact, not divergence.

**Then a controlled capability probe.** We ran the same basic-arithmetic
questions across base → SFT → DPO → think, no-think, greedy:
- base: just echoes (not instruct-tuned — expected).
- SFT/DPO: fluent but **0 arithmetic correct**.
- think: actually the **best** of the four (7×6=42 ✓, half of 80=40 ✓, correct
  *procedures*) — but slips on single-digit facts (17−5→16).

**Conclusion:** the failure is **arithmetic-fact errors inherited from weak
pretraining numeracy**, *not* a 2.88B "model too small" ceiling (well-trained 3B
models do this cold) and *not* "CoT SFT broke it" (CoT actually helped vs DPO).
The lever is **data**, upstream. This reframing drove everything after.

---

## 6. Data calibration — fix facts, format, and looping with better CoT data

If the problem is *what the model was trained on*, fix the diet. We built
stratified mixes with `build_sft_mix.py` and re-ran CoT SFT (same config).

**Mix v1 (~21k, `think_mix_ckpts`).** Added easy gsm8k, OpenMathReasoning, MATH
lvl1-3, high-quality Opus traces, and some *direct* (no-think) examples.
- Required a `cot-sft.py` change: `--allow_non_reasoning 1` keeps direct targets
  instead of dropping every non-`<think>` row.
- Result: MATH-with-CoT **0 → 3/10**; looping mostly gone; 7×6=42 now *survives*
  the CoT instead of self-overriding.
- But: GENERAL no-think **regressed 8 → 6** — too math-heavy a diet eroded
  everyday chat. (Lesson: a fine-tune is a zero-sum diet; over-index on one
  skill and others fade.)

**Mix v2 (~97k, `think_mix2_ckpts`) — all goals hit.** Rebalanced:
- Added a **synthetic-arithmetic tier** (15k short, verified `\boxed` traces) to
  *drill fact execution* directly.
- Pushed general/chat data back up to ~51% (tulu no-think + ultrachat) to undo
  the v1 regression.
- Result: MATH-with-CoT **6/10** (17−5=12, 8+3=11, 100÷4=25 all correct now;
  looping gone, all traces close `</think>`); MATH no-think ~5; GENERAL no-think
  back to **8**; GENERAL-with-CoT ~3-4.

**What we learned:** Data calibration *works* and is the highest-leverage knob:
fact drills fix facts, balance fixes regressions, easy data fixes looping. But
the *residual* failures — `2x+5=17`, `sum 1..10`, divisor counting, the
non-numeric logic puzzle — are **multi-step reasoning chains**, and those did
NOT yield to more supervised data. That's the boundary where RL comes in.

---

## 7. Building the inference engine — KV cache (`model.py`, `verify_cache.py`)

**What:** Implemented a proper **KV cache** (`past_key_values` + `use_cache`)
through GQA / RoPE / qk-norm / the block stack: prefill the prompt once, then
feed one token per step reusing cached keys/values.

**Why:** Naive generation recomputes the *whole sequence* every new token —
O(n²) and ~20s per problem. Any sampling-based method (STaR, GRPO) needs
*thousands* of rollouts; without a cache it's ~256× too slow to be practical.

**How we verified it (`verify_cache.py`):** prefill-once logits **bit-exact**
(diff 0.0) vs the no-cache forward, token-by-token argmax **100% match**, ~10×
faster. The training path stays byte-identical (`use_cache=False`). This also
fixed the broken `generate()`.

**Lesson:** correctness-gate your infra before building on it. A subtly wrong
cache would silently poison every downstream RL gradient.

---

## 8. STaR — offline RLVR by rejection sampling (`star_generate.py`)

**What:** STaR (Self-Taught Reasoner) is the *offline* form of RL with
verifiable rewards:
1. For each problem, sample **K** traces from the current model.
2. **Verify**: keep only traces whose final `\boxed{}` answer equals the gold.
3. **SFT** the model on its own verified-correct traces.

The reward (correct answer) is baked in by *filtering* — you only imitate
successes. We reuse the cached sampler and a numeric verifier (`extract_boxed`,
`norm`).

> Batching note: because the model has no padding support, we batch **K
> identical copies of one prompt** (same length, no padding) and iterate
> problems sequentially.

**Round 1** (from `think_mix2`, K=12, 1200 gsm8k): pass@12 ≈ 18.6%, **365**
correct traces kept (64% of samples never closed `</think>` within the token
budget — the yield ceiling). SFT on 365×4 + a 5k anchor → `think_star_ckpts`.
Result: marginal-but-real (fixed 2x+5=17, but 100÷4 regressed; net ~flat).

**Round 2** (from the round-1 model, K=12, gsm8k + MATH lvl1-3, **max_new 400→
512**): pass@12 **29%**, unclosed collapsed **64% → 24%**, **1530** traces kept.
Cumulative 1888 unique traces ×4 + 5k anchor (`build_star_sft.py`) → SFT from the
stable `think_mix2` base → `think_star2_ckpts`.
Result (4-quadrant eval):
- MATH-with-CoT 6 → **7** (100÷4 fixed back to 25; easy arithmetic stable).
- MATH no-think **regressed to ~3** — started dumping `import sympy` instead of
  answering directly. The trace set was 60% of the data; that over-specialized
  the model toward long solution-style output.
- GENERAL no-think 8 → 7 ("the sun is not a star, it is a planet" — yikes).

**What we learned:** **STaR saturates.** More verified traces buy marginal
fact-stability on the CoT path but do **not** fix reasoning-chain correctness
(the sum-loop, 2x+5, and logic puzzle persist across *all three* checkpoints).
And weighting the trace set too heavily (60%) erodes the direct-answer path.
Lessons: cap the STaR fraction (~≤30% held no-think in round 1), and recognize
that imitating filtered samples can't teach what the model can't already
occasionally do.

---

## 9. GRPO — online RLVR (`grpo.py`) ← we are here

**What:** Group Relative Policy Optimization puts reward on the **full rollout**,
online, instead of filtering+imitating:
1. For each problem, sample a **group** of G traces from the *current* policy.
2. Reward each: 1.0 if it closes `</think>` **and** the `\boxed` answer is
   correct, else 0.0 (a *verifiable* reward — no learned reward model).
3. **Group-relative advantage**: `A_i = (r_i − mean(group)) / (std(group)+ε)`.
   Above-average traces get pushed up, below-average down — the group is its own
   baseline (no critic network needed).
4. Policy-gradient update `−A·logπ`, with a **KL leash** to a frozen reference
   model (`β·KL`) so the policy improves without collapsing or forgetting.

**Why this can succeed where STaR couldn't:** STaR only ever *imitates* whole
correct traces. GRPO shapes the policy with a *continuous* signal over the
group — it can learn from the *difference* between a slightly-better and
slightly-worse attempt, and the KL term keeps general ability intact. It
directly optimizes the thing we actually want (verified correctness) rather than
a supervised proxy.

**Design decisions forced by this model:**
- **Right-padding is safe** (pure causal attention): a real token at position i
  never attends to trailing pad tokens, so we right-pad each group and mask the
  loss to real tokens — no attention mask needed.
- **Sample and score from the same distribution**: we sample at temperature T
  with *no* top-k/top-p truncation and compute log-probs as
  `log_softmax(logits/T)`, so the policy gradient is unbiased.
- **One inner update per batch** → importance ratio = 1, so the clipped
  surrogate reduces to the group-baseline policy gradient; we keep the KL (k3
  estimator: `exp(d)−d−1`, unbiased & non-negative).
- Start from `think_star2` (highest pass rate → densest reward → most learning
  signal), KL reference = same checkpoint, skip groups with zero reward variance
  (no signal).
- KV cache makes the rollout sampling fast enough to do this online at all.

**Status:** trainer written and smoke-tested (load → cached rollout → verify →
group advantage → KL → gradient step → checkpoint). Full run launching on the
gsm8k set; eval will use the same 4-quadrant probe to compare against
`think_mix2` / `think_star` / `think_star2`.

---

## The throughline (what this whole project teaches)

1. **Capability is set upstream.** Pretraining quality (here: numeracy) is the
   ceiling; fine-tuning calibrates and unlocks, it rarely creates from nothing.
2. **Diagnose before you train.** Half the early "model is broken" was actually
   decoder/eval bugs and a loss-reporting artifact. Cheap fixes first.
3. **Fine-tuning is a zero-sum diet.** Over-index on one skill and others
   regress; balance the mix and re-measure all axes (the 4-quadrant eval).
4. **Format ≠ reasoning.** CoT SFT teaches the *shape* of thinking; getting the
   *content* right needed data drills, then RL.
5. **Supervised imitation saturates.** STaR helps then plateaus, because you can
   only imitate successes you already produce.
6. **RLVR is the frontier lever.** Reward on verifiable outcomes (math answers)
   with a KL leash is how you push past the imitation ceiling — and a correct,
   fast inference path (KV cache) is the prerequisite that makes it tractable.

---

## File map

| File | Stage |
|---|---|
| `pretrain.py`, `midtraining.py`, `continue_pretrain.py` | §1 Pretraining / context extension |
| `sft.py` | §2 SFT |
| `dpo.py` | §3 DPO |
| `cot-sft.py` | §4, §6 CoT SFT (`--allow_non_reasoning`) |
| `model.py` | architecture + §7 KV cache |
| `eval_think.py`, `eval_numeracy.py` | §5 diagnostics / 4-quadrant eval |
| `build_sft_mix.py` | §6 calibrated data mixes |
| `verify_cache.py` | §7 cache correctness gate |
| `star_generate.py`, `build_star_sft.py` | §8 STaR (offline RLVR) |
| `grpo.py` | §9 GRPO (online RLVR) |
