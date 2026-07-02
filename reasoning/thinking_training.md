# Training a Reasoning Model from Scratch — Argonne 3.0

A walkthrough of every stage we went through to turn a freshly-initialized
2.88B transformer into a chain-of-thought ("thinking") model, written for
learning. Each section says **what** we did, **why**, and **what we learned** —
because most of the real lessons came from the things that *didn't* work.

> ## ⭐ LATEST RESULT (2026-07-02, §15): the recipe WORKS — it was always the base
>
> **One line:** running the *identical* reasoning recipe (intermix → SFT → DPO → CoT on the
> same `cot_sft_mix_v3`) on two **off-the-shelf** bases produced reasoning models that
> **pass all four quadrants** — something **no** Argonne from-scratch checkpoint ever did.
>
> | 4-quadrant (think) | **Qwen1.5-0.5B** | **Llama-3.2-1B** | best Argonne |
> |---|---|---|---|
> | MATH no-think / +CoT | **10 / 10** | **10 / 10** | ~1 / 5–7 |
> | GENERAL no-think / +CoT | **8 / 8** | **9 / 8** | ~8 / good |
>
> Both solve the §10 residuals with clean traces AND keep general chat. The **0.46B** Qwen
> matches the 1.24B Llama on math — so **base *quality* (numerate AND knowledgeable), not
> size, was the ceiling**; a small balanced base beats the 2.88B lopsided one. This closes
> the arc: throughline #1 ("capability is set upstream") proven in the affirmative — every
> §5–§13 lever (STaR/GRPO/data-calibration/FineMath) was fighting an upstream deficit the
> recipe never needed to fix given a healthy base. New base-agnostic harness:
> `reasoning/reason_control/`. Intermix was mildly *negative* on these already-good bases
> (it's a base-repair tool, not universal). Full details in **§15**.
>
> ---
>
> ## Earlier result (2026-06-28): the FineMath base BREAKS the §10 ceiling
>
> **The one-line takeaway:** the inherited-numeracy ceiling that blocked every
> prior stage (§5–§10) is **gone on the FineMath-midtrained base**. A thinking
> model built on it scores **10/10 on math-with-CoT and solves all four §10
> residual failures** (`2x+5=17`, `1+…+10=55`, perimeter, divisor-count) with
> clean, concise, correct traces — something *no* prior checkpoint (mix2, star2,
> grpo2, mix3) ever did. Data calibration / STaR / GRPO could never move these;
> **a better base did, immediately.** Capability is set upstream (throughline #1),
> now proven in the affirmative.
>
> **What was run (a deliberately CHEAP check, not the full recipe — see §11):**
> 1. **Few-shot base probe (no training, ~3 min):** raw bases on 20 arithmetic /
>    multi-step problems → `argonne-3.0-base` **3/20**, longmino Phase-1 **1/20**,
>    **FineMath Phase-2 16/20**. The math midtraining lifted base numeracy hugely;
>    residual misses were the §10 signature (right setup, one slipped step).
> 2. **Short direct CoT-SFT from the FineMath base** (`cot_sft_mix_v3`, 2500 steps,
>    **skipping general SFT/DPO** to stay cheap) → `think_finemath`, then the
>    4-quadrant eval vs the old-base baselines.
>
> **4-quadrant eval (`report/finemath_*.log`):**
>
> | quadrant | think_finemath (NEW base) | think_mix3 (OLD base, *same* v3 data) | think_mix2 | think_star2 |
> |---|---|---|---|---|
> | **MATH + CoT** | **10/10** ✅ | 6/10 | ~4/10 | 6/10 |
> | MATH no-think | ~7/10 | ~0/10 (degenerate `\boxed{first#}`) | ~4/10 | ~3/10 |
> | GENERAL no-think | **~1–2/10** ❌ | ~7/10 | ~8/10 | ~7/10 |
> | GENERAL + CoT | **~0/10** ❌ | ~6/10 | ~5/10 | ~5/10 |
>
> **The catch (and it's an artifact of the shortcut, not the base):** the cheap
> check skipped general SFT/DPO and trained only 2500 steps on a math-heavy mix,
> so `think_finemath` is a **math savant that's broken on general chat** (loops:
> "the capital of France is France itself…"). The old-base baselines went through
> the full SFT→DPO and keep general ability. This is the **zero-sum-diet lesson**
> (§6) in extreme form. The fix is known: run the **proper pipeline** on the
> FineMath base (general SFT → DPO → CoT with a *balanced* mix v4 that restores
> the no-think/general share) to keep the math win *and* recover general/no-think.
>
> **Net:** the FineMath base is a real, large step up for reasoning; the next run
> should do the full balanced pipeline on it (ideally on a *later*, more-trained
> FineMath checkpoint — this was an early ~1.9B-token snapshot). Details in §11.
>
> ---
>
> TL;DR of the journey: pretrain → SFT → DPO → CoT-SFT gave us a model that
> *formats* reasoning but can't reliably *reason*. We traced the failure to
> weak pretraining numeracy (not model size), fixed format/fact errors with
> calibrated CoT data, then hit a wall where supervised methods saturate. RLVR
> (STaR, then GRPO) cleaned up behavior but did **not** lift held-out
> multi-step reasoning. Targeted, verified multi-step data finally did — it's
> the only lever that taught the multi-step *procedure* (solving `2x+5=17`,
> counting divisors) — but it only **relocated** the ceiling: the model now
> structures the solution correctly yet still slips on the elementary
> *arithmetic* inside it. The bottleneck is now fact-execution (numeracy),
> which traces straight back to pretraining.

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

## 1.5 Handoff — from a midtrained base into the reasoning pipeline (VERIFIED)

This is the seam between "we just finished (mid)training a new base" and "now
make it reason." It is the entry point you (future agent) will use after the
**FineMath math-injection midtraining** (the Phase-2 auto-switch wired into
`midtraining.sh`/`weekend.sh`/`night.sh`) completes.

**What midtraining leaves you.** When a midtraining phase reaches its token
target, `midtraining.py:save_final_model_artifacts` writes a **plain HF model
dir** — `config.json` + `model.safetensors` + tokenizer + `chat_template.jinja`:
- Phase 1 (longmino): `/project/rcc/youzhi/models/midtrain/final_model_complete_longmino`
  (renamed in §16 — `models/midtrain` is now the live INTERMIX checkpoint dir, and a
  dir literally named `final_model_complete` there acts as the phase-done marker)
- Phase 2 (FineMath): `/project/rcc/youzhi/models/midtrain_finemath/final_model_complete`

That dir is a **base LM, not a chat model**, and it has **no `auto_map`/modeling
file** baked in. Two consequences:
1. You **cannot** `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`
   it directly (there's no remote code). You load it by supplying the
   architecture explicitly — which is exactly what `sft.py` (`from model import
   ArgonneModel`) and `cot-sft.py` (`--model_def <repo>/model.py`) already do.
   *Verified:* building `ArgonneModel` from its `config.json` and loading its
   `model.safetensors` gives 2.88B params, `unexpected=0`, only
   `missing=[lm_head.weight]` (the benign tied-embedding case).
2. Because it's a base model, you must run **SFT → DPO** before any CoT work —
   you can't CoT-SFT a raw base into a good reasoner (it won't follow chat
   format; that's literally what §2–§3 are for).

**The turnkey chain (run all of it from the repo root):**
```bash
BASE=/project/rcc/youzhi/models/midtrain_finemath/final_model_complete   # new numerate base

# §2 SFT  (root sft.py / sft_instruct.sh) -> new sft_ckpts
sbatch sft_instruct.sh        # set MODEL_PATH=$BASE
# §3 DPO  (root dpo.py / dpo_instruct.sh) -> new dpo_ckpts
sbatch dpo_instruct.sh        # starts from the new sft_ckpts
# §4/§6/§10 Reasoning CoT-SFT  (this dir) -> think_* ckpts
sbatch reasoning/cot_sft_instruct.sh   # MODEL_PATH=<new dpo_ckpts>, DATA_PATH=<a build_*_mix dataset>, --model_def <repo>/model.py
# grade every checkpoint
sbatch reasoning/star_eval.sh          # edit the M2/S2/M3 paths first
```
(Keep these manual/stage-gated — the whole project lesson is to *eval and decide*
between stages, §8 throughline. Do **not** auto-chain SFT→DPO→CoT-SFT unattended.)

**Why we did the FineMath phase at all, and what to try first on the new base:**
§10 showed the residual failure is *arithmetic-fact execution* (`8+3=7` inside an
otherwise-correct procedure) — an inherited-numeracy ceiling. FineMath midtraining
attacks that ceiling at the pretraining objective (digit-split tokenizer makes it
learnable). So the first reasoning experiment on the new base should be a
**mix v4**: keep the multi-step procedure tier from v3 but *restore* a strong
direct/no-think + arithmetic-fact-drill share (undo v3's no-think collapse) — now
that the base should actually *have* the numeracy for the drills to stick. Then,
with correct procedures + better facts, RLVR (§9) finally has a base worth amplifying.

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

## 9. GRPO — online RLVR (`grpo.py`)

**What:** Group Relative Policy Optimization puts reward on the **full rollout**,
online, instead of filtering+imitating:
1. For each problem, sample a **group** of G traces from the *current* policy.
2. Reward each trace with a *verifiable* reward (no learned reward model) — the
   `\boxed` answer is checked against the gold.
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

### Round 1 — a clean *null* result (and why)

First run: reward = **binary** (1.0 correct, else 0.0), from `think_star2`,
gsm8k, group G=8, 8 prompts/step, LR 1e-6, 400 steps. It ran cleanly for ~7h —
and **changed nothing**. The 4-quadrant eval of `think_grpo` was within noise of
`think_star2` (same wins, same failures, same "the sun is not a star" error).
The training logs said why before the eval did:
- **KL stayed pinned at ~0.0002 the entire run** → the policy never moved.
- **Reward never trended up** (flat 0.05–0.23).
- Only **~3 of 8 groups** carried any gradient each step.

**The root cause is a reward-design trap, and it's the key lesson of this
section.** With a binary reward and a *group-relative* advantage, a group
produces **zero gradient whenever all G traces get the same reward** — i.e. all
correct *or* all wrong. On a weak model most groups are all-wrong (and the easy
ones are all-right), so two-thirds of every batch was gradient-dead. Combine that
with a timid LR (1e-6) over only 400 steps and the net update is negligible.
GRPO didn't fail — it was *starved of signal*.

### Round 2 — the fix: dense reward shaping

The cheap, decisive fix is to **grade the reward** so groups almost always have
*variance* (and therefore a non-zero advantage), instead of an expensive
difficulty-curriculum pre-pass:

```
correct (closed + boxed==gold)      → 1.0
closed + has boxed, wrong answer    → 0.3
closed, no parseable boxed          → 0.15
stopped but never closed </think>   → 0.0
never stopped (truncated/looping)   → −0.2
```

Two things fall out of this ordering for free:
1. **Almost every group now carries gradient.** Even an all-wrong group differs
   on *how* wrong (some closed, some looped) → non-zero variance → signal.
2. **It directly attacks the dominant failure.** The biggest with-CoT pathology
   was degenerate enumeration loops that never close `</think>` (the logic puzzle
   literally counted `T=3…120`). Ranking *closed > looping* puts explicit
   downward pressure on exactly that behavior — something RL is good at and SFT is
   not. (We keep `is_correct` as a *separate* logged metric so accuracy stays
   honest — shaping can flatter the reward, never the gold-check.)

Plus the obvious knobs round 1 got wrong: **LR 1e-6 → 5e-6**, **8 → 12
prompts/step** (denser, lower-variance gradient), and run to a wall-clock budget
(~11h) instead of a fixed 400 steps.

**Smoke test confirmed the fix before committing a GPU slice:** `signal_groups`
jumped from ~3/8 to **6/6 every step**, with finite loss/grad-norm and no
instability. That single number — every group now contributes — is the whole
ballgame.

**Result — the policy moved, the capability didn't.** Round 2 ran on H200 (LR
5e-6, 12 prompts/step, G=8, ~510 steps before the wall cap). This time the
training signal was real: **KL climbed to ~0.0025 (12× round 1)**, the shaped
reward rose (degenerate groups collapsed toward `closed→1.0`), and training-set
accuracy showed noisy peaks ~0.29. The policy genuinely *moved*. But the
held-out 4-quadrant eval (H100) showed **zero gain**: `think_grpo2 ≈
think_star2`, and MATH-with-CoT even dipped 7 → 6. We had successfully maximized
a *shaped reward on gsm8k* without improving *held-out correctness* — a textbook
**reward-proxy / train-test gap**.

**What we learned:** fixing the gradient (round 1 → round 2) was necessary but
not sufficient. A moving policy that optimizes a verifiable reward *on the
training distribution* still doesn't generalize if the underlying capability
isn't there to amplify — RLVR sharpens what the model can already occasionally
do; it doesn't manufacture a missing skill. Three independent methods —
**data calibration (mix v2), STaR, and a properly-configured GRPO** — now agree
on the same verdict: the bottleneck is upstream capability, specifically
**multi-step reasoning chains**, not the RL recipe. So we returned to the one
lever with a proven track record of *moving* the held-out number: targeted data.

---

## 10. Targeted multi-step data (`build_mix_v3.py`)

**What:** A focused data tier aimed at the *exact* failures that survived every
prior stage. The 4-quadrant eval keeps failing four multi-step families across
*every* checkpoint (base → SFT → DPO → mix2 → star2 → grpo2):
1. two-operation linear algebra (`2x + 5 = 17 → x = 6`),
2. sequential / series sums (`1 + 2 + … + 10 = 55` — also the loop trap),
3. formula-then-substitute geometry (perimeter `= 2(l+w)`),
4. divisor counting (# positive divisors of 12 = 6).

`build_mix_v3.py` generates short, **correct-by-construction** `<think>` traces
for exactly these families — every number is computed in Python, and each trace's
`\boxed` answer is re-verified with the *same* extractor (`extract_boxed`,
`norm`) we use for RLVR. We keep **all of mix v2 as the anchor** (the zero-sum-diet
lesson from §6 — don't let a narrow tier erode general/no-think). Final mix:
~97k v2 anchor + a multi-step tier (`ms_algebra` 5000, `ms_series` 5000,
`ms_geometry` 5000, `ms_divisors` 1290 — the divisor family has a small natural
unique-question ceiling) = **113,341 rows**.

**Why this and not more RL:** §9 just showed RL can't amplify a skill the model
doesn't have. If the model has never reliably *seen* a correct two-step solution
for these families, give it clean, verified ones directly. This is the same lever
(data calibration) that took MATH-with-CoT 0 → 6 in §6 — applied surgically to
the residual failures instead of broadly.

**Result — it moved the PROCEDURE where RL couldn't, but not the arithmetic
ceiling.** We trained 1-epoch CoT-SFT (from `dpo_ckpts`, ctx 4096, LR 1e-5),
cancelled at ~77% of the epoch (loss 2.70 → 0.81, clean), and evaluated
`checkpoint-6000` on an H100 across all four quadrants vs `think_mix2` (baseline)
and `think_star2`. The headline is *qualitative*, not the raw counts:

- **The two hardest target families are now solved — only on this checkpoint.**
  `2x + 5 = 17 → x = 6` (full, correct derivation) and "divisors of 12 → 6"
  (`12 = 2²×3`, `(2+1)(1+1) = 6`). mix2, star2, *and* grpo2 all failed both
  (enumeration loops, `import sympy` dumps, or garbage). And the `</think>`-loop
  pathology is **gone** on the math-reasoning path — every trace closes.
- **But the residual misses are now pure single-step arithmetic-fact slips
  inside otherwise-correct procedures.** Sum 1..10: uses `n(n+1)/2` correctly but
  substitutes `n = 8` → 36. Perimeter: uses `2(l+w)` correctly but computes
  `8 + 3 = 7` → 14. Plus trivia it used to get right now wobble (`8+3=9`,
  `100/4=20`). **This is the inherited-numeracy ceiling laid bare: targeted data
  installed the multi-step *procedure* — exactly the thing RL couldn't move — but
  it cannot install the elementary arithmetic *facts* the procedure runs on.**

The cost of a 100%-thinking-format tier was real: **MATH no-think collapsed** to a
degenerate "The answer is `\boxed{X}`" with no reasoning (0/10), and GENERAL+CoT
got noisier (loops, a hallucinated multiple-choice frame on "Red Planet", "green"
listed as a primary color). **GENERAL no-think held at 8/10** — the mix v2 anchor
did its job (it even fixed star2's "the sun is a planet" error), confirming the
zero-sum-diet lesson once more.

**What we learned:** data calibration is, again, the *only* lever that visibly
moved multi-step reasoning — it did for the procedure what three rounds of RL/STaR
could not. But it merely *relocated* the bottleneck: from "can't structure a
multi-step solution" to "can't execute the arithmetic inside it." The next levers
follow directly: (1) re-add a strong direct/no-think + arithmetic-fact-drill share
to undo the no-think collapse and pressure fact accuracy (the tier was too
thinking-only); (2) RLVR is now *better* positioned than in §9 — with correct
procedures in place, a graded reward on final-answer correctness carries denser
signal than it did on grpo2's procedure-less base.

---

## 11. Re-running the recipe on the FineMath numerate base (DONE — base ceiling broken)

§10 left a sharp, testable hypothesis: the residual failures are *arithmetic-fact
execution* slips inside otherwise-correct procedures — an **inherited-numeracy
ceiling** from pretraining. The FineMath midtraining phase (`midtraining.py`,
launched via `night.sh`/`weekend.sh`) attacks exactly that. So the cleanest next
experiment is to **re-run the whole reasoning recipe on the FineMath base and
A/B it against the old base with the same CoT data**.

**Pinning the moving base (important gotcha).** FineMath midtraining was still
running, and it keeps only the latest `checkpoint_step_*.pt` (a new filename each
save, old ones deleted ~every 30 min) — there was **no `final_model_complete`
dir yet**. So we *pinned* a snapshot: copied `checkpoint_step_768847.pt` to
`/project/rcc/youzhi/models/midtrain_finemath_pinned/` and extracted a standalone
HF base dir (`final_model_complete/`) with `reasoning/extract_finemath_base.py`
(mirrors `midtraining.save_final_model_artifacts`: ctx 13568, **rope_theta=1e4**,
trims embeddings, copies tokenizer + chat_template). The pinned snapshot is an
*early* FineMath checkpoint — only ~1.9B FineMath tokens on top of the 16B Phase-1
(longmino) tokens (cumulative midtraining loss 1.83).

> **Verified base health (and a re-confirmation of the §5 lesson).** A CPU/fp32
> manual-load probe gave loss ~8.8 on plain English — alarming, until the control:
> the *known-good* `argonne-3.0-base` (which produced a working chatbot) scored
> ~9.5 on the **same harness**, and Phase-1 longmino ~9.6–10. The CPU eval path is
> the artifact, not the weights (exactly §5). On that probe the FineMath base is
> the **best** of the three bases — encouraging. The honest judge remains the GPU
> 4-quadrant eval.

**The chain (all in `reasoning/`, all on 1× H100, all auto-resubmitting slices).**
New output dirs throughout so the old-base baselines (`sft_ckpts`, `dpo_ckpts`,
`think_*_ckpts`) are preserved for comparison:
```
sft_finemath.sh  (base=pinned final_model_complete) -> sft_finemath
   └─auto─▶ dpo_finemath.sh                          -> dpo_finemath
              └─auto─▶ cot_finemath.sh (DATA=cot_sft_mix_v3, ROPE_THETA=1e4) -> think_finemath
                         └─auto─▶ eval_finemath.sh (4-quadrant)
```
H100-specific sizing keeps the effective batch identical to the H200 originals
(SFT 4×5=20; DPO 2×4=8; CoT 4×3=12 ≈ think_mix3's tbs 11). **CoT uses
ROPE_THETA=1e4** to match the FineMath base — the old cot launcher defaulted to
1e6 (correct only for the old θ=1e6 base); using 1e6 here would corrupt the model.

**The key A/B:** `think_finemath` (mix-v3 CoT on the numerate base) vs
`think_mix3_ckpts/checkpoint-6000` (**same** mix-v3 CoT, old θ=1e6 base) — isolates
exactly the math-injected base. If the §10 hypothesis is right, the arithmetic-fact
slips inside correct procedures should shrink. *(Reusing v3 — rather than building
the §1.5 "mix v4" — is the tighter experiment: one variable, the base. Mix v4
remains the documented follow-up.)*

**What we ACTUALLY ran (we pivoted from the full pipeline to a cheap check).**
The full `sft_finemath → dpo_finemath → cot_finemath` chain above *was* launched
and ran SFT to ~73% — but it's slow/expensive on 1 GPU (a day+), and the real
question was just "did the math base help reasoning?". So we **cancelled** the
full pipeline and answered it two cheap ways:

1. **Few-shot base probe** (`reasoning/quick_base_probe.py`, training-free, ~3 min):
   raw bases on 20 arithmetic/multi-step problems, `ArgonneModel.from_pretrained`
   (the self-healing loader), greedy, 4-shot. Result: `argonne-3.0-base` **3/20**,
   longmino Phase-1 **1/20**, **FineMath Phase-2 16/20**. The numeracy lift is
   real and large; residual misses were §10-style (right setup, one slipped step,
   e.g. `2x+5=17 → "17−5=12"`). This alone answers the question.
2. **Short direct CoT-SFT from the FineMath base** (`cot_finemath.sh` with
   `MODEL_PATH=<pinned base>`, `cot_sft_mix_v3`, `MAX_STEPS=2500`, **no general
   SFT/DPO**) → `think_finemath`, graded by `reasoning/eval_finemath.sh`.

**Result — the §10 ceiling is broken (4-quadrant, `report/finemath_*.log`):**

| quadrant | think_finemath | think_mix3 (old base, same v3) | think_mix2 | think_star2 |
|---|---|---|---|---|
| MATH + CoT | **10/10** | 6/10 | ~4/10 | 6/10 |
| MATH no-think | ~7/10 | ~0/10 (degenerate boxed) | ~4/10 | ~3/10 |
| GENERAL no-think | ~1–2/10 | ~7/10 | ~8/10 | ~7/10 |
| GENERAL + CoT | ~0/10 | ~6/10 | ~5/10 | ~5/10 |

- **MATH+CoT is a clean sweep.** `think_finemath` solves all four §10 residuals
  (`2x+5=17→6`, `1+…+10=55` via `n(n+1)/2`, perimeter `2(8+3)=22`, divisors of
  12 via `2²·3→(2+1)(1+1)=6`) with short correct traces. `think_mix3` (same v3
  data, OLD base) still slips: `8+3=9`, `100/4=20`, `sum→45 (n=9)`, `perim→18`.
  **Same CoT data, same residual-failure list — the ONLY change is the base.**
- **The cost is general ability**: `think_finemath` is a math savant that loops on
  general chat ("capital of France is France itself…"). Expected — it skipped
  general SFT/DPO and trained 2500 steps on a math-heavy mix from a base whose
  general English the FineMath phase had already narrowed. Pure zero-sum-diet (§6).

**The fix for next time** is the proper balanced pipeline on the FineMath base:
general SFT → DPO → CoT-SFT with a **mix v4** (restore the direct/no-think +
general share per §1.5) to keep the math win *and* recover general/no-think. Use a
**later FineMath checkpoint** than this early ~1.9B-token snapshot. The
`reasoning/{sft,dpo,cot}_finemath.sh` launchers are ready for that run.

### Operational lessons from this run (so the next agent doesn't re-pay them)
- **Pin the base immediately.** The live midtraining dir was *accumulating* a
  36 GB checkpoint every ~30 min (it does NOT auto-prune). Copy one out + extract
  an HF dir with `reasoning/extract_finemath_base.py` before it grows/changes.
- **The `test` partition has NO time cap here (`sinfo` TIMELIMIT=infinite).** Run
  each stage as ONE continuous job (`--time=1-00:00:00`, `EXIT_AFTER_CHECKPOINT_SAVE=0`,
  `SLICE_TIME_LIMIT=0`). The 30-min exit/resubmit "treadmill" was self-imposed and
  made wall-time ~60% non-compute (requeue + reload + HF's growing data-skip on
  every resume). Continuous is dramatically faster.
- **OOM is deterministic and will loop.** Per-device batch 4 OOM'd on a rare
  ~13k-token batch (SDPA full attention, peak ~ batch·seq²); on auto-resume it
  re-hit the same batch and looped. Fix: halve batch (`4→2`), double grad-accum to
  keep effective batch, resume. SFT/CoT here run **batch 2**.
- **`eval_numeracy.py` needs repo-root `model.py` on `sys.path`** to register the
  `argonne2` arch with `AutoModelForCausalLM` (else `KeyError: 'argonne2'`). Fixed
  in-script (it now adds `SCRIPT_DIR.parent`); these base/CoT dirs have no
  `auto_map`/modeling file baked in (§1.5).
- **Disk:** cleaned ~2.1 TB of accumulated/abandoned checkpoints; kept the pinned
  `midtrain_finemath_pinned/final_model_complete` (the base in use) and the live
  latest `midtrain_finemath/checkpoint_step_818607.pt`.

### New files added in `reasoning/` for this experiment
| Script | What |
|---|---|
| `extract_finemath_base.py` | Pin a midtraining `.pt` → standalone HF base dir (ctx 13568, θ=1e4). |
| `quick_base_probe.py` / `.sh` | Training-free few-shot numeracy probe across bases (the cheapest progress signal). |
| `sft_finemath.sh`, `dpo_finemath.sh`, `cot_finemath.sh` | H100, continuous-run, FineMath-pathed copies of the SFT/DPO/CoT launchers (θ=1e4, batch 2, auto-chain SFT→DPO→CoT→eval). |
| `eval_finemath.sh` | 4-quadrant eval of `think_finemath` vs `think_mix3/mix2/star2`. |

---

## 12. The balanced pipeline on the *latest* FineMath base (DONE — refutes §11's "use a later checkpoint" fix)

§11 ended with a prescription: run the **proper balanced pipeline** (general
SFT → DPO → CoT-SFT) on a **later FineMath checkpoint** to keep the math win *and*
recover the general ability that the cheap §11 check had lost. This section runs
exactly that experiment — and the answer is a clean, surprising **no**.

**What we ran.** Full reasoning recipe on the **latest** FineMath midtraining
checkpoint `checkpoint_step_833124.pt` (~20.5B midtraining tokens — an *order of
magnitude more* FineMath than §11's pinned `768847`/~1.9B snapshot):
```
extract_finemath_test_base.py (ckpt 833124 → HF base, ctx 13568, θ=1e4)
  └▶ sft_test.sh   (UltraChat general SFT)          -> midtrain_finemath_test/sft
       └▶ dpo_test.sh  (KatoHF chatbot_arena, chat_refine_strict) -> …/dpo
            └▶ cot_test.sh (cot_sft_mix_v3, θ=1e4)   -> …/think
                 └▶ eval_test.sh (4-quadrant)        -> report/finemath_test_*.log
```
1× H100, job name `test`, 1-hour auto-resubmitting slices, all in a throwaway
`midtrain_finemath_test/` dir. Reused **mix v3** (not the still-unbuilt mix v4) to
keep the experiment one-variable against §11 (the base is the only change vs
`think_mix3`; the SFT+DPO stages are the only change vs §11's `think_finemath`).

> **Bug worth remembering (cost a stalled day).** SFT auto-chains to DPO via
> `sbatch dpo_test.sh`, and SLURM's default `--export=ALL` **leaks the finished
> stage's exported env into the next job**. SFT's exported `DATA_PATH=ultrachat`,
> `OUTPUT_DIR=…/sft`, `DATASET_RECIPE=chat_refine_strict` clobbered DPO's
> `${VAR:-default}`s → DPO tried to build preference pairs out of UltraChat →
> "kept 0 unique rows" → `RuntimeError: Could not construct any valid DPO sample`
> → crash in ~1m47s, whole chain dead. **Fix:** `unset` every config var at the top
> of each chained launcher (done in `dpo_test.sh` and `cot_test.sh`), so the
> correct defaults always win; keep only `RESUME_FROM_CHECKPOINT` for self-resume.
> After the fix DPO read chatbot_arena / `…/dpo` and kept 204 valid pairs.

**Result — a hard math↔general trade-off that the balanced pipeline could NOT undo**
(`report/finemath_test_{math,gen}_{nt,th}.log`; NEW = `midtrain_finemath_test/think`):

| quadrant | **NEW (latest FineMath, full SFT→DPO→CoT)** | think_mix3 (old base, same v3) | think_mix2 | think_star2 |
|---|---|---|---|---|
| MATH no-think (greedy) | **10/10** clean & terse | ~0/10 (degenerate boxed) | ~1/10 | ~1/10 |
| MATH + CoT (sample) | **10/10** short correct traces | 5/10 | ~5/10 (loops) | ~5/10 (loops) |
| GENERAL no-think (greedy) | **~0/11** | ~8/11 | ~8/11 | ~7/11 |
| GENERAL + CoT (sample) | **~1/11** (only "Paris") | good | good | good |

- **Math is the best of any model we've trained, in *both* modes.** 10/10 no-think
  *and* 10/10 with-CoT, with short non-degenerate traces — it never falls into the
  `x=2 x=2 x=2…` / `7+3=10 → 4+3=7 →…` repetition loops that swallow mix2/star2's
  and even mix3's harder items. `think_mix3` (same v3 data, old base) still slips
  `8+3=9`, `100/4=20`, `sum→90`, `perim→14`. **The numerate base is doing all of it.**
- **General ability is catastrophically gone — worse than §11's savant, and the
  full general SFT+DPO did NOT rescue it.** No-think, NEW answers "capital of France
  is *France*", "the sun is a *planet*, not a star", "photosynthesis = breaking down
  food into CO₂", and loops on Shakespeare / primary colors. With-CoT it usually
  can't even *close* `</think>` on a general question (the trace loops until the
  token budget runs out); it recovers only "Paris". The old-base baselines answer
  all of these correctly.

**Why this refutes §11's fix.** §11 blamed the lost general ability on *skipping*
general SFT/DPO and guessed a *later* checkpoint + balanced pipeline would fix it.
Both guesses are wrong here: we **did** the full general SFT (UltraChat) + DPO, on a
**much later** checkpoint, and general got **worse**, not better. The culprit is the
**FineMath base itself**: ~20.5B tokens of math-heavy midtraining **catastrophically
forgot world/general knowledge** (§6's zero-sum diet, but on the *pretraining* side),
and ~250M tokens of downstream SFT cannot re-teach facts the base no longer holds.
More FineMath ⇒ better numeracy **and** deeper forgetting — the `768847`→`833124`
step moved *both* dials the wrong way for general. This is a genuine
**capability–capability trade-off in the base**, not a fine-tuning-recipe problem.

**Corrected recommendation (supersedes §11's "use a later checkpoint").**
1. **Use an *earlier* FineMath checkpoint**, not a later one — find the knee where
   numeracy has lifted (§11's `768847` already probed **16/20**) but general
   knowledge hasn't yet collapsed. Sweep a few early `checkpoint_step_*` with the
   training-free `quick_base_probe.py` **plus** a general-knowledge probe.
2. **Better: fix the midtraining recipe, not the checkpoint.** Interleave a general
   web/chat replay share into FineMath midtraining (anti-forgetting) so the base
   *keeps* Paris/Mars/oxygen while gaining arithmetic. Then the balanced pipeline
   (+ mix v4 for the no-think/general share, §1.5) can win both quadrants. **mix v4
   alone cannot save a base that has already forgotten the capital of France.**

All `midtrain_finemath_test/` checkpoints were deleted after grading (throwaway
A/B); the launchers `reasoning/{extract_finemath_test_base.py,sft_test,dpo_test,cot_test,eval_test}.sh`
remain for re-running against an earlier/replayed base.

---

## 13. What to do about general ability — and should we keep midtraining math?

Put §11 and §12 side by side and the shape of the problem is clear:

| base | FineMath tokens | downstream MATH (CoT) | general chat |
|---|---|---|---|
| old `argonne-3.0-base` | 0 | 5–6/10 (residual slips) | **~8/11 ✅** |
| FineMath `768847` (§11) | ~1.9B | **10/10 ✅** | broken (but skipped SFT/DPO) |
| FineMath `833124` (§12) | ~20.5B | **10/10 ✅** | **~0/11 ❌ (full SFT/DPO couldn't save it)** |

**The decisive observation: the math benefit saturated early, the forgetting did not.**
Going from ~1.9B → ~20.5B FineMath tokens bought **~0 additional** elementary-numeracy
(both bases already max our probe at 10/10) while turning a recoverable savant into a
base that has *forgotten the capital of France* and can't be fine-tuned back. On this
axis, every FineMath token past ~2B is nearly pure downside. *(Honest caveat: our math
probe is 10 easy items and saturates trivially — this says nothing about hard/competition
MATH, where more math tokens might still help. If hard math is a goal, measure it before
concluding. But for a general assistant with solid numeracy, the diet has overshot.)*

### Should we keep midtraining math?
- **Continue the *current* pure-FineMath run? No.** It's already idle at step `864124`
  and its elementary-numeracy gain saturated ~`768847`; more pure-math tokens only
  deepen catastrophic forgetting (`768847`→`833124` made general *worse*).
- **Keep injecting math at all? Yes — but never as a pure diet.** The right knob is a
  **replay mix**, not more math-only tokens.

### The plan for general ability, cheapest first
1. **Confirm the diagnosis + find the knee (do this first, ~10 min).** `quick_base_probe.py`
   is math-only; add a **general-knowledge base probe** (few-shot Paris/Mars/oxygen/…)
   and run it on `old-base` vs pinned `768847` vs latest `864124`. This directly tests
   the "the *base* forgot" claim (currently inferred from downstream, not measured on the
   base) and locates where general collapses. *(Note: `midtrain_finemath/` keeps only the
   latest `.pt`; the only early base we still have is the pinned `768847` — so the sweep
   is effectively old-base / 768847 / latest, not a dense curve.)*
2. **Balanced pipeline on the *earliest* good base (reuse §12 launchers).** Point
   `extract_finemath_test_base.py` + `{sft,dpo,cot,eval}_test.sh` at pinned `768847`
   instead of `833124`. If its base general-probe is healthier, the same SFT→DPO→CoT
   should retain more general while keeping the 10/10 math. Cheap — launchers exist.
3. **The durable fix — replay-mix midtraining (change the recipe, not the checkpoint).**
   Edit `midtraining.py`'s Phase-2 data recipe to interleave **~40–60% general/longmino
   replay** with FineMath (standard continued-pretraining anti-forgetting), and re-run.
   A base that *keeps* world knowledge while gaining numeracy is the only thing that lets
   the balanced pipeline (+ mix v4, §1.5) win **both** quadrants. Most compute, but it's
   the principled answer and it's what §12 proved we can't skip.
4. **Cheap side-experiment — model soup.** Weight-average the FineMath base ⊕ the old
   general base and probe both quadrants. A fast read on whether the two capabilities are
   even linearly reconcilable; if a merged base scores decently on both, it's a near-free
   base for the pipeline.

**Recommendation:** stop treating "more FineMath tokens" as progress (it saturated); do
(1) now to measure the trade-off honestly, run (2) as the cheap near-term model, and plan
(3) replay-mix midtraining as the real fix. Do **not** resume the pure-math run as-is.

### Measured — base probe on BOTH axes (2026-07-01, `report/base-probe-general.out`)

Step (1) above, run. `reasoning/base_probe_general.py` (few-shot greedy: 20 math +
15 world-knowledge items) on the three raw bases. The result **overturns §12's
"FineMath caused the forgetting" story and kills the intermix-from-longmino plan:**

| base | MATH /20 | GENERAL /15 | character of outputs |
|---|---|---|---|
| `pretrain/argonne-3.0-base` | 3 | **13** | knowledgeable, innumerate — the proven-general base |
| `midtrain/` longmino (Phase-1) | 2 | **5** | **degraded + degenerate** ("largest planet is the blue.krone", "opposite of hot is the red") |
| `midtrain_finemath/864124` (Phase-2) | **18** | 6 | numerate but factually hollow ("capital of Japan is Japan", "first president: Kennedy") |

- **Longmino — the proposed canvas — is NOT healthy.** It already fell 13→5 on general
  knowledge (and generates degenerately) while gaining *nothing* on math (3→2). The
  **Phase-1 context-extension midtraining did most of the world-knowledge damage —
  before FineMath ran at all.** Its only real contribution was long context.
- **FineMath is not the main culprit.** On top of longmino it went 5→6 general (flat)
  while lifting math 2→18. §12 attributed the catastrophic forgetting to FineMath; the
  base probe shows general was *already* destroyed at longmino.
- **Destruction, not suppression.** Few-shot prompting could NOT surface
  Tokyo/Washington/Portuguese from the FineMath base — consistent with §12's finding
  that full SFT+DPO couldn't recover general. Fine-tuning can't rebuild lost facts.
- *(Instrument caveat: 15-item keyword-matched general probe is rough, but the
  qualitative outputs — "blue.krone", "capital of Japan is Japan" — corroborate the scores.)*

**Corrected canvas + plan (supersedes the "intermix from longmino" idea above).**
The only base with intact world knowledge is **`pretrain/argonne-3.0-base` (13/15)**. So:
- **Do NOT intermix from longmino** (already forgotten), and do not treat longmino as a
  required step: the old general-good models (mix2/mix3, ~8/11) were SFT'd **directly
  from `argonne-3.0-base`**, getting long context via RoPE extrapolation at SFT time
  (§11 note) — longmino was never on the general-good path.
- **Intermix midtraining from `argonne-3.0-base`:** one balanced continued-pretraining
  run mixing FineMath (numeracy) + general/web replay (hold general near 13), **skipping
  longmino**. Target a base at ~13 general AND ~18 math. Smoke-test on a few-hundred-M
  slice, re-probe both axes with `base_probe_general.py`, then scale. The few-shot probe
  is cheap enough to tune the math:general ratio against directly before the full run.

---

## 14. Implementing the intermix fix — the midtraining launchers now do it (DONE 2026-07-01)

§13's corrected plan is now **wired into the production launchers**. `weekend.sh`,
`night.sh`, and `midtraining.sh` were switched from the old two-phase
longmino→FineMath chain to a **single INTERMIX phase**: seed the healthy pretrain
base and train it on a doc-shuffled FineWeb+FineMath mix, writing to `models/midtrain`.

**What changed (minimal, backward-compatible).**
- `midtraining.sh`: added a `DOC_SHUFFLE` knob (default 0) + `--doc_shuffle` on the
  torchrun call. **This was the critical missing piece** — the trainer never passed
  `--doc_shuffle`, so an intermix manifest would have been read in manifest order
  (all FineWeb docs, then all FineMath docs) = *sequential* training = the exact
  catastrophic forgetting we're fixing. `DOC_SHUFFLE=1` globally permutes docs each
  epoch so the two sources truly interleave.
- `weekend.sh` / `night.sh`: export `DATA_OVERRIDE=<intermix manifest>`,
  `DOC_SHUFFLE_OVERRIDE=1`, `ROPE_THETA_OVERRIDE=1000000` (match the argonne-3.0-base
  θ=1e6 regime, the proven-general path — NOT FineMath's θ=1e4), and `PHASE2_DATA=""`
  (single phase). Seed (`pretrain/checkpoint_step_329148.pt`) and output
  (`models/midtrain`) are the existing Phase-1 defaults, so nothing else moved. The
  auto-resubmit chain / slices / FSDP / wall-time saves are all untouched.
- Result: **`bash weekend.sh` (continuous chain) or `bash night.sh` (one 8h slice at
  23:00)** now runs the intermix midtraining from the base. `midtrain/` had no loose
  `.pt` checkpoints (only an old `final_model_complete`), so it seeds *fresh* from the
  base rather than resuming longmino.

**The intermix corpus (`reasoning/build_intermix.py`).** midtraining.py's
`DocManifestDataLoader` takes ONE `block_size`-token window per doc per epoch, so the
effective **token mix equals the DOC-count ratio**, not raw corpus size. The builder:
- references all 64 FineMath doc-bin shards directly (absolute paths), and
- carves a matching number of docs from **FineWeb** (`.../CC-MAIN-2025-21-binary/train.bin`
  — the pretrain corpus, the source of the base's 13/15 general knowledge) into
  `DOC_LEN`-token docs, writing one doc-bin shard + `.lengths.npy`, then
- emits a merged manifest with `tokenized_dir="/"` and absolute paths (so it can draw
  from two different tokenized_dirs). `GENERAL_RATIO` (default 1.0 = 50:50) tunes the
  split; bump to 1.5 (60:40 general) to lean harder against forgetting.
- **Gotchas:** `DOC_LEN` must be `> block_size` (13570 for the 13568 production block)
  or the loader raises "Short doc window"; and the FineWeb `.bin` has a **1024-byte
  header (256×int32)** before the uint32 tokens (see `pretrain.py: offset=256*4`) —
  read from byte 0 and you get garbage token IDs. Production manifest lives at
  `/project/rcc/youzhi/data/intermix/intermix_manifest.json` (28GB FineWeb slice +
  64 FineMath shards, 520,066 docs each, 50:50).

**Test-drive (validated it RUNS; NOT yet that the result is good).** Ran `weekend.sh`
~32 min: it seeded argonne-3.0-base (step 329148, phase token counter reset to 0, data
from the start), trained ~985 steps on the θ=1e6 doc-shuffled intermix, and saved
`midtrain/checkpoint_step_330133.pt` — then we cancelled. So the end-to-end pipeline
(seed→intermix data→train→checkpoint) is confirmed working. A real `weekend.sh` run
**resumes from 330133** (the test-drive work continues, not wasted).

**Still open (do this on the first real run).** We have NOT empirically confirmed the
50:50 mix preserves general knowledge — the smoke validation was cancelled before its
probe. On the first production checkpoint, probe both axes:
`EXTRA_CKPT=<midtrain .pt> EXTRA_THETA=1000000 python reasoning/base_probe_general.py`,
and if GENERAL slips below ~13/15, rebuild with `GENERAL_RATIO=1.5` and relaunch.
> **UPDATE (2026-07-02, §16): measured.** At ~644M intermix tokens the probe read
> MATH 12/20 / **GENERAL 11/15** — below the threshold, so the rule fired: the
> manifest was rebuilt at `GENERAL_RATIO=1.5` (60:40). Details + a full pipeline
> audit (several latent bugs found & fixed) in §16.

**New files (this experiment).**
| File | What |
|---|---|
| `reasoning/base_probe_general.py` / `.sh` | Few-shot BASE probe on BOTH axes (20 math + 15 world-knowledge); `EXTRA_CKPT`/`EXTRA_THETA` env add a checkpoint to compare. The §13 measurements + the intermix validator. |
| `reasoning/build_intermix.py` | Builds the production intermix manifest (FineWeb + all FineMath, DOC_LEN/GENERAL_RATIO configurable). |
| `reasoning/build_intermix_smoke.py`, `intermix_smoke.sh` | Small block-2048 smoke variant (build + short midtrain + auto-probe) for a fast read before the full run. |
| `weekend.sh`, `night.sh`, `midtraining.sh` (edited) | Now default to single-phase intermix from the base → `midtrain` (doc_shuffle, θ=1e6). |

---

## 15. The decisive control — run the recipe on REAL bases (Qwen1.5-0.5B, Llama-3.2-1B) — DONE 2026-07-02

Every section §1–§14 fought the same enemy: a *from-scratch* base whose capability was
set upstream (throughline #1). §13's base probe made it quantitative — no Argonne base
ever had numeracy AND world knowledge at once (argonne-3.0-base 3/20 math·13/15 gen;
longmino 2/20·5/15; FineMath-864124 18/20·6/15). The obvious, never-run experiment:
**take real, off-the-shelf, well-pretrained bases and run the IDENTICAL recipe.** If the
recipe is sound and only the base was the bottleneck, a good base should yield a reasoning
model that passes the 4-quadrant eval — which NO Argonne checkpoint ever did (each failed
≥1 quadrant). We ran two bases, spanning the "quality" axis: **Llama-3.2-1B** (1.24B, a
strong 1B base) and **Qwen1.5-0.5B** (0.46B — the "should be worse" base).

### The base probe overturns the premise: BOTH real bases are strong on BOTH axes
`reason_control/probe.py`, the same 20-math / 15-general few-shot probe as §13:

| base | params | MATH /20 | GENERAL /15 | character |
|---|---|---|---|---|
| argonne-3.0-base | 2.88B | 3 | 13 | innumerate (from-scratch) |
| longmino (Phase-1) | 2.88B | 2 | 5 | degraded both |
| FineMath-864124 | 2.88B | 18 | 6 | numerate, amnesiac |
| **Llama-3.2-1B** | 1.24B | **13–14** | **15** | strong both |
| **Qwen1.5-0.5B** | 0.46B | **14** | **14** | strong both |

**Even a 0.46B off-the-shelf base clears the numeracy ceiling that blocked the 2.88B
from-scratch model AND keeps world knowledge** — the both-axes health no Argonne base ever
had. Qwen's math-heavy pretraining shows even at 0.5B (it solved "divisors of 12 = 6" cold,
which the Llama-1B base missed). The hypothesis that the 0.5B model "should perform worse"
is already FALSE at the base-probe level. (Instrument caveat: greedy few-shot, single-fact;
±1 run-to-run bf16 wobble on math.)

### Method — a generic, base-agnostic recipe harness (`reasoning/reason_control/`)
The Argonne scripts are welded to the custom ArgonneModel + Qwen-3 tokenizer + FSDP/doc-bin
machinery. Rather than bend them, we re-implemented the SAME recipe as small, model-agnostic
scripts (plain HF + hand-rolled training loops; both bases are standard HF Llama/Qwen2):
1. **INTERMIX midtrain** (the §14 recipe, re-tokenized): the prebuilt intermix `.bin` is
   Qwen-3-tokenized — incompatible with these bases' own tokenizers — so we re-stream the raw
   FineWeb + FineMath parquet, tokenize with the *base's own* tokenizer, doc-shuffle 50:50,
   pack to 1024, continued-pretrain **80M tokens at a gentle LR 5e-5** (a strong base needs a
   light touch, not the from-scratch 1e-4).
2. **SFT** on UltraChat, **DPO** on argilla/dpo-mix-7k (hand-rolled DPO loss + frozen ref),
   **CoT-SFT** on the SAME `cot_sft_mix_v3` used for `think_mix3`/`think_finemath` — so the
   only variables vs the Argonne runs are the base + tokenizer.
3. **4-quadrant eval**: the exact §5 probes, {math,general}×{no-think greedy, with-CoT
   sampled}, auto-graded + full dumps.

Chat-family is auto-detected (ChatML for Qwen `<|im_start|>…<|im_end|>` vs Llama-3 headers);
training-time label masking and eval-time `apply_chat_template` are verified token-identical
per family. Runtime **HBM autotuner** sizes the batch for whatever H100 (80G/96G) SLURM gives.

### Intermix effect on an already-strong base (base → after 80M-token intermix @5e-5)
- Qwen1.5-0.5B: MATH 14→12   GENERAL 14→13   (mild drop both axes)
- Llama-3.2-1B: MATH 13→12   GENERAL 15→15   (general fully preserved; math −1)

CONFIRMS the expectation: on a base that's *already* numerate the intermix step is
neutral-to-mildly-**negative** (a little forgetting), NOT the big lift it gave the deficient
Argonne base (§14). The gentle 5e-5 LR kept the damage small (a 1e-4 smoke dropped Qwen math
to 11/20); the larger Llama-1B was more robust on general. **The intermix phase is a
base-repair tool, not a universal good — for a healthy base, skip it or keep the LR tiny.**

### 4-quadrant eval — YES: the recipe yields a working reasoning model on BOTH bases
| quadrant | **Qwen1.5-0.5B think** | **Llama-3.2-1B think** | best Argonne (think_*) |
|---|---|---|---|
| MATH no-think | **10/10** | **10/10** | ~0–1/10 (degenerate `\boxed`) |
| MATH + CoT | **10/10** | **10/10** | 5–7/10 |
| GENERAL no-think | **8/10** | **9/10** | ~8/11 |
| GENERAL + CoT | **8/10** | **8/10** | good |

**Both bases are a clean pass on all four quadrants — which NO Argonne checkpoint ever
achieved.** Each Argonne best failed ≥1 quadrant: `think_finemath`/`think_test` hit 10/10 math
but ~0/10 general (savant / catastrophic forgetting); `think_mix2/mix3` kept ~8/11 general but
**failed math no-think** (degenerate `\boxed{first#}`) and only reached 5–7/10 math-CoT. Here
the 0.46B Qwen MATCHES the 1.24B Llama on math (10/10 both modes) and trails by one on general
no-think (8 vs 9, tracking the base-probe gap 14 vs 15). Both produce textbook with-CoT traces
that solve all four §10 residuals: `2x = 17−5 = 12, x = 6`; `n(n+1)/2 = 10·11/2 = 55`;
`2·(8+3) = 22`; `12 = 2²·3 → (2+1)(1+1) = 6`. General misses were minor phrasing/keyword slips,
not the "capital of France is France" collapse of the FineMath base. Full runs finished in
~1h42 (Qwen) / ~3h (Llama) on one H100 each.

### What this control establishes
1. **Throughline #1, proven in the affirmative — the recipe was never the problem, the base
   was.** The exact pipeline that produced a quadrant-failing model on the 2.88B from-scratch
   base produces a **clean four-quadrant pass** on two off-the-shelf bases, using the *same*
   `cot_sft_mix_v3` data as `think_mix3`. All the §5–§13 heroics (STaR, GRPO, mix v1/v2/v3,
   FineMath midtraining) were fighting an upstream deficit; give the recipe a base healthy on
   **both** numeracy and world knowledge and it just works.
2. **"Smaller should be worse" is refuted — base QUALITY, not size, was the lever here.** The
   0.46B Qwen matched the 1.24B Llama on math and trailed by one on general, because it is
   already strong on both axes. A small-but-balanced base beats a large-but-lopsided one (the
   2.88B FineMath base was 18/20 math but amnesiac and unrecoverable). Capacity would likely
   bite on *harder* competition math; on this grade-school→early-algebra bar it did not.
3. **The intermix midtraining step is base-repair, not a universal good** (see above): mildly
   negative on both already-strong bases; SFT→DPO→CoT recovered (final general 8–9/10).
4. **Practical:** `reasoning/reason_control/` is a reusable, base-agnostic runner of the whole
   recipe (probe → intermix → SFT → DPO → CoT → 4-quadrant eval) on any HF base, with a runtime
   HBM autotuner for mixed 80G/96G H100s. Each full run fit in one H100 in <3.5h.

### Operational lessons (so the next agent doesn't re-pay them)
The model-agnostic re-implementation was easy; getting it to run *fast and full-HBM on a
mixed cluster* was where all the time went:
- **Memoize tokenizer family-detection.** `tok.get_vocab()` builds the full 128k–152k-entry
  dict; calling it inside the per-example chat renderer (twice) cost ~75 ms/example and made a
  data build *hang* ~50 min (looked like a training stall). Cache it once → ~80× faster (a
  113k-row CoT build drops to ~1 min). Also load the HF column once and shuffle *indices in
  memory* — `ds.shuffle()` + row iteration random-accesses the project FS and is pathological.
- **HBM autotuner: measure sustained RESERVED, not single-step ALLOCATED.** A single
  fwd+bwd+step on a fresh allocator reports `max_memory_allocated` ≈ 95% at a batch that then
  OOMs; OOM tracks `max_memory_reserved` (incl. fragmentation), which stabilizes only after
  ~10 steps. The LM-head **logits `(bs,seq,vocab)` upcast to fp32 for the loss** are the real
  ceiling (tens of GB), not the transformer. Force **grad_accum=1** (accumulation keeps a
  prior micro-step's param-sized `.grad` resident — memory the single-step measurement never
  saw). A CUDA OOM's **traceback pins the failed tensors**, so a naive reduce-and-retry
  *cascades* (28→10); null the step's locals + `gc.collect()` + `synchronize()` +
  `empty_cache()` before backing off. Net: autotuner selects ~96%, sustained settles ~80–96%
  after ≤1 clean backoff.

### New files (this experiment) — `reasoning/reason_control/`
| File | What |
|---|---|
| `common.py` | Chat-family autodetect + token-identical `render_chat`, HBM autotuner, time-boxed train loop, the §5 probe/eval question sets. |
| `probe.py` | Both-axes few-shot base probe (== §13's 20-math/15-general). |
| `midtrain.py` | Re-tokenized FineWeb+FineMath doc-shuffled intermix (base's own tokenizer). |
| `sft.py` / `dpo.py` / `cot.py` | UltraChat SFT / argilla DPO (hand-rolled) / `cot_sft_mix_v3` CoT-SFT. |
| `eval.py` | 4-quadrant auto-graded eval. |
| `run_all.sh` | One continuous, resumable, time-boxed H100 job; `BASE_MODEL_PATH` picks the base. |

**Bottom line:** stop trying to fix reasoning downstream. A good base + this modest recipe is
a *0.5B reasoning model that passes every quadrant* — better all-around than anything built on
the 2.88B from-scratch base. If a from-scratch Argonne base is still the goal, the target is
explicit: get it to ~14/20 math AND ~14/15 general *simultaneously* (what Qwen-0.5B already
has), then this recipe finishes the job.

---

## 16. First intermix probe + full pipeline audit (2026-07-02) — 50:50 → 60:40, five latent bugs fixed

The §14 intermix run had its first real slices (test-drive to step 330133, then one
8h night.sh slice to **345108** ≈ **644M intermix tokens**, ~78M tokens/h on 3×H200).
Two things happened today: the §14-prescribed probe was run on the newest checkpoint,
and the whole launcher/trainer/data pipeline got a line-by-line audit.

### The probe (`report/base-probe-intermix-345108.out`) — the decision rule fired

| base | MATH /20 | GENERAL /15 |
|---|---|---|
| argonne-3.0-base (seed) | 3 | 14 |
| longmino (Phase-1) | 2 | 5 |
| FineMath 864124 (Phase-2) | 18 | 6 |
| **intermix @ 644M tok** | **12** | **11** |

- **The intermix design works directionally**: math 3→12 with only ~322M math tokens
  (pure FineMath needed ~1.9B for 16/20), while general held at 11 instead of
  collapsing to 5–6 like both sequential phases. The training-log loss oscillation
  (~1.1 on math docs ↔ ~2.9 on web docs) is the visible signature of real doc-level
  interleaving.
- **But GENERAL slipped below the §14 threshold (~13/15)** → per the pre-registered
  rule, the manifest was **rebuilt with `GENERAL_RATIO=1.5` (60:40)**. Verified safe:
  55.2B FineWeb tokens available vs 10.6B needed; and `midtraining.py`'s
  metadata-mismatch resume branch handles an in-place manifest change cleanly (keeps
  the doc cursor, resets epoch — a harmless random skip). Bonus: 60:40 raises epoch
  capacity to ~17.6B tokens, so the 16B target is now reachable inside `max_epochs=1`
  (at 50:50 one epoch was only 14.11B — see the zombie-chain bug below).
  **Never rebuild while a slice is training** — `build_intermix.py` opens
  `fineweb_slice.bin` with `"wb"`, truncating the file a live loader is memmapping.
- Going forward: probe **every night slice** (`EXTRA_CKPT=<latest .pt>
  EXTRA_THETA=1000000`, ~15 min on 1×H100). Stop rule: **MATH ≥14 with GENERAL ≥13**
  (the §15 bar). If GENERAL <11 at 60:40, next levers: LR 3e-4 → 1e-4 (repair wants a
  lighter touch; §15 used 5e-5 on healthy bases) or `GENERAL_RATIO=2.0`.

### The audit — what was verified CORRECT
- `doc_shuffle` is a **global, per-epoch-seeded permutation across all 65 shards**
  (`midtraining.py:_refresh_doc_order`), and resume restores the exact permutation +
  doc position — no repeats, no skips.
- The data itself is clean: both sources decode to coherent text, uint32 dtype and the
  FineWeb 1024-byte header handled correctly, and **every doc in all shards is
  ≥13,570 tokens** (FineMath min 16,384) so the "Short doc window" crash can't fire.
- Checkpoint saves are atomic (tmp + `os.replace`) with a truncated-checkpoint
  fallback scan; θ=1e6 and all overrides survive every sbatch hop including
  auto-resubmit and failure-retry chains; slice 134 resumed exactly where the
  test-drive stopped.

### Bugs found & FIXED (all latent — none tainted the checkpoints trained so far)
1. **Zombie resubmit chain**: a finished single-phase run (token target or epoch end)
   wrote `final_model_complete` but `midtraining.sh` only had a done-check for the
   two-phase path — an AUTO_RESUBMIT=1 chain would launch 1-step slices forever, each
   writing a 36 GB checkpoint. Fixed: single-phase completion gate at slice start + a
   don't-resubmit guard when the marker was written this slice.
2. **Stale `midtrain/final_model_complete` (the old longmino final)** was sitting in
   the live intermix checkpoint dir. It (a) was the `PHASE1_DONE_MARKER` that would
   flip a slice to sequential FineMath if `PHASE2_DATA` ever leaked in, (b) would be
   silently overwritten when intermix finishes, (c) was the probes' longmino baseline
   path. **Renamed to `final_model_complete_longmino`**; all script references updated
   (`base_probe_general.py`, `quick_base_probe.py`, both extract scripts).
3. **`night.sh` PHASE2_DATA leak**: unlike `weekend.sh`, night.sh didn't pin
   `PHASE2_DATA=` (empty) in the GPU job's `--export` list, so a stale
   `export PHASE2_DATA=…` in the submitting shell (the §12 `--export=ALL` leak class)
   would silently re-enable the sequential FineMath phase. Fixed: appended
   `PHASE2_DATA=` to the wrap's EXTRA_EXPORT.
4. **`eos_token_id: null` written at the source**: the §5 eos bug was patched in
   checkpoint dirs but `midtraining.py`'s config construction never set eos, so every
   future `final_model_complete` would re-introduce it. Fixed: config now takes
   eos/bos/pad from the tokenizer (also in `extract_finemath_base.py`).
5. **`extract_finemath_base.py` hardcoded θ=1e4 + the FineMath ckpt path** — reusing
   it on an intermix checkpoint would write a corrupt-config base (§11's gotcha,
   reversed). Now env-parameterized (`CKPT`, `ROPE_THETA`, `OUT`, `TOKENIZER_SRC`),
   with a guard refusing to create a `final_model_complete` marker next to live
   checkpoints. This matters because stopping at the probe knee (~4B tokens) means no
   final artifacts exist — the extract script IS the handoff path.
6. **FSDP grad clipping** used the plain `torch.nn.utils.clip_grad_norm_`, which under
   `shard_grad_op` clips each rank by its LOCAL shard norm (underestimated,
   rank-inconsistent). Fixed to `FSDP.clip_grad_norm_`. (pretrain/continue_pretrain
   are DDP — theirs was already correct. Mostly benign historically since clip=1.0
   rarely binds.)
7. **WSD cooldown/warmup used GLOBAL scheduler steps** while `estimated_steps` is
   phase-local — on a seeded run (scheduler resumes at ~329k) setting
   `COOLDOWN_OVERRIDE` would have collapsed LR instantly instead of annealing at the
   end. Fixed: the schedule is now phase-local and anchored to
   min(epoch-end, token-target). Side effect (intended): a **freshly seeded** phase
   now actually performs its `--warmup_steps` warmup; the in-flight resumed run is
   unaffected (its phase step is ~16k, past warmup).

(Checkpoint pruning was flagged too — ~72 GB/h accumulation — but quota is ample, so
it was consciously skipped.)

**Timeline math**: ~570M tokens per 8h night slice → the ~4B-token knee (≈2B math at
50:50; a bit later at 60:40) is ~5–7 more night slices or ~2 days of a weekend chain.

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
6. **RLVR helps, but it amplifies — it doesn't create.** Reward on verifiable
   outcomes with a KL leash (and a fast KV-cache inference path to make it
   tractable) can sharpen what the model already does occasionally, but it can't
   manufacture a missing skill. Our GRPO round 2 *moved the policy* and maximized
   its shaped reward on gsm8k yet produced **zero held-out gain** — a reward-proxy
   / train-test gap. RLVR is a lever on capability you already have, not a
   substitute for it.
7. **Reward *density* is the make-or-break of RL, not the algorithm.** Our first
   GRPO run was a perfectly correct implementation that learned nothing, purely
   because a binary reward + group-relative advantage left most groups with zero
   gradient. Shaping the reward so every group has variance — and watching
   `signal_groups`/KL, not just the reward number — is what turns RL from a no-op
   into an update. Diagnose the *gradient*, not just the metric.
8. **The held-out eval is the only honest judge.** Low training loss, a moving
   KL, a rising shaped reward — every one of these looked healthy at some point
   while the held-out number stayed flat. Synthetic/templated data fits to low
   loss *by construction*. Never conclude "effective" from a training curve;
   conclude it from the 4-quadrant eval on held-out phrasings.
9. **When in doubt, go back to data.** Across the whole project, the only lever
   that ever *moved the held-out number* was calibrated data (mix v1, v2). RL and
   more imitation saturated. The current bet — targeted, verified multi-step
   traces for the exact residual failures (§10) — is that same lever applied
   surgically.

---

## Script & file guide — READ THIS FIRST (for the next agent / next time)

All reasoning/CoT work lives in **`reasoning/`** (this doc included). The base
training pipeline and shared infra stay at the **repo root** and are *not* moved,
because reasoning scripts depend on them. Below is what every script is, why it
exists, and how it's invoked — so you don't have to reverse-engineer it again.

### Directory layout & the golden rules
- **`reasoning/`** = everything specific to making the model *think* (data
  builders, the CoT trainer, STaR, GRPO, the evals, their launchers, this doc).
- **repo root** = shared infra used by *all* stages. **Do not move these:**
  - `model.py` — the architecture **and** the KV cache (§0, §7). Reasoning
    scripts use it two ways: `cot-sft.py` loads it via a **`--model_def
    <path>/model.py`** argument (dynamic import, not a Python `import`), and the
    samplers/evals load it through `AutoModelForCausalLM(..., trust_remote_code=True)`
    which reads the model code *copied into each checkpoint dir*. Either way the
    root `model.py` is the source of truth — that's why it can't live in `reasoning/`.
  - `verify_cache.py` — the §7 correctness gate. It does `from model import ...`
    directly, so it must sit next to `model.py` at root.
  - `pretrain.py`, `midtraining.py`, `continue_pretrain.py`, `sft.py`, `dpo.py`
    (+ their `.sh`) — the base pipeline (§1–§3) that produces `dpo_ckpts`, the
    checkpoint CoT-SFT starts from. Not reasoning-specific.
- **Operational gotchas:**
  - **Submit SLURM jobs from the repo root** (`sbatch reasoning/star_eval.sh`),
    not from inside `reasoning/`. The `#SBATCH --output=report/…` directives
    resolve against the *submission* directory (before the script's own `cd`), so
    submitting from root keeps logs in `report/` at root.
  - `.sh` files and `report/` are **git-ignored** — they exist on disk only.
  - Inside the launchers, paths to reasoning `.py` files are
    `${REPO_ROOT}/reasoning/<x>.py`; the `--model_def` path stays
    `${REPO_ROOT}/model.py` (root). Keep that distinction if you edit them.

### `reasoning/` — data builders (write datasets under `/project/rcc/youzhi/data`)
| Script | What / why | § |
|---|---|---|
| `build_sft_mix.py` | Builds the **calibrated CoT-SFT mixes** (v1, v2): stratified blends of easy gsm8k / MATH / Opus traces / a synthetic-arithmetic tier / general chat. Data calibration is the highest-leverage lever in the whole project. → `cot_sft_mix_v1/v2`. | §6 |
| `build_mix_v3.py` | Builds **mix v3** = the v2 anchor + a *targeted multi-step tier* (algebra/series/geometry/divisors), each trace correct-by-construction and re-verified. Imports the verifier from `star_generate.py`. → `cot_sft_mix_v3`. | §10 |
| `build_star_sft.py` | Assembles the **STaR SFT dataset**: cumulative verified traces (upsampled) + a stratified anchor. → `star_sft_v2`. | §8 |

### `reasoning/` — the CoT trainer & its launchers
| Script | What / why | § |
|---|---|---|
| `cot-sft.py` | The **CoT-SFT trainer** (HF `Trainer`). Distinct from root `sft.py`: parses `<think>…</think>` traces, supports `--allow_non_reasoning` (keep direct/no-think rows), reasoning-row filtering, and exit-after-checkpoint-save. Takes `--model_def <root>/model.py`. | §4,§6,§10 |
| `cot_sft_instruct.sh` | **The launcher actually used** for the mix2/star2/mix3 runs. Step-based saves, exit-after-save + auto-resubmit chain; SLURM job name `argonne-cot-sft-think`. | §6,§10 |
| `cot-sft.sh` | Alternative **self-resubmitting slice-chain** launcher for `cot-sft.py` (modeled on `weekend.sh`+`midtraining.sh`). | §4,§6 |
| `launch_mix3.sh` | **Idempotent guarded** wrapper: submits `cot_sft_instruct.sh` for the v3 run only if no `argonne-cot-sft-think` job is already queued/running (prevents double-submit). | §10 |

### `reasoning/` — STaR (offline RLVR)
| Script | What / why | § |
|---|---|---|
| `star_generate.py` | Core **sampler + verifier**: rejection-samples K traces/problem (KV-cached), keeps verified-correct `\boxed`. Also exports `extract_boxed`, `norm`, `load_problems`, `batched_sample` — the **shared verifier module** reused by `build_mix_v3.py` and `grpo.py`. (Batches K identical copies of one prompt since the model has no padding support.) | §8 |
| `star_gen.sh` | SLURM launcher for `star_generate.py`. | §8 |

### `reasoning/` — GRPO (online RLVR)
| Script | What / why | § |
|---|---|---|
| `grpo.py` | The **GRPO trainer**: group-relative advantage, *shaped* verifiable reward, k3 KL leash to a frozen ref. Imports the verifier/sampler from `star_generate.py`. Saves weights only (a "resume" is a warm restart). | §9 |
| `grpo.sh` | SLURM launcher for `grpo.py`. | §9 |
| `launch_grpo2.sh` | **Idempotent guarded** wrapper: submits `grpo.sh` only if no `argonne-grpo` job is queued/running. | §9 |

### `reasoning/` — the evals (the only honest judge — see throughline #8)
| Script | What / why | § |
|---|---|---|
| `eval_numeracy.py` | **The 4-quadrant probe** (math/general × no-think-greedy / with-CoT-sampled). The held-out judge run in diagnosis and after *every* training run. | §5 |
| `eval_think.py` | Sampling-based think-mode eval (longer traces). | §5 |
| `star_eval.sh` | **The launcher actually used to grade checkpoints**: runs `eval_numeracy.py` across 3 model paths × 4 quadrants on an H100. Per run, edit the `M2/S2/M3` model paths + the log prefix, and `rm` stale logs first. | §5,§8,§10 |
| `eval_numeracy.sh`, `eval_think.sh` | Thin SLURM launchers for the two eval scripts. | §5 |

### Root infra referenced above (kept at repo root on purpose)
| File | What / why | § |
|---|---|---|
| `model.py` | Architecture + KV cache; the source of truth all stages load. | §0,§7 |
| `verify_cache.py` | KV-cache correctness gate (`from model import …`; lives by `model.py`). | §7 |
| `pretrain.py`, `midtraining.py`, `continue_pretrain.py` | Pretraining / midtraining / context extension → the base + math-injection phases. | §1 |
| `sft.py`, `dpo.py` | General SFT and DPO → produce `dpo_ckpts`, the CoT-SFT start point. | §2,§3 |

### The end-to-end order (what produces what)
```
pretrain.py ─▶ [midtraining.py: longmino, then FineMath phase] ─▶ sft.py ─▶ dpo.py ─▶ dpo_ckpts
                                                                                         │
reasoning/: build_*_mix.py ─▶ cot_sft_instruct.sh (cot-sft.py) ─▶ think_*_ckpts ◀────────┘
                │                                                      │
                │  optional offline RLVR:  star_generate.py ─▶ build_star_sft.py ─▶ cot-sft.py ─▶ think_star*_ckpts
                │  optional online  RLVR:  grpo.py (grpo.sh / launch_grpo2.sh)    ─▶ think_grpo*_ckpts
                ▼
      every checkpoint is graded by reasoning/star_eval.sh (eval_numeracy.py, 4-quadrant)
```
