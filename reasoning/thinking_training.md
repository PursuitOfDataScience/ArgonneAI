# Training a Reasoning Model from Scratch — Argonne 3.0

A walkthrough of every stage we went through to turn a freshly-initialized
2.88B transformer into a chain-of-thought ("thinking") model, written for
learning. Each section says **what** we did, **why**, and **what we learned** —
because most of the real lessons came from the things that *didn't* work.

> ## ⭐ LATEST RESULT (2026-06-28): the FineMath base BREAKS the §10 ceiling
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
- Phase 1 (longmino): `/project/rcc/youzhi/models/midtrain/final_model_complete`
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
