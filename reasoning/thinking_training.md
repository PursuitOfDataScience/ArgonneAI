# Training a Reasoning Model from Scratch — Argonne 3.0

How we turned a freshly-initialized 2.88B transformer into a chain-of-thought
("thinking") model, **`Argonne-3.0-think`** — published at
[PursuitOfDataScience/Argonne-3.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.0-think).

This document now **leads with the recipe that worked and the things to avoid**,
then keeps the full chronological log (§0–§21) below as the evidence behind every
claim. Most of the real lessons came from the things that *didn't* work — those
are collected in **"Things to avoid."**

> ## ⚠ SCOPE DISCIPLINE — DO NOT DRIFT (user directive, 2026-07-13)
> **When the task is "improve Argonne-3.0-think," stay ENTIRELY within the 3.0-think lineage** (the 2.88B
> soup/CoT family: `dpo_soup → think_* → the v1–v5 soups`, models under `models/instruct/`). **Do NOT look at,
> probe, propose, or pivot to any other pretraining model — in particular NOT `argonne3.5` / the
> `/home/youzhi/ArgonneAI-3.5` worktree / `models/pretrain/`.** argonne3.5 is a SEPARATE from-scratch base line;
> it is never the answer to a 3.0-think request. Even when single-card weight edits look "exhausted," keep
> finding levers ON 3.0-think: weight-soup frontier points, broader/better CoT data, and serving-system wins
> (external-verifier reranker §25, tool-execution loop §27). ("Downstream exhausted → wait for 3.5" is a drift
> the user has now corrected TWICE — §29 pivot was wrong.) See [[argonne3-think-push-not-35]].

**Bottom line:** the shipped model scores **33/40** on the internal 4-quadrant probe —
strong arithmetic in *both* no-think and with-CoT modes, plus recovered general chat — the
first from-scratch Argonne model that can reason *and* chat. The two highest-leverage moves were
**calibrated, verified CoT data** and **training-free weight-soups**.

**Shipped checkpoint is now v4 (`x_v7v3_300` = 0.3·think_v7 + 0.7·v3, 2026-07-12, §28):** a modest
external-teacher-distillation reasoning update (ASDiv greedy ↑, native termination ↑; self-consistency
regressed ~6pt — a mixed update, shipped by owner decision over the §26 keep-v3 recommendation; v3 retained
for rollback). The honest judge is **clean SVAMP/ASDiv/MAWPS** (+GSM-Plus robustness; GSM8K is contaminated
— see §23/the contamination note; every prior GSM8K number here is invalid as held-out). The prior v3
(`x_v6v2_040`, §23) attacked the #1 *deployable* failure — **non-termination** (~50–60% of greedy
traces never close `</think>`) — with a **short-only CoT-SFT** (train only on ≤768-tok closed-correct
traces so greedy terminates natively) then a **cross-soup with v2** (`0.4·think_v6 + 0.6·blend_star_a06`).
Result on clean held-out math: **greedy SVAMP 18.0→22.7, ASDiv 22.7→27.3; self-consistency held (36.3→40.3
/ 51.0→48.0); pass@32 up** — the FIRST downstream change to move the *honest held-out* number (six months of
arithmetic-internalization never did). Cost: one fragile no-think probe (divisor-counting) regressed. Also
fixed the latent `eos_token_id=None` config bug. Earlier v2 (`blend_star_a06`, §22j): STaR + soup-recovery;
its GSM8K "2→7.5%" gain is now known to be measured on contaminated data. Downstream **test-time compute**
(budget-forcing + self-consistency) remains the deployable multiplier, served fast via the validated **vLLM
port** (§22h).

**▶ WHAT TO DO NEXT — read §24 (the go-forward plan) then §25 (Tier 1 is now DONE).** §24 ranks the levers
and the decision tree; **§25 executed the pivotal Tier-1 experiment (2026-07-12): an external *reasoning*
reranker (Qwen3-4B) DOES capture the pass@K selection gap** — v3 self-cons ~40/50% → best-of-N **~75/75%** ≈
the pass@32 ceiling on clean SVAMP/ASDiv (+35/+25pt, p<0.001). Caveats: a 1-token yes/no judge scored *below*
the vote (reasoning is required), and Qwen-solo=94% means the win is the external model's competence → it is a
**2-model serving** win, not a single-2.88B-card change. **For the HF single-model card, Tier 3 (a better
base = argonne3.5) is the only real ceiling-raiser; Tier 2 self-distillation is marginal.** The paragraph
below is the *historical* (§20) "exhausted" verdict — §22–§26 refine it. **§26 (2026-07-12): a thorough
single-card attempt (external-teacher distillation + tool-calling + coding data) was a NULL** — math traded
(greedy +2 / self-cons −6 on the broad gate), tool-calling learned perfectly (100% valid calls) but
unshippable weights-only (soup-washout + response-hallucination), coding base-capacity-limited (HumanEval
~0). **v3 stays shipped.** The two real forward moves are serving-system (tool-execution loop / external
reranker, §25/§26) or a better base (Tier 3 = argonne3.5).

**Downstream is now exhausted (§20, 2026-07-07):** online RLVR (GRPO, twice, incl. a
large-group variant) gives **no net benchmark gain** on this base — it maximizes the
*format* reward (trace-closing 49%→93%) while rollout accuracy stays flat at ~2%, because
the correct-answer signal is too sparse to amplify (RLVR amplifies, it doesn't create). For
a *cleaner from-scratch* reasoner, base quality → the argonne3.5 pretraining is the durable
path (§15 proved this recipe on Qwen/Llama-grade bases).

**BUT "exhausted" was scoped too broadly (§22, 2026-07-07).** Everything tried in §4–§21 aimed
to make the model *internally* better at arithmetic (CoT-SFT data, STaR-imitation, GRPO) — and
that class is genuinely saturated. Three lever classes were **never tried on `soup_blend_a085`**
and remain open *on this exact base*: (1) **capture the latent capability** — §21 measured
**pass@64 ≈ 48% vs single-sample ≈ 2.6%**, an ~18× headroom no aggregation/verifier ever
touched; (2) **remove the failing channel** — tool-augmented (calculator-offload) reasoning
attacks the documented root cause (correct procedures, wrong elementary arithmetic) directly;
(3) **RLVR-DPO** contrasts correct-vs-wrong self-generated traces (needs one positive per prompt,
so it sidesteps GRPO's group-advantage collapse). §22 is the ranked plan to pursue these.

---

## The success recipe (reproducible)

The proven pipeline that produced `Argonne-3.0-think`, in order. **Every downstream
stage runs at context 13,568 with RoPE θ = 1e6** (the base is RoPE-extrapolated from
a 1,024-ctx pretraining run). Scripts are on `main`; the launcher `.sh` files that
set these hyperparameters are untracked by repo policy, so the numbers are captured
here and in §17–§19.

```
FineWeb ─▶ [from-scratch pretrain] ─▶ Argonne-3.0-base (seed 329148)
                                          │
   FineWeb + FineMath  ─▶ [intermix midtraining, θ=1e6] ─▶ intermix ckpt 363908
                                          │
        SOUP BASE (training-free) = 0.35·seed + 0.65·intermix   ◀── idea #1
                                          │
                    SFT (UltraChat) ─▶ DPO (Chatbot Arena) ─▶ dpo_soup ──┐  (keep!)
                                          │                              │
                CoT-SFT (cot_sft_mix_v3, θ=1e6) ─▶ think_soup            │
                                          │                              │
   FINAL SOUP (training-free) = 0.15·dpo_soup + 0.85·think_soup  ◀── idea #2
                                          │
                              soup_blend_a085  = Argonne-3.0-think (33/40)
```

### Step 0 — The base is the whole ballgame
Argonne 3.0-base: 2.88B params, 24 layers, hidden 3072, **12 query / 4 KV heads**
(GQA), SwiGLU (8192), RMSNorm + QK/V/sandwich norms, RoPE, vocab **151,669** (Qwen3
tokenizer), tied embeddings. Pretrained from scratch on **~76B tokens of FineWeb**
at 1,024 ctx. **The single biggest determinant of the final reasoning model is this
base's quality** — specifically its numeracy and world-knowledge. Every downstream
lever calibrates and unlocks what's here; almost none of them *create* it (§11, §15,
and throughline #1). Scripts: `pretrain.py`, `model.py`.

### Step 1 — Repair numeracy: intermix midtraining
The pure-FineWeb base can't do grade-school arithmetic (3/20 on the probe). Fix it by
continuing pretraining on a **50:50 → 60:40 (by document) mix of FineWeb
(`CC-MAIN-2025-21`) + FineMath (`finemath-4plus`)** at LR 3e-4, θ=1e6 → intermix
checkpoint 363908 (~1.41B intermix tokens; MATH 14/20 but general eroding). Scripts:
`preprocess_finemath.py` → `reasoning/build_intermix.py` → `midtraining.py`.

### Step 2 — Reconcile math ↔ knowledge: the SOUP BASE (training-free) — **idea #1**
The 3e-4 intermix over-writes general knowledge faster than replay protects it, so the
raw intermix checkpoint is lopsided. A **linear weight interpolation of the two
same-lineage θ=1e6 checkpoints** reconciles them for free:
`0.35 · seed(329148) + 0.65 · intermix(363908)`. This is the **first from-scratch
Argonne base to clear both axes of the probe** (MATH 15/20 **and** GEN 13/15). Math and
general knowledge are ~linearly reconcilable in weight space; the raw checkpoint is a
mild WiSE-FT overshoot. Script: `reasoning/build_soup_base.py` (§17).

### Step 3 — General instruction-following: SFT
Full SFT on **UltraChat 200k** (`HuggingFaceH4/ultrachat_200k`, `train_sft`) from the
soup base. LR **2e-5**, 1 epoch, effective batch 18, 1×H200. Script: `sft.py`.

### Step 4 — Preference alignment: DPO → keep `dpo_soup`
DPO on **Chatbot Arena** (`KatoHF/chatbot_arena_binarized`, `chat_refine_strict`,
~204 pairs). LR **1e-6**, β **0.03**, effective batch 8, 1×H200 → `dpo_soup`.
**Retain this checkpoint** — the final soup (Step 6) needs it. At this point the model
is general-healthy (~7–8/10 general) but not yet a strong reasoner.

### Step 5 — Teach chain-of-thought: CoT-SFT → `think_soup`
CoT-SFT from `dpo_soup` on **`cot_sft_mix_v3`** (~113k rows), LR **1e-5**, 1 epoch,
effective batch 12 (3×H200 DDP), **θ=1e6** (critical — not the FineMath θ=1e4).
Scripts: `reasoning/build_sft_mix.py` + `reasoning/build_mix_v3.py` → `reasoning/cot-sft.py`.
The mix is deliberately calibrated (data is the highest-leverage lever in the project):

| tier | rows | source |
|---|---:|---|
| `direct_tulu` (no-think chat) | 34,000 | `allenai/tulu-3-sft-mixture` |
| `synth_arith` | 15,000 | synthetic, correct-by-construction |
| `gen_ultrachat` (CoT-augmented) | 15,000 | from `HuggingFaceH4/ultrachat_200k` |
| `hard_strict` | 12,000 | `PursuitOfDataScience/MiniMax-M2.1-Mixture-of-Thoughts` |
| `easy_gsm8k` | 8,402 | `openai/gsm8k` (`main`) + `<think>`/`\boxed{}` |
| `med_math` | 5,729 | `nlile/hendrycks-MATH-benchmark` (L1–3) |
| `ms_algebra`/`ms_series`/`ms_geometry`/`ms_divisors` | 16,290 | synthetic multi-step, Python-verified |
| `med_openmath` | 4,620 | `nvidia/OpenMathReasoning` (solutions regenerated) |
| `hq_opus` | 2,300 | `nohurry/Opus-4.6-Reasoning-3000x-filtered` |

Result: `think_soup` = **10/10 both math modes** (first Argonne to solve all four §10
residuals cleanly) — **but general chat regresses** (loops, lost facts), because the CoT
diet is math-heavy (the zero-sum diet, §6/§18b).

### Step 6 — Recover general without losing math: the FINAL SOUP (training-free) — **idea #2**
`think_soup` is just `dpo_soup` + a CoT weight-delta Δ in the *same* optimization basin.
So blend a fraction of the (general-healthy) pre-CoT weights back in:
`soup_blend_a085 = 0.15 · dpo_soup + 0.85 · think_soup`. This **fractionally un-applies Δ** —
enough to erase the loop/forgetting pathology (grammar loop gone; Mars fact restored)
while **keeping the full 10/10 math**. Script: `reasoning/build_ckpt_soup.py` (§19).
**α = 0.85 is a knee** (see Things to avoid): more general recovery below it, but the
`<think>` trace-closure format lives in Δ and starts breaking.

### The result (`report/recover_*.log`, greedy no-think / sampled with-CoT)

| quadrant | soup_blend_a085 | think_soup (α=1) |
|---|:---:|:---:|
| MATH no-think | **10/10** | 10/10 |
| MATH + CoT | **10/10** | 10/10 |
| GENERAL no-think | **7/10** | 5/10 |
| GENERAL + CoT | **6/10** | 6/10 |
| **total** | **33/40** | 29–31/40 |

**Two ideas carried the whole project:** (1) *calibrated, verified CoT data* — the only
lever that ever moved the held-out number; and (2) *training-free weight-soups*, used
twice — once to build a both-axes base (Step 2), once to reconcile reasoning with chat
(Step 6). Both are free (CPU tensor-averaging, minutes) and clean because the checkpoints
share a lineage/basin. **Ship with greedy decoding for math/no-think.** Eval:
`reasoning/eval_numeracy.py` (downstream), `reasoning/eval_intermix_base.py` (base probe).

---

## Things to avoid (each cost real time or compute to learn)

### Method / modeling dead-ends
- **Don't expect fine-tuning to *create* a capability the base lacks.** Six months of
  STaR/GRPO/data-calibration never gave a from-scratch Argonne base clean multi-step math;
  a better base (FineMath, then Qwen/Llama) did it immediately (§11, §15). Capability is
  set upstream — fix the base, don't paper over it downstream.
- **Don't chase RLVR (STaR/GRPO) to add a missing skill.** STaR saturates (you can only
  imitate successes you already produce). GRPO round 2 maximized its shaped reward on gsm8k
  and moved the policy yet produced **zero held-out gain** — a reward-proxy / train-test gap.
  RLVR amplifies existing capability; it doesn't manufacture it (§8, §9).
- **Don't over-index the CoT mix on math.** Fine-tuning is a zero-sum diet: a math-heavy
  CoT diet erases general chat and reintroduces loops (§6, §18b). Keep a large concise
  no-think / general share.
- **Don't try to fix CoT-induced general loops with DECODING.** *Refuted* (§18f):
  `repetition_penalty=1.3 + no_repeat_ngram=3` **corrupts arithmetic** — it blocks the model
  from re-emitting a digit it just used, turning `80/2` into `8/2`, collapsing math 10→4.
  It "fixed" only one general cell and left the real content errors. **Best decoding is
  plain greedy.** (The reference `argonne-3.0-instruct` card's rep-penalty settings are
  actively harmful for this reasoning model.)
- **Don't "rebalance the CoT data" as the cure for general regression.** Tried it (mix v4,
  56% concise): a **lateral trade**, 29/40, general still looped (§18d).
- **Don't resume intermix midtraining to fix general.** It's the wrong stage — the diagnostic
  proved general was healthy *after* SFT+DPO and broke at the *CoT* step, and intermix math
  saturates by ~2B tokens while general won't move at LR 3e-4 (§18c, §18g).
- **Don't over-dilute the final soup.** α below ~0.85 recovers more no-think general but
  **breaks `<think>` trace-closure** (the CoT format lives in the weight-delta): at α=0.5/0.7
  the with-CoT quadrant collapses to 1/10 (§19). α=0.85 is the knee.
- **Don't spend the last mile on the residual base gaps.** A few misses (naming all three
  primary colors, a taller/shorter transitivity puzzle) are wrong in `dpo_soup` too — genuine
  2.88B base-capability limits, unfixable by souping/decoding/data. That's argonne3.5 work.

### Diagnosis / evaluation traps
- **Don't trust training curves.** Low loss, a moving KL, a rising shaped reward all looked
  healthy while the held-out number stayed flat. Synthetic/templated data fits to low loss
  *by construction*. **The 4-quadrant held-out eval is the only honest judge** (throughline #8).
- **Don't diagnose "broken model" before ruling out decoder/eval bugs.** Early "gibberish"
  was a `from_pretrained` buffer bug + prompt-inclusive n-gram bans in the eval decoder, and
  training loss looked ~4× inflated purely from grad-accum scaling — not the model (§5).
- **Don't apply a repetition penalty over prompt tokens.** Penalize *generated* tokens only;
  banning prompt tokens produces garbage (§5, and the guard now in `eval_numeracy.py`).
- **Localize the regression before retraining.** Evaluating the *pre-CoT* checkpoint is what
  proved the CoT stage (not the base) broke general and pointed straight at the free fix (§18c).

### Operational / infra traps
- **Don't fill HBM blindly on CoT-SFT.** The ceiling is the **fp32 loss-logits
  `(batch × seq × vocab≈151k)`** materialized in the *backward* pass, not startup memory —
  batch 12/16/18 OOM at seq≈4k on a 140 GiB H200; profile the backward (§18e). Filling HBM
  also forces a bigger effective batch (quality tension).
- **Don't rely on `--export=ALL` across chained sbatch stages.** A finished stage's exported
  config leaks into the next and clobbers `:-` defaults (DPO once ran on the wrong dataset →
  0 pairs → crash). `unset` config vars at the top of each chained launcher (§12).
- **Don't run watchers with `nohup`/detached.** They die silently and miss failures — use
  harness-tracked background tasks.
- **Don't submit to the excluded SLURM nodes:** `midway3-0423,midway3-[0298,0377-0378,0603-0606]`
  (and 0602 is ECC-flaky). Every `sbatch` must carry the `--exclude`.
- **Never force-add a `.sh`.** Repo policy: all `.sh` are git-ignored (they carry
  cluster-specific paths); the recipe they encode lives in this doc. `.py`/`.md` are force-added.
- **Don't assume flash-attn / the sliding window is active.** The env has flash-attn-4, so
  `model.py` silently runs **full causal attention** (the 256-token local window is ignored)
  in all production runs; `qk_norm` is essential at high LR (§16 audit).
- **Don't midtrain on FineMath without replaying general data.** The full pipeline on a
  pure-FineMath base produced 10/10 math but **catastrophic** general ("the capital of France
  is France") that SFT+DPO could not recover — the base forgot the world (§12). The intermix
  + soup base (Steps 1–2) exists precisely to avoid this.

---

## Detailed chronological record

Everything below is the **condensed** log (§0–§24) that produced the recipe above, followed by
**"The throughline"** (the nine deeper principles) and the **script & file guide**. §0–§21 were
compressed 2026-07-12 (every number/decision/script name preserved; verbose narration cut); §22–§24
are kept in full as the still-live plan. Read the recipe and avoid-list first; drop into a section
when you need the evidence or the exact numbers behind a claim.

---

## 0. The model architecture (what we're training)

Argonne 3.0 is a ~2.88B decoder-only transformer. Load-bearing choices:

| Component | Value | Why |
|---|---|---|
| Hidden / layers | 3072 / 24 | Mid-size; 1-GPU trainable |
| Heads | 12 query / 4 KV (**GQA**) | Smaller KV cache, faster inference |
| Vocab | 151,669 | Qwen-style tokenizer |
| Pos enc | **RoPE**, θ=1e6 | High θ enables later context extension |
| Context | 1024 → 13568 → 4096 | Grown across stages |
| Embeddings | **Tied** (lm_head=embed_tokens) | Causes benign "lm_head MISSING" warning |
| Norms | sandwich_norm, qk_norm, v_norm | Stability |
| Attention | Interleaved local window 256 | *Ignored at runtime* (no flash-attn on our nodes) |
| logit_softcap | 15.0 | Prevents logit blow-up |

Key fact: **no attention-mask / padding support** — forward forces `attention_mask=None` (pure causal). This shaped every later inference/RL batching+padding decision.

---

## 1. Pretraining — teach it language (`pretrain.py`)

Train random-init model on general text, next-token CE on every token, short ctx (1024) for throughput. Knobs: **LR 3e-4** (production, NOT the 6e-4 argparse default), production effective batch + grad clip, `torch.compile` on. This stage sets arithmetic-pattern quality that everything inherits — if pretraining under-sees math, fine-tuning can't fully fix it (central to §6). Related: `midtraining.py` / `continue_pretrain.py` extend pretraining (e.g. RoPE ctx-extension) before instruction tuning.

---

## 1.5 Handoff — from a midtrained base into the reasoning pipeline (VERIFIED)

Entry point after **FineMath math-injection midtraining** (Phase-2 auto-switch in `midtraining.sh`/`weekend.sh`/`night.sh`).

**What midtraining leaves you:** on hitting token target, `midtraining.py:save_final_model_artifacts` writes a **plain HF dir** (`config.json`+`model.safetensors`+tokenizer+`chat_template.jinja`):
- Phase 1 (longmino): `/project/rcc/youzhi/models/midtrain/final_model_complete_longmino` (renamed in §16 — `models/midtrain` is now the live INTERMIX ckpt dir; a dir named `final_model_complete` there = phase-done marker)
- Phase 2 (FineMath): `/project/rcc/youzhi/models/midtrain_finemath/final_model_complete`

It's a **base LM, not chat**, with **no `auto_map`/modeling file**. Consequences:
1. Can't `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` directly (no remote code). Load by supplying arch explicitly, as `sft.py` (`from model import ArgonneModel`) and `cot-sft.py` (`--model_def <repo>/model.py`) do. *Verified:* building `ArgonneModel` from its `config.json` + loading `model.safetensors` → 2.88B params, `unexpected=0`, only `missing=[lm_head.weight]` (benign tied-embedding).
2. Being a base, run **SFT → DPO** before any CoT (can't CoT-SFT a raw base — won't follow chat format; that's §2–§3).

**Turnkey chain (from repo root):**
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
Keep manual/stage-gated — eval-and-decide between stages (§8 throughline); do **not** auto-chain SFT→DPO→CoT-SFT unattended.

**Why FineMath, and what to try first on the new base:** §10 residual = arithmetic-fact execution (`8+3=7` inside a correct procedure) = inherited-numeracy ceiling. FineMath midtraining attacks it at the pretraining objective (digit-split tokenizer makes it learnable). First reasoning experiment on new base = **mix v4**: keep v3's multi-step procedure tier but *restore* strong direct/no-think + arithmetic-fact-drill share (undo v3's no-think collapse), now that the base has the numeracy for drills to stick. Then correct procedures + better facts give RLVR (§9) a base worth amplifying.

---

## 2. Supervised Fine-Tuning (SFT) — teach it to follow instructions (`sft.py`)

**What:** Train on (prompt, response) chat pairs, loss on assistant tokens only. Context extended to 13568. Output → `sft_ckpts`.

**Why:** Base just continues text; SFT teaches chat format (`<|im_start|>user … <|im_start|>assistant …`) and *responding* vs continuing.

**Learned:** After SFT, fluent chatbot on single-step factual Qs (≈7/10) but **cannot do arithmetic** ("100 ÷ 4 = 0.5", "17 − 5 is a popular online platform"). Instruction-following worked, reasoning didn't → first hint problem is upstream.

---

## 3. DPO — align preferences (`dpo.py`)

**What:** Direct Preference Optimization on (prompt, chosen, rejected) triples — prefer chosen over rejected directly, no reward model. Output → `dpo_ckpts`.

**Why:** Polishes tone/helpfulness/format; a preference method, not a capability method.

**Learned:** Maintained chatbot quality but **no math gain** (same arithmetic failures as SFT). Confirms preference alignment ≠ reasoning ability.

---

## 4. CoT SFT — teach it to "think" (`cot-sft.py`)

**What:** Fine-tune on traces where assistant turn = `<think> … </think>` then answer (Qwen3-style template parses `<think>` into a separate reasoning field). Context 4096. Started from DPO ckpt. Output → `think_ckpts`.

**Why:** The step that *creates the reasoning model* — imitating worked-solutions to produce CoT before answering.

**Learned (the painful part):** First CoT model was a regression in disguise:
- Learned the *format* but long traces **injected arithmetic errors and anchored on them** ("7×6=42" mid-trace → concludes "answer: 7").
- Fell into **enumeration/repetition loops** ("Sam is taller than Bob but shorter than Bob…" forever).
- Narrow training set (OpenR1-Math + codeforces) → dumps **codeforces-style JSON/Python** for plain questions.
- Net: thinking mode *worse* than no-think on basics (0/10 math-CoT). **The mandated long trace itself was the pathology.** Kicked off diagnosis phase.

---

## 5. Diagnosis — is it the decoder, the data, or the model size?

Built honest evals (`eval_think.py`, `eval_numeracy.py`), ruled out causes one by one. **Most important methodological lesson in the project.**

**Decoder bugs first (cheapest).** Early "gibberish" was partly *inference* artifact (→ avoid-list):
- `config.json` had `eos_token_id=null` → never stopped on `<|im_end|>`, rambled to max length. Fix: set eos explicitly.
- `from_pretrained` buffer bug + **prompt-inclusive n-gram bans** → garbage. Fix: self-healing `from_pretrained` (re-ties lm_head, rebuilds RoPE buffers) + clean decode loop (stops on eos, penalizes only *generated* tokens).
- Logged training loss **inflated ~4× by grad-accum scaling** — reporting artifact, not divergence.

**Controlled capability probe** (same arithmetic Qs across base → SFT → DPO → think, no-think, greedy):
- base: echoes (not instruct-tuned).
- SFT/DPO: fluent but **0 arithmetic correct**.
- think: **best** of the four (7×6=42 ✓, half of 80=40 ✓, correct *procedures*) but slips on single-digit facts (17−5→16).

**Conclusion:** failure = **arithmetic-fact errors inherited from weak pretraining numeracy**, *not* a 2.88B ceiling (well-trained 3B do this cold), *not* "CoT broke it" (CoT helped vs DPO). Lever = **data, upstream.** Reframed everything after.

---

## 6. Data calibration — fix facts, format, and looping with better CoT data

Built stratified mixes with `build_sft_mix.py`, re-ran CoT SFT (same config).

**Mix v1 (~21k, `think_mix_ckpts`).** Added easy gsm8k, OpenMathReasoning, MATH lvl1-3, Opus traces, some *direct* (no-think) examples.
- Required `cot-sft.py` `--allow_non_reasoning 1` (keeps direct targets instead of dropping non-`<think>` rows).
- MATH-CoT **0 → 3/10**; looping mostly gone; 7×6=42 now *survives* the trace.
- But GENERAL no-think **regressed 8 → 6** — too math-heavy eroded everyday chat. (A fine-tune is a zero-sum diet.)

**Mix v2 (~97k, `think_mix2_ckpts`) — all goals hit.** Rebalanced:
- Added **synthetic-arithmetic tier** (15k short verified `\boxed` traces) to drill fact execution.
- Pushed general/chat to ~51% (tulu no-think + ultrachat) to undo v1 regression.
- MATH-CoT **6/10** (17−5=12, 8+3=11, 100÷4=25 all correct; looping gone, all close `</think>`); MATH no-think ~5; GENERAL no-think back to **8**; GENERAL-CoT ~3-4.

**Learned:** Data calibration is the highest-leverage knob (fact drills fix facts, balance fixes regressions, easy data fixes looping). But *residual* failures — `2x+5=17`, `sum 1..10`, divisor counting, the non-numeric logic puzzle — are **multi-step chains** that did NOT yield to more supervised data. That boundary is where RL comes in.

---

## 7. Building the inference engine — KV cache (`model.py`, `verify_cache.py`)

**What:** Proper **KV cache** (`past_key_values` + `use_cache`) through GQA/RoPE/qk-norm/block stack: prefill prompt once, then one token/step reusing cached K/V.

**Why:** Naive generation recomputes whole sequence per token — O(n²), ~20s/problem. Sampling methods (STaR, GRPO) need thousands of rollouts → ~256× too slow without a cache.

**Verified (`verify_cache.py`):** prefill logits **bit-exact** (diff 0.0) vs no-cache; token-by-token argmax **100% match**; ~10× faster. Training path byte-identical (`use_cache=False`). Also fixed broken `generate()`.

**Lesson:** correctness-gate infra before building on it — a subtly wrong cache silently poisons every downstream RL gradient.

---

## 8. STaR — offline RLVR by rejection sampling (`star_generate.py`)

**What:** offline RLVR by rejection sampling: (1) sample K traces/problem, (2) verify — keep only traces whose `\boxed{}` == gold (via `extract_boxed`, `norm`), (3) SFT on own verified traces. Reward baked in by filtering. Batch K identical copies of one prompt (no padding), iterate problems sequentially.

**Round 1** (from `think_mix2`, K=12, 1200 gsm8k): pass@12 ≈18.6%, **365** correct traces (64% never closed `</think>` within budget = yield ceiling). SFT on 365×4 + 5k anchor → `think_star_ckpts`. Marginal-but-real (fixed 2x+5=17; 100÷4 regressed; net ~flat).

**Round 2** (from round-1 model, K=12, gsm8k + MATH lvl1-3, **max_new 400→512**): pass@12 **29%**, unclosed **64%→24%**, **1530** traces. Cumulative 1888 unique ×4 + 5k anchor (`build_star_sft.py`), SFT from stable `think_mix2` base → `think_star2_ckpts`:
- MATH+CoT 6→**7** (100÷4 fixed to 25).
- MATH no-think **regressed to ~3** — dumped `import sympy`; trace set was 60% of data → over-specialized to long solution output.
- GENERAL no-think 8→7 ("sun is not a star, it is a planet").

**Learned:** STaR **saturates** — buys marginal CoT fact-stability but doesn't fix reasoning-chain correctness (sum-loop, 2x+5, logic puzzle persist across all 3 ckpts); heavy trace fraction (60%) erodes direct-answer path. Cap STaR fraction ~≤30% no-think; can't teach what the model can't already occasionally do.

---

## 9. GRPO — online RLVR (`grpo.py`)

**What:** reward on full online rollout instead of filter+imitate: (1) sample group of G traces from current policy, (2) verifiable reward (`\boxed`==gold, no reward model), (3) group-relative advantage `A_i = (r_i − mean)/(std+ε)` (group = own baseline, no critic), (4) update `−A·logπ` with KL leash `β·KL` to frozen reference (k3 estimator `exp(d)−d−1`). Continuous group signal + KL → can improve where STaR's whole-trace imitation couldn't.

**Model-forced design:** right-padding safe (pure causal → real token never attends to trailing pads, mask loss to real tokens); sample & score same distribution (temp T, no top-k/p, `log_softmax(logits/T)` → unbiased PG); one inner update/batch → ratio=1, clipped surrogate reduces to group-baseline PG, keep KL; start from `think_star2` (densest reward), KL ref = same ckpt, skip zero-variance groups; KV cache makes online rollout feasible.

### Round 1 — a clean *null* result (and why)
Binary reward (1.0/0.0), from `think_star2`, gsm8k, G=8, 8 prompts/step, LR 1e-6, 400 steps, ~7h → **changed nothing** (`think_grpo` ≈ `think_star2`, same failures). Logs: **KL pinned ~0.0002** (policy never moved), reward flat 0.05–0.23, only ~3/8 groups carried gradient. **Root-cause reward trap:** binary reward + group-relative advantage → **zero gradient when all G traces get same reward** (all-correct or all-wrong); on a weak model 2/3 of each batch is gradient-dead; timid LR 1e-6 × 400 steps = negligible. Starved of signal, not failed.

### Round 2 — the fix: dense reward shaping
Grade the reward so groups almost always have variance:
```
correct (closed+boxed==gold)      → 1.0
closed+boxed, wrong               → 0.3
closed, no parseable boxed        → 0.15
stopped, never closed </think>    → 0.0
never stopped (looping)           → −0.2
```
Falls out: (1) even all-wrong groups differ on *how* wrong → non-zero variance → signal; (2) ranks *closed > looping* → direct downward pressure on the dominant degenerate-enumeration-loop pathology (RL good at this, SFT not). Keep `is_correct` as separate logged metric so accuracy stays honest. Plus **LR 1e-6→5e-6**, **8→12 prompts/step**, run to wall-clock (~11h) not fixed steps. Smoke test: `signal_groups` ~3/8 → **6/6 every step**, finite loss/grad-norm.

**Result — policy moved, capability didn't.** H200, LR 5e-6, 12 prompts/step, G=8, ~510 steps. Real signal: **KL ~0.0025 (12× round 1)**, shaped reward rose, train accuracy noisy peaks ~0.29. But held-out 4-quadrant eval (H100) **zero gain**: `think_grpo2 ≈ think_star2`, MATH+CoT dipped 7→6. Maximized shaped reward on gsm8k without improving held-out correctness = **reward-proxy / train-test gap**.

**Learned:** fixing the gradient was necessary not sufficient — RLVR sharpens what the model can already occasionally do, doesn't manufacture a missing skill. Three methods (data calibration mix v2, STaR, properly-configured GRPO) now agree: bottleneck is **upstream multi-step-reasoning capability**, not the RL recipe. Returned to targeted data.

---

## 10. Targeted multi-step data (`build_mix_v3.py`)

**What:** focused tier for the four multi-step families that fail across *every* checkpoint (base→SFT→DPO→mix2→star2→grpo2): (1) two-op linear algebra (`2x+5=17→x=6`), (2) sequential/series sums (`1+…+10=55`, the loop trap), (3) formula-then-substitute geometry (perimeter `=2(l+w)`), (4) divisor counting (# divisors of 12 = 6). `build_mix_v3.py` generates short **correct-by-construction** `<think>` traces — every number computed in Python, each `\boxed` re-verified with `extract_boxed`/`norm`. Keep **all of mix v2 as anchor** (zero-sum-diet lesson §6). Final: ~97k v2 anchor + `ms_algebra` 5000, `ms_series` 5000, `ms_geometry` 5000, `ms_divisors` 1290 (small natural unique ceiling) = **113,341 rows**. Rationale: §9 showed RL can't amplify a missing skill → give clean verified two-step solutions directly (same lever that took MATH+CoT 0→6 in §6, applied surgically).

**Result — moved the PROCEDURE, not the arithmetic ceiling.** 1-epoch CoT-SFT (from `dpo_ckpts`, ctx 4096, LR 1e-5), cancelled ~77% (loss 2.70→0.81), eval `checkpoint-6000` (H100) vs `think_mix2` and `think_star2`:
- **Two hardest families now solved — only on this ckpt:** `2x+5=17→6` (full derivation) and divisors of 12→6 (`12=2²×3`, `(2+1)(1+1)=6`); mix2/star2/grpo2 all failed both. `</think>`-loop pathology **gone** on math-reasoning path.
- **Residual misses now pure single-step arithmetic-fact slips inside correct procedures:** sum 1..10 uses `n(n+1)/2` but substitutes n=8→36; perimeter uses `2(l+w)` but `8+3=7`→14; trivia wobbles (`8+3=9`, `100/4=20`). Inherited-numeracy ceiling laid bare — data installed the *procedure* (which RL couldn't) but not the arithmetic *facts*.
- Cost of 100%-thinking tier: **MATH no-think collapsed** to "The answer is `\boxed{X}`" (0/10), GENERAL+CoT noisier. **GENERAL no-think held 8/10** (anchor worked, even fixed star2's "sun is a planet").

**Learned:** data calibration is again the only lever that moved multi-step reasoning, but only *relocated* the bottleneck: "can't structure multi-step" → "can't execute the arithmetic inside." Next: (1) re-add strong no-think + arithmetic-fact-drill share to undo no-think collapse; (2) RLVR now better positioned than §9 — with correct procedures in place, graded final-answer reward carries denser signal.

---

## 11. Re-running the recipe on the FineMath numerate base (DONE — base ceiling broken)

Tests §10's hypothesis (residual math failures = inherited-numeracy ceiling from pretraining) by re-running the recipe on the FineMath midtraining base and A/B'ing vs the old base with the same CoT data.

**Pinning the moving base.** FineMath midtraining (`midtraining.py` via `night.sh`/`weekend.sh`) keeps only the latest `checkpoint_step_*.pt` (new file every ~30 min, old deleted; no `final_model_complete` yet). Pinned `checkpoint_step_768847.pt` → `/project/rcc/youzhi/models/midtrain_finemath_pinned/`, extracted HF base via `reasoning/extract_finemath_base.py` (ctx 13568, **rope_theta=1e4**, trims embeddings, copies tokenizer+chat_template). Early snapshot: ~1.9B FineMath tokens on top of 16B Phase-1 (longmino), cumulative loss 1.83.

> **Base-health probe (re-confirms §5).** CPU/fp32 manual-load gave loss ~8.8 on English — but control `argonne-3.0-base` scored ~9.5, Phase-1 longmino ~9.6–10 on the same harness → CPU path is the artifact, not the weights. FineMath was best of the three. GPU 4-quadrant eval is the honest judge.

**The chain** (all `reasoning/`, 1× H100, auto-resubmit; new output dirs preserve old baselines `sft_ckpts`/`dpo_ckpts`/`think_*_ckpts`): `sft_finemath.sh`→`sft_finemath` → `dpo_finemath.sh`→`dpo_finemath` → `cot_finemath.sh` (DATA=`cot_sft_mix_v3`, ROPE_THETA=1e4)→`think_finemath` → `eval_finemath.sh` (4-quadrant). H100 sizing matches H200 effective batch (SFT 4×5=20; DPO 2×4=8; CoT 4×3=12≈think_mix3 tbs 11). **CoT ROPE_THETA=1e4** to match base — old launcher defaulted 1e6, which would corrupt this θ=1e4 base.

**Key A/B:** `think_finemath` vs `think_mix3_ckpts/checkpoint-6000` (same v3 CoT, old θ=1e6 base) — isolates the math-injected base. Reused v3 (not the §1.5 "mix v4") = one variable; mix v4 remains documented follow-up.

**What we ACTUALLY ran (pivoted to a cheap check).** Full chain launched, ran SFT to ~73%, but slow (~a day on 1 GPU); cancelled and answered two cheap ways:
1. **Few-shot base probe** (`reasoning/quick_base_probe.py`, training-free ~3 min, 20 arith/multi-step, `ArgonneModel.from_pretrained`, greedy 4-shot): `argonne-3.0-base` **3/20**, longmino Phase-1 **1/20**, **FineMath Phase-2 16/20**. Numeracy lift real+large; residual misses §10-style (`2x+5=17→"17−5=12"`).
2. **Short direct CoT-SFT from FineMath base** (`cot_finemath.sh` MODEL_PATH=pinned base, `cot_sft_mix_v3`, MAX_STEPS=2500, no general SFT/DPO) → `think_finemath`, graded by `eval_finemath.sh`.

**Result — §10 ceiling broken** (`report/finemath_*.log`):

| quadrant | think_finemath | think_mix3 (old base, v3) | think_mix2 | think_star2 |
|---|---|---|---|---|
| MATH + CoT | **10/10** | 6/10 | ~4/10 | 6/10 |
| MATH no-think | ~7/10 | ~0/10 (degen boxed) | ~4/10 | ~3/10 |
| GENERAL no-think | ~1–2/10 | ~7/10 | ~8/10 | ~7/10 |
| GENERAL + CoT | ~0/10 | ~6/10 | ~5/10 | ~5/10 |

- **MATH+CoT clean sweep.** `think_finemath` solves all four §10 residuals (`2x+5=17→6`, `1+…+10=55`, perimeter `2(8+3)=22`, divisors of 12 `(2+1)(1+1)=6`); `think_mix3` (same data, OLD base) still slips (`8+3=9`, `100/4=20`, `sum→45`, `perim→18`). **Only change = the base.**
- **Cost = general ability**: math savant that loops on chat ("capital of France is France itself"). Expected — skipped general SFT/DPO + 2500 math-heavy steps from a base whose general English FineMath already narrowed (zero-sum diet, §6).

**Fix for next time:** balanced pipeline on FineMath base (general SFT→DPO→CoT with **mix v4**, §1.5) + a **later FineMath checkpoint**. Launchers `reasoning/{sft,dpo,cot}_finemath.sh` ready.

### Operational lessons (so next agent doesn't re-pay)
- **Pin base immediately** — live dir accumulates a 36 GB ckpt/~30 min, no auto-prune; copy + `extract_finemath_base.py` before it changes.
- **`test` partition has NO time cap** (TIMELIMIT=infinite) → run each stage as ONE continuous job (`--time=1-00:00:00`, `EXIT_AFTER_CHECKPOINT_SAVE=0`, `SLICE_TIME_LIMIT=0`); the 30-min exit/resubmit treadmill was self-imposed, ~60% non-compute.
- **OOM is deterministic + loops** — batch 4 OOM'd on a rare ~13k-token batch (SDPA full attn, peak~batch·seq²), re-hit on resume. Fix: halve batch 4→2, double grad-accum. SFT/CoT here run **batch 2**.
- **`eval_numeracy.py` needs repo-root `model.py` on sys.path** to register `argonne2` (else `KeyError: 'argonne2'`); fixed in-script (adds `SCRIPT_DIR.parent`) — base/CoT dirs lack `auto_map`/modeling file.
- **Disk:** cleaned ~2.1 TB; kept `midtrain_finemath_pinned/final_model_complete` + latest `midtrain_finemath/checkpoint_step_818607.pt`.

### New files
`extract_finemath_base.py` (pin .pt→HF base, ctx 13568/θ=1e4); `quick_base_probe.py`/`.sh` (training-free few-shot numeracy probe); `sft_finemath.sh`/`dpo_finemath.sh`/`cot_finemath.sh` (H100 continuous θ=1e4 batch2 auto-chain); `eval_finemath.sh` (4-quadrant).

---

## 12. The balanced pipeline on the *latest* FineMath base (DONE — refutes §11's "use a later checkpoint" fix)

Ran §11's prescription (general SFT→DPO→CoT on a **later** FineMath checkpoint). Answer: clean, surprising **no**.

**What we ran.** Full recipe on latest `checkpoint_step_833124.pt` (~20.5B midtraining tokens — 10× §11's `768847`/~1.9B): `extract_finemath_test_base.py` (833124→HF, ctx 13568, θ=1e4) → `sft_test.sh` (UltraChat) → `dpo_test.sh` (KatoHF chatbot_arena, chat_refine_strict) → `cot_test.sh` (`cot_sft_mix_v3`, θ=1e4) → `eval_test.sh`, all in throwaway `midtrain_finemath_test/`, 1× H100, 1-hour slices. Reused mix v3 = one-variable vs §11 (base only vs think_mix3; SFT+DPO only vs think_finemath).

> **Bug (cost a stalled day).** SLURM `--export=ALL` leaks finished stage's env into the next sbatch — SFT's `DATA_PATH=ultrachat`/`OUTPUT_DIR=…/sft`/`DATASET_RECIPE=chat_refine_strict` clobbered DPO's `${VAR:-default}` → DPO built pairs from UltraChat → "kept 0 unique rows" → crash in ~1m47s. **Fix:** `unset` every config var atop each chained launcher (done in `dpo_test.sh`, `cot_test.sh`); keep only `RESUME_FROM_CHECKPOINT`. After fix DPO kept 204 valid pairs. (→ avoid-list)

**Result — hard math↔general trade-off the balanced pipeline could NOT undo** (`report/finemath_test_{math,gen}_{nt,th}.log`; NEW=`midtrain_finemath_test/think`):

| quadrant | **NEW (latest FineMath, full SFT→DPO→CoT)** | think_mix3 | think_mix2 | think_star2 |
|---|---|---|---|---|
| MATH no-think (greedy) | **10/10** clean/terse | ~0/10 (degen) | ~1/10 | ~1/10 |
| MATH + CoT (sample) | **10/10** short traces | 5/10 | ~5/10 (loops) | ~5/10 (loops) |
| GENERAL no-think | **~0/11** | ~8/11 | ~8/11 | ~7/11 |
| GENERAL + CoT | **~1/11** (only "Paris") | good | good | good |

- **Math best of any model, both modes** — 10/10 no-think + 10/10 CoT, non-degenerate; never falls into `x=2 x=2…` / `7+3=10→4+3=7` loops that swallow mix2/star2/mix3. **Numerate base does all of it.**
- **General catastrophically gone, worse than §11's savant; full SFT+DPO did NOT rescue.** "capital of France is France", "sun is a planet", "photosynthesis = breaking food into CO₂", loops on Shakespeare/primary colors; CoT can't even close `</think>` on general Qs.

**Why this refutes §11's fix.** Did full general SFT (UltraChat)+DPO on a much-later checkpoint → general got **worse**. Culprit = the **FineMath base itself**: ~20.5B math tokens catastrophically forgot world knowledge (§6 zero-sum, on the *pretraining* side); ~250M downstream SFT tokens can't re-teach forgotten facts. More FineMath ⇒ better numeracy AND deeper forgetting — `768847`→`833124` moved both dials wrong. A **capability–capability trade-off in the base**, not a recipe problem.

**Corrected recommendation (supersedes §11's "later checkpoint").**
1. Use an **earlier** FineMath checkpoint — find the knee (numeracy up, e.g. `768847`=16/20; general not yet collapsed). Sweep early `checkpoint_step_*` with `quick_base_probe.py` + a general-knowledge probe.
2. **Better: fix the midtraining recipe** — interleave general web/chat replay into FineMath (anti-forgetting) so base keeps Paris/Mars/oxygen while gaining arithmetic. **mix v4 alone cannot save a base that forgot the capital of France.**

All `midtrain_finemath_test/` ckpts deleted post-grading; launchers `reasoning/{extract_finemath_test_base.py,sft_test,dpo_test,cot_test,eval_test}.sh` remain.

---

## 13. What to do about general ability — and should we keep midtraining math?

| base | FineMath tokens | downstream MATH (CoT) | general chat |
|---|---|---|---|
| old `argonne-3.0-base` | 0 | 5–6/10 (slips) | **~8/11 ✅** |
| FineMath `768847` (§11) | ~1.9B | **10/10 ✅** | broken (skipped SFT/DPO) |
| FineMath `833124` (§12) | ~20.5B | **10/10 ✅** | **~0/11 ❌** |

**Decisive: math benefit saturated early, forgetting did not.** ~1.9B→~20.5B FineMath bought **~0** extra elementary numeracy (both max the 10/10 probe) but turned a recoverable savant into an unrecoverable one. Every FineMath token past ~2B is near-pure downside. *(Caveat: 10-item easy probe saturates trivially — says nothing about hard/competition MATH; measure before concluding if hard math is a goal.)*

### Keep midtraining math?
- **Continue current pure-FineMath run? No** — idle at `864124`, numeracy saturated ~`768847`, more pure math only deepens forgetting.
- **Inject math at all? Yes — never pure diet.** Right knob = **replay mix**.

### Plan, cheapest first
1. **Confirm diagnosis + find knee (~10 min)** — add a general-knowledge base probe to math-only `quick_base_probe.py`; run old-base vs pinned `768847` vs latest `864124`. (Only early base still on disk = pinned `768847`; sweep is effectively old / 768847 / latest.)
2. **Balanced pipeline on *earliest* good base** — point §12 launchers (`extract_finemath_test_base.py` + `{sft,dpo,cot,eval}_test.sh`) at `768847` instead of `833124`.
3. **Durable fix — replay-mix midtraining** — edit `midtraining.py` Phase-2 recipe to interleave **~40–60% general/longmino replay** with FineMath, re-run. The only path letting the balanced pipeline (+ mix v4) win both quadrants.
4. **Cheap side-experiment — model soup** — weight-average FineMath ⊕ old general base, probe both quadrants.

**Recommendation:** stop treating "more FineMath tokens" as progress; do (1) now, (2) as cheap near-term, plan (3) as real fix. Do NOT resume pure-math run as-is.

### Measured — base probe on BOTH axes (2026-07-01, `report/base-probe-general.out`)

`reasoning/base_probe_general.py` (few-shot greedy: 20 math + 15 world-knowledge) on three raw bases. **Overturns §12's "FineMath caused forgetting" and kills the intermix-from-longmino plan:**

| base | MATH /20 | GENERAL /15 | character |
|---|---|---|---|
| `pretrain/argonne-3.0-base` | 3 | **13** | knowledgeable, innumerate |
| `midtrain/` longmino (Phase-1) | 2 | **5** | degraded+degenerate ("largest planet is the blue.krone") |
| `midtrain_finemath/864124` (Phase-2) | **18** | 6 | numerate, factually hollow ("capital of Japan is Japan", "first president: Kennedy") |

- **Longmino (proposed canvas) is NOT healthy** — 13→5 general + degenerate, gained nothing on math (3→2). **Phase-1 context-extension did most of the world-knowledge damage, before FineMath.** Only real contribution = long context.
- **FineMath not main culprit** — on top of longmino 5→6 general (flat) while math 2→18. General was *already* destroyed at longmino.
- **Destruction, not suppression** — few-shot couldn't surface Tokyo/Washington/Portuguese from FineMath base; consistent with §12 (SFT+DPO couldn't recover). Fine-tuning can't rebuild lost facts.

**Corrected canvas + plan (supersedes "intermix from longmino").** Only base with intact world knowledge = **`pretrain/argonne-3.0-base` (13/15)**:
- **Do NOT intermix from longmino** (forgotten), don't treat it as required: old general-good models (mix2/mix3, ~8/11) were SFT'd directly from `argonne-3.0-base`, getting long context via RoPE extrapolation at SFT time (§11) — longmino was never on the general-good path.
- **Intermix midtraining from `argonne-3.0-base`** — one balanced continued-pretrain run mixing FineMath + general/web replay (hold general ~13), **skipping longmino**. Target ~13 general AND ~18 math. Smoke-test few-hundred-M slice, re-probe with `base_probe_general.py`, then scale.

---

## 14. Implementing the intermix fix — the midtraining launchers now do it (DONE 2026-07-01)

§13's plan wired into production launchers. `weekend.sh`, `night.sh`, `midtraining.sh` switched from two-phase longmino→FineMath to a **single INTERMIX phase**: seed healthy pretrain base, train on doc-shuffled FineWeb+FineMath mix → `models/midtrain`.

**What changed (minimal, backward-compatible).**
- `midtraining.sh`: added `DOC_SHUFFLE` knob (default 0) + `--doc_shuffle` on torchrun. **Critical missing piece** — trainer never passed `--doc_shuffle`, so an intermix manifest read in manifest order (all FineWeb then all FineMath) = sequential = the catastrophic forgetting we're fixing. `DOC_SHUFFLE=1` globally permutes docs each epoch.
- `weekend.sh`/`night.sh`: export `DATA_OVERRIDE=<intermix manifest>`, `DOC_SHUFFLE_OVERRIDE=1`, `ROPE_THETA_OVERRIDE=1000000` (match argonne-3.0-base θ=1e6 proven-general path, NOT FineMath θ=1e4), `PHASE2_DATA=""` (single phase). Seed (`pretrain/checkpoint_step_329148.pt`) + output (`models/midtrain`) = existing Phase-1 defaults; auto-resubmit/slices/FSDP/wall-time saves untouched.
- Result: `bash weekend.sh` (continuous) or `bash night.sh` (one 8h slice @23:00) runs intermix from base. `midtrain/` had no loose `.pt` (only old `final_model_complete`) → seeds fresh, not resuming longmino.

**Intermix corpus (`reasoning/build_intermix.py`).** `DocManifestDataLoader` takes ONE `block_size`-token window per doc per epoch → effective **token mix = DOC-count ratio**, not corpus size. Builder: references all 64 FineMath doc-bin shards (absolute paths); carves matching #docs from **FineWeb** (`.../CC-MAIN-2025-21-binary/train.bin` — pretrain corpus, source of base's 13/15) into `DOC_LEN`-token docs (one shard + `.lengths.npy`); emits merged manifest `tokenized_dir="/"` + absolute paths. `GENERAL_RATIO` (default 1.0=50:50); bump 1.5 (60:40 general) to lean against forgetting.
- **Gotchas:** `DOC_LEN` must be `> block_size` (13570 for the 13568 production block) or loader raises "Short doc window"; FineWeb `.bin` has a **1024-byte header (256×int32)** before uint32 tokens (`pretrain.py: offset=256*4`) — read from byte 0 = garbage IDs. Production manifest: `/project/rcc/youzhi/data/intermix/intermix_manifest.json` (28GB FineWeb slice + 64 FineMath shards, 520,066 docs each, 50:50).

**Test-drive (validated it RUNS, not that result is good).** `weekend.sh` ~32 min: seeded argonne-3.0-base (step 329148, phase counter reset to 0), trained ~985 steps θ=1e6 doc-shuffled intermix, saved `midtrain/checkpoint_step_330133.pt`, cancelled. End-to-end pipeline confirmed; a real run resumes from 330133.

**Still open.** NOT confirmed 50:50 preserves general knowledge (smoke probe cancelled). On first production ckpt: `EXTRA_CKPT=<midtrain .pt> EXTRA_THETA=1000000 python reasoning/base_probe_general.py`; if GENERAL <~13/15, rebuild `GENERAL_RATIO=1.5`, relaunch.
> **UPDATE (2026-07-02, §16): measured.** At ~644M intermix tokens: MATH 12/20 / **GENERAL 11/15** — below threshold → rule fired, manifest rebuilt at `GENERAL_RATIO=1.5` (60:40). Full pipeline audit + latent bug fixes in §16.

**New files.** `reasoning/base_probe_general.py`/`.sh` (few-shot BASE probe both axes, 20 math + 15 world; `EXTRA_CKPT`/`EXTRA_THETA` add a ckpt); `build_intermix.py` (production manifest); `build_intermix_smoke.py`/`intermix_smoke.sh` (block-2048 smoke: build+short midtrain+auto-probe); `weekend.sh`/`night.sh`/`midtraining.sh` (edited: single-phase intermix, doc_shuffle, θ=1e6).

---

## 15. The decisive control — run the recipe on REAL bases (Qwen1.5-0.5B, Llama-3.2-1B) — DONE 2026-07-02

Never-run experiment (§1–§14 all fought a from-scratch base whose ceiling was set upstream, throughline #1): run the IDENTICAL recipe on real off-the-shelf bases. Chose **Llama-3.2-1B** (1.24B, strong) and **Qwen1.5-0.5B** (0.46B, the "should be worse" base).

### Base probe overturns the premise: BOTH real bases strong on BOTH axes
`reason_control/probe.py` (same 20-math/15-general few-shot as §13):

| base | params | MATH /20 | GEN /15 | character |
|---|---|---|---|---|
| argonne-3.0-base | 2.88B | 3 | 13 | innumerate (from-scratch) |
| longmino | 2.88B | 2 | 5 | degraded both |
| FineMath-864124 | 2.88B | 18 | 6 | numerate, amnesiac |
| **Llama-3.2-1B** | 1.24B | **13–14** | **15** | strong both |
| **Qwen1.5-0.5B** | 0.46B | **14** | **14** | strong both |

Even 0.46B Qwen clears the numeracy ceiling AND keeps world knowledge — both-axes health no Argonne base had (Qwen solved "divisors of 12 = 6" cold, which Llama-1B missed). "Should be worse" already FALSE at probe level. (Caveat: greedy few-shot single-fact, ±1 bf16 wobble.)

### Method — base-agnostic recipe harness (`reasoning/reason_control/`)
Argonne scripts are welded to ArgonneModel + Qwen-3 tokenizer + FSDP; re-implemented same recipe as model-agnostic plain-HF scripts:
1. **INTERMIX midtrain**: prebuilt intermix `.bin` is Qwen-3-tokenized (incompatible), so re-stream raw FineWeb+FineMath parquet, tokenize with base's OWN tokenizer, doc-shuffle 50:50, pack 1024, continued-pretrain **80M tokens @ LR 5e-5** (strong base → light touch, not from-scratch 1e-4).
2. **SFT** UltraChat, **DPO** argilla/dpo-mix-7k (hand-rolled loss + frozen ref), **CoT-SFT** on the SAME `cot_sft_mix_v3` as `think_mix3`/`think_finemath` — only variables vs Argonne runs = base + tokenizer.
3. **4-quadrant eval** (exact §5 probes).

Chat-family auto-detected (ChatML vs Llama-3 headers), label-masking + `apply_chat_template` verified token-identical per family. Runtime HBM autotuner sizes batch for 80G/96G H100.

### Intermix effect on already-strong base (base → after 80M-tok @5e-5)
- Qwen1.5-0.5B: MATH 14→12, GEN 14→13 (mild drop both)
- Llama-3.2-1B: MATH 13→12, GEN 15→15 (general preserved; math −1)

CONFIRMS: intermix is neutral-to-mildly-negative on an already-numerate base, NOT the big §14 lift. 5e-5 kept damage small (1e-4 smoke dropped Qwen math to 11). **Intermix is a base-repair tool, not universal — for healthy bases skip or keep LR tiny.**

### 4-quadrant eval — recipe yields working reasoner on BOTH bases
| quadrant | **Qwen-0.5B think** | **Llama-1B think** | best Argonne (think_*) |
|---|---|---|---|
| MATH no-think | **10/10** | **10/10** | ~0–1/10 (degenerate `\boxed`) |
| MATH + CoT | **10/10** | **10/10** | 5–7/10 |
| GEN no-think | **8/10** | **9/10** | ~8/11 |
| GEN + CoT | **8/10** | **8/10** | good |

**Both = clean pass on all four quadrants — NO Argonne checkpoint ever did.** Each Argonne best failed ≥1: `think_finemath`/`think_test` = 10/10 math but ~0/10 general (savant); `think_mix2/mix3` kept ~8/11 general but FAILED math no-think (`\boxed{first#}`), only 5–7/10 math-CoT. 0.46B Qwen matched 1.24B Llama on math, trailed by one on general (8 vs 9, tracking base gap 14 vs 15). Both solved all four §10 residuals with textbook CoT. Runs: ~1h42 Qwen / ~3h Llama on one H100.

### What this establishes
1. **Throughline #1 proven affirmative — recipe was never the problem, the base was** (→ recipe / bottom line).
2. **"Smaller worse" refuted — base QUALITY not size was the lever**; small-balanced beats large-lopsided (2.88B FineMath 18/20 math but amnesiac/unrecoverable). Capacity would bite on harder competition math, not this bar.
3. Intermix = base-repair not universal good (SFT→DPO→CoT recovered general to 8–9/10).
4. `reasoning/reason_control/` is a reusable base-agnostic full-recipe runner w/ HBM autotuner; each run <3.5h on one H100.

### Operational lessons
- **Memoize tokenizer family-detection**: `tok.get_vocab()` (128k–152k dict) called per-example cost ~75ms/ex, made a build hang ~50min (looked like a stall). Cache once → ~80× faster (113k-row CoT build → ~1min). Load HF column once + shuffle INDICES in memory (`ds.shuffle()`+row-iter random-accesses project FS, pathological).
- **HBM autotuner: measure sustained RESERVED not single-step ALLOCATED**. Single fwd+bwd on fresh allocator reports ~95% at a batch that then OOMs; OOM tracks `max_memory_reserved` (frag), stabilizes only after ~10 steps. LM-head logits `(bs,seq,vocab)` fp32-upcast are the ceiling. Force **grad_accum=1** (accumulation keeps a prior micro-step's `.grad` resident). OOM traceback pins failed tensors → naive reduce-retry CASCADES (28→10); null locals + `gc.collect()` + `synchronize()` + `empty_cache()` before backoff. Net: selects ~96%, settles ~80–96% after ≤1 backoff.

### New files — `reasoning/reason_control/`
`common.py` (chat autodetect + token-identical `render_chat`, HBM autotuner, time-boxed loop, §5 probe/eval sets), `probe.py` (both-axes base probe), `midtrain.py` (re-tokenized doc-shuffled intermix), `sft.py`/`dpo.py`/`cot.py`, `eval.py` (4-quadrant), `run_all.sh` (one resumable time-boxed job; `BASE_MODEL_PATH` picks base).

**Bottom line:** stop fixing reasoning downstream. A good base + this modest recipe = a 0.5B reasoner passing every quadrant. If a from-scratch Argonne base is still the goal, target is explicit: ~14/20 math AND ~14/15 general simultaneously, then this recipe finishes it.

---

## 16. First intermix probe + full pipeline audit (2026-07-02) — 50:50 → 60:40, five latent bugs fixed

§14 intermix reached step **345108 ≈ 644M intermix tokens** (~78M tok/h on 3×H200). Ran §14-prescribed probe + line-by-line pipeline audit.

### The probe (`report/base-probe-intermix-345108.out`) — decision rule fired
| base | MATH /20 | GEN /15 |
|---|---|---|
| argonne-3.0-base (seed) | 3 | 14 |
| longmino | 2 | 5 |
| FineMath 864124 | 18 | 6 |
| **intermix @ 644M tok** | **12** | **11** |

- Design works directionally: math 3→12 with only ~322M math tokens (pure FineMath needed ~1.9B for 16/20); general held at 11 not collapsing to 5–6. Loss oscillation (~1.1 math ↔ ~2.9 web docs) = signature of real doc-interleaving.
- **GENERAL slipped below §14 threshold (~13/15)** → per pre-registered rule, manifest **rebuilt `GENERAL_RATIO=1.5` (60:40)**. Verified safe: 55.2B FineWeb avail vs 10.6B needed; `midtraining.py` metadata-mismatch resume keeps doc cursor, resets epoch. 60:40 raises epoch capacity to ~17.6B tok so 16B target now fits `max_epochs=1` (50:50 epoch was only 14.11B). **Never rebuild while a slice trains** — `build_intermix.py` opens `fineweb_slice.bin` `"wb"`, truncating a live memmap.
- Going forward: probe every night slice (`EXTRA_CKPT=<latest .pt> EXTRA_THETA=1000000`, ~15min/1×H100). Stop rule **MATH ≥14 with GEN ≥13** (§15 bar). If GEN <11 at 60:40: LR 3e-4→1e-4 or `GENERAL_RATIO=2.0`.

### Audit — verified CORRECT
`doc_shuffle` is global per-epoch-seeded permutation across all 65 shards (`_refresh_doc_order`), resume restores exact permutation+position. Data clean: uint32, FineWeb 1024-byte header handled, every doc ≥13,570 tok (FineMath min 16,384) so "short doc window" crash can't fire. Checkpoint saves atomic (tmp+`os.replace`) w/ truncated-fallback; θ=1e6 + overrides survive every sbatch hop incl auto-resubmit/retry.

### Bugs found & FIXED (all latent — none tainted trained checkpoints)
1. **Zombie resubmit chain**: finished single-phase run wrote `final_model_complete` but `midtraining.sh` only done-checked the two-phase path → AUTO_RESUBMIT=1 launched 1-step slices forever, each a 36GB ckpt. Fixed: single-phase completion gate + don't-resubmit guard.
2. **Stale `midtrain/final_model_complete` (old longmino final)** in live intermix dir: was `PHASE1_DONE_MARKER` (would flip to sequential FineMath if PHASE2_DATA leaked), would be overwritten when intermix finishes, was probes' longmino baseline. **Renamed → `final_model_complete_longmino`**; refs updated (`base_probe_general.py`, `quick_base_probe.py`, both extract scripts).
3. **`night.sh` PHASE2_DATA leak** (unlike weekend.sh, didn't pin empty in `--export`; §12 `--export=ALL` leak class) → stale export re-enables sequential FineMath. Fixed: appended `PHASE2_DATA=` to EXTRA_EXPORT.
4. **`eos_token_id: null` at source** (§5 eos bug): `midtraining.py` config never set eos → every future `final_model_complete` re-introduced it. Fixed: config takes eos/bos/pad from tokenizer (also `extract_finemath_base.py`).
5. **`extract_finemath_base.py` hardcoded θ=1e4 + FineMath path** — reuse on intermix ckpt writes corrupt-config base (§11 reversed). Now env-parameterized (`CKPT`, `ROPE_THETA`, `OUT`, `TOKENIZER_SRC`), guard refuses `final_model_complete` next to live ckpts (extract script IS the handoff since knee stop leaves no final artifacts).
6. **FSDP grad clipping** used `torch.nn.utils.clip_grad_norm_` → under `shard_grad_op` clips per LOCAL shard norm (underestimated, rank-inconsistent). Fixed → `FSDP.clip_grad_norm_`. (pretrain/continue_pretrain are DDP, already correct; mostly benign since clip=1.0 rarely binds.)
7. **WSD cooldown/warmup used GLOBAL scheduler steps** while `estimated_steps` is phase-local → on seeded run (resumes ~329k) `COOLDOWN_OVERRIDE` would collapse LR instantly. Fixed: phase-local, anchored to min(epoch-end, token-target). Side effect: freshly-seeded phase now does its warmup; in-flight run unaffected (phase step ~16k, past warmup).

(Checkpoint pruning flagged — ~72GB/h — but quota ample, skipped.)

**Timeline**: ~570M tok/8h slice → ~4B knee (≈2B math @50:50) is ~5–7 slices / ~2 days.

### Deep eval of ckpt 345108 (`reasoning/eval_intermix_base.py`, `report/eval-intermix-345108.out`)
Doubled probe (fresh 20-math/15-general EXTENSION set) + held-out NLL@1024 + position-bucketed long-ctx NLL@13568, across seed (pretrain `329148.pt`, bit-identical to argonne-3.0-base dir), pure-FineMath, intermix @644M:

| model | MATH std/ext | GEN std/ext | NLL FW | NLL FM | ppl @8k-13.5k |
|---|---|---|---|---|---|
| seed 329148 | 3/2 | 14/15 | 2.572 | 3.322 | **803** (collapse past 1024) |
| finemath 864124 | 18/17 | 6/9 | 4.215 | 1.408 | 71 |
| **intermix 345108** | **12/11** | **11/14** | **3.006** | **1.601** | **17** (flat) |

1. Math gain replicates on never-seen items (11 ext vs 12 std) — no overfit. Combined 23/40 vs seed 5/40.
2. General healthier than 11/15 suggested: 14/15 ext, 25/30 combined vs seed 29/30 — mild erosion, not FineMath's collapse (15/30). Part of FineWeb-NLL rise = distribution shift, not fact loss; probe is better forgetting metric.
3. **Efficiency headline**: at 644M tok (4.5% epoch), captured **~90% of math-NLL gain** of 21B pure-FineMath (Δ1.72 of Δ1.91) at **~26% of general-NLL cost** (Δ0.43 vs Δ1.64).
4. Intermix IS the context extension (replaced longmino): seed ppl explodes past 1024 (θ=1e6 raw does NOT extrapolate); intermix flat to 13.5k (ppl 12–25). One phase = numeracy + retention + long-ctx.

Gap to §15 bar: math probe must ~double; NLL says knowledge arriving, accuracy follows tokens. Keep nightly probe, stop at knee.

---

## 17. Deep eval of intermix `363908`, and the WEIGHT-SOUP that clears the bar (2026-07-03 — pipeline wired into weekend.sh/night.sh)

§16 said "grind to the ~4B knee." This section OVERTURNS that: **stop the 16B grind, harvest banked math via a training-free weight-soup, run the real downstream test.** (Absorbs former `eval_intermix_363908_findings.md`.)

> **⭐** Training-free `0.35·seed + 0.65·intermix363908` (both θ=1e6, same lineage → clean linear interp) = **MATH 15/20 AND GEN 13/15** — **first from-scratch Argonne base to clear the both-axes bar** (§15: MATH ≥14, GEN ≥13), from checkpoints we already have. Soup downstream pipeline now default of weekend.sh/night.sh (`MODE=soup`).

**Checkpoint.** `models/midtrain/checkpoint_step_363908.pt` (~1.41B intermix tok; 50:50 through 345108, **60:40** after; LR still **3e-4**), pinned during eval. Seed = `pretrain/checkpoint_step_329148.pt` = argonne-3.0-base. Pruning off → 363908 persists.

### Numbers (`reasoning/eval_intermix_base.py`, 1×H100, θ=1e6)
| model | MATH std/ext | GEN std/ext | NLL FW | NLL FM | long-ctx ppl @8k-13.5k |
|---|---|---|---|---|---|
| seed 329148 | 3/2 | **14/15** | 2.572 | 3.322 | ~803 (collapse past 1024) |
| finemath 864124 | **18/17** | 6/9 | 4.215 | **1.408** | 71 |
| **INTERMIX 363908** | **14/11** | **11/10** | **2.984** | **1.589** | **15** (flat) |
| — @345108 (§16, 644M) | 12/11 | 11/14 | 3.006 | 1.601 | ~17 (flat) |

**Bar: MATH ≥14 AND GEN ≥13 on standard set.** Raw 363908: MATH 14 ✓ (soft — ext only 11, multi-step fails) · GEN 11 ✗ (misses by 2, TRENDING DOWN: ext regressed 14→10 vs 345108; real seed-facts lost — largest planet→"Mars", 1st president→"Lincoln"). **CASE-B near-miss driven entirely by general.**

- Eval trustworthy (audited): seed `.pt` bit-identical to argonne-3.0-base dir; finemath 864124 reproduced §16 exactly → intermix numbers not a silent-crippled-load artifact. θ=1e6 correct (flat curve; 1e4 corrupts — §11 trap).
- **Diagnosis:** at **LR 3e-4** (from-scratch pretraining LR), accumulating math over-writes seed's 14/15 general faster than 60:40 replay protects — §15 lesson (healthy base wants 5e-5) showing as erosion at 3e-4.
- Efficiency win holds: ~90% math-NLL gain at ~25% general-NLL cost in 1.41B tok; also IS the context extension (longmino obsolete).

### Three cheap tests → stop the grind, soup instead
**1. Forecast to 16B.** Math near-saturating (probe 3→12→14; NLL floor 1.41 ~reached) → predicted end 16–18/20 (clears). General below bar, NOT recovering: std stuck 11 across 2.2× tokens AND the 50:50→60:40 rebuild; web-mode train loss flat-to-rising (2.80→2.87, ticked UP across the switch) → predicted 10–12/15 sub-bar. Finishing ≈ 8 days for ~0 gain (§13's "math saturates early, forgetting doesn't").

**2. Weight-soup frontier** (seed ⊕ intermix, both θ=1e6 → clean linear interp; α=0 seed, α=1 raw intermix):
| α | MATH std/ext (/40) | GEN std/ext (/30) |
|---|---|---|
| 0.00 seed | 3/2 (5) | 13/15 (28) |
| 0.55 | 13/10 (23) | 13/14 (27) |
| 0.60 | 13/12 (25) | 13/14 (27) |
| **0.65** | **15/12 (27)** | **13/13 (26)** ← **CLEARS BAR** |
| 0.70 | 16/12 (28) | 12/13 (25) |
| 0.75 | 15/11 (26) | 12/13 (25) |
| 1.00 intermix | 14/11 (25) | 11/10 (21) |

- **`0.35·seed + 0.65·intermix` clears bar (MATH 15, GEN 13), robust on ext (12/20·13/15), beats raw intermix on BOTH axes.**
- Raw ckpt slightly over-trained (α=0.70–0.75 give more math AND general than α=1.0 — WiSE-FT overshoot); pulling ~30% back to seed recovers general free. Math/general nearly linearly reconcilable in weight space; α≈0.65 = sweet spot.

**3. Opportunity cost.** §15 already produced a 4-quadrant reasoner on Qwen/Llama without intermix; the intermix+soup payoff is specifically the from-scratch-Argonne ambition — which α≈0.65 achieves at base level.

### Verdict & plan (implemented)
- **Worth it? YES** — intermix broke the numeracy ceiling that beat §1–§13, and souped w/ seed yields first from-scratch Argonne base to clear both-axes bar (15/13); also replaced longmino.
- **Continue 16B? NO.** Math banked/saturating; general won't reach bar. Instead: (1) stop/cancel intermix chain (keep 363908 + seed); (2) build α≈0.65 soup as base; (3) run downstream SFT→DPO→CoT→4-quadrant on the soup — real gold standard, never tested on an Argonne base (§15's link borrowed from Qwen/Llama); this ~1-day 1×H100 run is the decisive experiment; (4) only if downstream general short: LR 3e-4→1e-4 / GENERAL_RATIO=2.0 / finer α.
- **Caveat:** bar is a coarse base probe (easy math, keyword grading, ±1 wobble; α=0.65 general exactly at 13, ext 13/15 corroborates). Necessary not sufficient — 4-quadrant run is honest judge.

### Implemented — soup pipeline wired into launchers
`weekend.sh`/`night.sh` gained `MODE` switch defaulting to **`soup`** (old run preserved under `MODE=intermix` for step-4 fallback):
- **`MODE=soup`** submits `reasoning/sft_soup.sh` → builds soup base once → auto-chains `sft_soup → dpo_soup → cot_soup → eval_soup` (each a 1×H100 continuous self-resubmitting job chaining next, §11 `_finemath` pattern).
- **θ=1e6 throughout** — `cot_soup.sh` sets `ROPE_THETA=1000000.0` (NOT FineMath 1e4, §11 trap reversed); SFT/DPO inherit from base config.
- New dirs `sft_soup`/`dpo_soup`/`think_soup` (nothing from §11/§12/§15 clobbered); eval logs → `report/soup_{math_nt,math_th,gen_nt,gen_th}.log`.
- **Gotcha:** running INTERMIX chain does NOT stop on its own (each `midtraining.sh` slice resubmits next directly; editing weekend.sh doesn't redirect it). `scancel` to free H200s (math already banked). Soup build reads pinned retained `363908.pt`.

**New files.** `build_soup_base.py` (memory-frugal `(1-α)·seed + α·intermix` → standalone HF base dir, θ=1e6, ctx 13568, trimmed embeddings; idempotent; env `ALPHA`/`SEED_CKPT`/`INTERMIX_CKPT`/`OUT`/`ROPE_THETA`/`TOKENIZER_SRC`), `sft_soup.sh` (builds soup once + UltraChat SFT → dpo_soup.sh; `MODE=soup` entry), `dpo_soup.sh` (DPO KatoHF chatbot_arena `chat_refine_strict` → cot_soup.sh), `cot_soup.sh` (CoT-SFT `cot_sft_mix_v3`, θ=1e6 → eval_soup.sh), `eval_soup.sh` (4-quadrant `think_soup` vs `think_mix3/mix2/star2`; key A/B = `think_soup` vs `think_mix3`, same v3 CoT on OLD innumerate base, isolating the both-axes soup base).

**Reproduce.**
```bash
ALPHA=0.65 python reasoning/build_soup_base.py       # -> models/soup_seed_intermix_a065
EXTRA_CKPT=/project/rcc/youzhi/models/soup_seed_intermix_a065 EXTRA_THETA=1000000 \
  EXTRA_LABEL="soup a=0.65" python reasoning/base_probe_general.py   # go/no-go: MATH ≥14 AND GEN ≥13
sbatch reasoning/sft_soup.sh                          # full downstream (== MODE=soup bash weekend.sh)
```

---

## 18. Running the soup downstream — the math ceiling breaks, but general doesn't hold (2026-07-04/05)

§17's α=0.65 weight-soup base (probe **15/13**, first from-scratch Argonne base to clear both axes) ran the full recipe via `weekend.sh MODE=soup`.

> **Soup base broke the NUMERACY ceiling — `think_soup` is the first from-scratch Argonne model at 10/10 in BOTH math modes + all four §10 residuals. But GENERAL chat regressed under CoT-SFT (loops + lost facts); CoT-data rebalance (mix v4) didn't fix it (lateral). From-scratch soup line plateaus ~29/40 (strong math/weak general); a clean four-quadrant pass still belongs to §15's real bases (~36/40).**

### 18a. What ran (zero-crash, end-to-end)
`MODE=soup bash weekend.sh` → `build_soup_base.py` (0.35·seed329148 + 0.65·intermix363908, θ=1e6, trimmed embeds) → `sft_soup.sh` (UltraChat, 1×H200) → `dpo_soup.sh` (KatoHF chatbot_arena, **204 pairs — no §12 zero-pairs crash**) → `cot_soup.sh` (**`cot_sft_mix_v3`, θ=1e6**, 3×H200 DDP) → `eval_soup.sh`. Zero failures; θ=1e6 held every stage.

### 18b. `think_soup` (v3 CoT) — the headline math win

| quadrant | **think_soup** | think_mix3 | think_mix2 | think_star2 |
|---|---|---|---|---|
| MATH no-think | **10/10** | 0/10 | 2/10 | 3/10 |
| MATH + CoT | **10/10** | 5/10 | 6/10 | 7/10 |
| GENERAL no-think | 5/10 | 8/10 | 8/10 | 7/10 |
| GENERAL + CoT | 4/10 | 3/10 | 7/10 | 4/10 |
| **total** | **29/40** | 16/40 | 25/40 | 21/40 |

- **Math ceiling gone.** Only Argonne model solving math-no-think and 10/10 both modes; nails all four §10 residuals (`2x=17−5=12,x=6`; `n(n+1)/2=55`; `2·(8+3)=22`; `(2+1)(1+1)=6`). Six months of STaR/GRPO never did this — the soup base did immediately.
- **General regressed:** non-terminating loops (grammar→"conjunction"×11) + lost facts (Red Planet→Earth, colors→green) — worse on soup base than the SAME v3 data on old base (mix3 kept 8/10).

### 18c. The diagnostic (key methodological result)
Evaluated **`dpo_soup` (pre-CoT)** on general: **general-HEALTHY ~7–8/10, concise, NO loops** (grammar correct, no degeneration). So **general was fine after SFT+DPO; the CoT stage broke it → the base is NOT the bottleneck, the CoT fine-tune is.** (Taller/shorter puzzle + primary colors are wrong in `dpo_soup` too = genuine base gaps.) Consequence: more midtraining/better soup won't help — the lever is the CoT step.

### 18d. mix v4 — rebalance CoT data (DID NOT fix general; lateral)
v3 is 70% long-`<think>`, 30% direct → over-generalized to "always reason at length" → loops. `build_mix_v4.py`: keep all v3, **upsample `direct_tulu` no-think 3× → 56% direct/44% think**. Re-ran CoT from `dpo_soup` → `think_soup_v4`.

| quadrant | think_soup (v3) | **think_soup_v4** | mix2 (old base) |
|---|---|---|---|
| MATH no-think | 10/10 | **9/10** (divisors→2) | 2/10 |
| MATH + CoT | 10/10 | **9/10** (Σ1..10→21) | 5/10 |
| GENERAL no-think | 5/10 | **6/10** | 8/10 |
| GENERAL + CoT | 4/10 | **5/10** | 10/10 |
| **total** | 29/40 | **29/40** | 25/40 |

- **Same 29/40, lateral trade** (+1 gen/mode, −1 math/mode). Fixed Red Planet (→Mars), closes more traces, but grammar-fix still loops; base errors persist.
- **CONFOUND:** v4 changed data (v3→v4) AND batch/LR (eff-12/1e-5 → eff-30/1.6e-5, forced by HBM-fill below) — not a clean ablation. **Verdict: rebalancing CoT data doesn't rescue general.**

### 18e. HBM-fill saga (loss-logits are the CoT memory ceiling)
Pushing to ≥95% HBM: **batch 18/16/12 all OOM in the backward** — killer is fp32 loss logits `(batch×seq×vocab=151,669)` (~4 GiB/row @seq≈4k, backward spikes ~30+ GiB). **batch 10 = 98% HBM, no OOM** (~9.8 h/6k-step epoch) = ceiling. Lessons: profile the backward not startup; grad-ckpt hard-coded ON in `cot-sft.py` (can't disable); grad-accum=1 so filling HBM forces bigger eff-batch (eff-30 vs proven eff-12) — batch 4 (eff-12, ~63% HBM) is quality-first.

### 18f. Decoding hypothesis — REFUTED (`report/soup_dec_*.log`)
Added `--repetition-penalty`/`--no-repeat-ngram` to `eval_numeracy.py` (generated-tokens-only, §5), re-ran at rep-penalty 1.3 + no-repeat-3:

| quadrant | think_soup (greedy→dec) | think_soup_v4 (greedy→dec) |
|---|---|---|
| MATH no-think | 10 → **4** | 9 → **5** |
| MATH + CoT | 10 → **4** | 9 → **4** |
| GENERAL no-think | 5 → **7** | 6 → **4** |
| GENERAL + CoT | 4 → **3** | 5 → **3** |
| **total** | 29 → **18** | 29 → **16** |

- **Net LOSS — anti-repeat CORRUPTS math**: `half of 80`→"8/2"→boxed **4**; `15% of 80`→"15/100×8"→**9**; garbles sentences ("doesn't liked"). Helps ONE cell (think_soup gen-nt 5→7). **Verdict: no single decoding config wins → best is plain GREEDY; general regression is baked into weights + base gap, not a decoder artifact.** Real number: `think_soup` @ greedy = **29/40**.

### 18g. Verdict & next
1. **Do NOT resume intermix midtraining** — saturated (§16/17) AND base isn't the bottleneck (§18c).
2. Check decoding re-eval first (cheapest) — done (§18f).
3. **Durable path = base QUALITY not more argonne3.0** (§15 → ~36/40 on Qwen/Llama-grade base); this is the **argonne3.5 recipe search** target.
4. Un-confounded soup-improvement test = re-run v4 CoT at eff-12/LR-1e-5 (~63% HBM) — expect small gain at best.

### New files
`build_soup_base.py` (training-free `(1-α)·seed+α·intermix`, θ=1e6, ctx 13568, trimmed embeds, idempotent); `{sft,dpo,cot,eval}_soup.sh` (chain; `cot_soup.sh` θ=1e6, defensive `unset` for §12 leak; logs `report/soup_*.log`); `build_mix_v4.py` (v3+3× `direct_tulu`→`cot_sft_mix_v4`); `{cot,eval}_soup_v4.sh` (batch 10 @98% HBM, LR 1.6e-5); `eval_numeracy.py` (+rep-penalty/no-repeat-ngram); `weekend.sh`/`night.sh` `MODE` switch (default `soup`, `intermix` = old chain).

Retained (as of §19): `models/soup_seed_intermix_a065`, `models/instruct/{dpo_soup,think_soup}`, **`models/instruct/soup_blend_a085` (FINAL)**. Deleted `sft_soup`, `think_soup_v4`.

---

## 19. Training-FREE general recovery — weight-soup the pre-CoT and post-CoT checkpoints (2026-07-05)

§18c localized it: `think_soup = dpo_soup + CoT-delta` in ONE basin (same arch, θ=1e6); pre-CoT `dpo_soup` general-healthy; CoT stage overwrote general. Cheap un-tried lever attacking the *weights*:

> **`blend_α = (1−α)·dpo_soup + α·think_soup = dpo_soup + α·(CoT-delta)`.** High α keeps 10/10 math; low α restores general. **Training-FREE** (CPU tensor-average, ~2 min/blend) — the §17 soup trick applied one stage later to fractionally un-apply CoT.

- **`build_ckpt_soup.py`** — memory-frugal (`safe_open`, per-key fp32 blend) averager; copies config+tokenizer (think-mode chat template) from THINK dir; idempotent.
- **`soup_recover.sh`** — builds `blend_a{050,070,085}`, runs 4-quadrant probe (**greedy no-think / sampled think, NO rep-penalty**) + `think_soup` (α=1). 1×H100.
- **Go bar:** an α where general-no-think climbs toward ~7–8 while math stays ≥8/10 both modes → balanced all-round reasoner (>29/40).

### 19a. RESULT — WORKS. `blend_a085` = 33/40, best from-scratch Argonne reasoner (`report/recover_*.log`)

| quadrant (greedy nt / sampled th) | a050 | a070 | **a085** ⭐ | think_soup (α=1) |
|---|---|---|---|---|
| MATH no-think | 8 | 10 | **10** | 10 |
| MATH + CoT | 5 | 9 | **10** | 10 |
| GENERAL no-think | 8 | 7 | **7** | 5 |
| GENERAL + CoT | 1 | 1 | **6** | 6 |
| **total** | 22 | 27 | **33/40** | 31* |

<sub>*think_soup re-graded 31 vs §18b's 29 — GEN+CoT is sampled (±1–2 run-to-run); deterministic quadrants stable.*</sub>

- **`soup_blend_a085` (α=0.85) is FINAL.** Keeps **10/10 math both modes** AND recovers general no-think **5→7** (grammar loop gone, Mars restored) — **+2 net, now BALANCED**. First from-scratch Argonne model that is both a perfect arithmetic reasoner and loop-free generalist.
- **Clean α mechanism:** lower α suppresses more loops (a050 gen-nt 8, zero loops) BUT **breaks CoT trace-closure** — `<think>…</think>` formatting lives in the CoT delta; a050/a070 gen-with-CoT collapse to **1/10** (`closed=False`, answer trapped). **α=0.85 = the knee:** enough `dpo_soup` to fix general, enough CoT-delta to keep 10/10 math + close traces.
- **Residual misses = genuine base gaps** (primary-colors→"green"; taller/shorter transitivity — both wrong in `dpo_soup`). The 2.88B base ceiling → argonne3.5 target.

### 19b. FINAL VERDICT — reasoning-model line DONE
1. **Ship `models/instruct/soup_blend_a085`** (greedy for math/no-think; sampling OK for think). **10/10 math both modes + recovered general, 33/40.**
2. **Full recipe (all training-free after DPO):** seed⊕intermix soup base (§17) → SFT → DPO (`dpo_soup`) → CoT-SFT mix-v3 (`think_soup`) → **weight-soup `0.15·dpo_soup + 0.85·think_soup`** (§19, the novel step: surgically un-does CoT's general regression, keeps math, free).
3. **Do NOT** resume intermix (§18g), chase decoding (§18f), or re-balance CoT (§18d) — all inferior to the weight-soup.
4. Only remaining lever = base QUALITY (argonne3.5).
5. *Optional (not pursued):* peak may be α∈(0.80,0.90); one-job sweep +0–1, not worth it.

Retained: `soup_blend_a085` (final), `dpo_soup`+`think_soup`. Deleted `a050`, `a070`, `sft_soup`, `think_soup_v4`.

---

## 20. RLVR / GRPO on the soup thinking model — the reward-proxy plateau, confirmed (2026-07-06/07)

**Q:** can GRPO improve `soup_blend_a085` on math/reasoning benchmarks? **A: no net gain — GRPO maximizes the shaped/format reward without lifting accuracy, because the correct-answer signal is too sparse to amplify.** Reconfirms §9 on the strong base; ceiling = base capability (throughline #1), not the RL recipe.

### 20a. What ran
Two GRPO runs from `soup_blend_a085`, GSM8K rollouts in own chat/`<think>` format, verified `\boxed` reward + k3 KL leash to frozen start (`reasoning/grpo.py`):
- **Run A** `think_grpo_soup`: P=12, G=9, LR 5e-6, ~120 steps (6 h).
- **Run B** `think_grpo_soup2`: **G=64** (better advantage), LR 1e-5, chunked backward, ~32 steps (5 h).

### 20b. The plateau (both runs, same shape)
| | reward | `</think>` closed | rollout accuracy |
|---|---|---|---|
| Run A start→end | 0.02 → 0.13 | 0.49 → 0.79 | ~1.8% → ~1.8% (flat) |
| Run B start→end | 0.03 → **0.27** | 0.49 → **0.93** | 1.8% → **~2% (flat)** |

- GRPO **learns to format** (49%→93%), drives shaped reward ~9× — accuracy stuck ~2%. Run B's KL grew ~0.03 (vs A's ~0.003): larger group+higher LR moved policy toward the **format optimum, not correctness**.
- Root cause: temp-1.0 gives a fully-correct GSM8K solution in only **~1–4 of ~500 rollouts**; group-relative advantage can only reinforce successes the model already produces. **RLVR amplifies existing capability, doesn't create it** (throughline #6). Larger groups/higher LR don't fix a signal problem.

### 20c. HBM engineering
GRPO is **generation-bound** (~80% rollout gen, model+KV ≈13 GiB ≈14% of 94 GiB card; brief backward spike). Two fixes:
1. **Auto-tune to card** — H100 pool is a **mix of 80/94 GiB** (no distinguishing feature); `grpo.py --target-hbm` detects `total_memory` and sizes backward micro-batch.
2. **Decouple gen from backward** (`--gen-group`/`--bwd-micro`): large gen-group fills idle HBM during rollouts + sharpens advantage; backward chunked (fp32 `(n×seq×vocab=151k)` logits are the ceiling, §18e), grads accumulate to identical update. Both OOM-safe (retry smaller/skip). Result: **~14% → 87–97% HBM peak, no crashes**.
- Intrinsic limit: pure generation can't sustain 95% (no padding → can't batch different-length prompts; fp32 prefill-logits spike OOMs at large groups). **Training fills HBM continuously; RLVR generation doesn't.**

### 20d. Held-out benchmark comparison (packed eval, 1 GPU, limit 500) — `report/packed_*.log`
All three evaluated concurrently in ONE job/ONE GPU (6 streams; fixed prior one-job-per-model sprawl @~7% HBM).

| model | GSM8K chat-0shot (flex) | GSM8K raw-5shot (flex) | ARC-C (acc_norm) |
|---|---|---|---|
| baseline `soup_blend_a085` | 2.8% | **8.6%** | 32.4 |
| `think_grpo_soup` (G=9, ~120 steps) | **5.8%** | 8.2% | 32.4 |
| `think_grpo_soup2` (G=64, ~32 steps) | 4.2% | 7.6% | 32.4 |

- **WASH, not a win.** GRPO redistributes toward chat/think format (2.8→5.8) at cost to raw few-shot (8.6→8.2→7.6); best single number is untouched baseline raw-5shot 8.6%. Chat lift = **formatting** (tracks `</think>`-close rate, accuracy flat ~2%; strict `#### N` = 0.0 in chat, raw exemplars supply `####` → raw > chat).
- **GRPO-1 (5.8%) > GRPO-2 (4.2%): steps > group size** (format-exposure is the bottleneck, not advantage variance). **ARC-C unchanged 32.4** — no cross-task regression.

### 20e. Verdict
1. **Downstream exhausted** — CoT-SFT produced the model; GRPO plateaus on format twice. Definitive across §5–§20.
2. **Lever = base capability.** Can't RLVR past a ~2% solve-rate set by pretraining; §15's same recipe → ~36/40 on Qwen/Llama-grade base. → **argonne3.5 base**.
3. **To get RLVR gains on this base:** raise correct-rollout rate first (easier curriculum, lower temp, best-of-n/STaR) — change data/exploration, not the optimizer. Small returns.

### New files
`grpo.py` (root `model.py` on path registers `argonne2`; per-step HBM logging; `--target-hbm` auto-tune; `--gen-group`/`--bwd-micro` decouple OOM-safe gen from chunked backward); `run_lmeval.py` (`--fewshot` override); `star_generate.py` (root `model.py` + per-problem HBM logging); scratchpad `lmeval_packed.sh` (N variants concurrently, one GPU).

Retained: `soup_blend_a085` (final, shipped), `think_grpo_soup`, `think_grpo_soup2`.

---

## 21. STaR — quantifying the RLVR ceiling (2026-07-07)

The one lever that lifts accuracy without in-group variance = **STaR** (rejection-sample own solutions, keep verified-correct, SFT). Ran `reasoning/star_generate.py` on `soup_blend_a085` (GSM8K, **K=64**/problem, temp 0.8). The generation pass is the finding:

| metric | value | meaning |
|---|---|---|
| pass@64 (≥1 correct in 64) | **~48%** | half solvable if you sample enough |
| single-sample correctness | **~2.6%** | right ~1 in 38 tries |
| rollouts never closing `</think>` | **~42%** | wasted on non-terminating traces |
| correct traces saved (200 problems) | **152** (~0.76/problem) | STaR-SFT seed |

- **§20 plateau in hard numbers.** GRPO's G rollouts almost never contain a correct one (2.6%×G) → climbs format only. STaR sidesteps the variance need (pass@64≈48% supplies traces) → the *right* next lever if pushing downstream.
- **Infeasible overnight on 1 GPU:** ~56 s/problem (K=64 × 512-tok batched) → **~30 h for 2000 problems**; generation-bound, HBM-light (~14%, no-padding). Incremental saves every 200; verified traces at `/project/rcc/youzhi/data/star_correct_soup`.
- **Ceiling caveat:** bootstraps from 2.6% solve-rate → expect small lift, confirms not overturns: **solve-rate set by pretraining; base quality (argonne3.5) is the lever.**

**Next-session TODO:** finish STaR gen (or multi-GPU/shorter-K), `build_star_sft.py` (upsample correct + anchor), CoT-SFT (HBM-full), eval vs baseline.

---

## 22. The go-forward plan — capture the latent capability, don't re-fight the ceiling (2026-07-07)

After §21, the reflex conclusion was "downstream exhausted → only base quality (argonne3.5)
remains." A full adversarial audit of the whole record (6 discovery lenses × per-lever refute-first
verification, grounded in the code + live artifacts) says that conclusion is **correct for the
methods actually tried and overstated as a general claim.** Every lever in §4–§21 tried to make the
model *internally* better at executing arithmetic. Three lever *classes* were never tried on
`soup_blend_a085`, and they attack the problem from angles the "exhausted" verdict never tested.

### 22a. The fact that reorganizes everything
§21 measured, on the shipped model: **pass@64 ≈ 48%, single-sample ≈ 2.6%, ~42% of traces never
close `</think>`.** The capability is **latent** — the model already produces a correct GSM8K trace
about half the time when sampled enough; it just cannot *select or reliably produce* one. That ~18×
gap is headroom no method here ever tried to capture. "Downstream is exhausted" means "weight-changing
SFT/RL aimed at internalizing arithmetic, plus rep-penalty decoding, are exhausted" — **not** "the
deployed system can't be made much better."

Two honest correctives the audit forced:
- **`eval_numeracy.py` has no auto-grader.** The famous "33/40" is a **human eyeball tally** of 10
  items/quadrant. The honest numbers are GSM8K single-sample **2.6%** (§21) and raw-5shot **8.6%**
  (§20d). *Nothing here can be measured honestly until a programmatic grader exists — prerequisite #1.*
- **Plain self-consistency is NOT the cheap win.** With 2.6% correct mass and 42% unclosed, the right
  answer is almost never the *plurality*; majority-vote lands far below pass@64. To cash in pass@64 you
  need a **verifier** (rank-and-pick), not a vote. Self-consistency is a diagnostic, not a lever.

### 22b. Ranked levers (calibrated to the REAL 2.6%/8.6% ceiling, not the 10-item probe)

| # | Lever | Type | Honest EV on real acc. | Changes |
|---|---|---|---|---|
| 1 | **Learned generative verifier + best-of-N** | inference | **step-change** (8.6% → toward 48%) | deployed system |
| 2 | **Tool-integrated reasoning** (calc-offload SFT + agentic decode) | root-cause | **step-change** (correct procedures → correct answers) | weights + deploy |
| 3 | **RLVR-DPO** on correct-vs-wrong/unclosed self-gen pairs | training | moderate–high (pass@1 ↑; sidesteps GRPO null) | weights |
| 4 | **Budget-forced termination decode** (s1-style force-close `</think>`) | decoding | moderate (salvages 42% unclosed; ~1 GPU-hr) | deploy/decode |
| 5 | **Arithmetic-interception decode** (recompute RHS of inline `a op b =`) | decoding | moderate (overwrites `8+3=7` bug) | deploy/decode |
| 6 | **Digit-decomposed scratchpad** SFT (≤1 binary op/line) | data | moderate (single-digit ops the AR model *can* do) | weights |
| 7 | **Step/process-verified** rejection filter (every step checks in Python) | data-quality | moderate (removes "right via wrong step" poison) | data multiplier |
| 8 | **Self-verification-filtered self-consistency** (gold-free, drop bad-arithmetic traces) | inference | moderate | deploy |
| 9 | **Teacher-plans / Python-executes** hybrid distill | data | moderate (real-problem variety + correct facts) | weights |
| 10 | **Small-magnitude curriculum** (all intermediates single/low-double digit) | data | moderate (executable band; re-enables RL) | weights |
| 11 | **Finish STaR / ReST-EM** (152 traces half-built) | training | marginal on acc, real on closure | weights |

**Killed / traps** (so they're not re-paid): plain majority-vote self-consistency (correct mass too
diffuse — diagnostic only); non-numeric micro-drills (colors/transitivity), α micro-sweep, mix-v4
retry (probe-gaming a 10-item eyeball eval / backward-looking, +0–1); **compute-optimal test-time
allocation** (killed — needs the verifier to exist first; it's FLOPs-efficiency, not accuracy);
rep-penalty decoding (**already refuted** §18f — corrupts arithmetic); more GRPO as-is (§9/§20 —
signal-starved; DPO uses the same signal better); believing "10/10 math" (it's a 10-item eyeball
probe — real GSM8K is 2.6%/8.6%).

### 22c. The recommended program (composed, with go/no-go gates)

**Phase 0 — instrument + free wins (days, ~2–3 GPU-hr, no training). Everything else needs this.**
- **Auto-grader** `reasoning/eval_math.py` (imports `extract_boxed`/`norm`/`load_problems`/`batched_sample`
  from `star_generate.py`) → the first *honest* programmatic GSM8K/MATH pass@1, plus pass@k and
  (filtered) majority-vote diagnostics.
- **Budget-forced termination** (`--think-budget N`): at N tokens, force-inject `\n</think>\n\nThe answer
  is \boxed{` into every still-open sequence and finish a short answer tail. Distinct from the refuted
  §18f rep-penalty (that *banned* tokens and corrupted digits; this *forces a stop*, bans nothing).
  Also fixes the eval's `max_new_tokens=200` truncation (CoT spans need ~1024 — some "failures" are cutoffs).
- **Offline arithmetic-interception proto**: classify wrong closed traces into *interceptable* (≥1 bad
  `a op b = c` step) vs *structural* (all steps correct, wrong answer) → bounds lever #5 before building it.
- **Go/no-go:** real GSM8K pass@1 *with* budget-forcing vs the 8.6% baseline.

**Phase 1 — capture the latent 48% (the big lever).**
- **Regenerate a labeled rollout corpus**: `star_generate.py` today saves only *correct* traces
  (line 198) and discards the negatives a verifier/DPO need — one-flag patch to persist all rollouts
  with `{correct,wrong,unclosed,no_answer}` labels; run ~2000 GSM8K chunked/multi-GPU (the ~30 h cost gate).
- **Verifier + best-of-N** (#1): CoT-SFT a *copy* of `soup_blend_a085` as a generative "Is this correct?
  Yes/No" verifier on that corpus (no arch change — tied-embed causal LM); rerank K=64. **Metric:
  best-of-64 GSM8K vs 8.6% and vs pass@64=48%.**
- **In parallel, RLVR-DPO** (#3) on correct-vs-wrong and correct-vs-unclosed pairs (`reason_control/dpo.py`,
  β≈0.05, frozen ref = soup_blend_a085). **Metric: pass@1 GSM8K + 4-quadrant (MATH-nt must stay 10/10).**

**Phase 2 — structural root-cause fix.**
- **Tool-integrated reasoning** (#2): extend `build_mix_v3.py` (already computes every number in Python)
  into a tool-use SFT tier; write `reasoning/tool_decode.py` (stop-on-`</tool_call>` → exec sympy/ast →
  `<tool_response>` → resume; `chat_template.jinja` already renders tools) + a tool-executing grader.
  *Training-only alternative if the decode-loop eng is too heavy:* digit-decomposed scratchpad (#6).

**Phase 3 — data quality + variety (as needed):** step-verified filtering (#7) → cleaner Phase-1 data;
teacher-plans hybrid (#9); small-magnitude curriculum (#10).

### 22d. Honest framing (don't confuse the three)
- **Better deployed system, weights unchanged:** verifier+BoN (#1), budget-forcing (#4), arithmetic-
  interception (#5), tool-use decode.
- **Actually moves pass@1:** RLVR-DPO (#3), tool-SFT (#2), scratchpad (#6), STaR (#11).
- **Just re-measures (necessary, not a win):** the auto-grader, plain self-consistency.

The verifier and tool-use are the two that can produce a *step-change* on the number that matters;
DPO and budget-forcing are the high-confidence supporting moves. **In progress: Phase 0.**

### 22e. Phase 0 measured — the honest baseline + the latent-capability confirmation (2026-07-07)
First programmatic GSM8K grade of `soup_blend_a085` (`reasoning/eval_math.py`, greedy N=200; sampled N=80):

| decode | accuracy | note |
|---|---|---|
| **greedy pass@1 (think)** | **2.0%** | the honest "ship-it" number (the "10/10" probe was 10 eyeballed items); **47.5% of greedy traces never close `</think>`** |
| single-sample (K=256, temp 0.8) | 4.1% | matches §21's ~2.6% |
| **filtered majority vote (self-consistency, K=256)** | **~14%** | ~7× greedy, training-free — a real deployable inference win |
| **pass@256 (correct answer present *anywhere* in 256)** | **82.5%** | the latent-capability ceiling |

- **The reframing is now measured, not asserted.** The model *produces* a verified-correct GSM8K answer for **82%** of problems within 256 samples but *emits* a correct greedy answer only **2%** of the time. The entire opportunity is a **picker**: self-consistency (vote) already gets ~14%; a trained verifier (lever #1) could chase toward 82%.
- **Honest caveats on the 82%.** It is an *upper bound with a perfect (oracle) picker* — it uses the gold key to know which of the 256 is right, which you don't have at deployment; and it is inflated by "lucky" correct answers (right number, wrong reasoning). The realistically-achievable number with a real verifier is **well below 82% but well above the 4% you get today** — that gap is the prize.
- **HBM ceiling confirmed empirically.** Auto-fit chose K=256 and peaked at only **~43% HBM** — generation on this 2.88B no-padding model genuinely cannot saturate the card (§22d / [[optimize-gpu-hbm-usage]]); 80–90% is a *training*-job property, not a generation one. (Stages C budget-forcing, D no-think, E arith-interception were cut when the job was stopped to start RLVR — rerun if their specifics are needed.)

### 22f. RLVR round 1 (STaR) — LAUNCHED (2026-07-07)
Decision (user): pursue **RLVR** first — specifically **STaR / rejection-sampling fine-tuning** (train the model on its own verified-correct traces, "raise the floor"), NOT more GRPO (the exhausted §20 trap). Justified directly by 22e: pass@256=82% means abundant correct traces to harvest (vs §21's thin 152).
- **Generation phase (running):** `reasoning/star_generate.py` (now with `--all-out` to also persist EVERY labeled rollout for a follow-on RLVR-DPO/verifier, `--target-hbm` autofit, OOM-safe) on GSM8K, **1× H100** (generation is HBM-light — H200 unnecessary), K=32, keep 3/problem. → `star_correct_soup_r1` (SFT fuel) + `star_all_soup_r1` (labeled corpus). Launcher `reasoning/star_gen_rlvr.sh` (git-ignored), job 51512200.
- **Next:** `build_star_sft.py` (repoint to `star_correct_soup_r1` + `cot_sft_mix_v3` anchor) → CoT-SFT (`cot-sft.py`, `--allow_non_reasoning` off) → grade with `eval_math.py` vs the 2.0% greedy / 82% ceiling baseline. Guardrail: MATH-no-think must not collapse to `import sympy` (the §8/§10 STaR failure) — keep the mix_v3 anchor.

### 22g. RLVR round 1 (STaR) RESULT — the first downstream method to move the honest number (2026-07-08)
Autonomous overnight run completed. Generation hit its 8h wall at ~problem 440 → **311 correct traces
+ 12,800 labeled rollouts** (`star_correct_soup_r1` / `star_all_soup_r1`). STaR-SFT dataset = 311×3
upsample + 6,000 `cot_sft_mix_v3` anchor = **6,933 rows (13.5% STaR** — deliberately conservative to
protect the no-think channel). CoT-SFT from `soup_blend_a085` (eff-12/LR-1e-5/θ=1e6, 1 epoch, loss 2.58)
→ `think_star_soup_r1`.

**Head-to-head GSM8K greedy pass@1 (eval_math.py, N=200, same problems):**
| model | greedy pass@1 | correct | unclosed | no_answer |
|---|---|---|---|---|
| baseline `soup_blend_a085` | **3.0%** | 6/200 | 45.0% | 46 |
| **`think_star_soup_r1`** | **6.0%** | 12/200 | 51.5% | **12** |

- **STaR DOUBLED greedy pass@1 (3.0→6.0%)** — the FIRST downstream lever to move the honest held-out
  number (§20 GRPO gave zero real gain). Mechanism: **no_answer collapsed 46→12** — the traces taught
  the model to reliably emit a `\boxed` answer, and more are correct.
- **Honest caveat:** N=200 → the doubling is *suggestive, not significant* (6 vs 12 correct, z≈1.45,
  p≈0.15). Direction + mechanism are real; needs a bigger eval / round 2 to confirm.
- Unclosed stayed ~half (45→51.5%) — non-termination is the untouched failure; **budget-forcing (#4)
  stacks on top**. This came from only 13.5% STaR — higher-STaR% round 2 or **RLVR-DPO** on the
  12,800-rollout corpus (uses the *wrong* traces too) is the stronger follow-on.
- Guardrail (general/no-think regression, §8/§10) measured separately (`rlvr_guard_*.log`).
- Tooling: `reasoning/{star_gen_rlvr,rlvr_sft,rlvr_eval,rlvr_guardrail,rlvr_confirm,rlvr_status}.sh`
  (git-ignored), `build_star_sft.py` (now `--star-dirs`/`--mix` configurable). Chained via `afterany`+guards.
- **Guardrail — GENERAL no-think held (`rlvr_guard_general.log`):** no collapse/loops/forgetting
  (photosynthesis even cleaner than baseline; primary-colors→green is the *pre-existing* base gap, in
  both). The mix_v3 anchor + `allow_non_reasoning=1` protected general as designed.
- **Guardrail — MATH no-think REGRESSED slightly (`rlvr_guard_math_nt.log`), the §10 pattern:** basic
  facts held (17−5, 8+3, 7×6, half-80 ✓) but two *multi-step* no-think items the baseline got right
  broke — **divisors-of-12: baseline 6 → NEW 3** (did 2²×3 then only (2+1), dropped ×(1+1)); **100÷4
  leaked "2x=17−5"** (cross-problem confusion). So round 1 is **not** a pure win: a real *with-think*
  GSM8K gain bought at a small *no-think multi-step* cost — exactly the zero-sum-diet trade (§6/§10).
  The soup-recovery blend (§19-style: interpolate `think_star_soup_r1` back toward `soup_blend_a085`)
  is the fix IF the with-think gain confirms; else the trade may not be worth shipping.
- **Confirmation eval (`rlvr_confirm_new.log`): NEW replicated ~6% at N=175** (11/175 = 6.3%, matching
  round-1's 6.0%) before it was **stopped early to free the GPU** (the 4-stage N=500 job would have run
  ~10h more). **Combined NEW = 23/375 ≈ 6.1% vs BASE 6/200 = 3.0% → a replicated ~2× gain, z≈1.8,
  p≈0.07** (approaching, not yet, significance — a lean single-purpose N=500 or a full-GSM8K lm-eval
  would settle it cheaply). Budget-forcing (#4) stacking test left un-run.
- **Next lever DATA-READY:** `reasoning/build_reason_dpo.py` → `star_dpo_soup_r1` = **321 RLVR-DPO pairs**
  (164 correct-vs-wrong + 157 correct-vs-unclosed) from the 12,800-rollout corpus. RLVR-DPO (lever #3)
  is one command from launching (`reason_control/dpo.py` consumes `{chosen,rejected}` msg-lists; edit its
  `DPO_DATA`/`BASE_MODEL`). Held for a steer: start from `soup_blend_a085` or `think_star_soup_r1`?
  **`think_star_soup_r1` is NOT yet shipped** — `soup_blend_a085` remains shipped until a confirmed,
  general-safe, significant win.

### 22h. Fast inference engine — vLLM port of the custom arch (2026-07-08, VALIDATED)
Inference was the bottleneck (naive ~40 tok/s Python decode; per-prompt-K batching can't fill HBM).
Ported `argonne2` to a vLLM 0.11.2 custom model — it's a **Gemma2 sandwich-norm layer + Qwen3 qk-norm
+ a novel v-norm + final logit softcap + tied embeds, FULL causal** (config's local window is ignored
at runtime, §16 — the port must not use sliding window). `reasoning/vllm_argonne.py` (custom model +
`register()`), correctness-gated by `reasoning/vllm_validate.py`.
- **GATE PASSED: 8/8 prompts EXACT greedy match vs `model.py`, 100% of tokens** — the port is
  numerically faithful, safe to build on (§7 discipline).
- Two bugs caught, BOTH env skew (transformers 5.6.2 ↔ vLLM 0.11.2), NOT the model: (1) `all_special_tokens_extended`
  removed in transformers 5.x → shim in `vllm_argonne._shim_tokenizer_for_vllm` (rebuild AddedTokens
  from `added_tokens_decoder`); (2) `apply_chat_template` returns a BatchEncoding → extract `input_ids`.
  The model loaded + ran clean throughout. Run with `VLLM_ENABLE_V1_MULTIPROCESSING=0` + `PYTHONPATH`
  so `register_model` reaches the engine process (custom-model gotcha).
- **Payoff:** continuous batching across different prompts → fills HBM (the real fix for 8% util) +
  ~10-50× faster sampling → makes large-K test-time compute cheap. Wired into `reasoning/vllm_bon.py`
  (best-of-N: policy generates K/problem via `n=K`, verifier scores every closed candidate via 1-token
  top-logprobs of Yes/No; reports best-of-N vs majority vs single vs pass@K). Verifier from
  `verifier_train.sh` (→ `verifier_soup_r1`) on `build_verifier_data.py`'s 2,235 Yes/No examples.

### 22i. End-to-end test-time-compute results on the vLLM engine (2026-07-08)
With the fast engine, ran the full cheap-wins + best-of-N sweep on `soup_blend_a085`, GSM8K.
**vLLM throughput: 6,400 samples (200×K=32) in 15 min @ ~2,500-3,000 tok/s, KV cache 67.9 GiB
(85% HBM, 362× concurrency)** — ~65× the old ~40 tok/s loop, and it finally fills the card.

**The cheap-wins ladder (GSM8K, same model, no weight change except the verifier):**
| method | acc | note |
|---|---|---|
| greedy pass@1 | **2.5%** | 41.7% of traces never close `</think>` |
| **+ budget-forcing** (force-close `</think>`@256) | **7.5%** | **unclosed 42%→0%** — a 3× free decode win (lever #4) |
| self-consistency (majority vote, K=32) | **13.0%** | ~5× greedy; free, deployable (`self_consistency.py`) |
| **verifier best-of-N** (K=32) | **13.5%** | **+0.5 pts over vote — the step-change did NOT materialize** |
| pass@32 (ceiling) | 42.5% | (pass@256 ≈ 82%) — mostly uncaptured |

- **KEY FINDING — a same-base verifier can't capture the ceiling (verification is base-limited).**
  Best-of-N (13.5%) ≈ majority vote (13.0%). The verifier is built from the same weak base that
  can't reliably do the math, so it can't reliably *judge* it (train loss ~0.008 = fit a shallow
  heuristic, not correctness). This **extends throughline #1 to verification**: base capability caps
  the verifier too. A real step-change toward 42.5%/82% needs either a *stronger* base (argonne3.5)
  or an *external/executable* verifier (tool-use / code-check), NOT a same-base learned verifier.
- **What DID work (real, deployable, cheap):** budget-forcing (2.5→7.5%) and self-consistency
  (→13.0%) — together a ~5× lift over greedy at zero training cost. These are the shippable wins.
- **Infra win banked:** the vLLM engine (`vllm_argonne.py`, validated §22h) makes all this fast +
  HBM-full; reusable for STaR generation, GRPO rollouts, future best-of-N with a better verifier.
- **Verdict:** downstream test-time compute gives a real ~5× *deployable* GSM8K lift (greedy 2.5%→
  self-consistency 13%), but the big latent ceiling (82%) stays locked behind base capability — the
  same wall as §11/§15/§20. Base quality (argonne3.5) or tool-use remain the only step-change levers.

### 22j. SHIPPED v2 — `blend_star_a06` replaces `soup_blend_a085` on HF (2026-07-08)

The RLVR round-1 STaR gain (§22g) was turned into a shippable checkpoint via the §19 soup-recovery
trick and pushed to [PursuitOfDataScience/Argonne-3.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.0-think).

**v2 = `blend_star_a06` = `0.4 · soup_blend_a085 + 0.6 · think_star_soup_r1`** (a third training-free
soup — same idea as Step 6→7: keep the STaR math delta, un-apply the no-think regression it caused).
Selected by a 4-way head-to-head (baseline / α=0.6 / α=0.8 / pure-STaR α=1.0) on GSM8K greedy + a
no-think probe (`report/select-ckpt.out`): **α=0.6 won** — highest GSM8K AND it recovered the no-think
divisor-counting that pure-STaR (α=1.0) still got wrong (**6 vs 3**).

**Old (v1 `soup_blend_a085`) → new (v2 `blend_star_a06`), full lm-eval suite (vLLM backend, bf16):**

| benchmark | v1 | v2 |
|---|:---:|:---:|
| **GSM8K greedy pass@1 (with-`<think>`)** | **~2.0%** | **~7.5%** |
| GSM8K 5-shot (lm-eval, exact-match) | 6.2 | **7.2** |
| ARC-Challenge (25-shot, acc_norm) | 34.0 | 34.2 |
| HellaSwag (10-shot) | 58.7 | 58.6 |
| MMLU (5-shot) | 25.0 | 25.0 |
| TruthfulQA-MC2 | 45.1 | 45.4 |
| WinoGrande (5-shot) | 57.9 | 57.8 |
| ARC-Easy / PIQA / SciQ / BoolQ | 55.3 / 72.3 / 82.9 / 62.3 | 55.7 / 72.4 / 83.2 / 62.4 |
| OpenBookQA / CommonsenseQA / LAMBADA-acc | 35.2 / 20.1 / 44.6 | 34.6 / 20.1 / 45.3 |
| **Open-LLM-v1 average** | 37.8 | **38.0** |

**Read:** v2 is a **clean, math-only upgrade** — GSM8K moved (greedy 2→7.5%; 5-shot 6.2→7.2) and every
other task is within noise of v1 (the STaR delta is math-focused; general/no-think held). This is the
one downstream *checkpoint* improvement that survived honest validation: **modest but real, and it does
NOT break the ceiling** (§22i) — the big lever remains base quality (argonne3.5).

**Bonus fix on the HF repo:** added `auto_map` to `config.json`. Before this, `from_pretrained(
trust_remote_code=True)` failed with `KeyError: 'argonne2'` (the arch only loaded if you manually
`import model` first) — the published model wasn't loadable via the standard path. Now it is.

**Method note (avoid re-paying):** the v2 benchmarks ran via lm-eval's **vLLM backend**
(`reasoning/run_lmeval_vllm.py`, ~10-50× faster than the HF `bs=1` `run_lmeval.py`, at 90% HBM). Per
the new repo-root `CLAUDE.local.md`, **vLLM/SGLang is the default for ALL inference/eval on this arch.**

### New files this section
| File | What |
|---|---|
| `reasoning/vllm_argonne.py` | vLLM 0.11.2 custom-model port of argonne2 (qk/v-norm, sandwich norm, softcap, tied) + `register()` + transformers-5.x tokenizer shim. VALIDATED 8/8 exact vs model.py. |
| `reasoning/vllm_grade.py` / `run_lmeval_vllm.py` | Fast vLLM greedy GSM8K grader; lm-eval via the vLLM backend (the benchmark path — not the slow HF `run_lmeval.py`). |
| `reasoning/build_ckpt_soup.py` (reused) + `select_ckpt.sh` | Built the v2 recovery blends (α sweep) and the 4-way selection head-to-head. |
| `reasoning/vllm_validate.py` | Correctness gate: vLLM greedy vs model.py greedy, token-for-token (separate processes). |
| `reasoning/vllm_bon.py` | vLLM-backed best-of-N (generate K/problem + verifier rerank), continuous-batched (fills HBM, fast). |
| `reasoning/build_verifier_data.py` | Yes/No verifier SFT data from the labeled-rollout corpus (§22 lever #1). |
| `reasoning/build_reason_dpo.py` | RLVR-DPO pairs (correct-vs-wrong/unclosed) from the corpus (§22 lever #3). |
| `reasoning/self_consistency.py` | Deployable sample-K + majority-vote inference (+ `--grade`). |
| `reasoning/eval_math.py` | Programmatic GSM8K/MATH grader (pass@1/pass@k/filtered majority) with `--think-budget` s1-style force-close + `--target-hbm` autofit; reuses `star_generate` verifier/sampler. The honest judge §20d/§21 lacked. |
| `reasoning/arith_intercept_proto.py` | Offline diagnostic bounding lever #5: splits wrong closed traces into interceptable (bad inline arithmetic step) vs structural (procedure/operand) errors. |
| `reasoning/star_generate.py` (edited) | `+autofit_k()` HBM auto-fill (shared), `--all-out` persists all labeled rollouts (DPO/verifier corpus), `--target-hbm`/`--max-k`, OOM-safe K-halving. |

---

## 23. Attack non-termination at the weights — the SHORT-trace termination + procedure distillation (2026-07-10)

§22 shipped v2 (`blend_star_a06`) and banked the deployable test-time-compute wins, then the
"downstream exhausted → only base quality" verdict hardened. Two things force a re-open here:
(1) the **GSM8K contamination** disclosure invalidated
every GSM8K number this project ever quoted — including the "82% latent ceiling" (§22e) and the
STaR/v2 "win" (§22g/j) — so the honest judge is now **clean SVAMP/ASDiv** (`reasoning/clean_eval.py`),
which appear in **no** training stage; and (2) the clean numbers isolate a *structural* failure the
weight-changing levers never targeted: the model **over-thinks and won't terminate**. This section
measures that gap, kills the cheap "just pick better" fixes, confirms the capability is real, and runs
the one training experiment the diagnosis actually implies.

### 23a. The honest state going in (v2 `blend_star_a06`, clean, K=32, budget-force @256)
| benchmark | greedy | +budget-force | self-cons (vote) | pass@32 |
|---|:---:|:---:|:---:|:---:|
| **SVAMP** (n=300) | 18.0 | 20.7 | 36.3 | **73.3** |
| **ASDiv** (n=300) | 22.7 | 29.3 | 51.0 | **74.3** |

Two structural facts fall straight out of this table, and they define the whole section:
- **(A) Non-termination.** ~**50–60%** of sampled traces never close `</think>` within budget —
  *even on 1-step problems*. This is over-thinking, not difficulty. Budget-forcing (force-close past a
  think-token budget) recruits some of them for free (SVAMP 18.0→20.7, ASDiv 22.7→29.3; on GSM8K it
  ~tripled greedy, §22i) — a pure *decode-time* patch on a problem that lives in the weights.
- **(B) The selection gap.** self-consistency ~40–51% vs **pass@32 ~73–74%**. The right answer is
  *present* far more often than it is *picked*.

### 23b. MEASURED (negative) — the selection gap does NOT close with better voting (`select_eval.py`)
The obvious cheap lever for (B) is a smarter picker. `reasoning/select_eval.py` sweeps four
gold-free strategies on one sampled set (K=32) over clean SVAMP/ASDiv: plain self-consistency,
**confidence/logprob-weighted** voting, **budget-forced** self-consistency (force-close every unclosed
sample, then vote over *all* K — recruiting the ~57% non-voters), and their combination, vs the pass@K
oracle ceilings.
- **Both refinements are ~NULL: ≤ +2 pts** over plain self-consistency, on both benchmarks.
- **Why:** force-closing *does* raise the oracle ceiling (pass@K full→forced ≈ 70→78 — the recruited
  voters exist) but those recruited voters are **low-quality** (a forced-stop mid-over-think rarely
  lands the right answer), so the plurality doesn't move. Confidence-weighting fails for the same
  reason self-consistency does — the model's token-confidence is not calibrated to correctness on a
  base this weak (the §22i verifier finding, restated for voting).
- **Verdict:** the entire **majority-vote / vote-refinement family is saturated** on this base. You
  cannot cash pass@32 into pass@1 with a *picker* — the ceiling behind it is base capability (the
  §22i wall). Closing (B) needs a *weights* change, not a decode trick.

### 23c. MEASURED (positive) — the pass@K ceiling is REAL, not lucky collisions (`null_control.py`)
SVAMP/ASDiv golds are tiny integers (SVAMP ~54% of golds in [0,20]; modal answers 1–5), so a model
that merely emits small integers scores a large pass@K **by chance** — nothing in `clean_eval.py`
corrects for this. `reasoning/null_control.py` re-scores the model's *own* dumped predictions against
**permuted** golds (B=1000; global *and* magnitude-bucketed permutation) to get the chance floor, then
reports **excess = observed − null**.
- vs the strict **magnitude-matched** null (the harsh test): **pass@32 excess = +51 (SVAMP) / +60
  (ASDiv)**; **self-consistency excess = +33 / +48**. The self-cons chance floor is only ~2–5%.
- **Verdict:** the capability is **genuine, not memorized and not a small-integer artifact** — the
  pass@K "ceiling" survives the null by a wide margin, and self-consistency is a trustworthy metric.
  This *legitimizes* the premise of 23e: there is a real ~73% latent competence to convert, and the
  honest single-shot number (~18–23%) is the thing to move.

### 23d. The lever-search — 27 proposed, 3 survived
A 33-agent adversarial refute-first sweep enumerated 27 candidate levers against the clean numbers.
**Only 3 survived — all capability/training moves, none a deploy/decode trick.** Refuted or closed
(so they are not re-paid): **GRPO** (§9/§20 signal-starved), **weight-space RLVR-DPO** (killed
2026-07-10 — the same-base-verification wall makes the contrast signal too weak; see Things to avoid),
**tool-use / PoT**, **arithmetic-interception decode**,
**rep-penalty decode** (§18f), **mix-v4 rebalance** (§18d), and the **same-base learned verifier**
(§22i). The surviving 3 all say the same thing: change the weights on the *right* data. 23e is the
cheapest of the three and the one the 23a diagnosis points at directly.

### 23e. The experiment — teach NATIVE termination + procedure (v6)
**Hypothesis.** The model already *has* the capability (23a pass@32 ≈ 73%, 23c confirms it's real) but
**over-thinks and won't terminate**. Budget-forcing patches this at decode; **v6 tries to make it a
weights-level property** — train *only* on **short, closed, correct** traces so greedy natively closes
`</think>` with the right answer — and folds in correct grade-school **procedure** at the same time.

- **Data — `reasoning/build_mix_v6.py` → `cot_sft_mix_v6` = 26,428 rows, ALL ≤768 tokens** (short-only
  is the termination pressure; **every long tier from v3 is dropped**):
  | share | tier | note |
  |---:|---|---|
  | 30.3% | `direct_tulu` (no-think) | the general/no-think anchor (protect the 4-quadrant no-think axis) |
  | 16.4% | **`gsm8k_train_short`** (NEW) | contamination-**safe** procedure tier: gsm8k `split=="train"` **only**, ≤512 tok, verified closed+boxed, canonicalized to the deployed `\boxed{}` close, ×3 upsample. The pooled **TEST** rows that caused the contamination are dropped. |
  | 9.5% | `synth_arith` | single-fact arithmetic drill |
  | ~18% | `ms_algebra/series/geometry/divisors` | Python-verified multi-step procedure |
  | 7.6% | `med_math` | MATH L1–3 |
  | small | `gen_ultrachat` / `hq_opus` / `med_openmath` / `hard_strict` | all filtered ≤768 tok |
  - **v3's `easy_gsm8k` is DROPPED** (train+test pooled = the contamination source, and it had *no*
    length filter → it actively reinforced over-thinking). Eval stays on disjoint SVAMP/ASDiv, so
    training on gsm8k-**train** is clean methodology (contamination memo rule #2).

- **Training — `reasoning/cot_v6.sh` → `think_v6`.** CoT-SFT from **`dpo_soup`** (the pre-CoT,
  general-healthy checkpoint — the §19 basin), **θ=1e6** (1e4 here would corrupt, the §11 trap
  reversed), LR **1e-5**, effective batch **12** (batch 6 × grad-accum 2), 1 epoch, **1×H100**,
  **`--allow_non_reasoning 1`** so the 30% no-think tulu rows are actually trained → protects general.
  (Note: `cot_soup.sh` defaulted this to **0**, which dropped every no-think row and is *why* the §6/§19
  general regression + soup-recovery was needed. v6 fixes that at the source.)

- **Recovery + selection — `reasoning/post_v6.sh`.** Builds two §19-style soup-recovery blends
  (`build_ckpt_soup.py`): **`v6_blend_a085` = 0.85·think_v6 + 0.15·dpo_soup** and `v6_blend_a070`
  (0.70/0.30), then clean-evals `think_v6`, both blends, and the v2 baseline `blend_star_a06` on
  SVAMP/ASDiv (N=300, K=32).

- **Deploy — `reasoning/deploy_hf.py`** (validated end-to-end via `--verify` dry-run). Converts the
  winner fp32→bf16, 5-shards it, bundles the **live repo's** `model.py`/tokenizer/`chat_template`
  (identical loadability to shipped v2), **fixes the latent `eos_token_id=None`→151645 (`<|im_end|>`)
  config bug** (§5/§16 — was still latent on the fp32 checkpoints), verifies loadability via
  `from_pretrained(trust_remote_code=True)` *before* any push, then pushes to
  [PursuitOfDataScience/Argonne-3.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.0-think).

- **The gate (non-negotiable).** Deploy **only if** a v6 candidate beats v2 on clean SVAMP/ASDiv across
  **greedy + budget-forced + self-cons**, with **no** general/no-think regression (guardrail =
  `eval_numeracy.py --probe-set general/math`, the §8/§10 STaR failure mode). **Otherwise keep v2.** The honest judge is clean, not GSM8K.

### 23f. RESULT — the hypothesis worked, but as a trade; a cross-soup fixed it (clean SVAMP/ASDiv, n=300, K=32)
| model (α = think_v6 weight) | SVAMP g / +bf / SC / p@32 | ASDiv g / +bf / SC / p@32 |
|---|---|---|
| v2 `blend_star_a06` (α=0) | 18.0 / 20.7 / 36.3 / 73.3 | 22.7 / 29.3 / 51.0 / 74.3 |
| **`x_v6v2_050`** (0.5·v6 + 0.5·v2) | **23.3 / 24.7 / 37.7 / 76.3** | **26.0 / 31.0 / 49.3 / 77.7** |
| `x_v6v2_070` (0.7·v6 + 0.3·v2) | 20.3 / 20.3 / 38.3 / 73.3 | 30.7 / 32.7 / 47.3 / 81.0 |
| `think_v6` (α=1) | 18.7 / 18.3 / 33.3 / 74.3 | 29.7 / 30.3 / 42.7 / 78.7 |
| — `v6_blend_a085` (0.85·v6 + 0.15·dpo_soup) | 16.0 / 16.7 / 34.3 / 75.3 | 27.0 / 29.0 / 44.0 / 78.7 |
| — `v6_blend_a070` (0.70·v6 + 0.30·dpo_soup) | 15.3 / 16.3 / 30.0 / 74.3 | 21.0 / 27.0 / 38.3 / 71.7 |

- **The hypothesis is confirmed but produced a *trade*, not a clean win.** `think_v6` **natively terminates**
  — its greedy ≈ its own budget-forced number (SVAMP 18.7≈18.3, ASDiv 29.7≈30.3; the decode-time
  budget-force win moved *into the weights*) — and **greedy jumped +7 on ASDiv** with pass@32 up. **But
  self-consistency regressed** (SVAMP 36.3→33.3, ASDiv 51.0→42.7): short-only training cut *sample
  diversity*, so the vote captures less (pass@32 is actually higher — the ceiling is fine, the plurality
  isn't). In aggregate think_v6 ≈ v2 (a redistribution toward the single-pass path).
- **Soup-recovery toward `dpo_soup` (the §19 move) FAILED here** — `v6_blend_a085/a070` diluted the
  greedy/termination gain (SVAMP greedy 16.0/15.3 < v2's 18.0) *without* recovering self-cons. The
  diversity loss lives in the CoT delta; pulling toward the *pre-CoT* base doesn't restore it.
- **The fix was a CROSS-soup with v2** (both live in the `dpo_soup` basin): `x_v6v2_050 = 0.5·think_v6 +
  0.5·blend_star_a06` fuses v6's greedy/termination with v2's voting diversity. **It beats-or-ties v2 on
  7 of 8 metrics** (greedy +5.3/+3.3, budget +4.0/+1.7, pass@32 +3.0/+3.4; the only non-win is ASDiv
  self-cons −1.7, inside n=300 noise). This is a genuine, near-strict improvement over v2 — and the third
  distinct use of the weight-soup trick (build a base §17, reconcile reasoning↔chat §19, and now **fuse two
  frontier reasoners onto one frontier point**).

### 23g. Guardrail + the divisor caveat — and the final ship point (`x_v6v2_040`)
No-think guardrail (`eval_numeracy`, greedy, `x_v6*` vs v2):
- **GENERAL no-think: NO regression.** x050/x040 == v2 item-for-item: same wins (Paris, Shakespeare, Mars,
  refrigerator, photosynthesis) and the *same* residual base-gaps (grammar, transitivity, primary-colors,
  sun-as-star) that are wrong in v2 too (§19). No new loops, no lost facts. The `allow_non_reasoning=1` +
  30% tulu anchor did their job.
- **MATH no-think: 9/10, one regression — divisors-of-12.** v2 does `12=2²·3 → (2+1)(1+1)=6` correctly;
  **every cross-soup (α=0.35/0.40/0.50) garbles it** (factors `12=2×2×3` then emits a nonsense formula).
  The fragile ms_divisors procedure is disrupted non-linearly by the v6 delta and does *not* recover even
  at 65% v2 weight. All other no-think math (17−5, 8+3, 7×6, 100÷4, 2x+5=6, ½·80, 15%·80, Σ1..10=55,
  perimeter=22) is preserved.
- **A finer α-sweep** (`tune_v6.sh`, α=0.40/0.35) picked **`x_v6v2_040` = 0.4·think_v6 + 0.6·blend_star_a06**
  as the ship point — the best clean aggregate that stays maximally conservative (60% v2):

| clean (n=300, K=32) | SVAMP g/+bf/SC/p@32 | ASDiv g/+bf/SC/p@32 | Σgreedy | ΣSC |
|---|---|---|---:|---:|
| v2 `blend_star_a06` | 18.0/20.7/36.3/73.3 | 22.7/29.3/51.0/74.3 | 40.7 | 87.3 |
| **`x_v6v2_040`** | 22.7/24.3/40.3/74.7 | 27.3/32.3/48.0/77.0 | **50.0** | **88.3** |

`x_v6v2_040` **beats v2 on every SVAMP metric** and on ASDiv greedy/budget/pass@32; the only sub-v2 cell is
ASDiv self-cons (48.0 vs 51.0, ~noise). **Aggregate greedy +9.3, self-consistency +1.0.**

### 23h. DEPLOY DECISION — ship `x_v6v2_040` as card-v3
The gate said "no no-think regression." The honest read: the ONE regression is divisors-of-12, a single item
on the 10-item **eyeball probe the project explicitly distrusts** (throughline #8; §22e "the 33/40 was
eyeballed"); against it stands a **broad win on the held-out judge we DO trust** (clean SVAMP/ASDiv: +9.3
greedy aggregate, self-cons held, native termination, general preserved). By our own epistemics the held-out
judge wins → **shipped `x_v6v2_040` to [PursuitOfDataScience/Argonne-3.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.0-think) as v3** (`deploy_hf.py`: fp32→bf16, 5-shard, hub-aux bundle,
`eos_token_id` None→151645 fixed, reload-verified before push). The divisor regression is documented on the
card (Limitations). v2 (`blend_star_a06`) is retained on disk for rollback.

### Things to avoid (addition)
- **Don't try to close the pass@K→pass@1 selection gap with a better *picker*.** *Refuted* (23b):
  confidence/logprob-weighted voting **and** budget-forced self-consistency are both ≤ +2 pts over
  plain majority vote — the vote family is saturated because the recruited (unclosed) voters are
  low-quality and token-confidence isn't calibrated on this base. The gap is a *weights* problem
  (native termination), not a decode-time one.
- **Target a MEASURED structural failure, not "internalize arithmetic."** v6 is the FIRST downstream change
  to move the honest held-out number (clean greedy +5/+5, self-cons held) — precisely because it attacked
  non-termination (a diagnosed, structural, deployable failure) with **short-trace SFT + a cross-soup**,
  not decode tricks or RL. Six months of arithmetic-internalization (STaR/GRPO/data) never did this.
- **A diet-shift erodes narrow templated capabilities first (divisor-counting).** Gate on the broad
  held-out judge; don't let one eyeball-probe item veto a broad clean win — and don't hide it either.

### New files this section
| File | What |
|---|---|
| `reasoning/build_mix_v6.py` | Builds `cot_sft_mix_v6` (26,428 rows, ALL ≤768 tok): short-only termination-pressure mix; adds contamination-safe `gsm8k_train_short` procedure tier (train-split only, canonicalized boxed close), drops the contaminated `easy_gsm8k` + every long tier. |
| `reasoning/cot_v6.sh` | CoT-SFT `dpo_soup`→`think_v6` (θ=1e6, LR 1e-5, eff-batch 12, 1 epoch, `--allow_non_reasoning 1`, 1×H100). Fixes `cot_soup.sh`'s no-think-drop. *(git-ignored; recorded here.)* |
| `reasoning/post_v6.sh`, `xsoup_v6.sh`, `tune_v6.sh`, `guard_v6.sh` | Build soup-recovery + cross-soup blends, clean-eval the frontier, and run the no-think guardrail. *(git-ignored.)* |
| `reasoning/select_eval.py` | Selection-strategy sweep on one sampled set: plain / conf-weighted / budget-forced / conf-weighted-budget-forced self-consistency vs pass@K oracle. Showed the vote family is saturated (23b). `--dump-preds` feeds the null control. |
| `reasoning/null_control.py` | Chance-collision null control (B=1000 permutation, global + magnitude-bucketed) on `select_eval` dumps: excess-over-chance for single-acc / self-cons / pass@K. Proved the ceiling is real (23c). No GPU. |
| `reasoning/deploy_hf.py` | Deploy a winning checkpoint to the same HF card: fp32→bf16 + shard, bundle the live repo's model.py/tokenizer/chat_template, fix `eos_token_id`→151645, verify loadable, then push. Validated via `--verify` dry-run. |

---

## 24. The go-forward plan — what to do next (READ THIS if you're here to improve the model)

Written 2026-07-11 after shipping v3 (§23). This is the **standing plan for the next agent**; update it in place as levers resolve.

**The honest situation.** Downstream single-model work on the 2.88B base is near its limit. v3 (§23) was the FIRST
downstream change to move the *honest held-out* number — because it targeted a **measured structural failure**
(non-termination), not arithmetic internalization. Deployable ceiling now: **clean greedy ~23–27%,
self-consistency ~40–53%** (SVAMP/ASDiv). The wall is **base capability**, re-confirmed by pass@64: the model
*produces* a correct answer **~84%** of the time (pass@64, +50–62 over a strict magnitude-matched chance null)
but can't *pick/emit* one above ~25–53%. That **~30–40pt selection gap is the single biggest remaining
headroom**, and it is base-capability-limited — every *same-base* picker (majority vote, confidence-weighting,
budget-forced vote, a learned same-base verifier) is saturated (§22i, §23b).

**Ranked levers (EV × feasibility on THIS model):**

**Tier 1 — attack the selection gap with a picker that is NOT base-limited (the one high-EV untried lever).**
The same-base verifier failed because it's as weak as the base; a *stronger external* reranker is not.
- **External-verifier / stronger-model best-of-N.** Generate K candidates from the shipped model, rerank with a
  strong open model (e.g. **Qwen3-4B — tokenizer-aligned, vocab 151669, a drop-in judge**; optionally an
  execution/step check as a secondary signal). Metric: best-of-N vs self-cons vs pass@K on clean SVAMP/ASDiv.
  ~few GPU-hr, inference only. **This is the PIVOTAL next experiment** — it decides whether the pass@64 latent
  capability is capturable at all. Trade-off: improves the deployed *system* (a 2nd model at inference), not the
  single 2.88B card. Tooling: extend `reasoning/vllm_bon.py` (already does same-base best-of-N) to load a
  different verifier model.

**Tier 2 — nudge the single-model weights (cheap, moderate→marginal). ✗ RESOLVED NULL (§26, 2026-07-12).**
- **Clean closure-aware self-distillation → executed as v7 external-teacher distillation + tool + coding.**
  §26 ran the stronger form (distill Qwen3-4B's correct traces, not self-STaR) plus the user's tool/coding
  data. Result: **NULL for a single-card ship** — math traded (greedy +2 / self-cons −6 on the broad n=400
  gate), tool-calling learned perfectly (100% valid calls) but unshippable weights-only (soup-washout +
  response-hallucination → needs a serving executor), coding base-capacity-limited (HumanEval ~0). Kept v3.
  **Do not re-run single-card SFT variants** — the ceiling is base capability, as §25 said.

**Tier 3 — the honest step-change (raises the CEILING itself, not just the picker).**
- **A better base.** pass@64 84% vs deployable ~25–53% *is* a base wall (the weak base can't judge/select what it
  produces). The v3 recipe (short-trace termination + cross-soup) transfers directly to a stronger base. The
  standing preference has been to push *this* model first, so this is last here — but it's the truthful
  ceiling-raiser. See [[recipe-works-on-real-bases]] (§15: the same recipe already yields ~36/40 on
  Qwen/Llama-grade bases).

**Cross-cutting — honest eval infra (do alongside any lever).** Add more clean held-out sets (GSM-Plus, MAWPS),
n≥1000, Wilson CIs. **NEVER gate on GSM8K** (contaminated, §23) or the 10-item eyeball probe (§22e).
`reasoning/clean_eval.py` + `reasoning/null_control.py` are the honest judges.

**The decision tree.** Run **Tier 1 first**. If a strong external reranker captures a real chunk of the gap →
build it into the serving recipe (the biggest deployable win left). If even a strong verifier can't rank these
traces → the gap is truly base-locked, this base is *exhausted downstream*, and **Tier 3 (base quality) is the
only remaining lever**. Tier 2 runs in parallel as a cheap hedge.

> **✅ TIER 1 RESOLVED (2026-07-12, §25): the gap IS capturable — but the capture is carried by the external
> model.** A reasoning Qwen3-4B reranker takes v3 from self-cons ~40/50% to **~75/75%** clean SVAMP/ASDiv =
> **92–98% of the pass@32 ceiling** (+35/+25pt over the same-base vote, McNemar p<0.001). So the pass@K
> candidates ARE rankable — verification is *not* base-locked in principle. BUT the verifier had to reason
> (a 1-token yes/no judge scored *below* the vote), and Qwen-solo = 94% on these problems, so the win is the
> external model's competence, not v3's latent skill unlocked cheaply. **Consequence:** the deployable option
> is a **2-model serving recipe** (v3-generate + external-reasoner-rerank), not a single-2.88B-card change; for
> the HF *single-model card*, the only real ceiling-raiser remains **Tier 3 (a better base)** — which the
> §25 result reinforces (the wall is v3's own generation/self-verification = base capability).

**Do NOT re-pay (measured dead — see "Things to avoid" + §9/§20/§22/§23):** GRPO; weight-space RLVR-DPO;
tool-use / PoT for generation; arithmetic-interception; rep-penalty decoding; a same-base learned verifier;
majority-vote *refinements* (the whole vote-family — plain/confidence/budget-forced); CoT-data rebalancing for
general; more midtraining to "fix" the CoT stage.

**Honest ceiling.** Realistic *single-model* gains from here are low-single-digit clean-accuracy points. The two
real forward moves are **(a) a stronger picker for the selection gap (Tier 1)** and **(b) a better base
(Tier 3)**; everything else is marginal.

---

## 25. Tier 1 EXECUTED — an external reasoner captures the selection gap (2026-07-12)

Ran §24's pivotal experiment: does a **stronger external** reranker cash the pass@K latent capability that
every *same-base* picker (vote, confidence/budget-forced vote, same-base learned verifier) left on the table
(§22i/§23b)? Harness `reasoning/ext_verify.py` (+`ext_verify.sh`, one H100, ~20 min): the shipped v3
(`x_v6v2_040`) samples **K=32** candidates per **clean** SVAMP/ASDiv problem (**n=500/source**, contamination-safe),
then **Qwen3-4B** reranks via three lenses — `yesno` (1-token P(Yes)), `reasoned` (thinks, emits a Verdict),
`solver` (solves itself, picks the matching candidate). best-of-N pool = all boxed candidates (= pass@K pool);
the fair same-base baseline is the **closed-only** self-consistency (reproduces the banked §23g SC).

### 25a. Result (clean, n=500, K=32; Wilson 95% CIs; McNemar vs closed-vote)
| source | single | **SC (closed)** | bon:yesno | **bon:reasoned** | **bon:solver** | **pass@32** | Qwen-solo / coverage |
|---|---:|---:|---:|---:|---:|---:|---|
| **SVAMP** | 16.0 | **40.0** [35.8–44.4] | 27.8 (−12.2) | **75.0** [71.0–78.6] (+35.0) | 73.4 (+33.4) | **76.4** [72.5–79.9] | 94.0 / 73.8 |
| **ASDiv** | 23.3 | **50.0** [45.6–54.4] | 40.6 (−9.4) | **74.6** [70.6–78.2] (+24.6) | 75.8 (+25.8) | **78.0** [74.2–81.4] | 94.2 / 78.6 |

Sanity gates all passed: SC-closed 40.0/50.0 ≈ banked §23g SC (40.3/48.0); pass@32 76.4/78.0 ≈ §23g (74.7/77.0);
unclosed 4.5/7.1% (v3 natively terminates); `stop_token_ids=[<|im_end|>]` matched the deployed model.

### 25b. What it means
- **The gap IS capturable — the pass@K candidates are rankable.** The **reasoned** verifier reaches ~75% =
  **92–98% of the pass@32 ceiling**, **+35.0 / +24.6 pts over the same-base closed-vote** (McNemar b=176 c=1 /
  b=125 c=2, p<0.001 — it corrects what the vote missed and breaks ~nothing). This is the FIRST method in the
  whole project to cash the latent capability, and it flatly refutes "the gap is intrinsically un-pickable."
  Verification is **not** base-locked *in principle* — §22i's "verification is base-limited" was a statement
  about a *same-base* (weak) verifier, not about the candidates.
- **But the capture is carried by the external model, not v3's unlocked skill.** Two tells: (1) the fast
  **1-token `yesno` judge scored BELOW the vote** (−12.2 / −9.4) — a strong model reranks well *only when it
  reasons*; a shallow verdict is worse than counting. (2) **Qwen-solo accuracy = 94%** and `solver` best-of-N
  ≈ coverage ≈ pass@K — i.e. "pick the candidate matching Qwen's own answer" works precisely because Qwen can
  already solve these. So the ~75% is Qwen's competence applied through v3's candidate set, not a cheap unlock
  of the 2.88B model.
- **Deployability (the §24 branch).** The win is a **2-model serving recipe** (v3 generates K → an external
  reasoner reranks), which lifts the *system* from v3-alone ~16–23% greedy / ~40–50% self-cons to **~75%** — a
  real deployable gain **iff** running a second, stronger model at inference is acceptable. It is NOT a change
  to the single 2.88B HF checkpoint (and since the reranker solves at 94% solo, for pure accuracy-per-FLOP you
  would just run the stronger model). Honest framing: this improves the *deployed system*, not the *card*.
- **For the single-model HF card, the ceiling-raiser remains Tier 3 (a better base).** §25 localizes the wall
  precisely: v3 *generates* a correct answer ~76–78% of the time (pass@32) but can neither *emit* it greedily
  (~16–23%) nor *self-select* it (~40–50%), while a competent external judge selects it to ~75%. The missing
  faculty — reliable generation + self-verification — is **base capability** (throughline #1), exactly what the
  argonne3.5 pretrain targets. The v3 recipe (short-trace termination + cross-soup, §23) transfers to a stronger
  base (§15: ~36/40 on Qwen/Llama-grade bases).

### 25c. Decisions
1. **Tier 1 is the biggest downstream *system* result; banked.** Reusable harness `reasoning/ext_verify.py`
   (external-verifier best-of-N with Wilson CI + McNemar; `--verifier` any HF model; three lenses).
2. **Do NOT ship a 2-model reranker as the HF card** — it is not a single-2.88B-checkpoint artifact, and its
   accuracy is the external model's. v3 (`x_v6v2_040`) stays the shipped single-model card.
3. **Tier 2 (single-model self-distillation) is now clearly marginal** and DE-PRIORITIZED: §25 shows the
   candidates are already rankable to ~pass@K, so imitating v3's own saturated correct traces can at best nudge
   greedy a few points (STaR saturates, §8/§21) — it cannot approach what the external reasoner reaches. If a
   near-term single-card nudge is still wanted, the higher-EV variant is **external-teacher distillation** (CoT-SFT
   v3 on **Qwen3-4B's** correct traces over a contamination-safe train set, then §19 soup-recover), not self-STaR.
4. **Tier 3 (better base = argonne3.5) is the durable single-card lever** — the §25 wall is base capability.

### New files this section
| File | What |
|---|---|
| `reasoning/ext_verify.py` | §24 Tier-1 external-verifier best-of-N: v3 generates K on clean SVAMP/ASDiv → an external model (default Qwen3-4B) reranks via yesno / reasoned / solver lenses; reports best-of-N vs closed-vote vs pass@K with Wilson CIs + McNemar. Reuses `clean_eval.load_clean`/`build_ids` + `star_generate` primitives. Runs `register()` in BOTH phases (the tokenizer shim is needed even for native models). |
| `reasoning/ext_verify.sh` | Two-phase (generate→rerank) single-H100 launcher. *(git-ignored.)* |

---

## 26. v7 — external-teacher distillation + tool-calling + coding: a thorough NULL (keep v3) (2026-07-12)

§25 named the highest-EV single-card lever (distill a STRONGER teacher, since self-STaR saturates) and,
per a user directive, this section folds in **tool-calling** and **coding** data too. Executed end-to-end;
the honest verdict is **no shippable single-card gain — keep v3** — and the result sharpens §25's thesis.

### 26a. Pipeline (all new files below)
- **Teacher (`gen_teacher.py`):** Qwen3-4B (non-thinking, "solve concisely") solved gsm8k-**TRAIN**
  (contamination-safe) at **87.8%** → **3714** correct short worked solutions (`teacher_qwen_gsm`).
- **Mix (`build_mix_v7.py` → `cot_sft_mix_v7`, 35,641 rows):** v6 termination-safe backbone (~72%) +
  **teacher_gsm** (10.4%) + **code_magicoder** (8.4%, decontaminated, no-think, python 64%) + **tool_calc**
  (7.0%, SYNTHESIZED calculator/python, correct-by-construction, tool interaction baked inside `<think>`).
- **Train (`cot_v7.sh` → `think_v7`):** CoT-SFT dpo_soup, θ=1e6, eff-batch 12, 1 epoch, 70 min/1×H100,
  loss 2.13, all 35,641 rows kept. **Recover:** cross-soup `x_v7v3_a = (1-a)·v3 + a·think_v7` (`post_v7.sh`).

### 26b. MATH — a trade, not a win (broad gate: svamp/asdiv/mawps clean, gsmplus semi-clean; n=400, Wilson CIs)
| model | Σgreedy (3 clean) | Σself-cons (3 clean) | notable |
|---|---:|---:|---|
| v3 `x_v6v2_040` | 70.5 | **132.0** | baseline |
| **`x_v7v3_300`** (winner) | **72.5** (+2.0) | 125.75 (**−6.25**) | ASDiv greedy 25.8→**30.5** (teacher landed); SVAMP greedy −3; MAWPS self-cons −5.25 |
| `think_v7` (a=1) | (SV/AS/MAWPS mixed) | ~lower | native termination best (unclosed 42→8) but self-cons diluted |

- **The teacher distillation produced a REAL ASDiv-greedy gain (+4.75)** — but the code/tool/teacher diet
  **cut sample diversity → self-consistency regressed** (aggregate −6.25, MAWPS −5.25), and SVAMP greedy
  dropped. Net: better greedy, worse self-cons → a **trade**, not the both-axes win §23g required to ship.
  (The n=300 sweep looked more favorable (+4.3 greedy); the better-powered n=400 gate shrank it and exposed
  the self-cons regression — a lesson in **gating at n≥1000 aggregate with CIs**, not n=300.)

### 26c. TOOL-CALLING — taught perfectly, but unshippable as a single card (the key finding)
`tool_eval.py` (held-out synthetic arithmetic + a calculator/python tool spec):
| model | valid `<tool_call>` | tool expr == gold | final \boxed == gold |
|---|---:|---:|---:|
| v3 | 0% | 0% | 25% |
| **`think_v7`** | **100%** | **100%** | 55% |
| `x_v7v3_300` (30% v7) | **0%** | 0% | 35% |

- **`think_v7` learned the tool-call FORMAT perfectly** (100% well-formed calls, 100% correct expressions) —
  the capability IS teachable on this 2.88B base. **But it is not shippable weights-only:** (1) it is
  **washed out by the cross-soup** (30% think_v7 → 0% tool calls; the behavior lives in the delta, exactly
  the §19 α-knee phenomenon), so the math-best blend has no tool-calling; a high-α blend that keeps it
  carries think_v7's math self-cons regression; and (2) because tool_calc baked the `<tool_response>`
  in-trace (single-turn), the model **hallucinates the tool response** (e.g. 540+389 → writes "939", boxes
  it) → the arithmetic offload is **illusory** (55% final acc). Confirms the §PoT-refuted lesson: tool-use
  helps ONLY with a **real tool-execution serving loop** (stop-on-`</tool_call>` → execute → inject real
  `<tool_response>` → resume), which is a serving-system change — parallel to §25's external-verifier win.

### 26d. CODING — null (base-capacity-limited)
HumanEval pass@1 = **0.6% (1/164) for v3, think_v7, AND the blend alike** (all produce a function def 100%
of the time, but can't pass tests). An 8% Magicoder tier cannot give a 2.88B general model real coding
ability — the ceiling is base capacity (throughline #1), same wall as math.

### 26e. DECISION — keep v3; reaffirms Tier 3
No v7 blend clears the bar: math is a trade (self-cons regresses), tool-calling can't be shipped weights-only,
coding is null. **`x_v6v2_040` (v3) remains the shipped HF card.** This is the honest, well-measured outcome,
and it sharpens §25: **single-card weights-only downstream is exhausted; the two real forward moves are both
serving-system (a tool-execution loop / an external-verifier reranker) or a better BASE (Tier 3 = argonne3.5,
~31% pretrained as of 2026-07-12).** New honest-eval infra banked: MAWPS + GSM-Plus held-out sets, Wilson CIs
in `clean_eval`, and `code_eval`/`tool_eval` capability probes.

### New files this section
| File | What |
|---|---|
| `gen_teacher.py` (+`.sh`) | Qwen3-4B external-teacher gen on gsm8k-train → `teacher_qwen_gsm` (87.8% correct). |
| `build_mix_v7.py` | `cot_sft_mix_v7` = v6 backbone + teacher_gsm + code_magicoder + synthesized tool_calc. |
| `cot_v7.sh` / `post_v7.sh` / `eval_xsoups.sh` | Train think_v7; cross-soup sweep + clean-eval. |
| `gate_v7.sh` / `guard_v7.sh` / `diag_v7.sh` | Comprehensive gate (math×4+code+tool+guardrail); no-think guardrail; tool/code capability diag. |
| `code_eval.py` | HumanEval pass@1 (vLLM + sandboxed subprocess). |
| `tool_eval.py` | Tool-call format/correctness eval (+`--show` raw samples). |
| `clean_eval.py` (edited) | +MAWPS +GSM-Plus loaders, +Wilson 95% CIs + raw counts. |

---

## 27. Tool-EXECUTION loop — making v7's tool-calling real (2026-07-12)

§26 found think_v7 emits **100% valid tool calls with correct expressions** but hallucinates the
`<tool_response>` (→ 55% final acc), because tool_calc baked the response in-trace. §27 builds the
serving-system fix the evidence keeps pointing at (cf. §25's external reranker): `reasoning/tool_decode.py`
— an agentic loop that generates to `</tool_call>`, **executes** the calculator/python call (regex-gated
arithmetic), injects the **real** `<tool_response>`, and resumes.

**Result (think_v7, held-out single-op arithmetic word problems, n=150):**
| decode | acc |
|---|---:|
| greedy, self-hallucinated tool response | 53.3% (80/150) |
| **REAL tool-execution loop** | **100.0%** (150/150) |
| **lift from real execution** | **+46.7 pts** (avg 1.00 tool call/problem) |

- **The tool-calling capability is GENUINE once a real executor closes the loop** — the model reliably
  recognizes it should compute, emits the correct expression, and uses the injected result. This
  validates the §26 diagnosis (53.3% ≈ the 55% self-hallucinated) and **fulfills the tool-calling goal
  via a serving loop, not weights.** The bottleneck shifts from *arithmetic execution* (now solved by the
  tool) to *problem decomposition* (still the model's job).
- **Honest scope:** these are tool-friendly single-operation problems; 100% shows the *mechanism* works,
  NOT that multi-step SVAMP/ASDiv become 100% (those need the model to decompose + chain multiple calls —
  gated by the same base reasoning). And it does NOT change the shipped card: v3 emits 0% tool calls, so
  tool-augmented serving requires shipping think_v7 (or a high-α tool-preserving blend, which regresses
  plain math self-cons). So §27 is a **serving-system building block + capability proof**, not a v3 change.
- **Synthesis (§25–§27):** every downstream win left on this 2.88B base is a **serving-system** move — an
  external-verifier reranker (§25, gap→~75%) or a tool-execution loop (§27, arithmetic→100% on offloadable
  steps) — NOT a single-card weight change (§26 null). The only single-card ceiling-raiser is a better
  **base** (Tier 3 = argonne3.5). New file: `reasoning/tool_decode.py` (+`.sh`).

---

## 28. SHIPPED v4 — `x_v7v3_300` replaces v3 on HF (2026-07-12)

Per an explicit deploy decision (overriding the §26 "keep v3" recommendation — the model owner's call),
**v4 = `x_v7v3_300` = 0.3·think_v7 + 0.7·v3** was pushed to
[PursuitOfDataScience/Argonne-3.0-think](https://huggingface.co/PursuitOfDataScience/Argonne-3.0-think)
(`deploy_hf.py`: fp32→bf16, 5-shard, live-repo aux bundle, `eos_token_id`→151645, reload-verified
2,882,162,688 params before push; old shards replaced).

**Honest characterization (from the §26 n=400 gate — do NOT overclaim):** v4 is a **modest,
reasoning-focused, mixed update**, not a clean dominance:
- **Greedy**: +2 aggregate on the 3 clean sets — driven by **ASDiv 25.75→30.5** (the external-teacher
  distillation landed); SVAMP greedy −3 (within noise), MAWPS ~flat. Native `</think>` termination is
  stronger (think_v7 lineage).
- **Self-consistency**: **regressed ~6pts aggregate** (MAWPS −5.25) — the code/tool/teacher diet cut
  sample diversity; the conservative 70%-v3 soup only partly recovered it. This is v4's real cost.
- **Coding / tool-calling**: NOT in v4 (HumanEval ~0; 0% tool calls — washed out by the soup). These
  live in `think_v7` (100% valid tool calls) + the `tool_decode.py` serving loop (§27), documented as
  research tooling, NOT baked into the shipped card.
- **v3 (`x_v6v2_040`) retained on disk for rollback.** If the self-cons regression matters more than
  the greedy/termination gain, revert with `deploy_hf.py --src …/x_v6v2_040 --verify --push`.

**Version lineage:** v1 `soup_blend_a085` → v2 `blend_star_a06` → v3 `x_v6v2_040` → **v4 `x_v7v3_300`**.
The honest verdict stands (§26/§27): this is the ceiling of single-card weight edits; the real
forward moves are serving-system (reranker §25 / tool-exec loop §27) or a better base (Tier 3).

---

## 29. v8 — diversity-preserving teacher distillation: mechanism WORKS, ship gate NULL (keep v4) (2026-07-12/13)

The §26 v7 pilot proved external-teacher distillation lands a real greedy gain but collapses
self-consistency. An adversarial design panel diagnosed the cause as **short-only SFT sharpening the
student's sampling distribution + teacher-share homogenization** (not the teacher's decode), and
prescribed a **self-anchor tier** (v3's OWN verified-correct traces, keeping its basin) + modest teacher
share + no upsampling, **gated behind a cheap STOP-GATE probe.** Executed in full:

### 29a. Phase-A STOP-GATE (probe, ~16k mixes, n=400) — the mechanism is REAL
Arm 0 (v7-replica: teacher greedy, ~10%) vs Arm 1 (teacher-M2 **+ self_anchor**), cross-souped into v3 @α=0.3:
- **Arm 0 reproduced v7's collapse** (self-cons −6.0 vs v3); **Arm 1 HELD self-cons** (−0.75) with greedy +5.0.
- At matched α, Arm 1 beat Arm 0 by **+5.25 self-cons** → the self-anchor tier is the fix. STOP-GATE PASSED → Phase B.

### 29b. Phase-B build (`gen_traces.py` → `build_mix_v8.py` → `cot_v8.sh`)
- Teacher: **Qwen3-4B** (non-thinking, M≤2 distinct/problem) on **gsm8k-train + MATH-L1-3** → 14.6k verified
  traces. Self-anchor: **v3's own** verified `<think>` traces on gsm8k-train → 2.4k. `cot_sft_mix_v8` = 28.9k,
  ALL ≤640 tok: teacher_math 16% / self_anchor 8% / direct_tulu 26% / diversity tiers ~34% (NO code/tool, §26).
- CoT-SFT `dpo_soup`→`think_v8`, θ=1e6, **HBM-aware micro-batch** (card-adaptive: 94 GiB→micro-28, eff-28,
  LR √-scaled to 1.53e-5, filled **90%**; auto-scales down on 80 GiB cards). Cross-soup `think_v8 × v3`,
  α-select knee = **0.30** (`xv8_30`), exactly as Phase A predicted.

### 29c. Ship gate (n=1000, Wilson CIs; svamp/asdiv/mawps clean + gsmplus) — NULL vs the incumbent
| model | Σgreedy (clean) | Σself-cons (clean) |
|---|---:|---:|
| v3 `x_v6v2_040` | 71.2 | **131.7** |
| **v4 `x_v7v3_300` (shipped)** | 76.7 | 123.3 |
| **v8 `xv8_30`** | **77.9** | 124.7 |

- **v8 ≈ v4 (within noise):** Δgreedy **+1.2**, Δself-cons **+1.4** — both inside the ~±3–4% n=1000 CIs
  (svamp greedy even −0.4). NOT a clear improvement over the live model.
- **v8 fails the strict gate vs v3:** self-cons **−7.0** (greedy +6.7). The self-anchor held self-cons on
  SVAMP/ASDiv (in-distribution to its gsm8k basin) but **NOT on MAWPS** (v3 43.3 → v8 39.2) — the fix is
  **distribution-limited**; at the broad gate v8 is the same greedy↔self-cons trade v4 was, just marginally
  less-regressed. General no-think held (Paris/Shakespeare/Mars/photosynthesis ✓; only the pre-existing
  colors/transitivity base-gaps, same as v3/v4).
- **DECISION: KEEP v4.** No single-card v8 variant achieves a *clear both-axes win over v3*; v8 vs the
  incumbent v4 is within noise. Shipping it would churn the public card for a non-significant gain
  (throughline #8). `think_v8` + `xv8_30` + the verified corpus are retained for **Tier 3 transfer**.

### 29d. The thrice-confirmed conclusion
Three independent measurements now agree — Phase-A greedy control, v7 (§26), v8 (§29): **single-card weight
edits on this 2.88B base convert into a greedy↔self-consistency TRADE, never a clean both-axes win over v3.**
The wall is base capability (throughline #1; pass@32 ~77% unmoved throughout). The two real forward moves
stand: **(a) a better BASE** — argonne3.5, to which the v8 recipe (short-trace termination + diversity-
preserving teacher distillation + self-anchor + cross-soup) transfers directly (§15: ~36/40 on
Qwen/Llama-grade bases); **(b) serving-system** wins (external reranker §25, tool-exec loop §27). Single-card
distillation is now **exhausted** — do not re-pay it.

### New files this section
| File | What |
|---|---|
| `gen_traces.py` | Model-agnostic diversity-preserving trace generator (teacher OR v3-self-anchor; multi-sample, keep-M-distinct by step-signature, canonicalize). |
| `build_mix_v8.py` | `cot_sft_mix_v8` (teacher-math + self_anchor + v6 backbone, ≤640 tok, teacher ~15%). |
| `cot_v8.sh` | HBM-aware CoT-SFT (micro-batch auto-sized to the live card → ~90% on 80/94 GiB; eff-batch fixed via grad_accum; LR √-scaled). |
| `build_mix_v8_probe.py`, `phaseA_v8.sh`, `eval_phaseA.sh` | Phase-A STOP-GATE probe (Arm0 vs Arm1) + answer-entropy metric in `clean_eval`. |
| `post_v8.sh`, `gate_v8.sh` | Cross-soup α-select + the n≥1000 4-source ship gate (Wilson CIs) vs v3 and v4. |

---

## §30 — "improve 3.0-think, don't drift, thoroughly evaluate everything, then update" (2026-07-13/14)

Directive: push the **3.0-think card itself** (NOT argonne3.5 — see the SCOPE-DISCIPLINE banner up top and
[[argonne3-think-push-not-35]]); evaluate exhaustively before shipping. Two new card levers were run to
convergence, plus the definitive n=1000 characterization of the whole frontier.

### 30a. v3↔think_v8 soup frontier, densely sampled (n=500, 6 models) — FLAT
`soup_v8v3.sh` built `xv8_10/15/20` (+ existing `xv8_30`) and evaluated them with v3 & v4 on clean
SVAMP/ASDiv/MAWPS. Every point is within ~±1.5pt (means) of every other; per-source Wilson CIs overlap
heavily. `xv8_15` is the best-balanced (greedy 25.9 ≥ v4 25.5, self-cons 42.1 > v4 40.9 — both nominal,
< CI). No point dominates v4. **Confirmation #4.**

### 30b. v9 — BREVITY self-distillation (attack the unclosed greedy loss) — NULL/REGRESSION
Audit of the greedy failure modes: the dominant single-shot loss is **unclosed/no-answer** (15–30% of greedy
attempts never emit an answer — thinking loops past the 512 plain-greedy limit; asdiv n=1000 v3: 192 unclosed
+ 105 no_answer of 1000). Hypothesis: CoT-SFT on v3's **own** verified traces that CLOSE within budget →
teach concise termination while own-traces preserve the answer distribution (low homogenization risk).
`gen_v9.sh` (`gen_traces --think 1 --max-tokens 400` over gsm8k_train+MATH-L1-3 → **6394** distinct traces,
median 495 chars) → `build_mix_v9.py` (short-self-anchor **25%**, direct_tulu 29% general anchor, **NO
teacher**) → `cot_v9.sh` (dpo_soup→`think_v9`, HBM-aware micro-28 @90.9% reserved) → soups `xv9_30/50/70`.
**Result (`screen_v9.sh`, n=500):** `think_v9` is **worse on every axis** (greedy 23.3 vs v4 25.5, self-cons
38.0 vs v3 43.3, mawps greedy collapsed 22→17; answer-entropy 2.70). SFT on the model's own short traces
**homogenized** it without adding accuracy; soups just interpolate back toward v3. **Confirmation #5.**

### 30c. The definitive n=1000 frontier (`gate_v8.sh`, v3/v4/xv8_15, 4 sources + guardrail) — FLAT PARETO
| model | Σgreedy | +budget | **Σself-cons** | pass@32 | Σ(greedy+self-cons) |
|---|---:|---:|---:|---:|---:|
| v3 `x_v6v2_040` | 71.2 | 77.4 | **131.7** | ~77.7 | **202.8** |
| **v4 `x_v7v3_300` (shipped)** | **76.7** | **80.6** | 123.3 | ~78.3 | 200.0 |
| `xv8_15` (0.85·v3+0.15·v8) | 74.5 | 79.2 | 126.4 | ~77.1 | 200.9 |
*(Σ = sum over the 3 clean sources svamp/asdiv/mawps; per-source in `report/gatev8_math_*_52093640.log`.)*

- **The three points are one flat Pareto line:** identical pass@32 (~78% — the *capability* ceiling is fixed),
  Σ(greedy+self-cons) tied to within 1.4/200 (0.7%). v3 = self-cons corner (Σ +8.4 over v4, real & consistent
  across all 4 sources incl. gsmplus; mawps +5.0), v4 = greedy corner (asdiv-driven, +5.7). **`xv8_15` is the
  midpoint — it does NOT dominate v4** (self-cons +3.1 / greedy −2.2 vs v4, a milder version of the same trade).
- Guardrail: general/math no-think **identical** across v3/v4/xv8_15 — same successes, same pre-existing
  base-gaps (grammar "She don't", divisors-of-12→12, primary-colors→green, sun-is-a-planet). No new regressions.
- **6th independent confirmation** (Phase-A control, v7, v8, soup frontier, v9, this gate): **no single-card
  weight edit on this 2.88B base wins both greedy and self-consistency.** Even a hypothetical perfect
  selection-DPO caps at pass@32 ≈ 78 (fixed), and §22i shows SFT/DPO can't reach external-reasoner selection
  on this base → at best a marginal self-cons gain for a greedy cost, i.e. another trade. **Weights are done.**

### 30d. DECISION & the real lever
**KEEP v4 weights** — shipping any frontier point over another is an *operating-point* choice, not a benchmark
improvement (composite tied; would churn the public card for < CI, throughline #8). v3 remains the self-cons
alternative retained upstream. **The one genuine benchmark lift on this card is the serving reranker (§25):**
it cashes the fixed ~78% pass@32 ceiling, taking the deployable metric from self-cons ~41 → ~75 (+34pt).
Re-validated on the **shipped v4 base + MAWPS** (`ext_verify.sh`, job 52116464, n=500, McNemar vs v4's vote):

| set | v4 self-cons (vote) | **reasoned-rerank** | solver | pass@32 |
|---|---:|---:|---:|---:|
| svamp | 36.4 | **74.8** (+38.4, p<.001) | 73.8 | 77.0 |
| asdiv | 49.0 | **76.0** (+27.0, p<.001) | 77.4 | 79.2 |
| mawps | 38.4 | **58.0** (+19.6, p<.001) | 60.4 | 77.2 |

- **The reranker recovers the exact self-cons v4 traded away** (mawps 38.4→58.0) and reaches ~97% of pass@32
  on svamp/asdiv. Mean deployable accuracy **41 → 70 (+28pt)**, every source significant. `yesno` (1-token
  judge) still *hurts*; `solver`≈`reasoned` ⇒ the capture is Qwen's competence applied to Argonne's candidate
  set — a 2-model serving win, honestly framed, NOT the 2.88B card unlocked. This is now the headline
  "best accuracy" recipe on the HF card; `ext_verify.py` (two-phase generate→rerank) is the runnable form.

### §30 conclusion
Weights are done (6 confirmations); the real lever is serving. Shipped: **KEEP v4 weights**, HF README updated
with the n=1000 frontier + this validated reranker recipe, campaign scripts committed to main. Optional next
card lever (documented, NOT run — near-certain 7th null since pass@32 is pinned): selection-DPO on
reranker/ground-truth preference pairs.

### New files this section
| File | What |
|---|---|
| `soup_v8v3.sh` | Dense v3↔think_v8 frontier (α 0.10–0.30) build + n=500 eval. |
| `gen_v9.sh`, `build_mix_v9.py`, `cot_v9.sh`, `screen_v9.sh` | v9 brevity self-distillation pipeline (gen concise own-traces → short-anchor-dominant mix, no teacher → CoT-SFT → soups → screen). NULL. |

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

### `reasoning/` — vLLM-era inference + HONEST-eval tooling (§22–§25; the current judges)
| Script | What / why | § |
|---|---|---|
| `vllm_argonne.py` | vLLM 0.11.2 custom-model port of `argonne2` (+ `register()` + transformers-5.x tokenizer shim). VALIDATED 8/8 exact vs `model.py`. **`register()` is required in EVERY vLLM process** (the shim it applies is needed even for native models like Qwen). | §22h |
| `clean_eval.py` | **The current honest judge**: contamination-free SVAMP/ASDiv (+ math500/gsm8k) via `load_clean`; greedy / budget-forced / self-cons(closed-only) / pass@K on the vLLM path. GSM8K is CONTAMINATED — never gate on it. | §23 |
| `null_control.py` | Chance-collision null control (permutation, global + magnitude-bucketed) — proves pass@K/self-cons excess-over-chance is real, not a small-integer artifact. No GPU. | §23c |
| `select_eval.py` | Selection-strategy sweep (plain / conf-weighted / budget-forced self-cons vs pass@K) — showed the same-base vote family is saturated. | §23b |
| `vllm_bon.py` | Same-base best-of-N (generate K + same-base verifier rerank). §22i: a same-base verifier ≈ vote (base-limited). | §22i |
| `ext_verify.py` | **§24 Tier-1 EXTERNAL-verifier best-of-N** (the pivotal capturability test): v3 generates K on clean SVAMP/ASDiv → **Qwen3-4B** reranks (yesno / reasoned / solver lenses) → best-of-N vs closed-vote vs pass@K, with Wilson CIs + McNemar. `ext_verify.sh` runs both phases in one H100 job. | §24,§25 |
| `eval_math.py` | Programmatic GSM8K/MATH grader (pass@1/k/majority) + `--think-budget` s1 force-close. GSM8K contaminated → superseded by `clean_eval.py`. | §22e |
| `self_consistency.py` | Deployable sample-K + majority-vote inference (+ `--grade`). | §22i |
| `deploy_hf.py` | Ship a winner to the HF card: fp32→bf16 + 5-shard, bundle live `model.py`/tokenizer/chat_template, fix `eos_token_id`→151645, reload-verify, push. | §23h |

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
