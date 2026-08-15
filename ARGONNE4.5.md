# Argonne 4.5 — what to change in PRETRAINING, and why

Written 2026-08-12, after auditing the two frontier open releases of the last 48 hours against
everything the Argonne line has actually done and measured. Branch `argonne4.5`, worktree
`/home/youzhi/ArgonneAI-4.5` (main clone stays on `argonne4.0` — other work is running).

Ground truth for both reference models comes from the **local weights** in
`/project/rcc/youzhi/toxic-models` (`toxic` alias), not from blog posts — configs, tensor shapes and
parameter names, which are considerably more informative than what either vendor wrote up.

---

## 0. FINAL CONFIG — set from 51 probe arms, 2026-08-14

**a4.5 is argonne4.0's architecture at 2.06B with the systems settings fixed. Nothing else survived.**

Two probe campaigns (51 arms, 3× H100, `exp/EXPERIMENTS.md`) tested every lever proposed in the
sections below. The result was almost entirely negative, and this page overrides them.

### Measured throughput and how long the run takes

From the real production slice (`report/1-train.out`, job 53353941, 3× 94GB H100, micro 16 ×
accum 11), **not** from a probe:

| quantity | measured |
|---|---|
| step time | **10.56 s** (324 steps in 3420 s of training) |
| throughput | **51,222 tok/s** at 540,672 tok/step |
| model FLOPs | 224 TFLOP/s per GPU ≈ **22.7% MFU** vs H100 bf16 dense peak |
| slice startup | 72 s (warm inductor cache) |
| checkpoint save | ~21 s (18 s write + 1.7 s verify), 23.1 GiB |
| **full run** | **203,450 steps ≈ 24.9 days pure training, ~26 days wall clock** |

Wall clock assumes ~628 one-hour slices at 95% duty. It is **card-dependent**: an H200 slice runs
micro 22 × accum 8 (fewer accumulation boundaries) and should be faster; an 80GB H100 runs
micro 11 × accum 16 and slower. The effective batch is pinned at 540,672 on all three, so only
speed changes, never the schedule.

22.7% MFU is modest but expected for a 2B model at seq 1024 with activation checkpointing — small
matmuls and recompute, not idle silicon. The tested throughput levers are already banked
(`loss_chunk_size=0` +22.9%, `checkpoint_stride=2` +6.1%, ≈+30.5% together). The one untested
lever left is `torch.compile(mode="max-autotune")`, which needs a free GPU to A/B.

### What to run

```bash
cd /home/youzhi/ArgonneAI-4.5
./weekend.sh --dry-run          # inspect
./weekend.sh                    # continuous chain on H100 (default)
./weekend.sh --h200             # same chain on H200
./night.sh --h200               # ONE slice at 23:00, H200
```

**Card selection.** a4.0 hardcoded pretrain to H100; a4.5 makes it selectable (`--h100` / `--h200`,
or `PRETRAIN_CARD=`). Micro-batch and grad-accum follow the card so the **effective batch stays
540,672 either way** and the LR-6e-4 recipe is unaffected:

| card | HBM | micro × accum | effective |
|---|---|---|---|
| H100 (default) | 94 GB NVL | 16 × 11 | 540,672 |
| H200 (`--h200`) | 141 GB | 22 × 8 | 540,672 |

**Runtime card adaptation.** The worker reads HBM at slice start and picks micro/accum to hold the
effective batch at 540,672 on any card — H200 141GB → 22×8, H100 94GB → 16×11, H100 80GB → 11×16.
This matters because `0426` and `0372` advertise an IDENTICAL feature string but are 94GB and 80GB,
which OOM'd a probe arm before it was caught. Detection failure falls back to the smallest batch.

⚠️ **This cluster REJECTS OR-constraints.** `--constraint="H100|H200"` and `"[H100|H200]"` both fail
verification with "Access/permission denied" (plain `H100` passes), so a single job cannot take
whichever card frees first — it must commit to one. `--any-card` exists in the launchers but is
non-functional here.

⚠️ **H200 availability is the practical catch.** Only `midway3-0600` / `0601` survive the node
policy, and through this whole campaign they were **8/8 booked** by other users' multi-day jobs —
the first free 3-GPU slot was ~7.5 h out. H100 `0426` was idle throughout, which is why every probe
arm ran there and why H100 remains the default. Also note **all 51 probe measurements are H100 NVL
numbers**; the +30.5% systems win is a ratio and should carry, but the absolute tok/s will not.

| axis | value | basis |
|---|---|---|
| params | **2,063,667,712** | hidden 2560 / 24L / 10Q-2KV / head_dim 256 / SwiGLU 7040 / tied |
| architecture | **identical to a4.0** | every proposed change refuted or unresolved (below) |
| `loss_chunk_size` | **0** | e01→e03: 32,205 → 39,593 tok/s, **+22.9%** |
| ckpt stride | **2** | e03→e04: → 42,016 tok/s, **+6.1%**, HBM 65→84% |
| micro-batch | **16** × block 1024 | forced by chunk=0: 16·1024 rows = 9.9 GiB fp32 logits (a4.0's 170 would need 105 GiB) |
| effective batch | 16·3·1024·**11** = **540,672** | within 3% of the LR-6e-4-validated 524,288 |
| LR / warmup / cooldown | **6e-4 / 8000 / 0.15** | inherited; probes could NOT resolve these (see σ below) |
| data | 50/30/20 edu/math/code | inherited; mixture unresolved |
| precision | fp8 + lm_head, bf16 autocast, torch.compile | a4.0 |

**Net measured gain over a4.0's settings: +30.5% throughput ≈ 7.9 days off a 34-day run.**

### Every proposed lever, and what happened to it

| lever | verdict |
|---|---|
| LLLG sliding window | **REFUTED** — +0.038 CE *and* −14.6% throughput. No flash-attn-2 here, so the window forces an explicit SDPA mask and abandons the fused `is_causal` kernel. |
| NoPE global layers | **REFUTED** — worse at every length; my "improved slope" claim was retracted (an FFN-only control reproduced the same slope shift). |
| intra-document masking | **REFUTED at iso-compute** — a real −0.034 iso-token win, but −27.5% throughput means the baseline wins by 0.353. |
| real MTP module | **REFUTED** — +0.127 iso-compute. Its loss still reaches the trunk via `h_prev`. |
| RHO-1 selective loss | **REFUTED** — +0.449 token-matched, 23% slower. Dropping 40% of tokens removes 40% of the gradient. |
| ReLU² FFN | **UNRESOLVED** — three iso-param measurements, three different signs. |
| gated attention / untied embeddings | **UNRESOLVED** — 0.7σ and 0.5σ against the true noise floor. Campaign 1 called both wins using a σ that was 4.6× too small. |
| MoE | **not tested.** It is a capacity question, and the probe is provably blind to capacity (see below). |

### ⚠️ Measurement caveats that bound all of the above

1. **σ = 0.130 at the probe operating point** (three runs of one config: 4.7263 / 4.8393 / 4.9858),
   not the 0.028 measured at the *old* operating point. 2σ resolution is **0.368**. Anything smaller
   than that in this campaign is unresolved, not null.
2. **The probe cannot answer capacity questions.** It burns ~1 GPU-hour against production's ~24
   GPU-days — 4,600× less — where the Chinchilla-optimal model is ~43M params. Both size and block
   size came out "smaller is better," which is exactly what theory predicts down there and carries
   no information about 154B tokens. MoE would fail the same way.
3. **The batch finding is the one large surviving result** (61,440 beats 122,880 by 0.994 iso-token
   and 0.569 iso-compute, 3–5σ) — but its mechanism is *step-limitation* at 568 steps, and
   production runs ~147,000 steps and is not step-limited. **Direction transfers, magnitude does not.**
   This is why the shipped batch stays near a4.0's, not at the probe optimum.
4. Throughput on `midway3-0426` varies **±12%** with node co-tenancy, so any throughput delta under
   ~15% measured across time is unusable.

### Can a4.5 beat argonne3.5?

Honestly: **coin flip at 154B tokens.** Pricing with the only slopes measured on this line —
params 2.88B→2.06B is **−5.5 pt** (the sole *production-scale* datum, from a4-vs-3.5), tokens
64B→154B is +3 to +6, mix is +2 to +4 ⇒ **net ≈ +2 pt**. The params term is the dominant negative
and the probe cannot overturn it.

**If the goal is specifically to beat 3.5, match its size.** At fixed compute 2.06B@154B ≈ 2.88B@110B,
and the larger model deletes the −5.5 pt term instead of trying to out-token it (≈ +5 pt net).

### The one thing to verify before committing 34 days

A **scaled-up batch run** — a few billion tokens, not 35M — to test whether the batch direction
survives outside the step-limited regime. That single assumption underpins the entire schedule
recommendation and is the one thing a 30-minute probe structurally cannot test.

---

---

## 1. The two reference models, as measured

### 1.1 NVIDIA Nemotron 3.5 Lightning — `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`

`NemotronHForCausalLM`, 30B total / ~3B active, 52 layers, hidden **2688**, vocab 131,072, untied.

| Axis | Value |
|---|---|
| Layer sequence (52) | `MEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMEMAEMEMEMEME` |
| Composition | **23 Mamba-2 · 23 MoE · 6 attention** (attention at 5,12,19,26,33,42) |
| Mamba-2 | d_inner 4096 (64 heads × 64), state 128, n_groups 8, conv 4, chunk 128; `in_proj` = 10304 = z+x+B+C+dt |
| Attention | GQA 32Q/2KV, head_dim 128, **rope_theta 10,000** at **1M context**, fp8 KV (`k_scale`/`v_scale`) |
| MoE | **128 routed experts, top-6, 1 shared** (inter 3712 = 2× routed 1856), `routed_scaling_factor` 2.5 |
| Expert shape | `up_proj [128,1856,2688]` + `down_proj [128,2688,1856]` — **2 matrices, no gate** |
| MLP activation | **`relu2`** (ReLU²), not SwiGLU |
| Load balancing | `gate.e_score_correction_bias` → **aux-loss-free** (DeepSeek-V3 style bias nudging) |
| MTP | `num_nextn_predict_layers: 1`; `mtp_layers_block_type: ['attention','moe']` with `eh_proj`/`enorm`/`hnorm` |
| Precision | pretrained under an **NVFP4** recipe; BF16 reference released alongside |
| Data | >20T tokens, 19 natural + 43 programming languages, "crawled and synthetic" |

The three things that matter for us: **(a)** experts are 2-matrix ReLU², so an expert costs 2/3 of a
SwiGLU expert at equal width; **(b)** MTP is a *separate module* with its own parameters, plus a
dedicated MTP-boost phase after pretraining — it is not an auxiliary term bolted onto the main head;
**(c)** rope_theta stays at 10,000 while claiming 1M context, because only 6 of 52 layers are
attention and Mamba carries the long-range state. Position encoding stops being the long-context
mechanism.

### 1.2 Meta Muse Glimmer 30B — `meta-models/Muse-Glimmer-30B`

`MuseGlimmerForConditionalGeneration`, dense, 29.6B text + ~1.8B ViT-G/14, 52 layers, hidden 6656,
inter 19968, vocab 202,048, untied, 131k context, distilled from Muse Spark.

| Axis | Value |
|---|---|
| Layer sequence (52) | `sssFsssFsssF…` → **39 sliding (window 2048) : 13 full attention** |
| **Position encoding** | RoPE θ=500,000 on sliding layers; **`layer_rope_theta = 0` on every full-attention layer → NoPE** |
| Attention | GQA 32Q/2KV, head_dim 128, **gated** (`self_attn.gate_proj`), no q_norm/k_norm |
| Scaling | `qk_scale_factor: 3.87`, `output_multiplier: 0.196…` (= 1/√26 = 1/√(L/2)), `final_logit_softcapping: 20.0` |
| Norms | `input` + `post_attention` + `pre_feedforward` + `post_feedforward` — sandwich on both sub-blocks, `post_norm_eps 1e-8` |
| FFN | SwiGLU, 3× hidden |
| Serving | DFlash block-diffusion drafter, 5 layers, 16-token blocks, 3.1× on RTX 5090 |

The headline for us is the position-encoding design: **the global layers have no positional encoding
at all.** Locality is supplied by the 3-in-4 sliding layers; the global layers are position-agnostic
and therefore have nothing to extrapolate incorrectly. This is the current best answer to the exact
failure recorded in `[[rope-extrapolation-fails-phaseb-works]]`.

### 1.3 What the two agree on

Independently, both labs converged on:

1. **Most layers are cheap; few layers are global.** Nemotron 11.5% attention, Muse 25% full attention.
   Nobody runs dense full attention at every layer any more.
2. **Untied embeddings** at 30B (both).
3. **A drafter/MTP shipped with the model** — decode speed is a first-class deliverable, not an afterthought.
4. **Distillation is the default construction method for a small model,** not an optional extra.
   Muse Glimmer is *defined* as "distilled from Muse Spark"; the Nemotron family is distill+prune.
5. **Synthetic data is load-bearing,** stated explicitly by both.
6. Neither uses a novel optimizer. AdamW-class, WSD-ish. Our optimization recipe is not the gap.

---

## 2. Where Argonne actually stands

Production `pretrain.py` / `model.py` today (argonne4.0, 1.04B):

| Have | Don't have |
|---|---|
| GQA 6Q/2KV, head_dim 256 | MoE (never trained one) |
| SwiGLU inter 4096 | ReLU² / 2-matrix FFN |
| RMSNorm, sandwich norm, qk-norm, v-norm | Gated attention |
| RoPE θ=1e6, all layers | NoPE layers / per-layer rope |
| Logit softcap 15.0 | µP / depth-aware output scaling |
| Tied embeddings (233M of 1.04B) | Untied at real horizon |
| AdamW fused, WSD, warmup 8k, cooldown 0.15, grad_clip 0.4 | — (this part is well-searched, keep it) |
| FP8 tensorwise + lm_head, torch.compile | NVFP4 (needs Blackwell — we're on Hopper) |
| Chunked CE, selective activation ckpt | Real MTP module |
| Weighted multi-source mixture 50/30/20 | Intra-document attention masking |
| Doc-manifest loader, decontamination | RHO-1 / selective loss |
| Reasoning anneal + context-extension phases | Pretrain-scale KD; synthetic data; agentic traces in pretrain |

**Two things this audit verified that were not previously known to be true:**

### ⚠️ Finding A — the sliding window has never been active in any production run

`model.py` only applies `sliding_window` on the flash-attn-2 path. In the live AI env:

```
flash_attn spec: True
flash_attn_func import FAILED: ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
```

so `_flash_attn_available = False`, the SDPA path runs, and `sliding_window` is dropped on the floor.
`pretrain.py` has shipped `ENABLE_INTERLEAVED_LOCAL_ATTENTION = True, LOCAL_ATTENTION_WINDOW = 256`
since 3.0 — **every Argonne pretrain to date has run full attention on all layers.** We have been
paying full O(L²) attention cost on every layer while believing half of them were windowed at 256.

Notably, `experiments/model.py` **does** have the fix (an explicit SDPA window mask, added during the
arch sweep with a comment naming the flash-attn-4 problem) — it was never ported back to production.
Same failure mode as the `--cooldown 0` bug in `[[anneal-no-lr-decay-and-general-forgetting]]`: a fix
lands in one tree and not the other. The arch sweep's "window 64 won at 450 steps" result is real;
the production config it was supposed to inform is inert.

### ⚠️ Finding B — our MTP null tested a different technique than the one NVIDIA shipped

`model.py`'s MTP re-uses the **same `lm_head`** on the **same hidden state**, shifted:
`logits[:, :-shift]` vs `labels[:, shift:]`, added to the main loss at `mtp_loss_weight`. There is no
MTP module, no extra parameters, and it competes directly with the t+1 objective for the trunk's
capacity.

The evidence against MTP is exactly one run: `exp045_mtp2`, horizon 2, weight 0.3, base `exp042`
(3.8513) → **3.9346, +0.083 LOSS**, at **450 steps**. Correct verdict for what was tested. It is not
evidence about DeepSeek-V3/Nemotron MTP, which adds a *separate* block (`mtp.layers.0` attention +
`mtp.layers.1` MoE, with `eh_proj` concatenating `[emb(t+1), h(t)]`, plus `enorm`/`hnorm`) so the
trunk's t+1 objective is untouched. Per `[[a4think-selection-is-the-lever]]`: never price a lever by
an exhausted family's best — and this family was never actually opened.

---

## 3. The lever list

Ranked within tiers by (value to a reasoning base at ~1B on 3 GPUs) ÷ (engineering + risk).
"Untried" means never attempted in Argonne pretraining at any scale.

### Tier 1 — do these for 4.5

**T1.1 RHO-1 / selective language modeling — UNTRIED. Highest value per hour of work.**
Train (or reuse) a small reference model on high-quality data; at pretrain time, compute per-token
excess loss `L_student − L_ref` and backprop only the top-k% of tokens (typically 60%). Reported
2–10× data efficiency on math specifically. Our binding constraint is documented in
`base-for-reasoning.md` §1.1: the a4 run finished at **62.1B tokens on 1.0385B params = 60
tokens/param** (measured from `report/argonne4.0/`, not the corpus size — the run repeated data),
against the 300–450 that makes small models capable. RHO-1 attacks that ratio directly, costs
nothing at inference, and needs no new data. The reference model can be an existing `a4_dose` checkpoint or
Qwen3-0.6B-Base (same tokenizer family).
*Cost: a scoring pass over the corpus + a masked-CE path in `pretrain.py`. Risk: low — it degenerates to standard CE at k=100%.*

**T1.2 Real MTP module + a dedicated MTP-boost phase — effectively UNTRIED (see Finding B).**
Separate block(s) with `eh_proj`/`enorm`/`hnorm`, own head, own loss; trunk objective unchanged.
Two independent payoffs: a denser training signal per token (which is what we are short of), and a
**free speculative-decoding drafter**. The second one compounds: our deployable reasoning recipe is
self-consistency at K=8–32 (`[[tooluse-pot-refuted-procedure-is-wall]]`), so a 2–3× decode speedup is
2–3× more samples per GPU-hour on every future experiment, not just a serving nicety.
*Cost: ~150 lines in `model.py` + an anneal-phase flag. Risk: moderate — must be validated at ≥5B tokens, not 450 steps.*

**T1.3 NoPE global layers + RoPE sliding layers (Muse pattern) — UNTRIED.**
`sssF` with θ=500k on sliding@2048 and **θ=0 (NoPE) on full-attention layers**. This targets a
specific, documented Argonne failure — `[[rope-extrapolation-fails-phaseb-works]]`: "a
block-1024-trained base is BLIND past 1024. Never assume θ=1e6 extrapolates." NoPE global layers have
no frequency basis to break. If it works, the phase-B context-extension stage gets much cheaper or
disappears, and `[[context-window-limits-measured]]` stops being a hard wall.
*Cost: per-layer rope config + a real SDPA window mask (which also fixes Finding A). Risk: moderate — validate length generalization explicitly, don't infer it from CE.*

**T1.4 Intra-document attention masking — UNTRIED.**
`ArgonneModel.forward` hard-sets `attention_mask = None`, and the production flat/weighted loaders
slice a contiguous window out of a packed `.bin`. At block 1024 over FineMath and code, a single
window spans many documents and every token attends across the boundaries. This is a plain
correctness defect in the training objective, and it bites hardest exactly on our target domains
(short math problems, short code files). Fix with a block-diagonal mask from document ids, or
`flash_attn_varlen_func` if a 2.x-compatible build gets installed.
*Cost: emit doc-id/cu_seqlens alongside tokens in the loader; mask in attention. Risk: low.*

**T1.5 Break the 38.6B-token wall with synthetic data — UNTRIED.**
Named in `[[argonne4-design]]` as "WRAP", never built. Both reference models state synthetic data is
central. We have Qwen3-14B/30B locally and a validated vLLM path (`reasoning/vllm_argonne.py`,
`vllm_bon.py`) — rephrasing FineWeb-Edu into textbook/QA/dialogue styles is throughput-bound, not
research-bound. This is the only lever that changes the *numerator* of tokens/param without more
crawling. Combine with the ≤4× repetition budget (Muennighoff) already noted in `ARGONNE4.0.md`.
*Cost: generation GPU-hours (large but parallel and interruptible) + a build script. Risk: low quality-wise, high in wall-clock.*

**T1.6 µP (maximal update parametrization) — UNTRIED.**
Explicitly listed as next-step (e) in the campaign's own morning summary and never done. Right now
LR 6e-4 was tuned at 24K batch on 2.88B and is used at ~1M batch on 1.04B by hand-argument (the
comment block at `pretrain.py:56-65` is that argument). Under µP the 78M-token proxy runs would
actually transfer, which retroactively upgrades the whole experiment methodology — including the
cross-GPU noise problem that already forced one retraction (KEY FINDING 14).
Muse's `output_multiplier = 1/√(L/2)` and `qk_scale_factor` are the same family of idea.
*Cost: init-std and per-group LR scaling by width; a coord-check script. Risk: low, high methodological payoff.*

### Tier 2 — high value, real engineering; decide explicitly

**T2.1 MoE — UNTRIED, and the largest capacity-per-FLOP lever available.**
Our documented conclusion across six post-training methods is that base capability is the wall
(`base-for-reasoning.md`, `[[grpo-rlvr-exhausted]]`). MoE is the standard way to buy capacity without
buying FLOPs: a ~4B-total / ~0.8B-active model trains at roughly 0.8B cost and holds ~4B of
knowledge. Copy Nemotron's specifics rather than inventing: **2-matrix ReLU² experts** (2/3 the params
of SwiGLU at equal width), **1 shared expert always on**, **top-k routed**, and
**aux-loss-free balancing via a `e_score_correction_bias` nudged from expert load** — no auxiliary
loss term fighting the LM objective.
*Cost: high — routing, balancing, capacity/dropless kernels, checkpoint size, and our save/resume path assumes a dense state dict. Risk: high. But this is the honest answer to "we need more capability per token."*

**T2.2 Pretrain-scale distillation — tried, but the null is scope-limited; worth re-opening.**
The campaign found KD neutral-to-harmful (KEY FINDINGS 6–8): α=0.5 hurt, α=0.3 was mildly net-positive
on web-only, and redundant once the mix had math/code. That verdict is honest **for what it tested**:
offline logit KD, Qwen3-**1.7B** teacher, **78M tokens** — and the campaign itself flagged that regime
as unable to validate KD ("distillation's proven wins are at 500B+ tok… the 78M proxy likely can't
validate KD").
Three reasons to re-open it, not to overturn the finding:
- Muse Glimmer, a 30B model, is *entirely* built this way. So is the Nemotron family.
- Our single best post-training result ever — `[[a4think-on-policy-distillation]]`, +5.20 at p=4.2e-10,
  acc|ANS +8.8pp — was a distillation objective, and the decisive variable was **teacher/student
  distribution matching**, not teacher strength. A 1.7B teacher at α=0.5 against a 78M-token student
  is maximal mismatch, which is consistent with the observed harm.
- What is untried: KD at ≥10B tokens, from a *large* same-tokenizer teacher (Qwen3-14B/30B, local),
  with **top-K sparse logits** precomputed and stored (makes it cheap and removes the teacher forward
  from the training loop), and α annealed from ~0 upward as the student stops being undertrained.
*Cost: a top-K logit dump over the corpus + a KD term. Risk: moderate; the prior is genuinely mixed and it must be gated on a ≥10B-token arm.*

**T2.3 Gated attention — UNTRIED.** One extra `hidden → n_heads·head_dim` projection, sigmoid-gating
the attention output per head. Muse ships it and drops q_norm/k_norm entirely. We currently carry
qk_norm + v_norm + sandwich norm (4 extra RMSNorms per block); if gating subsumes some of that, it is
net-neutral on params and cheaper on kernels. Reported to remove attention sinks and improve stability
at high LR — and `[[argonne-next-arch-search]]` records that "qk_norm is ESSENTIAL" at our LR, i.e. we
are already living at the stability edge that gating is designed to fix.
*Cost: small. Risk: low. Needs a proxy run at a real horizon.*

**T2.4 ReLU² / 2-matrix FFN — UNTRIED.** Drops `gate_proj`. At iso-params the FFN can be ~50% wider;
at iso-width it is 33% cheaper. Our sweep tested SwiGLU *ratios* (2.0× worse, 2.75× optimal, 3.5×
neutral) but never the activation family. Pairs naturally with T2.1 since it is what makes Nemotron's
128 experts affordable.
*Cost: trivial. Risk: low, but it is a from-scratch-only change.*

**T2.5 Untie embeddings — tested, but only at 450 steps.** `exp044` untie = **+0.089 LOSS**, and the
decisions doc names the reason: "untied lm_head starts random and can't be learned in 450 steps." That
is a short-horizon trap, not a property of untying — both 30B references untie, as does essentially
every modern model. The real question for us is different and is about *size*, not steps: at vocab
151,936 × hidden 1536, untying adds **233M params to a 1.04B model (+22%)**. That is a defensible
reason to stay tied at 1B, and a bad one at 2B+.
*Cost: none. Risk: none — but re-measure at ≥10B tokens before believing either answer.*

**T2.6 Raise pretrain context from 1024 → 2048–4096 — UNTRIED at pretrain.**
Called out in `base-for-reasoning.md` §2.5 as an unmeasured capability lever: block 1024 was chosen
for loss/GPU-hour, and multi-step reasoning traces need room. Cheaper than it used to be now that
chunked CE exists, and much cheaper once T1.3's sliding window is actually active (Finding A means we
are currently paying full quadratic attention, so the marginal cost of longer context is worse than
it needs to be).
*Cost: throughput. Risk: low.*

### Tier 3 — real, but not for 4.5

- **T3.1 Mamba-2 hybrid.** The most interesting thing in the Nemotron config, and the reason it can
  claim 1M context with rope_theta=10,000. But it is a new kernel dependency, a new state-management
  story for `generate`/vLLM, and it would invalidate the vLLM port (`[[vllm-argonne-port]]`). Betting
  a full run on it is not justified until the Tier-1 items are banked.
- **T3.2 NVFP4 pretraining.** Nemotron's headline precision result. **Requires Blackwell; we are on
  H100/H200 (Hopper).** Not available. The Hopper-side increment is fp8 blockwise/MXFP8 over our
  current tensorwise, which is a throughput tweak, not a capability lever.
- **T3.3 Multilingual (19 languages) + 43 programming languages.** Correct for a general model, off-strategy
  for a reasoning base on 3 GPUs.
- **T3.4 Vision tower / multimodal.** Both references are multimodal. Out of scope.
- **T3.5 Agentic/tool traces in the pretrain mix.** Argued in `base-for-reasoning.md` §4, and both
  reference models are agent-first. `Toucan-1.5M` and BFCL are already in the HF cache. Deferred only
  because it competes for the same anneal budget as reasoning traces — but this is the first Tier-3
  item to promote if the target moves toward agents.

### Explicitly not on the list

A new optimizer. Neither reference model uses one, `[[recipe35-training-search]]` already found AdamW
beats Muon iso-time here, and `[[argonne-next-arch-search]]` searched this axis to exhaustion. The
optimization recipe (LR 6e-4, WSD warmup 8k / cooldown 0.15, grad_clip 0.4, qk-norm, FP8) is the part
of Argonne that is genuinely well-tuned. Keep it.

---

## 3b. Sizing: the measured budget (2026-08-12)

Everything below is calibrated on the **actual a4.0 run**, reconstructed from the 239 slice logs in
`report/argonne4.0/`. Two corrections to what was assumed:

- The run was on **3× H100** (`--constraint=H100`, nodes midway3-0426/0372), not H200.
- It finished at **62.1B tokens** (step 109,620) = **60 tokens/param**, not the 37 the corpus size implies.

| Phase | config | median tok/s | slices |
|---|---|---|---|
| pretrain | block 1024, micro-batch **170**/GPU, accum 1 | **60,096** | 136 |
| anneal A | block 1024, micro-batch **32**/GPU, accum 6 | **86,353** | 57 |
| phase B | block 13568, micro-batch 2/GPU | 39,300 | 44 |
| whole run | 62.1B tokens / 11.8 calendar days | 61,030 | — |

### ⚠️ Finding C — RETRACTED as a batch-size effect. It is a cross-GPU artifact.

The anneal phase ran **44% faster** than pretrain at the same block size, and the first reading here
attributed it to the micro-batch (32×accum-6 vs 170×accum-1) via CE chunk count. **That mechanism is
wrong, and the comparison is confounded:**

- Chunk count per *optimizer step* is invariant to the micro-batch split — 42.5 vs 48 across the two
  configs, essentially identical. Chunking cannot explain a 44% gap.
- `run_full_training.sh:454` `midtrain_card_now()` switches the SLURM constraint to **H200 between
  23:00–07:00** once `.pretrain_complete` exists. Pretrain ran entirely on **H100** (0426/0372); the
  one anneal slice that names its node ran on **midway3-0601 = H200**.

So the 1.44× is at least mostly **H200-vs-H100**, which is a plausible ratio for a bandwidth-bound
small model (H200 4.8 TB/s vs H100 3.35). This is the same artifact class that forced KEY FINDING 14's
retraction, and it is a reminder that on this cluster *any* cross-phase throughput comparison is
cross-GPU unless pinned.

**What survives:** the H100 rate is measured (60,096 tok/s at 1.038B, block 1024, 3 GPUs, 136 slices).
The H200 speedup and any micro-batch effect are **unmeasured and entangled** — pin the card and
measure both in the bake-off. Still worth probing there: gradient checkpointing off (≈33% recompute)
and the micro-batch, since `[[sft-length-grouping-beats-hbm-fill]]` and
`[[optimize-gpu-hbm-usage]]` both say HBM fill is not the objective. Just don't bank 1.44× on it.

### Throughput and calendar

Planning rate: 5.27 B tok/day measured (3×H100, 1.038B) → **~9 B tok/day on 4×H200** at block 2048
(×4/3 GPUs, ×~1.44 H200, ×0.88 for block 2048+LLLG). Treat 9 as a planning figure with ±20%, not a
measurement. `R(N) ≈ 9.34/N` B tokens/day on 4 GPUs; double it on 8.

The `gpu` QOS ceiling is **16 GPUs / 4 nodes / 36h**; the launcher is `--nodes=1 --gres=gpu:3`. So
4×H200 on one node is free, and **8 GPUs costs about a day of multi-node launcher work and doubles
everything permanently** — the single highest-leverage piece of infrastructure on this list.

| N | 4×H200 | 8×H200 | 154B @4 | 154B @8 | 300B @8 | tok/param @154B |
|---|---|---|---|---|---|---|
| 1.04B | 9.0 B/d | 18.0 B/d | 17 d | 9 d | 17 d | 148 |
| 1.5B | 6.2 | 12.5 | 25 d | 12 d | 24 d | 103 |
| **2.06B** | **4.5** | **9.1** | **34 d** | **17 d** | **33 d** | **75** |
| 2.88B | 3.2 | 6.5 | 48 d | 24 d | 46 d | 53 |
| 3.5B | 2.7 | 5.3 | 58 d | 29 d | 56 d | 44 |

### The size decision: 2.06B — 2.0× a4.0, 0.72× a3.5

**Decided 2026-08-12 (owner call): 2,063,667,712 params.** The reasoning below establishes that
1.04B is too small and that the loss optimum for our budget sits at 2.4–3.8B with a *flat bottom
across 2.5–3.5B*. 2.0B sits just under that band and costs **~+0.003 predicted loss at fixed
compute** — and buys two things the extra 0.88B does not:

- **154B tokens in 34 days on FOUR GPUs.** Multi-node stops being a prerequisite and becomes an
  accelerator (17 days if it lands). At 2.88B the run would have *needed* 8 GPUs to stay under a month.
- **Compute for the levers.** 4.5's thesis is five untried mechanisms. Their unmeasured upside
  plausibly exceeds 0.003 loss; 0.88B more dense parameters does not buy a hypothesis, it buys a
  slightly lower number. Spending the delta on ablating RHO-1 / MTP / NoPE / doc-masking / ReLU²
  properly is the better bet.

The size argument that follows is unchanged in direction — go **up** from a4.0, and the two-model
evidence for that is below. Only the stopping point moved.

#### Why up from 1.04B at all

**The direct experiment has already been run and it says parameters win at our token scale.**

| | params | tokens | tok/param | data mix | think 5-set greedy |
|---|---|---|---|---|---|
| argonne3.5 | **2.882B** | ~64B | 22 | web-heavy 85/15, **no code** | **55.16** |
| argonne4.0 | 1.038B | 62.1B | 60 | edu/math/code 50/30/20 | **38.37** |

At essentially the same token count, **2.8× the parameters beat 2.7× better tokens/param *and* a
better data mix, by 16.79 points** — worse on all five pools at p ≤ 1e-5, and `pass@8` is also
−12.72, so the *ceiling* moved. That is capability, not post-training
(`[[a4think-recipe-transfer-refuted]]`).

ARGONNE4.0.md said plainly that the campaign "does not prove 1.04B beats the 2.88B model — that needs
a full run." The full run has now happened, and the answer is **no**. The data-efficiency result was
real *iso-token at 1B-vs-1B*; it does not survive being cashed in for a 2.8× parameter cut.

The earlier "hold 1.04B" reading in this document weighted comparability against **a4.0**. That was
the wrong anchor, and so was the follow-up that anchored on 3.5. Size should come from the budget:

- **The optimum for our budget is 2.4–3.8B, with a flat bottom.**
  Minimising the Chinchilla fit `L = 1.69 + 406.4·N^-0.34 + 410.7·D^-0.28` at fixed compute
  `C = 6ND`, using our measured throughput, gives:

  | budget (8×H200) | N·D | loss-optimal N | its D | its D/N |
  |---|---|---|---|---|
  | 16 GPU-days | 3.0e20 | 2.4B | 126B | 53 |
  | **24 GPU-days** | 4.5e20 | **2.9B** | 157B | 55 |
  | 32 GPU-days | 6.0e20 | 3.2B | 184B | 57 |
  | 46 GPU-days | 8.6e20 | 3.8B | 225B | 59 |

  For any calendar we would plausibly run the optimum is **2.4–3.8B**, and the curve is
  flat-bottomed: at 24 GPU-days, 2.0B costs **+0.0033** predicted loss and 4.0B costs +0.0030. So the
  whole band **2.0–3.5B** is within a rounding error of optimal, and the choice inside it should be
  made on grounds the loss curve cannot see — calendar, risk, and how much compute is left for the
  levers. **2.06B is the floor of that band, and that is the call.**
  ⚠️ An earlier draft said "at D = 154B the loss-optimal model is ≈7.7B." Wrong framing — it treats
  data as free and compute as unconstrained. At fixed *compute*, the actual constraint, it is ~2.9B.
- **Sub-2B is genuinely out.** Below ~2B the loss is worse *and* the compute-matched token count
  exceeds the corpus: 1.5B wants 299B tokens at a 24-GPU-day budget and we can build 154B.

#### The chosen shape

| | value | why |
|---|---|---|
| params | **2,063,667,712** | 2.0× a4.0, 0.72× a3.5 |
| hidden / layers | **2560 / 24** | aspect 107, between a3.5's 128 and Gemma-2-2B's 89 |
| heads Q / KV | **10 / 2**, head_dim 256 | head_dim 256 is Phase-1-confirmed; GQA ratio 5 → **48 KB/token** KV cache, 2.5× smaller than ratio 2 — it matters because the deployable recipe is self-consistency at K=8–32. Both reference models run 16:1, so 5:1 is conservative. |
| FFN | **SwiGLU 7040 → ReLU² 10560** | MLP **2.75×**, the sweep's one HIGH-confidence arch optimum (2.0× was +0.042 worse). Iso-param, Δ+0 verified. |
| layers = 24 | | `LLLG` divides evenly → **6 global / 18 sliding, and the last layer is global** (Muse's `sssF`×13 also ends on a full-attention layer) |
| embedding share | 18.8% | vs 22.5% at a4.0's 1.04B |

Two remaining notes on the size argument:

- **Qwen3-0.6B is not a counterexample.** It beats a4 at 58% of the params on **36T tokens** =
  ~60,000 tok/param. That regime is three orders of magnitude away; inside the regime we can actually
  reach, the table above is the relevant evidence.

- **The FFN swap stays exactly iso-param at this size**: at hidden 2560 / 24L / 10Q / 2KV /
  head_dim 256, SwiGLU inter 7040 (3 matrices) and **ReLU² inter 10560** (2 matrices) are both
  **2,063,667,712** params (verified, Δ+0). So ReLU² remains a clean single-variable test with
  depth, width, attention and total size held constant.

**The cost, stated plainly.** 2.06B gives up 0.82B parameters against 3.5 and has to win them back
from tokens and levers. The scaling fit prices that at ~0.003 predicted loss versus 2.88B at equal
compute — small — but the *two-model evidence* (16.79pt for 2.8× params) is a downstream measurement
and the noisier of the two, so this is a real bet, taken deliberately. What it buys is a **34-day run
on four GPUs we already have** instead of a 48-day run that needs multi-node first, and the freed
compute goes into ablating the five levers that are the actual thesis of 4.5.

**What would change this answer:** MoE. If parameters are what we are short of and compute is the
wall, an MoE buys parameters without buying training FLOPs — a ~1.5B-active / 8B-total model trains
at 1.5B speed. That is the correct long-run response to this evidence and it is why T2.1 should be
promoted to the 5.0 design, started in parallel now rather than after 4.5 finishes.

## 4. Proposed argonne4.5

**Thesis: 4.0 proved composition is the per-token lever and then ran out of tokens. 4.5's job is to
raise the value of each token (T1.1, T1.2) and the number of tokens (T1.5), fix two defects
(Finding A, T1.4), and adopt the one architectural idea that solves a failure we have already
measured (T1.3).** Capacity via MoE (T2.1) is the separate, larger bet.

### The config

| Axis | 4.0 (1.04B) | 3.5 (2.88B) | **4.5 (decided)** | why |
|---|---|---|---|---|
| Params | 1,038,509,568 | 2,882,196,480 | **2,063,667,712** | 2.0× a4.0; floor of the flat-bottomed optimum band, so 4 GPUs suffice |
| hidden / layers | 1536 / 32 | 3072 / 24 | **2560 / 24** | aspect 107; 24L makes `LLLG` divide evenly and end on a global layer |
| heads Q/KV, head_dim | 6 / 2, 256 | 12 / 4, 256 | **10 / 2, 256** | head_dim 256 Phase-1-confirmed; GQA 5:1 → 48 KB/token KV cache for K=32 sampling |
| FFN | SwiGLU 4096 | SwiGLU 8192 | **ReLU² 10560** (iso-param w/ SwiGLU 7040) | MLP 2.75× = the sweep's one HIGH-confidence optimum; gated on the bake-off |
| Attention pattern | 1:1 interleave, window 256 (**inert**) | same, inert | **`LLLG`, sliding 1024, actually applied** | Finding A; Muse's 3:1 layout |
| Position encoding | RoPE θ=1e6 all layers | same | **RoPE θ=500k on L, NoPE on G** | targets `[[rope-extrapolation-fails-phaseb-works]]` |
| Block size | 1024 | 1024 | **2048** | reasoning traces need room; ~12% cost with the window active |
| Doc boundaries | cross-doc attention | same | **block-diagonal doc mask** | plain objective defect |
| Objective | plain CE | plain CE | **CE + RHO-1 top-60%** | the tokens/param lever |
| MTP | degenerate, off | off | **real module, depth 1, w 0.3** + boost phase | denser signal + free drafter |
| Embeddings | tied | tied | **tied** (untie = bake-off arm) | untying adds 388M (+19%) — worth measuring at 2B, unlike at 1B |
| Optimizer | AdamW 6e-4, WSD 8k/0.15, clip 0.4, fp8 | same | **unchanged** (+ µP) | the well-searched part — do not touch |
| Effective batch | 522,240 | ~1.01M | **1,048,576** | 3.5's validated value at ~this size |
| Micro-batch | 170/GPU, accum 1 | — | **probe it** (start 16/GPU × 2048) | Finding C is retracted — measure on a pinned card, don't assume |
| Grad checkpointing | on | on | **probe off** | ≈33% recompute; the useful way to spend HBM |
| Hardware | 3× H100 | 3× H100 | **4× H200 (8 if multi-node lands)** | 34 d @154B on 4; 17 d on 8 |
| Token budget | 62.1B (60:1) | ~64B (22:1) | **154B → 300B (75–145:1)** | 154B is buildable today; 300B needs the synthetic expansion |

Everything above is config-gated and defaults OFF in the code, so `argonne4.5` can still reproduce a
4.0 run byte-for-byte.

### The three things that gate the run

1. **Multi-node DDP — an accelerator, no longer a gate.** At 2.06B the run is 34 days on the 4 GPUs
   we already have, 17 if multi-node lands. The launcher is `--nodes=1 --gres=gpu:3` and the QOS
   allows 16 GPUs / 4 nodes, so it is still ~a day of work for a 2× that compounds across every
   future run — do it, just not on the critical path.
2. **Data.** 154B usable exists today (38.6B unique × 4×) and is enough to start — it is already
   3.4× 3.5's tokens/param. 300B needs the synthetic expansion (T1.5) to ~75B
   unique; that work runs in parallel and lands before the cooldown, not before the run starts.
3. **A 10B-token arch bake-off, on one pinned card.** 4 arms: (a) 3.5 arch at block 2048;
   (b) + `LLLG`/NoPE/doc-mask; (c) + ReLU²; (d) + MTP module. Plus a fifth cheap arm for
   untie-vs-tie, which at 2.06B (+388M, +19%) our 450-step null cannot answer.
   ~7 GPU-days at 2.06B — 5% of the run it protects, and the only defence against another 450-step
   trap. **Pin the GPU model:** Finding C's retraction is what happens otherwise.

## 5. How to validate — don't repeat the 450-step mistake

The arch sweep produced two retractions and two short-horizon traps (shallow-wide, untie) because
450 steps cannot see effects that need scale. Every lever here is one of those. Rules for 4.5:

1. **Minimum horizon 5–10B tokens** for anything touching the objective (RHO-1, MTP, KD). The 78M/450-step
   proxy is valid only for optimizer and shape questions.
2. **Same GPU for every arm in a comparison.** Cross-GPU noise is 0.05–0.10 CE — larger than most
   effects here, and it already caused KEY FINDING 14's retraction.
3. **Gate on downstream, not CE.** `reasoning/clean_eval.py`; never GSM8K
   (`[[gsm8k-contaminated-all-argonne-evals]]`); watch the ±2-item noise floor from
   `[[argonne4-pretrain-effective-but-code-regressing]]` — never gate on one checkpoint.
4. **Length generalization gets its own test.** T1.3's whole claim is about behavior past the training
   block. CE at block 2048 says nothing about token 8192. Measure it directly.
5. **One job at a time** (`[[experiments-one-job-at-a-time]]`), design each arm from the last.

## 6. Code in this branch

All changes are additive and default-off; a 4.0 config reproduces 4.0 exactly.

- `model.py` — per-layer rope/NoPE + attention pattern, **SDPA sliding-window mask (fixes Finding A)**,
  gated attention, ReLU² MLP, real MTP module, document-boundary masking.
- `test_a45_arch.py` — CPU smoke test: proves defaults are numerically unchanged and each new path runs.

Not started here, and deliberately: MoE (T2.1), RHO-1 scoring pass (T1.1), synthetic data build
(T1.5), µP (T1.6). Each is a separate piece of work with its own validation, and none should be
started before the flags above are proxy-checked.
