# Argonne 4.5 — what to change in PRETRAINING, and why

Written 2026-08-12, after auditing the two frontier open releases of the last 48 hours against
everything the Argonne line has actually done and measured. Branch `argonne4.5`, worktree
`/home/youzhi/ArgonneAI-4.5` (main clone stays on `argonne4.0` — other work is running).

Ground truth for both reference models comes from the **local weights** in
`/project/rcc/youzhi/toxic-models` (`toxic` alias), not from blog posts — configs, tensor shapes and
parameter names, which are considerably more informative than what either vendor wrote up.

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

### ⚠️ Finding C — the production micro-batch costs ~1.44× throughput

Same block size, same effective batch (~0.52M vs 0.59M), **44% apart**. The mechanism is chunked CE:
`loss_chunk_size 4096` over a 170×1024 micro-batch is **42.5 chunks**, over 32×1024 it is **8**. Each
chunk is a separate `@torch.compiler.disable` eager kernel with its own checkpoint recompute, and
`[[chunked-ce-is-the-throughput-lever]]` already recorded that cost scales with *chunk count*. The
big micro-batch was chosen to fill HBM; `[[sft-length-grouping-beats-hbm-fill]]` and
`[[optimize-gpu-hbm-usage]]` both say fill is not the objective here — s/step is.
**Fix: shrink the micro-batch and raise grad-accum.** Free 1.44×, no quality change.

Spend the freed HBM on **disabling gradient checkpointing** (≈33% recompute) rather than on a bigger
micro-batch — that is the useful way to satisfy the fill target.

### Throughput and calendar

Calibrated: 5.27 B tok/day measured on 3×H100 → **10.7 B tok/day on 4×H200** (×4/3 GPUs, ×1.2 H200,
×1.44 batch fix, ×0.88 for block 2048+LLLG). The `gpu` QOS ceiling is **16 GPUs / 4 nodes / 36h**, and
the launcher is `--nodes=1 --gres=gpu:3` today — so 4×H200 on one node needs **no code change** and is
+2× over what a4 actually used.

| tokens | tok/param | 4× H200 | 8× H200 | unique corpus needed @4× repetition |
|---|---|---|---|---|
| 100B | 96 | 9 d | 5 d | 25B |
| 200B | 193 | 19 d | 9 d | 50B |
| **300B** | **289** | **28 d** | **14 d** | **75B** |
| 400B | 385 | 37 d | 19 d | 100B |
| 600B | 578 | 56 d | 28 d | 150B |

**Compute is not the constraint. Data is.** Our 38.6B unique corpus at ≤4× repetition is 154B usable
= 148 tok/param at 1B — still 2–3× short of the target, and no amount of GPU fixes that.

### The size decision: hold 1.0387B

At a fixed corpus, tokens/param scales as 1/N, so smaller looks better on paper. It isn't, here:

- **We cannot fill even 1B.** At 148 tok/param we are data-limited, not capacity-limited. Shrinking to
  0.6B raises the ratio to 257 but does not add a single token of information, and pushes the
  embedding to **33% of the model** (Qwen3's 151,936 vocab is a fixed 233M at hidden 1536).
- **Growing is strictly worse.** 1.5B at the same corpus = 103 tok/param, and 61 days for 500:1.
- **Holding the exact a4.0 param count buys a controlled comparison.** Every 4.5 lever then reads
  against a4.0's real downstream numbers with size held fixed — and our history
  (two retractions, a cross-GPU artifact, 450-step traps) says confound control is worth more than a
  few percent of any scaling curve.
- The thesis of 4.5 is *value per token*, not parameter count. Changing both at once measures neither.

Conveniently the FFN swap is **exactly iso-param**: SwiGLU inter 4096 (3 matrices) and ReLU² inter
6144 (2 matrices) are both 1,038,509,568 params at hidden 1536 / 32L / 6Q / 2KV / head_dim 256. So
ReLU² can be tested with depth, width and attention all held constant.

## 4. Proposed argonne4.5

**Thesis: 4.0 proved composition is the per-token lever and then ran out of tokens. 4.5's job is to
raise the value of each token (T1.1, T1.2) and the number of tokens (T1.5), fix two defects
(Finding A, T1.4), and adopt the one architectural idea that solves a failure we have already
measured (T1.3).** Capacity via MoE (T2.1) is the separate, larger bet.

### The config

| Axis | 4.0 | **4.5 (decided)** | why |
|---|---|---|---|
| Params | 1,038,509,568 | **1,038,509,568 — identical** | controlled comparison; we are data-limited, not capacity-limited |
| hidden / layers | 1536 / 32 | **1536 / 32** | unchanged |
| heads Q/KV, head_dim | 6 / 2, 256 | **6 / 2, 256** | head_dim 256 confirmed by the Phase-1 sweep |
| FFN | SwiGLU 4096 | **ReLU² 6144** (iso-param) | 2-matrix FFN, 50% wider at equal params; gated on the bake-off |
| Attention pattern | 1:1 interleave, window 256 (**inert**) | **`LLLG`, sliding 1024, actually applied** | Finding A; Muse's 3:1 layout |
| Position encoding | RoPE θ=1e6 all layers | **RoPE θ=500k on L, NoPE on G** | targets the documented `[[rope-extrapolation-fails-phaseb-works]]` failure |
| Block size | 1024 | **2048** | reasoning traces need room; ~12% cost with the window active |
| Doc boundaries | cross-doc attention | **block-diagonal doc mask** | plain objective defect |
| Objective | plain CE | **CE + RHO-1 top-60%** | the tokens/param lever |
| MTP | degenerate, off | **real module, depth 1, w 0.3** + MTP-boost phase | denser signal + free drafter |
| Embeddings | tied | **tied** | untying adds 233M (+22%) at this vocab |
| Optimizer | AdamW, LR 6e-4, WSD 8k/0.15, clip 0.4, fp8 | **unchanged** (+ µP) | the well-searched part — do not touch |
| Micro-batch | 170/GPU, accum 1 (42 CE chunks) | **16/GPU, accum 4 @ 4 GPUs** (8 chunks) | Finding C: 1.44× free |
| Effective batch | 522,240 | **524,288** | the LR-6e-4-validated value |
| Grad checkpointing | on | **probe off** | the freed HBM's best use (~33% recompute) |
| Hardware | 3× H100 | **4× H200, single node** | no launcher change; +2× |
| Token budget | 62.1B (60:1) | **300B (289:1)** | 28 d at 4×H200; needs 75B unique |

Everything above is config-gated and defaults OFF in the code, so `argonne4.5` can still reproduce a
4.0 run byte-for-byte.

### The two things that gate the run

1. **Data, not compute.** 300B tokens needs ~75B unique at 4× repetition; we have 38.6B. The
   synthetic expansion (T1.5) is on the critical path — it is the only item that must finish before
   the main run can be worth its calendar time. Everything else can be developed in parallel.
2. **A 10B-token arch bake-off before committing 28 days.** 4 arms × 10B tokens ≈ 4 GPU-days total at
   4×H200 — trivial next to the run it protects, and the only defence against another 450-step trap:
   (a) baseline = 4.0 arch at block 2048; (b) + `LLLG`/NoPE/doc-mask; (c) + ReLU²; (d) + MTP module.
   Ship whatever (b)-(d) actually win, at the same GPU, gated on `reasoning/clean_eval.py`.

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
