# Argonne 3.5 — training-recipe revision

Argonne 3.5 is **not a new architecture**. It is Argonne 3.0's model, unchanged, trained with a
better **optimization/schedule recipe** discovered by the 50-experiment proxy search in
`experiments/` (`experiments/results/decisions.md`, "Recommended Argonne-Next recipe" = *recipe B*).

This document records (1) the evaluation that decided the recipe is worth adopting, (2) exactly what
changed and why, (3) the caveats that bound each change, and (4) the validation to run before
committing a full multi-day 2.88B run.

---

## 1. Verdict — is the recipe better than the current Argonne 3.0 base?

**Yes, net-positive — but the win is the optimization/schedule axes, not the architecture.** The
proxy search's headline (−0.46 to −0.64 nats held-out CE on a 1.03B proxy) does **not** transfer as a
number, and its literal LR (1.6e-3) must **not** be copied. What transfers is the *direction*, and —
critically — the search exposed that the **current production launcher leaves documented, low-risk
wins on the table**:

| Deficiency in the launched Argonne 3.0 | Evidence | 3.5 fix |
|---|---|---|
| `--cooldown 0` → the WSD schedule has **no decay phase**; LR is flat at peak forever (`min_lr_ratio` is dead code on that path, `pretrain.py` `if cooldown<=0: return 1.0`) | `run_full_training.sh`; `pretrain.py` lr_lambda | add a ~15% terminal cooldown (`--cooldown_frac 0.15`) |
| `--lr 3e-4` — below `pretrain.py`'s **own** validated default `6e-4` and its stated safe band `4e-4..1e-3` | `pretrain.py:44-54`; proxy LR sweep (dominant lever, monotone) | raise to `6e-4` (probe up to `1e-3`) |
| `--grad_clip 1.0` — looser than `pretrain.py`'s **own** default `0.4` | `pretrain.py:65`; proxy exp021/exp047 | `0.4` |
| `--warmup_steps 1000` (~1.1% of ~89k steps) — too short for a raised LR | proxy exp11/12/13 (LR×warmup interaction) | `8000` (~9%) |

Every 3.5 change moves a knob from a demonstrably-conservative setting toward the codebase's own
documented best practice **and** the search's high-transfer findings. A sensibly-ported recipe being
*worse* than the launched base is not plausible; the residual uncertainty is only *how much* better.

### How the search was independently checked (not taken on faith)
- **Every headline number reproduces exactly** from `experiments/results/results.jsonl` (re-parsed):
  baseline seeds 4.5176/4.5109/4.5073 (σ=0.0043), recipe 3.8807, seed-confirm 3.8907, and the
  800-step paired run (baseline 3.9952 · recipe 3.5342 · shallow-wide 3.5391). The full 46-row
  leaderboard and every recipe-table delta are arithmetically consistent.
- **Infra finding corroborated directly**: 279 lines across `report/*.out` show production trains with
  `SDPA/math ... full attention — local_attention_window=256 ... IGNORED` (flash-attn-2 absent), so
  the interleaved sliding window is inert in production today. The search found local attention
  quality-neutral at block 1024, so there is no quality cost to this — but it means "local attention"
  is not actually doing anything in the current runs.
- **Harness/eval audited**: eval is pure next-token CE (`labels=None` path, no z-loss/MTP leak),
  train/test never overlap (max train reach 59M ≪ test offset 20.6B), `block_size` pinned 1024, vocab
  pinned 151936, LR schedule identical across runs, SDPA window applied equally to both arms.

### Independent adversarial re-audit (confirms the above, and shapes the caveats)
Two further auditors were run. A dedicated bug-hunt concluded **"no defect invalidates the
baseline-vs-recipe comparison"** (the 0.637-nat gap is ~148σ and LR-dominated). A devil's-advocate
argued the port is **net-positive only if** four things hold — all four are honored here:
1. **Drop GQA ratio 2** — done (§2): near-noise gain vs production's actual ratio 3, permanent ~1.5×
   KV-cache tax at inference. Not adopted.
2. **Do not trigger the continue-stage cooldown bug** — done (§5): the cooldown is applied only to the
   pretrain stage; the continue stage keeps `cooldown 0`.
3. **Raise LR only to a scale-swept value, not the proxy's 1.6e-3** — done (§3–4): ships `6e-4`,
   gated on a probe.
4. **Re-validate grad_clip 0.4 at the swept LR** — in the validation plan (§4); grad_clip's benefit is
   coupled to a high LR, so it must be re-checked at whatever LR the probe selects.

Two honest limits on the numbers (neither changes the go decision): the single-axis "wins" were gated
against σ=0.0043 measured at the *baseline* LR, but noise at the *recipe* LR is ≈0.010 — so the small
attention-shape wins (window, head_dim) are weaker than the leaderboard's 2σ gate implies (irrelevant
here: window is excluded, head_dim 256 is already production). And the "cooldown helps" leg was never
A/B'd against `cooldown 0` in the proxy — it rests on standard WSD practice + the mechanism (production
literally omits WSD's decay phase), not on a proxy datapoint. The **LR direction** and the
**missing-decay fix** are the two robust, load-bearing legs.

---

## 2. What changed (and what did NOT)

The only **tracked** change is one small enabling addition to **`pretrain.py`** (`--cooldown_frac`);
the rest of the recipe is a set of **launcher flags**. Per this repo's convention, shell scripts are
git-ignored (`*.sh`) and kept local — so `run_full_training.sh` is **not committed on this branch**; the
exact flags to set in your local copy are in §2.2 below, and this doc is their authoritative record.
**The model architecture is byte-for-byte Argonne 3.0.**

### Changed — `pretrain.py`
- Added `--cooldown_frac` (float, default 0.0). When > 0 it overrides `--cooldown` and sets the WSD
  cooldown to `int(cooldown_frac × estimated_steps)`. Expressed as a fraction so the terminal anneal
  lands at the run's true end regardless of corpus size or how the run is sliced across wall-time
  SLURM jobs (`estimated_steps` is recomputed identically every resume). No other logic changed.

### 2.2 Changed — launcher flags (edit your local, untracked `run_full_training.sh`)
| Flag | 3.0 | 3.5 | Confidence |
|---|---|---|---|
| `--lr` | `3e-4` | `6e-4` | direction HIGH, magnitude MEDIUM (probe) |
| `--grad_clip` | `1.0` | `0.4` | MEDIUM (safe — tightening a clip cannot destabilize) |
| `--warmup_steps` (pretrain) | `1000` | `8000` | direction MEDIUM (co-tune with LR) |
| `--cooldown` → `--cooldown_frac` (pretrain) | `0` (no decay) | `0.15` | HIGH that *some* decay helps; fraction MEDIUM |

Exact flags to set (everything else in `run_full_training.sh` is unchanged from 3.0):

```bash
# --- pretrain.py torchrun block (exports the base model; gets the terminal WSD anneal) ---
    --lr 6e-4 \
    --grad_clip 0.4 \
    --warmup_steps 8000 \
    --schedule wsd \
    --cooldown_frac 0.15 \      # replaces the old  --cooldown 0
    --min_lr_ratio 0.1 \

# --- continue_pretrain.py torchrun block (2nd stage; NO cooldown — see §5) ---
    --lr 6e-4 \
    --grad_clip 0.4 \
    --warmup_steps 0 \
    --schedule wsd \
    --cooldown 0 \
```

### Deliberately NOT changed (kept at Argonne 3.0 defaults — validated or out-of-scope)
- **Architecture** — `hidden 3072, 24 layers, 12Q/4KV, head_dim 256, SwiGLU 8192, tied embeddings,
  qk/v/sandwich norm, softcap 15, rope 1e6`. The search validated all of these as near-optimal (e.g.
  qk_norm OFF *diverges* at high LR; head_dim 256 is a clean interior optimum; MLP 2.67× is a floor).
- **GQA ratio 2 (12Q/6KV) — NOT adopted.** The proxy's +0.013 win was measured vs the proxy's ratio 4
  baseline, but production is already ratio 3; the true ratio-3→ratio-2 gain is ≈0.006 (below the
  0.0085 noise gate), single-seed, and it raises the inference KV-cache ~1.5×. It is the one genuine
  *architecture* lever, but the evidence is too thin to justify changing the arch (and the search's
  own one-line summary is "keep the architecture, change the optimization"). Left as an optional,
  probe-gated experiment (see §4).
- **Short-horizon artifacts, explicitly excluded** (the search proved these vanish with more data):
  shallow-wide depth (edge gone at 800 steps), tiny local attention window (64), depth-scaled residual
  init. Production depth/width, window ≥256, and the large residual init are kept.
- **Continue-stage cooldown** — kept at 0 on purpose (see §5).

---

## 3. Why NOT the proxy's literal numbers (the transfer caveats)

The proxy trains at ~0.057 tokens/param (450 steps, 59M tokens); production trains at ~7 tokens/param
(20.8B tokens) — **~125× more trained**. In that under-trained proxy regime a higher LR wins almost by
construction, so:
- **LR magnitude is a short-horizon artifact.** The proxy optimum (1.6e-3) is inflated by both the
  short horizon and the narrower proxy width; the at-scale optimum is lower. The advantage already
  shrank 0.637→0.461 going from 450→800 steps. We port **6e-4** (a modest, validated 2× raise), not
  1.6e-3, and gate the final value on a probe. Supporting evidence that 6e-4 is *conservative*, not
  aggressive: the real effective batch is only **233,472 tokens** (`38×3×1024×2`), ~1.78× the proxy's
  131K — *not* the ~1M that `pretrain.py`'s LR comment assumes. Sqrt-scaling the exp_317-validated
  `6e-4@24K-batch` to this actual batch lands near **1.9e-3** — so the probe should genuinely explore
  toward the top of the `4e-4..1e-3` band, while watching for *late* instability the proxy can't see.
- **Warmup fraction is a short-horizon artifact.** 10–15% of ~89k steps would be 9k–13k warmup steps;
  8000 (~9%) is a defensible, cheaper compromise — co-tune with the LR probe.
- **Cooldown-vs-zero was never directly A/B'd** in the search (every proxy run had a cooldown; the
  A/B'd points were 10/20/40%). "Add a terminal decay" therefore rests on standard WSD practice + the
  mechanism (production literally omits WSD's decay phase), not on a direct proxy datapoint. It is
  low-risk and the fraction is capped modest (a *longer* 40% cooldown hurt in the proxy).

---

## 4. Validation before the full run (do this first)

1. **Production-scale LR probe (highest priority).** Runs at `{6e-4, 8e-4, 1e-3}` with the 3.5 schedule
   (warmup 8000). Make them **long enough to catch *late* divergence** (attention-entropy collapse,
   logit-norm growth, loss spikes at 10k–40k steps) — a short "higher-was-better-at-2B, ship it" probe
   is exactly the trap, because in a short run a higher LR wins by construction. Watch `nan_loss_steps`
   and val loss to the end of the probe. Pick the best *stable* LR; set both `--lr` flags to it.
   `pretrain.py`'s comment warns linear-scaled 2.5e-2 "would diverge" — stay ≤1e-3.
2. **Re-validate `grad_clip 0.4` at the swept LR.** Its benefit is coupled to a high LR (it curbs the
   larger updates a high LR produces). If the probe selects an LR near the low end, re-check 0.4 vs 1.0
   at that LR rather than assuming the proxy's high-LR result carries over. (Tightening a clip can't
   destabilize, so 0.4 is safe either way; this is about confirming it still *helps*.)
3. **(Optional, likely skip) GQA ratio 2 A/B** at the chosen LR: set `NUM_KV_HEADS = 6` in `pretrain.py`
   / `continue_pretrain.py` (12%6==0, head_dim 256 preserved) and compare vs 12Q/4KV. The prior is
   negative — the prod-relevant ratio-3→ratio-2 gain is ≈0.006 (sub-noise) against a permanent ~1.5×
   KV-cache tax — so only adopt if it *clearly* clears noise at scale.
4. Confirm the cooldown fires: the startup log prints `Cooldown: N steps (frac=0.15, ~15.0% of run)`;
   ensure the run actually reaches its planned end (the terminal anneal only helps if the last slice
   completes; if a run is stopped early it simply never anneals — no worse than 3.0's constant LR).

---

## 5. Known limitation — continue-stage cooldown

`continue_pretrain.py` steps its LR scheduler on the **cumulative** `global_step` (carried across the
pretrain→continue boundary) while `estimated_steps` is stage-local. A nonzero cooldown in the continue
stage would therefore compute `cooldown_start = estimated_steps − cooldown` against a step counter that
is already far past it, collapsing the LR to `min_lr` on the first continue step (the entire continue
stage would silently train at `0.1×LR`). That is *why* production used `cooldown 0`. This is a **latent
bug in `continue_pretrain.py` that predates 3.5**; 3.5 **avoids triggering it by design** — the terminal
WSD anneal is placed at the **end of pretrain** (fresh, stage-local scheduler via `reset_schedule`; and
pretrain is the stage that exports the base model), and the continue stage is left at `cooldown 0`. So
after pretrain cools to `min_lr` and exports, the continue stage's `cooldown 0` lambda returns 1.0 and
the LR returns to full `6e-4` for continued pretraining (a standard WSD re-warm), never stuck at min.
**Do not** set a nonzero `--cooldown`/`--cooldown_frac` on the continue stage until the scheduler is
rebuilt stage-local (reset `last_epoch` to 0) on the first `reset_schedule` slice — deferred as a
follow-up, not needed for the base model.

---

## 6. Files in this branch vs argonne3.0
- `pretrain.py` — added `--cooldown_frac` and wired it into the WSD scheduler + startup log.
- `run_full_training.sh` — **NOT tracked** (repo-wide `*.sh` git-ignore; launchers stay local). The
  recipe flags to apply to your local copy are the authoritative record in §2.2 above.
- `ARGONNE3.5.md` — this document.
- Everything else (incl. `model.py`, `continue_pretrain.py`) is unchanged from argonne3.0.
