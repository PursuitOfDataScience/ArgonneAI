# a5 campaign: measurement rules, noise floors, and retracted claims

This file exists because the campaign's most expensive mistakes were **comparator errors**, not
compute errors. Every retraction below was self-inflicted and each one initially looked like a
result. Read this before trusting any Δ in [`FINDINGS.md`](FINDINGS.md).

---

## The noise floor is not a constant

⚠️ **σ is both horizon-dependent AND config-dependent:**

| steps | σ |
|---|---|
| 145 | 0.0287 |
| 242 | 0.0060 |
| 282 | 0.0010 |

⛔ **A single config's floor was quoted for ~30 comparisons and overstated significance by 6.1×.**

⚠️ **One seed pair does not measure a noise floor.** The floor was originally set from ONE seed pair
(σ_diff 0.0129); a second pair **tripled it to 0.0297**.

⚠️ **A same-seed rerun does NOT reproduce.** σ ≈ 1e-3, which equals the 3-seed σ. Read
`train_loss_ema` alongside `tgt`, it is free and it separates optimization noise from transfer.

⚠️ **σ_run for the KD family is ≥0.037 in LEVEL**: about **2.5× the value imported** from a non-KD
config. Importing a floor across families is the same error as importing it across horizons.

### The saving grace for KD: the seed shift is common-mode

Because the seed shift is common-mode, **same-seed differences are stable** even when levels are not.
Two independent pairs gave `doc_mask` −0.0480 and −0.0437: agreeing to 0.0043.

⛔ That effect was first called "straddling zero" **off a cross-seed comparison**: the campaign's own
comparator error, committed while writing a correction about noise floors.

---

## ⭐⭐ There are TWO floors, and the campaign has only ever measured one of them

**Init floor** = spread across seeds at fixed data order. **Run-to-run floor** = spread when
*nothing at all* changes: same seed, same config, same node. They are different quantities, and the
second is the bar that a **same-seed** comparison must clear. Almost every floor this campaign has
quoted is the first kind.

⛔ **The run-to-run floor, where it exists, is 3× the init floor.** `a21_variance_check` re-ran `a12`
**verbatim**: configs identical but for `id`/`note`, same seed 444, same node midway3-0426, same 569
steps, and landed:

| | tgt | edu | math | code |
|---|---|---|---|---|
| **verbatim rerun**, 2.064B / 35.0M tok / block 2048 / fp8=1 / H100 NVL | **0.1130** | 0.0422 | 0.1095 | **0.1872** |
| init floor (3 seeds), same cell | 0.0515 | 0.0369 | 0.0377 | 0.1099 |

`a21`'s own note called this in advance: *"land near 4.99 and the resolution is ~0.37, which voids
most of this campaign's verdicts and I rebuild from measured noise instead of assumed noise."* It
landed at 4.8393, so the resolution at that operating point is **~0.11, not the 0.028 that had been
quoted ever since**, and the rebuild never happened.

**Why a fixed seed does not imply a fixed result:** non-deterministic backward atomics, cuBLAS /
Inductor kernel selection, and, in that cell specifically: **fp8 per-tensor scaling** (the scale
factors track observed dynamic range, so any divergence compounds) plus **3-rank NCCL reduction
order**. The current n3xx cell has `fp8=0` and `world=1`, which removes the two largest sources, so
it *should* be far tighter. "Should" is the word that produced the 0.028 error, so `n318` measures it.

⭐ **Only 1 of 34 cells has a run-to-run floor at all**: found by extending `build_index.py`
(2026-08-23), which now computes both floors per cell mechanically and labels them differently. Rule
#1 said never import a floor; the tool now makes the in-cell number impossible to miss, and prints an
explicit warning for cells that have neither.

⚠️ **Where this bites right now:** the KD verdict rests on `n313` vs `n315`, which is a **same-seed**
comparison (both 444, −0.0700 tgt). The seed floors from `n316`/`n317` are the wrong bar for it. The
right bar is the run-to-run floor in that cell, which is unmeasured: hence `n318`, a verbatim rerun
of `n313`.

## ⭐⭐⭐ THREE floors, not two: and the third is the one that kills cross-cell claims

Measured in the same cell (2.06B / block 1024 / 146 steps / A40 / fp8=0), each strictly larger than
the last:

| floor | tgt | code | bounds what |
|---|---|---|---|
| **run-to-run** (verbatim rerun, same seed) | 0.0073 | 0.0163 | a LEVEL comparison at fixed seed |
| **init** (across seeds, 4 draws) | 0.0465 | 0.1360 | a LEVEL comparison across seeds |
| **effect-size reproducibility** (same contrast, two seeds) | **0.0125** | **0.0655** | whether an EFFECT SIZE is bigger in cell A than cell B |

The third one is new (2026-08-23, `n330`) and it is the one that bites. It is **1.7× / 7.9× / 4.6× /
4.0×** the run-to-run bar on tgt / edu / math / code. It killed a finding that had cleared its own bar
by 3.25×: "code's α benefit shrinks with horizon" was a legitimate seed-444 difference-of-differences,
but seed 445 put the 146-step effect at −0.1576 against seed 444's −0.0922, straddling the 73-step
−0.1605. The effect size varies as much across seeds as it did across the horizon.

**Rule: a claim that an effect is LARGER in one cell than another requires the contrast replicated at
a second seed, even when every arm in it clears its own bar.** Signs are robust at one seed;
magnitudes and magnitude *comparisons* are not.

## ⚠️ Rule #1 is easiest to break in the arm that is ABOUT horizon

2026-08-23: I designed a horizon test for the α lever (73 vs 146 steps) and wrote `BAR: 0.0073` into
both configs: a floor measured **at 146 steps**. The arm exists to detect a horizon effect, and its
bar was imported across the very axis under test.

Two reasons this is worse than ordinary floor-importing:

1. **σ grows at shorter horizons.** The recorded ladder gives 0.0287 @145 steps, 0.0060 @242, 0.0010
   @282, so a 73-step floor is likely *larger* than the 146-step 0.0073, and an imported bar is
   optimistic in the direction that manufactures findings.
2. **The quantity at stake is a difference of differences** ("did the α effect shrink?"), whose noise
   is √2× a single difference's. Under-stating the bar there is how `n314`'s interaction claim came to
   be called "marginal" when it was not resolved.

**Discipline: a horizon pair needs THREE arms, not two**, the two conditions plus a verbatim rerun at
the new horizon to measure its own floor. Any effect quoted before that third arm lands is provisional
and must be labelled as such.

### …and the cost was measured, not argued (2026-08-23)

`n329` supplied the 73-step floor. It is **tgt 0.0342 (4.68× the 146-step 0.0073), edu 0.0657
(82.1×), math 0.0236 (5.02×), code 0.0132 (0.81×)**. Re-scoring the horizon test with the imported
bar vs the measured one: the imported bar turns the `tgt` change (+0.0125) from **0.36×, not
resolved** into **1.7×: "resolved"**, i.e. an invented finding, and inflates math's significance 5×.
Code's verdict survives either way.

⭐ **Floors are not uniformly larger at shorter horizons: code's is SMALLER (0.0132 vs 0.0163)** while
tgt/edu/math grow 5-82×. A single scalar bar per cell throws that away: measure per domain.


## ⭐ A cheap early-warning that is 3-for-3: the matched-step training loss

For any arm that changes only an optimiser hyperparameter (same seed, same data order, same step
count), the **loss at step 25 of 73** predicts the final verdict at ~1/3 of the arm's cost:

| arm | step-25 loss | vs the best reference (7.0538) | final |
|---|---|---|---|
| `muon_lr` 0.08 | **7.0538** | n/a | best |
| `muon_lr` 0.04 | 7.0936 | above | worse ✓ |
| `muon_lr` 0.16 | 7.1588 | above | worse ✓ |
| `adam_lr` 4e-3 | 7.2575 | above | worse ✓ |

Why it works here and should not be over-extended: these arms share seed **and** data order, so the
trajectories are directly comparable point-by-point: the comparison is not available for arms that
change the data mixture, the horizon, or the loss function. ⛔ In particular it is **not** valid across
an α change or a KD on/off change, because those alter what the training loss *measures* (see the
`train_loss_ema` trap below).

**Use it to abort or to pre-register a threshold, not to replace the arm**: the final `tgt` is still
what gets quoted.

⛔⛔ **AND IT FAILED ONE ARM LATER. It is 3-for-4, and the failure is instructive.** `adam_lr` 1e-3 read
step-25 loss **6.9729: below the 7.0538 reference, i.e. predicting BETTER**, and delivered
tgt **+0.0251** with code **+0.0860 (6.5× bar, worse)**. Its `train_loss_ema` was the **lowest of the
three** (6.0624 vs 2e-3's 6.0661) while its `tgt` was worse: **lower training loss, worse transfer.**

The scope limit I wrote above was too narrow. The real condition is: **the diagnostic is valid only
when the knob does not trade optimisation against transfer.** An LR on the *embedding/norm* group does
exactly that: a gentler embedding LR disturbs the representation less (lower train loss) while
learning less that transfers. Mixture / horizon / loss-function changes are only the obvious cases.


**⭐ AND HERE IS THE RATIO, MEASURED (2026-09-16, INFRA §M+349).** The trap above is qualitative; this
prices it. Over the **verified** matched-config draw set at h1600 (`_h1600_by_draw_floor`: n1157,
n1162, n_dfill4441, n_dfill7717, n_dfill9901: config diff confirmed to be `data_seed`/`id`/`note`
only), the same five runs give
**sd(`tgt`) = 0.00371** against **sd(`train_loss_ema`) = 0.0475**, i.e. `train_loss_ema` is
**12.8× noisier**. Gating a lever on it therefore needs an effect ~13× larger than gating on `tgt`,
which is why it has produced inverted reads. Held-out CE bins split the same way: edu 0.0017 and math
0.0024 are ~10× quieter than code2 0.0217 and codetask 0.0180, so a code-axis claim needs a much
bigger effect than an edu one.
⚠️ Computed on the VERIFIED set on purpose: my first pass used an ad-hoc five including `dfill5337`
and `dfillrep`, which are not in the config-verified group, and got 0.0391/0.00272 (14×). Same
conclusion, wrong denominator: the registry's set exists because the config diff was checked.

Corollary, which is the campaign's older rule arriving from a new direction: **read `train_loss_ema`
next to `tgt`, and when they disagree, believe `tgt`.** A knob that improves the training loss and
worsens validation CE has been found twice now (α=1.0 for a different reason, and this).

## ⭐⭐ `tgt` is an UNWEIGHTED mean: the wrong scorecard for a code-targeted arm

`tgt = mean(edu, math, code)` (probe.py:488), so the **deficit domain carries 1/3 weight**. But the
recorded absolute deficits vs Qwen3-0.6B-Base are **edu +0.071 / math +0.093 / code +0.497**: code is
~5× the others and is the entire reason this line exists.

⚠️ **CAVEAT added 2026-09-16 (INFRA §M+267/§M+269): these three absolute numbers may each be inflated by up to ~0.03.** Four scripts, including two CE probes, imported `ENABLE_INTERLEAVED_LOCAL_ATTENTION` from `continue_pretrain`, where it was `True/256` until 2026-09-15 23:4x, and it is not yet established which tool produced this table. A measured window-on/off toggle on the finished weights costs **+0.0278 CE mean (edu +0.032 / math +0.019 / code +0.032)**. The offset is near-uniform, so the **ratio**, code ≈5× the others: is unaffected and in fact strengthens if the offset is removed; only the absolute values are uncertain. Quote as ≈+0.07 / ≈+0.09 / ≈+0.50 until the val-bin on/off run settles it.


Worked example (the 35/30/35 mixture arm): code −0.4610, edu +0.2549, math +0.0968.

| scorecard | effect |
|---|---|
| `tgt`, equal weight | **−0.0364**: 1.1× bar, "marginal" |
| deficit-weighted (0.107/0.141/0.752) | **−0.3056**: 8.4× larger |

**Always report the three domains separately, and both scorecards, for anything aimed at code.** A
`tgt`-only read would have filed a 35×-bar code win as marginal. Conversely, a knob that improves `tgt`
while degrading code (e.g. `muon_momentum` 0.90) should be rejected, and the deficit weighting is the
principled reason rather than an instinct.

## ⛔ Derive a step rate from the SAME step count, never by scaling

2026-08-23: a 146-step Muon arm was sized from a 73-step arm's 8.25 s/step. The real rate at 146 steps
was **8.80 s/step (+6.7%)**, so the arm hit `wall_guard` at **142 of 146 steps** and its comparison
against a 146-step control was silently contaminated by a 2.7% token deficit. A 146-step Muon arm in
fact needs ~1,655 s of the 1,680 s cap: **25 s of margin, i.e. it does not reliably fit.**

Two rules:
1. **Size an arm from a run at the same step count.** Per-step cost is not constant across horizons
   here (compile amortisation, checkpoint cadence, allocator state).
2. **When an arm truncates, prefer a MATCHED CONTROL over an arithmetic correction.** Correcting
   requires an assumed slope, which is the ingredient behind this campaign's one retraction. A control
   at the truncated step count costs one slot and needs no assumption.

⚠️ And check the sign of any correction against the prose that motivates it. The correction above was
first computed as −0.8243 when the reasoning ("the truncation *understates* Muon") demands a value more
negative than the measured −0.8472. The right answer is **−0.8701**.

## ⛔ FLOP ratios predict wall clock ONLY at fixed shape

Two predictions from the same FLOP model of the optimiser step, one right and one wrong:

| change | predicted | measured | shapes |
|---|---|---|---|
| `ns_steps` 5 → 3 | +5.2% throughput | **+5.0%** ✅ | unchanged |
| ReLU² under Muon (MLP 7040→10560) | +4 to +5% **wall** | **−0.4% wall, +2.0% tok/s** ⛔ | changed |

NS FLOPs genuinely rise 32% in the second case (51.7 → 68.3 TFLOP, 20.4% → 26.9% of a 253.6 TFLOP
step) and it costs nothing, because a (10560,2560) matmul uses the card better than a (7040,2560) one.
**Kernel efficiency varies with shape and a FLOP count does not capture it.** Scale work at fixed
shape and the model predicts; change the shape and it does not.

## ⛔ Do the division before naming the mechanism (2026-08-23)

`ns_steps` 5 → 3 removed 40% of the Newton-Schulz iterations and bought **+5.0%** throughput. I
concluded the 14.2% Muon penalty was therefore mostly *loop overhead* and built a batched
implementation to recover it. It was exact (tgt agreed to 0.0005 over a full 73-step run) and **4.0%
slower**.

The arithmetic that refutes the inference takes one line: if the penalty is pure NS work, removing 2 of
5 iterations leaves 14.2% × 3/5 = 8.5%, and throughput rises by 1.142/1.085 − 1 = **+5.2%**: the
measured value. Independently: NS is **51.7 TFLOP** against a **253.6 TFLOP** step = **20.4%**, and
3,360 kernel launches × 10 µs is **0.26%** of a 13 s step.

**The measurement was consistent with the simplest model all along; I named a mechanism instead of
testing the null.** Before attributing a cost to overhead, compute what the obvious cause predicts.

## ⚠️ "Untested" is not "too low"

2026-08-23: `muon_lr` 0.04 → 0.08 paid −0.0883 because the recorded value was tuned for a different
parameter split. I then applied the identical argument to `adam_lr` 2e-3 → 4e-3 and predicted a win; it
came back **+0.0579 tgt, worse on three of four metrics.** The principle ("inherited hyperparameters in
a changed configuration need testing") was sound and had already paid once. The *direction* was an
unforced addition to it. Test the knob; do not assume which way it moves.

## Coarse-grid behaviour is asymmetric

⚠️ Coarse effects **inflate 1.6-2.8×**. But coarse **NULLS fail in both directions**: a null at low
resolution is not evidence of absence. Do not treat a coarse null as a decision.

---

## Retracted / corrected claims

| claim | what was wrong | status |
|---|---|---|
| "the embedding drift converges to parity (36%)" | computed as a **MEDIAN over an 8.7× monotone DEPTH RAMP** | **RETRACTED.** True value ~6%. **When a claim rests on a median, print the distribution.** |
| KD is width-dependent | compared a **72-step arm against 121-step arms** | **withdrawn**: a fake width effect, caught before publication |
| dCE/doubling = −0.87 (from probes) | 4-6× too steep | use the **live run's slice logs: −0.21** |
| mixture 50/25/25 wins | short-horizon artifact; **reversed at 282 steps** (+0.034, 30σ) | **REVERSED** |
| "5e-6 catastrophe" | Adam moments put g_eq at 0.006-0.054 | **retired** |
| the MTP null | tested the **DEGENERATE** variant, at 450 steps | not a test of MTP |
| the KD null | 1.7B teacher at only **78M tokens** | superseded; KD is now the biggest non-optimizer lever |
| a4.5 size (first two calls) | both were **comparability arguments**, i.e. wrong reasoning | size must come from the **fixed-compute optimum** (2.4-3.8B, flat bottom) |

---

## Errors of reasoning worth not repeating

⛔⛔ **The forward-only memory error, made TWICE**: the second time in the arm immediately after
writing the rule down. **Count the BACKWARD pass.** The real fix was not a bigger estimate but
algebra: α-mixed KD ≡ soft-target CE ⇒ one softmax.

⛔ **A ladder of horizons defends against noise, not against a bias shared by every rung.** The
mixture ladder had every rung agree, and every rung was wrong.

⛔ **A null does not generalise out of its regime.** (Established on the reasoning line, applies
here: `--exclude-terminators` read as a null, but was measured against a hazard-*matched* teacher, 
i.e. precisely the regime where the mask is unnecessary.)

⛔ **Never price a lever by an exhausted family's best value.** On the reasoning line a lever was
priced at +1.2 because that was the best of 13 exhausted methods; the next arm hit +4.85.

---

## The `tgt` weighting defect: audited across all 96 arms (2026-08-22)

`tgt` is the **UNWEIGHTED MEAN** of edu/math/code: verified for all 92 `ok` arms carrying per-domain
CE (max deviation 8.9e-16; against the incumbent 50/30/20 objective it differs by a median of 0.120).
So every campaign finding was judged on a metric that values code at 1/3, while the model is trained
toward 50/30/20. On the resumed campaign's `n303` arm that gap **inverted a conclusion's sign**
(equal-weight −0.0823 win → 50/30/20 +0.1476 loss), so the whole recovered campaign was re-scored.

**Result: the defect does NOT contaminate the recorded campaign.**

| check | result |
|---|---|
| matched cells where the winner changes | **0 / 10** |
| within-cell pairwise comparisons flipping sign | 11 / 881 (1.2%) |
| …of those, with an equal-weight Δ above the floor (i.e. reportable) | **0** |

Every sign flip is inside noise. Nothing in [`FINDINGS.md`](FINDINGS.md) needs revising for this.

⭐ **Why it was latent for 92 arms, and why it went live immediately on the 93rd.** The optimizer
swap, KD, ReLU² and LR arms move edu/math/code *in the same direction*, so every positive weighting
agrees on the sign: the choice of weights only rescales the magnitude. A **data-reallocation** arm is
the one kind that moves domains in **opposite** directions, and the mixture ladder pinned edu at 50%
on every rung, so the campaign never generated a heterogeneous-effect comparison. The defect went
live on the first arm that broke the ladder's shape.

⭐ **The rule this buys, which is cheaper than re-auditing:** check the objective weighting when the
intervention **reallocates data**. Optimizer and architecture arms are immune by construction. This is
the same lesson as "a ladder is not a control" (rule 4 below) seen from the metric side: a ladder
that varies one thing cannot expose a defect in how the *aggregate* trades that thing off.

---

## Rules to carry into the next campaign

1. **Measure σ in the configuration and at the horizon you are comparing at.** Never import it.
2. **Same-seed differences, not cross-seed levels**, when σ_level is large but the shift is
   common-mode.
3. **Report the paired number first**, and report the spread of the units you paired over.
4. **A ladder is not a control.** Add a rung that differs in *kind*, not just in size.
5. **Verify the gate before trusting a null**, e.g. the teacher-CE gate for KD. A misconfigured
   pipeline produces a clean, believable zero.
6. **When a summary statistic carries the claim, print the distribution.**
7. **After changing what an optimizer covers, re-test the hyperparameters of everything it no longer
   covers.** That is where the 65σ result was hiding.
