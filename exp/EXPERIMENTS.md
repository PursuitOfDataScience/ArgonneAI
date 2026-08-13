# argonne4.5 probe campaign — 30 arms, 3× H100, ≤30 min each

Sequential, one job at a time, each arm designed after reading the previous arm's result.
Nothing pre-queued, nothing auto-submitted. Completed 2026-08-13.

## Setup

| | |
|---|---|
| Hardware | **3× H100 NVL 94 GB**, hard-pinned to `midway3-0426` |
| Model | the decided a4.5 size: hidden 2560 / 24L / 10Q-2KV / head_dim 256 / **2,063,667,712** |
| Data | production mix `edu 50 / math 30 / code 20` (`/project/rcc/youzhi/data/argonne4_pretrain`) |
| Metric | **TGT** = mean pure next-token CE over held-out `val_edu` / `val_math` / `val_code` |
| Standard budget | 35 M tokens, block 2048, micro 8 × accum 5 × 3 = 245,760 tok/step |
| Files | `probe.py` (harness), `run.sh` (sbatch, untracked per repo policy), `configs/`, `results/`, `logs/` |

**H100 not H200, and why.** Both H200 nodes surviving the exclusion policy (0600/0601) were 8/8
allocated by other users' multi-day jobs; the first free 3-GPU slot was ~7.5 h out and would have
had to be re-won for each of 30 arms. `midway3-0426` was idle — and it is one of the two nodes
argonne4.0 actually trained on, so throughput here is directly comparable to its measured
5.27 B tok/day. Arm 1 confirmed that: production scaled by 1/N predicts 30,679 tok/s, measured 32,205.

---

## ⚠️ Read this before using any number below

**The seed noise is σ ≈ 0.047 TGT (3 seeds), not the 0.008 I used for most of the campaign.**
Seeds 444/445 landed 0.0078 apart and I generalised from that pair for 20+ arms. Seed 446 came in
0.085 away from seed 444. Every "N× noise" multiple quoted mid-campaign for a single-arm comparison
was overstated by roughly 6×. The corrected yardstick for two single-seed arms is
σ_diff = 0.047·√2 = **0.066**.

Throughput is unaffected — it varies only 0.9% across arms at fixed config — so **all systems
results stand exactly as measured. It is the CE-based architecture claims that soften.**

| finding | Δ TGT | σ | verdict |
|---|---|---|---|
| full recipe @48M vs old recipe | −0.429 | 6.5 | **ESTABLISHED** |
| token scaling 35M→48M | −0.387 | 5.9 | **ESTABLISHED** |
| full recipe @35M vs old recipe | −0.366 | 5.5 | **ESTABLISHED** |
| LR 1.2e-3 (worse than 6e-4) | +0.346 | 5.2 | **ESTABLISHED** |
| warmup 2→14 steps @ LR 6e-4 | −0.184 | 2.8 | **ESTABLISHED** |
| stack (gate + untie) | −0.141 | 2.1 | **ESTABLISHED** |
| LR 6e-4→3e-4 (at warmup 2) | −0.140 | 2.1 | **ESTABLISHED** |
| untie alone | −0.093 | 1.4 | suggestive |
| gated attention alone | −0.057 | 0.9 | not established |
| ReLU² on stack @48M | +0.040 | 0.6 | not established |
| doc-mask (iso-token) | −0.034 | 0.5 | not established |
| real MTP module | +0.033 | 0.5 | not established |
| cooldown 0.15→0.30 | −0.011 | 0.2 | not established |
| block 2048 vs 512 | −0.006 | 0.1 | not established |

---

## The recommendation for argonne4.5

**Ship (systems — measured, seed-independent):**

1. **`loss_chunk_size = 0`** and **gradient checkpointing at `stride 2`**. 32,205 → 42,016 tok/s,
   **+30.5%**, HBM 55% → 84%. ≈ **7.9 days off the 34-day production run**. The single largest
   result in the campaign. Chunked CE runs `@torch.compiler.disable` *and* recomputes each chunk in
   backward; removing it is worth 6× more than removing gradient checkpointing was.
2. Keep `--mem=48G` (19.1 GiB per-task MaxRSS badly understates the 3-rank cgroup + memmap need).

**Ship (schedule — large and established):**

3. **Warmup ≥10% of steps.** Worth −0.184 at LR 6e-4. The probe's inherited 2-step warmup was
   pathological and was distorting every LR reading taken before arm 15.
4. **Do not raise LR above 6e-4.** 1.2e-3 costs +0.346 — equivalent to discarding 26% of the token
   budget. The arch sweep's 1.6e-3 preference does **not** transfer from 1B/131k-batch to
   2.06B/246k-batch.

**Adopt with a caveat (architecture):**

5. **Gated attention + untied embeddings**, together **−0.141 (2.1σ)** and 94% additive. The *stack*
   is established; the individual components are not (0.9σ and 1.4σ) — adopt them as a pair, and do
   not cite either number alone. Untie is nearly free in FLOPs; the gate costs ~3.6% throughput.
   Note this **overturns the 450-step arch sweep's untie verdict** (+0.089 loss at 1B → −0.093 win
   at 2.06B), and the difference is size and operating point, not horizon — my run is *shorter*.

**Do not ship:**

| lever | why |
|---|---|
| LLLG sliding window | +0.038 CE **and** −14.6% throughput. No flash-attn-2 here, so the window forces an explicit mask into SDPA and abandons the fused `is_causal` kernel. Windowing *costs* on this cluster. |
| NoPE global layers | worse at every length tested; my "improved slope" claim was retracted (arm 8's control has identical position handling and shows the same slope shift). |
| intra-document masking | genuine iso-token win (−0.034) but −27.5% throughput. At iso-compute the baseline wins by **0.353**. Refuted on cost, not on merit. |
| real MTP module | +0.033 CE, −7.8% throughput, +0.127 iso-compute. Its loss backprops into the trunk via `h_prev`, so a separate module *reduces* but does not remove competition with t+1. |
| ReLU² FFN | **unstable**: +0.032 standalone, −0.030 on the stack @35M, +0.040 on the stack @48M. Three measurements, three answers, drifting away from a win as tokens grow. |

**Unresolved — the probe cannot answer these:**

- **Block size.** 512/1024/2048 are within 0.006 at 2048-eval while differing by **+63%** in
  throughput. A 512-trained model loses nothing at 2048 context, which means the model is not using
  context at 35 M tokens and *the instrument is blind to this axis*. The +37%/+63% throughput
  numbers are real; the "at equal quality" half is not transferable.
- **Anything context-dependent**, for the same reason — which is why the sliding-window and NoPE
  refutations are downgraded to "lost on cost; intended benefit unmeasurable at this horizon."
- **Batch transfer.** Arm 25 doubled the batch at fixed tokens, which halves the steps (142→71) and
  confounds the two. It does show step count dominates batch at this scale (+0.466), which is itself
  a caveat: a 142-step probe is *step-limited* and production runs ~147,000 steps, so the LR/warmup
  **direction** transfers but the **values** should not be copied.

**Instruments built (reusable):**

- **Token exchange rate: dTGT/d(ln tokens) = −1.13 to −1.22** (3 points: 20/35/48 M). Converts
  throughput into quality units and is what turned doc-mask from a win into a decisive loss.
- **Parameter exchange rate: dTGT/d(ln N) = −1.02** (from arm 18's iso-param control). Params and
  tokens have nearly equal marginal value here, and Chinchilla's N-term underpredicts the parameter
  effect ~11×. Independent support for the a4-vs-3.5 evidence that params are underweighted.
- Multi-length eval, EOS-derived document ids, and a 21-test CPU parity suite.

---

## Bugs and near-misses found

| | |
|---|---|
| `set -u` + conda `activate.d` | job died in 2 s (`ADDR2LINE` unbound). |
| `--mem=26G` | host-RAM OOM at 22 s. `sacct` MaxRSS is **per-task**; the cgroup covers 3 ranks + memmap page cache. |
| MTP under fp8 | `_scaled_mm` needs rows %16; depth-k truncation gave 8×2047 = 16,376. The MTP path had passed 21/21 CPU tests **that structurally cannot reach fp8**. |
| **0426 vs 0372** | identical feature string `gold-6346,512g,H100`, but **94 GB NVL vs 80 GB**. Arm 22 OOM'd on 0372. Audited all 21 prior arms — all on 0426, campaign clean — but that was luck. Now hard-pinned. Confirms `target-80gb-not-94gb-hbm`. |

## Claims I made and then withdrew

Four, each killed by a control I ran afterwards. This is the campaign's real methodology result:
a lever measured once, at one operating point, on one base, with one seed is not a finding.

1. **NoPE improves length generalisation** (arm 7) — killed by arm 8, whose FFN-only change shows the
   same slope shift with identical position handling. The slope metric is confounded with quality.
2. **LR 6e-4 is too high** (arm 13) — killed by arm 15. It was a 2-step-warmup artifact; with proper
   warmup the inherited 6e-4 beats every reduced LR tried.
3. **Block 1024 is a −0.361 iso-compute win** (arm 22) — killed by arm 23. Block 512 ties too, so the
   probe is blind to context.
4. **ReLU² is a −0.066 win on the stack** (arm 27) — killed by arm 29 at 48 M tokens.

And a fifth, self-inflicted: the **noise floor itself** (0.008 → 0.047), which softened six verdicts
at the very end.

## All 30 arms

| # | arm | tested | TGT | tok/s |
|---|---|---|---|---|
| 01 | `e01_baseline` | calibration: tok/s, compile, HBM | wall-guard | 32,205 |
| 02 | `e02_nockpt` | gradient checkpointing OFF | wall-guard | 33,648 |
| 03 | `e03_chunk0` | `loss_chunk_size=0` | 6.4373 | 39,593 |
| 04 | `e04_stride2` | checkpoint stride 2 | 6.4317 | 42,016 |
| 05 | `e05_lllg_rope` | LLLG sliding window 1024 | 6.4696 | 35,899 |
| 06 | `e06_base_lengths` | replication + length curve | 6.4280 | 42,297 |
| 07 | `e07_lllg_nope` | LLLG + NoPE globals | 6.4859 | 35,978 |
| 08 | `e08_relu2` | ReLU² standalone | 6.4598 | 43,515 |
| 09 | `e09_docmask` | intra-document masking | 6.3940 | 30,681 |
| 10 | `e10_base_48M` | exchange rate + doc-mask iso-compute | 6.0409 | 42,104 |
| 11 | `e11_base_20M` | exchange rate, 3rd point | 7.0611 | 42,387 |
| 12 | `e12_lr12e4` | LR 1.2e-3 | 6.7742 | 42,073 |
| 13 | `e13_lr3e4` | LR 3e-4 | 6.2884 | 42,183 |
| 14 | `e14_lr15e5` | LR 1.5e-4 | 6.2851 | 42,362 |
| 15 | `e15_lr6e4_warm10` | warmup confound test | 6.2440 | 42,257 |
| 16 | `e16_lr3e4_warm10` | closes the LR × warmup 2×2 | 6.2030 | 42,254 |
| 17 | `e17_attngate` | gated attention | 6.0713 | 40,752 |
| 18 | `e18_attngate_isoparam` | gate, ISO-PARAM control | 6.1456 | 41,800 |
| 19 | `e19_untie` | untied embeddings | 6.1100 | 42,090 |
| 20 | `e20_gate_untie` | do the winners stack? | 6.0619 | 41,528 |
| 21 | `e21_mtp` | real MTP module | 6.2355 | 38,960 |
| 22 | `e22_block1024` | block 1024 | 5.9913 | 57,049 |
| 23 | `e23_block512` | block 512 — falsification test | 6.0683 | 67,505 |
| 24 | `e24_stack_48M` | **recipe scale-robustness** | **5.6118** | 41,467 |
| 25 | `e25_batch2x` | batch transfer (confounded) | 6.5282 | 42,689 |
| 26 | `e26_stack_seed445` | seed 445 | 6.0541 | 41,470 |
| 27 | `e27_relu2_stack` | ReLU² on the stack | 6.0321 | 42,831 |
| 28 | `e28_cooldown30` | cooldown 0.30 | 6.0212 | 42,598 |
| 29 | `e29_final_48M` | ReLU² on the stack @48M | 5.6520 | 42,780 |
| 30 | `e30_final_seed446` | seed 446 — **noise floor correction** | 5.9773 | 41,577 |

**Best measured config: arm 24** — gated attention + untied embeddings + SwiGLU, LR 3e-4,
warmup 10%, `chunk=0`, ckpt stride 2 — **TGT 5.6118 at 48 M tokens**, vs 6.0409 for the old recipe
at the same tokens.
