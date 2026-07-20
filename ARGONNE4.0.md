# Argonne 4.0 — a ~1.04B data-efficient reasoning base (recipe + ops)

argonne4.0 is the production line coming out of the **argonne4 data-efficiency campaign**
(2026-07-18/20; 49 iso-token proxy runs; full record in
`/project/rcc/youzhi/argonne4_experiments/CAMPAIGN_data_efficiency.md`). The campaign's answer:
hold the architecture at argonne3.5's shape scaled to ~1B and make the **data composition** the
lever. This branch codes that up end-to-end, reusing the argonne3.5 self-resubmitting training
workflow verbatim.

Branch `argonne4.0`, worktree `/home/youzhi/ArgonneAI-4.0` (main clone stays on `argonne3.5`).

## What changed vs argonne3.5 (workflow is IDENTICAL; only model/data/sizing differ)

| Axis | argonne3.5 | argonne4.0 |
|---|---|---|
| Params | 2.88B (hidden3072/24L/12h/4kv/inter8192) | **1.04B** (hidden1536/32L/6h/2kv/**head_dim256**/inter4096) |
| Arch set in | `pretrain.py` / `continue_pretrain.py` module constants | same files, edited to the 1B constants |
| Pretrain data | flat 85/15 FineWeb/FineMath bin (`--data_path`) | **50% edu / 30% math / 20% code** weighted mix (`--train_sources`) |
| Data loader | flat `DataLoader` | new **`WeightedMultiLoader`** (per-micro-batch weighted sampling, DDP-synced, resumable) |
| Checkpoint | ~32 GiB | ~11 GB → pre-kill **save margin cut 300s→150s** (more training/slice) |
| Ckpt dirs | `models/pretrain`, `models/midtrain` | `models/pretrain4`, `models/midtrain4` (fresh) |
| Recipe knobs | LR 6e-4, grad_clip 0.4, WSD warmup8000/cooldown0.15, fp8 | **same** (all transferable per the campaign) |

`head_dim` is derived (`hidden//n_head = 1536//6 = 256`), not a knob; every fp8 GEMM dim
(1536, 6·256, 2·256, 4096) is /16-divisible so fp8 + lm_head stay eligible.

## The data recipe — 50% general (FineWeb-Edu) / 30% math (FineMath) / 20% code

Pinned on both axes at 300M tokens and passing the translate-test (advantage over web-only GREW
with scale). Realized by `pretrain.py --train_sources "EDU:50,MATH:30,CODE:20"`: it samples ONE
source per micro-batch by weight (no pre-built blend bin; ratio decoupled from raw source sizes).

**Data readiness (2026-07-20).** Raw docbin tokenized today (`build_a4_data.py` sources):

| Source | docbin available | note |
|---|---|---|
| FineWeb-Edu | ~2.0B tok (24 shards) | **the gap** — highest weight (50%), smallest corpus |
| FineMath-4plus | ~9.5B tok (64 shards) | ≈ all of FineMath-4plus; FineMath-3plus (~34B) can extend |
| github_code | ~7.5B tok (16 shards) | |

`build_a4_data.py` builds the full per-source flat bins (`--out`, all shards by default) + held-out
val bins. For a large run, tokenize more FineWeb-Edu into `EDU_DOCBIN_DIR` (upstream ~1.3T) — at 50%
weight, edu is the binding source. Multi-epoch repetition ≤~4× is ~free (Muennighoff), so ~19B
combined supports a meaningful run today; a full 100B+ run wants more edu.

## The pipeline (marker-gated, self-resubmitting — same as 3.5)

`weekend.sh` / `night.sh` → `run_full_training.sh` (the per-slice worker). Three auto-resuming,
marker-gated stages, each resuming from its own latest checkpoint:

1. **pretrain** — `pretrain.py`, 50/30/20 weighted mix → `models/pretrain4`; writes `.pretrain_complete`
   + `final_model_complete/`. Terminates at the `--train_tokens` budget (the sampler flips "epoch"
   at the budget so the completion + transition fire).
2. **midtrain phase A** — `continue_pretrain.py`, reasoning anneal (`reasoning_anneal_flat.bin`,
   block 1024), same dir, fresh WSD seeded from the pretrain final; writes `.continue_pretrain_complete`.
3. **midtrain phase B (gated)** — `continue_pretrain.py` context-extension to block 13568 →
   `models/midtrain4`, gated on `models/midtrain4/.midtrain_armed` (OFF by default, exactly like 3.5
   before arming; the chain cleanly stops after phase A until you arm it and smoke-test its batch).

Crash-resilient: a crashed slice resubmits, resumes from the last checkpoint, and excludes the dead
node (≤ `FAILURE_RETRY_MAX`). Clean wall-time slices pre-submit the next slice on the SLURM `USR1`.

## How to run

```bash
cd /home/youzhi/ArgonneAI-4.0

# continuous self-resubmitting chain (dry-run first to inspect the sbatch):
./weekend.sh --dry-run
./weekend.sh

# or ONE slice tonight at 23:00:
./night.sh
```

**Before a REAL run** (defaults point at the campaign PROXY bins so the pipeline runs today for a
smoke — the worker prints a loud warning if the corpus is small):

```bash
# 1) build the scale per-source bins (all shards):
python build_a4_data.py --out /project/rcc/youzhi/data/argonne4_pretrain
# 2) point the launcher at them + set a real token budget:
A4_EDU=/project/rcc/youzhi/data/argonne4_pretrain/edu_flat.bin \
A4_MATH=/project/rcc/youzhi/data/argonne4_pretrain/finemath_flat.bin \
A4_CODE=/project/rcc/youzhi/data/argonne4_pretrain/code_flat.bin \
A4_TRAIN_TOKENS=100000000000 \
./weekend.sh
```

Key env overrides (all have sensible defaults in `run_full_training.sh`): `A4_EDU/A4_MATH/A4_CODE`,
`A4_W_EDU/A4_W_MATH/A4_W_CODE` (weights), `A4_TRAIN_TOKENS` (schedule length), `A4_BATCH`,
`A4_GRAD_ACCUM`, `A4_LR`, `A4_WARMUP`, `A4_GRAD_CKPT`, `CKPT_DIR_OVERRIDE`.

**Batch sizing.** Default `A4_BATCH=64 A4_GRAD_ACCUM=3` → effective 589,824 tok/step (~ the
campaign-validated 524,288; LR is flat over 6–8e-4). block-1024 is compute-bound, so a bigger batch
buys ~0 throughput — check HBM on the first slice with `slurmwatch` and only lower `A4_BATCH` if it
OOMs (chunked CE `loss_chunk_size 4096` frees the 151k-vocab fp32-logit transient). For max speed,
`A4_GRAD_CKPT=0` (the 1B likely fits without checkpointing; recompute-free is faster).

## Honest bounds (per no-premature-optimism)

The campaign proves the recipe is far more data-efficient PER TOKEN than 3.5's web-heavy recipe
(1B-vs-1B decisive on math/code; edge grows with tokens). It does **not** prove 1.04B beats the
2.88B model — that needs a full run. General/world-knowledge is where the 2.88B capacity edge likely
persists; math/code/reasoning is genuinely winnable. Proxy CE is blind to downstream, so a
decontaminated eval suite (GSM8K/MMLU/HumanEval) is a prerequisite before trusting end-to-end.

## Files (this branch)

- `pretrain.py` — 1B arch constants; `WeightedMultiLoader` + `--train_sources`/`--train_tokens`; save margin 150s.
- `continue_pretrain.py` — 1B arch constants; save margin 150s.
- `build_a4_data.py` — build the scale per-source flat bins from docbin shards.
- `run_full_training.sh`, `weekend.sh`, `night.sh` — the workflow (untracked per repo policy; never committed).
- `model.py` — unchanged (arch comes from the constants above; validated on this exact config by the campaign).
