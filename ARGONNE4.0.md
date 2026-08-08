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

**Data BUILT (2026-07-20)** at `/project/rcc/youzhi/data/argonne4_pretrain/` (the launcher default):

| Source | tokens | note |
|---|---|---|
| FineWeb-Edu (`edu_flat.bin`) | **20.6B** | fully tokenized — all 218 arrow shards (was a 2.2B replay anchor) |
| FineMath-4plus (`finemath_flat.bin`) | ~10B | ≈ all of FineMath-4plus; FineMath-3plus (~34B) can extend |
| github_code (`code_flat.bin`) | ~8B | |
| **combined** | **~38.6B** | + held-out `val_{edu,math,code}.bin` (3M each) |

Built by `build_reasoning_corpus.py tokenize --source fineweb_edu_a4` (46-core CPU job, 28 min) →
`build_a4_data.py`. At 50/30/20 over ~38.6B, each source runs ~1× (edu is no longer the bottleneck;
math is now first to repeat). ~38.6B is ~1.5× Chinchilla for a 1B; raise `A4_TRAIN_TOKENS` for a more
overtrained run — ≤~4× per-source repetition is ~free (Muennighoff), i.e. up to ~130–150B before
edu/math repeat past 4×.

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

The launcher defaults are now the BUILT scale bins + the probed batch, so `./weekend.sh` runs the
real recipe with **no overrides needed**. Optional knobs (sensible defaults in `run_full_training.sh`):
`A4_TRAIN_TOKENS` (schedule length; 0 = combined ~38.6B ≈ 1×/source), `A4_EDU/A4_MATH/A4_CODE` + `A4_W_*`
(sources/weights), `A4_BATCH`/`A4_GRAD_ACCUM`/`A4_LR`, `A4_GRAD_CKPT`, `CKPT_DIR_OVERRIDE`. For a more
overtrained run: `A4_TRAIN_TOKENS=100000000000 ./weekend.sh` (edu/math ~2.5–3×, ≤4× = ~free).

**Batch (probed on 3×H200, 2026-07-20).** Default `A4_BATCH=170 A4_GRAD_ACCUM=1` → effective
**522,240 tok/step == the campaign-validated 524,288** (LR 6e-4 directly validated). The probe showed
batch 170 already saturates the cards (**51% HBM @ 100% GPU util**) — block-1024 is compute-bound, so a
bigger batch buys ~0 throughput. `A4_BATCH=288` fills ~76% HBM (effective ~885K → set `A4_LR≈7.8e-4`)
if you insist on HBM fill, but there's no throughput reason to. Chunked CE (`loss_chunk_size 4096`) is
what frees the 151k-vocab fp32-logit transient so the big single-pass batch fits (the 3.5 phase-2 trick).

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
