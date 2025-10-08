# ArgonneAI (Argonne 2.0 Work Branch)

## Status
- **Argonne 2.0** pretraining is currently running; checkpoints and logs are still being produced while we continue large-scale experiments.
- Expect rapid iteration on the training scripts in this branch until training stabilises enough to merge back into `main`.

## Project overview
This repository hosts the in-progress training stack for the Argonne 2.0 language model family. The codebase focuses on large-scale pretraining with distributed data processing, pipeline parallelism, and robust recovery utilities so that long jobs can survive hardware interruptions.

## Repository layout
- `model.py` – Defines the `ArgonneModel` transformer architecture and its `ArgonneConfig`, including rotary attention, RMSNorm layers, and backwards-compatible aliases for Argonne 1.x checkpoints.
- `data_processing.py` – Handles tokenizer loading/fallback training, streaming parquet datasets, cached offline preprocessing, and assembling fixed-length token chunks for training.
- `training_utils.py` – Provides shared helpers such as dataset shard discovery/logging and a cosine learning rate scheduler with warmup for use across entrypoints.
- `training.py` – Pipeline-parallel training entrypoint with dataset streaming, resumable shard tracking, and automatic batch-size backoff when CUDA or compilation OOMs occur.
- `train_with_fsdp.py` – Alternative Fully Sharded Data Parallel (FSDP) launcher for multi-GPU jobs that prefer sharded data parallelism over pipeline stages.
- `resume_pretrain.py` – Legacy resume script maintained for compatibility with earlier experiments; kept while we migrate to the unified `training.py` flow.

## Default dataset location
- Training scripts now target the Common Crawl derived shards stored at `../data/CC-MAIN-2025-26/*.parquet`. Shards are consumed in natural numeric order so `000_00000.parquet` is seen before `000_00001.parquet`, ensuring deterministic sequential coverage of the crawl export.

## How this branch differs from `main`
The `work` branch diverges from the stable `main` branch to incubate Argonne 2.0. Key enhancements under active development here include:

1. **Pipeline model parallelism and smarter failure recovery.** `training.py` orchestrates multi-stage pipeline parallelism with automated batch-size reduction and retry logic when Torch Dynamo or CUDA OOMs arise—capabilities we are exercising here before promoting them to `main`.
2. **Dataset streaming with resumable positions.** The new token generator and data trackers keep shard/offset state so long-running jobs can resume without re-tokenising or replaying entire shards.
3. **Shared infrastructure cleanup.** Utilities for tokenizer validation, shard discovery, and cosine scheduling were consolidated into `training_utils.py` so both the pipeline and FSDP flows share one implementation instead of the duplicated helpers that live on `main`.

These changes will be merged back once Argonne 2.0 completes training and the tooling proves stable.
