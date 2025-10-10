# ArgonneAI (Argonne 2.0 Work Branch)

## Status
- **Argonne 2.0** pretraining is currently running; checkpoints and logs are still being produced while we continue large-scale experiments.
- Expect rapid iteration on the training scripts in this branch until training stabilises enough to merge back into `main`.

## Project overview
This repository hosts the in-progress training stack for the Argonne 2.0 language model family. The codebase focuses on large-scale pretraining with distributed data processing, pipeline parallelism, and robust recovery utilities so that long jobs can survive hardware interruptions.

## Model architecture
The Argonne 2.0 configuration that is currently exercised by the training scripts is a decoder-only transformer with the following key hyperparameters:

- **Depth:** 24 transformer blocks with grouped-query attention (24 query heads, 8 key/value heads) and SwiGLU feed-forward layers.【F:training.py†L286-L301】【F:model.py†L67-L313】
- **Width:** 4,096 hidden dimensions with a 8/3 × expansion (rounded to 256) in the SwiGLU MLP, yielding an 11,008-unit intermediate width.【F:training.py†L286-L301】【F:model.py†L79-L301】
- **Context length:** 4,096 token context window per block, with rotary position embeddings parameterised by a 500,000 base theta to extend extrapolation headroom.【F:training.py†L286-L301】【F:model.py†L121-L142】
- **Normalization & dropout:** Pre- and post-attention RMSNorm layers with dropout kept at 0.0 in both attention and MLP modules for baseline pretraining stability.【F:training.py†L286-L295】【F:model.py†L107-L319】
- **Efficiency features:** Flash attention is enabled when available, residual projection weights are marked for fused residual kernels, and word embeddings remain untied to allow custom output heads.【F:training.py†L286-L301】【F:model.py†L200-L276】

## Tokenizer
Argonne 2.0 pretraining uses the **Qwen2.5-3B-Instruct** tokenizer. Important characteristics of this tokenizer include:

- 151,936 token vocabulary with dedicated `<|im_start|>`, `<|im_end|>`, and multimodal delimiters for future tool-use and vision extensions.
- Special tokens covering conversational (`<|endoftext|>`, `<|im_*|>`), tool calling (`<tool_call>`), and multimodal padding markers (`<|vision_pad|>`, `<|image_pad|>`, `<|video_pad|>`).
- Model max length set to 131,072 tokens, allowing us to safely expand beyond the 4,096-token training window if longer-context finetuning is required.
- Uses the Qwen chat template for tool-aware prompts; training scripts ensure a pad token is present and adapt `model_max_length` to match the 4,096-token block size.

## Repository layout
- `model.py` – Defines the `ArgonneModel` transformer architecture and its `ArgonneConfig`, including rotary attention, RMSNorm layers, and backwards-compatible aliases for Argonne 1.x checkpoints.
- `data_processing.py` – Handles tokenizer loading/fallback training, streaming parquet datasets, cached offline preprocessing, and assembling fixed-length token chunks for training.
- `training_utils.py` – Provides shared helpers such as dataset shard discovery/logging and a cosine learning rate scheduler with warmup for use across entrypoints.
- `training.py` – Pipeline-parallel training entrypoint with dataset streaming, resumable shard tracking, and automatic batch-size backoff when CUDA or compilation OOMs occur.
- `train_with_fsdp.py` – Alternative Fully Sharded Data Parallel (FSDP) launcher for multi-GPU jobs that prefer sharded data parallelism over pipeline stages.
- `resume_pretrain.py` – Legacy resume script maintained for compatibility with earlier experiments; kept while we migrate to the unified `training.py` flow.

## Running the FSDP continuation script
On a single DGX node you should launch `train_with_fsdp.py` with `torchrun` so that all eight GPUs participate as individual FSDP ranks. The script follows the exact same resume logic as `resume_pretrain.py`—it restores checkpoints written by the legacy flow, reuses the shared `DataPosition` tracker, and advances the cosine schedule in lockstep—while letting every GPU process its own micro-batch concurrently for higher throughput. A minimal command matching the resume workflow is:

```bash
torchrun --standalone --nproc_per_node=8 train_with_fsdp.py \
  --tokenizer-path ../Qwen2.5-3B-Instruct
```

Additional flags exposed by the script include `--data-glob` for alternate shard locations, `--batch-size` for the per-rank batch, and precision options like `--bf16`/`--fp16`. Only rank 0 performs dataset/tokeniser work; batches are broadcast to the other ranks, which means you can scale to multi-node runs by exporting the usual rendezvous variables (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`) before invoking `torchrun`. The script automatically discovers the latest checkpoint (or accepts `--checkpoint-path`) and restores the tokenizer/training step metadata before continuing. 【F:train_with_fsdp.py†L340-L427】【F:train_with_fsdp.py†L452-L610】

## Default dataset location
- Training scripts now target the Common Crawl derived shards stored at `../data/CC-MAIN-2025-26/*.parquet`. Shards are consumed in natural numeric order so `000_00000.parquet` is seen before `000_00001.parquet`, ensuring deterministic sequential coverage of the crawl export.

## How this branch differs from `main`
The `work` branch diverges from the stable `main` branch to incubate Argonne 2.0. Key enhancements under active development here include:

1. **Pipeline model parallelism and smarter failure recovery.** `training.py` orchestrates multi-stage pipeline parallelism with automated batch-size reduction and retry logic when Torch Dynamo or CUDA OOMs arise—capabilities we are exercising here before promoting them to `main`.
2. **Dataset streaming with resumable positions.** The new token generator and data trackers keep shard/offset state so long-running jobs can resume without re-tokenising or replaying entire shards.
3. **Shared infrastructure cleanup.** Utilities for tokenizer validation, shard discovery, and cosine scheduling were consolidated into `training_utils.py` so both the pipeline and FSDP flows share one implementation instead of the duplicated helpers that live on `main`.

These changes will be merged back once Argonne 2.0 completes training and the tooling proves stable.
