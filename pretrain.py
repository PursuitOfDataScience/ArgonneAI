"""
DDP training for Argonne model.
Supports pretraining, continued pretraining on new data (--reset_schedule),
and automatic checkpoint resume.
"""

import os
import re
import sys
import glob
import time
import argparse
import json
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Model architecture -- argonne4.0: the argonne3.5 arch SCALED TO ~1.04B (the
# data-efficiency campaign's fixed "3.5-at-1B" shape; see ARGONNE4.0.md). Only the
# four width/depth constants change from 3.5's 2.88B; head_dim is DERIVED
# (hidden_size // num_heads = 1536 // 6 = 256) and every fp8 GEMM dim stays
# /16-divisible (1536, 6*256, 2*256, 4096), so fp8 + lm_head remain eligible.
HIDDEN_SIZE = 1536
NUM_LAYERS = 32
NUM_HEADS = 6
NUM_KV_HEADS = 2  # GQA (num_heads % num_kv_heads == 0 required)
INTERMEDIATE_SIZE = 4096  # 3.5 lets this fall to the model.py default (8192); argonne4.0 PINS 4096
ENABLE_QK_NORM = True
ENABLE_V_NORM = True
Z_LOSS_WEIGHT = 0.0
ENABLE_SANDWICH_NORM = True
ROPE_THETA = 1000000.0
ENABLE_MTP = False
MTP_HORIZON = 1
MTP_LOSS_WEIGHT = 0.0
ENABLE_INTERLEAVED_LOCAL_ATTENTION = True
LOCAL_ATTENTION_WINDOW = 256
LOGIT_SOFTCAP = 15.0

# ---------------------------------------------------------------------------
# argonne4.5 -- 2,063,667,712 params. SET FROM MEASUREMENT, 2026-08-14.
#
# 51 probe arms on 3x H100 (exp/EXPERIMENTS.md) tested every lever proposed in ARGONNE4.5.md.
# The result was almost entirely negative, and this block reflects that: a4.5 is a4.0's
# ARCHITECTURE at a larger size, with the systems settings fixed. Nothing else survived.
#
#   REFUTED outright: LLLG sliding window (+0.038 CE AND -14.6% throughput -- no flash-attn-2 here,
#     so the window forces an explicit SDPA mask and loses the fused is_causal kernel); NoPE global
#     layers; intra-document masking (a real -0.034 iso-token win, but -27.5% throughput makes the
#     baseline win by 0.353 iso-compute); a real DeepSeek-style MTP module (+0.127 iso-compute);
#     RHO-1 selective loss (+0.449 token-matched, 23% slower).
#   UNRESOLVED, therefore NOT adopted: ReLU^2 FFN (three measurements, three different signs);
#     gated attention and untied embeddings (0.7 and 0.5 sigma against a measured sigma of 0.130 --
#     campaign 1 called both wins using a noise floor that was 4.6x too small).
#
# So: SwiGLU, tied embeddings, no gate, full attention, RoPE everywhere -- all a4.0 defaults.
# The ONLY architecture change from a4.0 is the size (1536/32L -> 2560/24L, head_dim 256 kept).
# Every fp8 GEMM dim stays /16-divisible: 2560, 10*256, 2*256, 7040.
# ⚠️ THE LEGACY WINDOW MUST BE TURNED OFF EXPLICITLY, in BOTH branches. ENABLE_INTERLEAVED_LOCAL_
# ATTENTION=True / LOCAL_ATTENTION_WINDOW=256 above are inherited from a4.0, where they were DEAD
# CODE: only flash-attn-2 honored `window_size`, this env ships flash-attn-4 without the 2.x
# interface, so every Argonne pretrain since 3.0 actually ran full attention. This tree's model.py
# then gained the argonne4.5 SDPA fix (Finding A) that makes the window real on the SDPA path --
# which silently RE-ACTIVATED the legacy flag here: verified 2026-08-14 that with these flags the
# 256-token window goes live on all 12 odd layers and changes the logits (max|diff| 2.06), while
# the startup banner still printed "full attention ... IGNORED on this path".
#
# That contradicts the intent stated below (full attention), contradicts every prior Argonne
# pretrain, and contradicts our own measurement: arm-tested sliding window was REFUTED at +0.038
# CE and -14.6% throughput, precisely because without flash-attn-2 the window forces an explicit
# SDPA mask and loses the fused is_causal kernel. So leaving it on costs quality AND speed.
ENABLE_INTERLEAVED_LOCAL_ATTENTION = False
LOCAL_ATTENTION_WINDOW = None

A45 = True    # argonne4.5 production. Set False to reproduce a4.0 exactly.
if A45:
    HIDDEN_SIZE = 2560
    NUM_LAYERS = 24
    NUM_HEADS = 10
    NUM_KV_HEADS = 2
    INTERMEDIATE_SIZE = 7040
    MLP_TYPE = "swiglu"
    ATTN_PATTERN = None
    SLIDING_WINDOW_SIZE = 2048
    NOPE_GLOBAL = False
    DOC_MASK = False
    MTP_MODULE_LAYERS = 0
    ATTN_GATE = False
else:
    # argonne4.0 reproduction path: every 4.5 flag off, so this file still builds the exact
    # 1.04B a4.0 config and every existing a4.0 checkpoint/launcher keeps working.
    MLP_TYPE = "swiglu"
    ATTN_PATTERN = None
    SLIDING_WINDOW_SIZE = 2048
    NOPE_GLOBAL = False
    DOC_MASK = False
    MTP_MODULE_LAYERS = 0
    ATTN_GATE = False

# Parse arguments
parser = argparse.ArgumentParser()
# Paths
parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
parser.add_argument("--data_path", type=str, required=False, default=None, help="Single training source: a flat llm.c .bin OR a doc-manifest .json. Mutually exclusive with --train_sources; exactly one is required.")
parser.add_argument("--train_sources", type=str, default=None, help="argonne4.0 weighted data mixture: 'PATH:WEIGHT,PATH:WEIGHT' (e.g. edu.bin:50,math.bin:30,code.bin:20). Samples ONE source per micro-batch by weight (DDP-synced across ranks, resumable). Overrides --data_path.")
parser.add_argument("--train_tokens", type=int, default=0, help="Total training-token budget defining the WSD schedule length when --train_sources is used (estimated_steps = train_tokens / effective_batch, so the cooldown lands at the run's true end). 0 = default to the combined source size (~one weighted pass).")
parser.add_argument("--doc_shuffle", type=int, default=0, choices=[0, 1], help="If 1 and --data_path is a doc-manifest .json, globally shuffle document order each epoch (REQUIRED for an intermix manifest so fineweb+finemath are interleaved, not trained sequentially)")
parser.add_argument("--doc_shuffle_seed", type=int, default=1337, help="Base seed for doc-manifest document shuffling; ALSO seeds the --train_sources source sampler (identical on every rank so source picks stay in lockstep)")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory for checkpoints")
# Training hyperparameters
# Production LR: 6e-4
# Inherited from exp_317 (2.88B arch, tuned at 24K effective batch in nextrun3 search).
# Production effective batch is ~1M tokens, ~41x larger. Batch-scaling rules suggest:
#   - Linear: 6e-4 * 41 = 2.5e-2  (too aggressive, would diverge)
#   - Sqrt:   6e-4 * 6.4 = 3.8e-3  (aggressive for cold start at this scale)
# Counter-balancing effect: 2.88B is 2.2x larger than the 1.3B llm.c baseline,
# which used LR=3e-4. Larger models typically want slightly lower LR at fixed batch.
# Net: 6e-4 is a conservative, exp_317-validated starting point that survives
# both adjustments. Verify at scale with a short probe (2-5B tokens) before
# committing to a full run. Safe range to explore: 4e-4 to 1e-3.
parser.add_argument("--lr", type=float, default=6.0e-4, help="Learning rate")
parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of LR")
parser.add_argument("--batch_size", type=int, default=19, help="Batch size per GPU")
parser.add_argument("--total_batch_size", type=int, default=1011712, help="Total batch size in tokens")
parser.add_argument("--block_size", type=int, default=1024, help="Sequence length")
parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--schedule", type=str, default="wsd", choices=["cosine", "wsd"], help="LR schedule")
parser.add_argument("--cooldown", type=int, default=4000, help="Cooldown steps at end of WSD schedule")
# Argonne-3.5 recipe: express the WSD cooldown as a fraction of the estimated run instead of a
# fixed step count. When >0 this OVERRIDES --cooldown and is recomputed from estimated_steps on
# every resume, so the terminal LR anneal lands correctly regardless of corpus size or how the
# run is sliced across wall-time-limited SLURM jobs. The launcher's old --cooldown 0 left the
# WSD schedule with NO decay phase at all (LR flat at peak forever); 0.15 restores the anneal.
parser.add_argument("--cooldown_frac", type=float, default=0.0, help="WSD cooldown as a fraction of estimated steps; if >0 overrides --cooldown. Argonne-3.5 recipe: 0.15.")
parser.add_argument("--grad_clip", type=float, default=0.4, help="Gradient clipping")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")
# Argonne-3.5 FP8: torchao float8 (tensorwise dynamic) on the Linear matmuls + lm_head. Recipe-search
# result: ~1.25x H200 throughput at neutral quality (needs torch_compile=1; master weights stay fp32).
parser.add_argument("--fp8", type=int, default=0, choices=[0, 1], help="Enable FP8 training via torchao float8 (requires torch_compile=1)")
parser.add_argument("--fp8_lm_head", type=int, default=1, choices=[0, 1], help="Also FP8 the (tied) lm_head — recipe default; tie is preserved")
parser.add_argument("--loss_chunk_size", type=int, default=0, help="If >0, chunked cross-entropy over this many (batch*seq) rows/chunk — frees the full-logit fp32 transient so batch can grow (fill HBM higher). 0 = off. NOTE: raising batch mid-run shifts the WSD cooldown (cooldown_frac × estimated_steps); use only on a FRESH run with LR/cooldown set for the bigger batch.")
parser.add_argument("--flash_attention", type=int, default=1, choices=[0, 1], help="Use flash attention")
parser.add_argument("--checkpoint_interval", type=int, default=1800, help="Checkpoint interval in seconds")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum epochs to train")
parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Use gradient checkpointing")
parser.add_argument("--checkpoint_stride", type=int, default=1, help="Selective activation checkpointing (ported from argonne4.0). 1=checkpoint ALL layers (default, prior behavior). >=2=checkpoint every layer EXCEPT store (un-checkpoint) every Sth layer (store ceil(n_layers/S), recompute the rest) -> smaller S stores MORE = more HBM + less recompute = faster (too-small S OOMs). Numerically identical; requires --gradient_checkpointing 1.")
parser.add_argument("--torch_compile", type=int, default=1, choices=[0, 1], help="Use torch.compile for speedup")
parser.add_argument("--torch_compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], help="torch.compile mode")
parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint file")
parser.add_argument("--wall_time", type=int, default=0, help="Wall time in seconds. If > 0, save checkpoint 5 min before this limit. 0 = disabled.")
parser.add_argument("--save_deadline_epoch", type=int, default=0, help="Absolute wall-clock deadline (unix epoch seconds): when time.time() reaches it, save ONE checkpoint and exit cleanly. SLURM-clock-relative (run_full_training.sh sets it = job_start + slice_seconds - lead), so it fires a FIXED time before the SLURM kill regardless of compile/startup drift -> the reliable save trigger. 0 = disabled.")
parser.add_argument("--reset_schedule", type=int, default=0, choices=[0, 1], help="Reset LR schedule, step counter, and data position when resuming. Use for continued pretraining on new data.")
parser.add_argument("--val_data_path", type=str, default=None, help="Optional path to held-out validation data (.bin)")
parser.add_argument("--val_batch_size", type=int, default=16, help="Micro-batch for validation eval — small + independent of the train batch so the val logit tensor never OOMs at a large train batch.")
parser.add_argument("--periodic_val_every", type=int, default=0, help="If >0, eval val every N optimizer steps during training (rank-0 + barrier), logging 'PERIODIC_VAL step=.. tokens=.. val_loss=..'. 0 = only the end-of-run eval (production default).")
parser.add_argument("--final_model_dir", type=str, default=None, help="Optional directory for the final Hugging Face model export.")
parser.add_argument("--completion_marker", type=str, default=None, help="Optional marker file written only after max_epochs is completed and the final model export succeeds.")
parser.add_argument("--seed", type=int, default=444, help="Base random seed")
args = parser.parse_args()
assert args.data_path or args.train_sources, "provide exactly one of --data_path or --train_sources"

# Distributed setup
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

RANK, LOCAL_RANK, WORLD_SIZE = setup_distributed()
IS_MAIN = RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}"

BASE_SEED = int(args.seed)
RUN_SEED = BASE_SEED + RANK
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
torch.manual_seed(RUN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RUN_SEED)

# Compute gradient accumulation
TOKENS_PER_MICRO = args.batch_size * WORLD_SIZE * args.block_size
GRAD_ACCUM_STEPS = args.total_batch_size // TOKENS_PER_MICRO
assert GRAD_ACCUM_STEPS >= 1, (
    f"total_batch_size ({args.total_batch_size}) too small for "
    f"{WORLD_SIZE} GPU(s) x batch_size {args.batch_size} x block_size {args.block_size}"
)
ACTUAL_TOTAL_BATCH = GRAD_ACCUM_STEPS * TOKENS_PER_MICRO

# Wall time: save the checkpoint before the hard kill. This training-ELAPSED margin is a BACKUP;
# the PRIMARY save trigger is --save_deadline_epoch (absolute wall clock, startup-immune).
#
# The 180s is MEASURED for a4.5, not inherited (2026-08-14). The a4.5 checkpoint is 23.1 GiB
# (fp32 weights + AdamW moments for 2.06B), and the real slice in report/1-train.out wrote it in
# ~18s (~1.3 GiB/s to /project). The save path then re-reads it to verify before committing --
# that costs 1.7s in practice, because the bytes are still in page cache moments after the write,
# and 29.5s only on a fully cold read. So:
#     torch.save ~18s + gate1 <1s + gate2 1.7s typical (29.5s cold) = ~21s typical, ~49s worst
# against a 180s reserve, i.e. 131s of headroom even in the worst case. Re-measure this if the
# model grows or the checkpoint gains state; do not just scale the constant by feel.
WALL_TIME_SAVE = args.wall_time - 180 if args.wall_time > 0 else 0

# Autocast setup
if args.precision == "bf16":
    AUTOCAST_DTYPE = torch.bfloat16
    USE_AUTOCAST = True
elif args.precision == "fp16":
    AUTOCAST_DTYPE = torch.float16
    USE_AUTOCAST = True
else:
    AUTOCAST_DTYPE = None
    USE_AUTOCAST = False

# Data loading
def load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        magic = header[0]
        if magic != 20240801:
            raise ValueError(f"Unknown magic number: {magic}")
        tokens = np.memmap(filename, dtype=np.uint32, mode='r', offset=256*4)
    return tokens

class DataLoader:
    def __init__(self, filename, B, T, rank=0, world_size=1, start_token_offset=0):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.tokens = load_data_shard(filename)
        self.start_token_offset = int(start_token_offset)
        self.current_position = self.start_token_offset + rank * B * T
        self.epoch = 0
        if self.current_position + (B * T + 1) > len(self.tokens):
            raise ValueError(
                f"Start offset {self.start_token_offset} is too close to end of dataset "
                f"for batch_size={B}, block_size={T}, world_size={world_size}"
            )
        if rank == 0:
            print(f"DataLoader: {len(self.tokens):,} tokens (start_offset={self.start_token_offset:,})")

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long).pin_memory()
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.world_size
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = self.start_token_offset + self.rank * B * T
            self.epoch += 1
            if self.rank == 0:
                print(f"\n*** Epoch {self.epoch} completed ***\n")
        return x, y

    def get_position(self):
        return self.current_position

    def set_position(self, position):
        self.current_position = position


# ---------------------------------------------------------------------------
# Weighted multi-source loader (argonne4.0): realize a train-data mixture at an
# EXACT ratio by sampling a SOURCE per micro-batch by weight, without pre-building a
# blend .bin per ratio. Ported from the validated data-efficiency probe
# (argonne4_probe.py) and made DDP-aware + resumable for the wall-time-sliced run.
#   DDP correctness: the source-selection RNG is seeded IDENTICALLY on every rank
#     (seed carries no rank term), so at each draw all ranks pick the SAME source and
#     stay in lockstep; within a source the per-rank DataLoader shards disjointly
#     (offset rank*B*T, stride B*T*world_size) => no token is seen by two ranks.
#   Resume: get_position() (called on rank 0) returns the shared rng state + each
#     source's rank-0 cursor; on resume every rank restores the rng and sets each
#     source cursor to rank0_cursor + rank*B*T (the per-source analogue of the flat
#     loader's resume math). Because it is a RANDOM sampler over corpora far larger
#     than any single slice, an imperfect resume only re-samples a little data (no
#     contiguous skip/repeat) -- strictly more robust than a single-pass flat bin.
def parse_train_sources(spec):
    """'PATH:WEIGHT,PATH:WEIGHT' -> [(path, weight)] (rsplit on ':' so paths keep slashes)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, w = part.rsplit(":", 1)
        out.append((path.strip(), float(w)))
    return out


class WeightedMultiLoader:
    def __init__(self, sources, B, T, rank=0, world_size=1, seed=1337, train_tokens=0):
        # sources: list of (path, weight); reuse the DDP-aware flat DataLoader per source
        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.paths = [p for p, _ in sources]
        self.loaders = [DataLoader(p, B, T, rank, world_size) for p, _ in sources]
        w = np.array([wt for _, wt in sources], dtype=np.float64)
        assert (w > 0).all() and w.sum() > 0, "source weights must be positive"
        self.p = w / w.sum()
        self.n = len(self.loaders)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.counts = [0] * self.n
        # Progress is tracked in TOKENS, not draws. `counts` stays a per-source draw
        # tally (mixture telemetry only); `drawn_base` holds tokens consumed BEFORE
        # this process started and `draws` counts this process's own draws. Deriving
        # progress from `counts * B` instead made it rescale by B_new/B_old whenever a
        # resume changed the micro-batch, silently misreporting corpus consumption and
        # breaking the epoch stop (regression test: scratchpad/test_wml_accounting.py).
        self.drawn_base = 0
        self.draws = 0
        self.epoch = 0
        total = sum(len(l.tokens) for l in self.loaders)
        # num_tokens sets the WSD schedule length (estimated_steps = num_tokens/eff_batch).
        # Default = combined corpus size (~one weighted pass); override with --train_tokens.
        self.num_tokens = int(train_tokens) if train_tokens and int(train_tokens) > 0 else int(total)
        if rank == 0:
            frac = ", ".join(f"{os.path.basename(p)}:{self.p[i]:.3f}" for i, p in enumerate(self.paths))
            print(f"WeightedMultiLoader: {self.n} sources [{frac}] combined={total:,} tok; "
                  f"schedule num_tokens={self.num_tokens:,} (seed={self.seed})")

    def next_batch(self):
        i = int(self.rng.choice(self.n, p=self.p)) if self.n > 1 else 0
        self.counts[i] += 1
        self.draws += 1
        xy = self.loaders[i].next_batch()
        # Define an "epoch" as one full pass over the token BUDGET (num_tokens): total tokens
        # drawn across all ranks = (per-rank draws) * B * T * world_size. This makes
        # `train_loader.epoch >= max_epochs` fire in pretrain.py's stop check so the pretrain
        # stage TERMINATES at the budget and writes .pretrain_complete -- WITHOUT this an
        # infinite sampler never triggers the epoch-completion transition to continue_pretrain.
        # Only THIS process's draws are scaled by the current B; earlier progress is carried
        # as a token count so a micro-batch change across a resume cannot rescale it.
        drawn = self.drawn_tokens()
        self.epoch = int(drawn // self.num_tokens)
        return xy

    def drawn_tokens(self):
        """Total tokens consumed across all ranks, invariant to micro-batch changes."""
        return int(self.drawn_base + self.draws * self.B * self.T * self.world_size)

    def get_position(self):
        # rank-0 aggregate state (save_checkpoint calls this on IS_MAIN only). Picklable
        # -> stored verbatim as checkpoint['data_position'].
        return {
            "wml": True,
            "rng": self.rng.bit_generator.state,
            "positions": [int(l.current_position) for l in self.loaders],
            "counts": list(self.counts),
            "epoch": int(self.epoch),
            # Persist progress in TOKENS (and the B that produced it) so a resume at a
            # different micro-batch restores consumption exactly. Older checkpoints lack
            # both keys; resume falls back to the authoritative tokens_processed instead.
            "drawn_tokens": self.drawn_tokens(),
            "B": int(self.B),
        }

    def resume_from_checkpoint_position(self, state, drawn_tokens=None):
        # Presence of this method routes the loader through the manifest-style resume
        # branch (pretrain.py passes the raw data_position, no external rank offset).
        if not isinstance(state, dict) or not state.get("wml"):
            if self.rank == 0:
                print("WeightedMultiLoader: no compatible resume state; starting sampler fresh")
            return
        try:
            self.rng.bit_generator.state = state["rng"]
        except Exception as e:  # pragma: no cover - defensive
            print(f"WeightedMultiLoader: rng restore failed ({e}); reseeding {self.seed}")
            self.rng = np.random.default_rng(self.seed)
        off = self.rank * self.B * self.T
        for l, pos0 in zip(self.loaders, state.get("positions", [])):
            newpos = int(pos0) + off
            if newpos + (self.B * self.T + 1) > len(l.tokens):  # wrap safety
                newpos = l.start_token_offset + self.rank * self.B * self.T
            l.current_position = newpos
        self.counts = list(state.get("counts", self.counts))
        # Restore token progress, most authoritative source first:
        #   1. tokens_processed from the checkpoint (exact; passed by the caller)
        #   2. drawn_tokens persisted by a post-fix checkpoint
        #   3. legacy derivation counts * B, using the B that WROTE the state if recorded
        # Never counts * self.B -- that is the rescaling bug this replaces.
        if drawn_tokens is None:
            drawn_tokens = state.get("drawn_tokens")
        if drawn_tokens is None:
            b_old = int(state.get("B", self.B))
            drawn_tokens = sum(self.counts) * b_old * self.T * self.world_size
        self.drawn_base = int(drawn_tokens)
        self.draws = 0
        self.epoch = int(state.get("epoch", 0))
        if self.rank == 0:
            print(f"WeightedMultiLoader: resumed {self.n} source cursors + rng "
                  f"(epoch {self.epoch}, {self.drawn_base:,} tok consumed, micro_batch={self.B})")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_position(self, position):
        # Compat shim: pretrain.py's flat-loader reset path may hand back an int; a
        # random sampler has nothing contiguous to seek, so only a dict state is honored.
        if isinstance(position, dict):
            self.resume_from_checkpoint_position(position)


# ---------------------------------------------------------------------------
# Doc-aware manifest loader (ported verbatim from midtraining.py): reads a
# fineweb+finemath docbin manifest (.json), samples ONE block_size window per
# doc per epoch, and (with doc_shuffle) interleaves the sources. build_train_loader
# dispatches to the flat llm.c DataLoader for a .bin and to this for a .json.
# ---------------------------------------------------------------------------
class DocManifestDataLoader:
    def __init__(
        self,
        manifest_path,
        B,
        T,
        rank=0,
        world_size=1,
        cache_size=4,
        shuffle_docs=False,
        shuffle_seed=1337,
    ):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.manifest_path = os.path.abspath(manifest_path)
        self.epoch = 0
        self.shuffle_docs = bool(shuffle_docs)
        self.shuffle_seed = int(shuffle_seed)
        self._cache_size = max(1, cache_size)
        self._shard_cache = OrderedDict()

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        tokenized_dir = manifest["tokenized_dir"]
        files = [item for item in manifest["files"] if int(item["docs_kept"]) > 0]
        if not files:
            raise ValueError(f"No kept documents found in manifest: {self.manifest_path}")

        self.shards = []
        doc_offsets = [0]
        total_docs = 0
        for item in files:
            bin_path = os.path.join(tokenized_dir, item["bin_path"])
            lengths_path = os.path.join(tokenized_dir, item["lengths_path"])
            docs_kept = int(item["docs_kept"])
            self.shards.append(
                {
                    "bin_path": bin_path,
                    "lengths_path": lengths_path,
                    "docs_kept": docs_kept,
                    "source_relpath": item["source_relpath"],
                }
            )
            total_docs += docs_kept
            doc_offsets.append(total_docs)

        self.doc_offsets = np.asarray(doc_offsets, dtype=np.int64)
        self.total_docs = total_docs
        self.total_tokens = int(manifest["qwen_tokens_kept"])
        self.docs_per_global_step = self.B * self.world_size
        self.usable_docs = (self.total_docs // self.docs_per_global_step) * self.docs_per_global_step
        if self.usable_docs <= 0:
            raise ValueError(
                f"Doc-aware dataset is too small for B={self.B}, world_size={self.world_size}: "
                f"total_docs={self.total_docs}"
            )

        self.num_tokens = self.usable_docs * self.T
        self.current_position = self.rank * self.B
        self.doc_order = np.arange(self.usable_docs, dtype=np.int64)
        self._refresh_doc_order()

        if self.rank == 0:
            print(f"DocManifestDataLoader: {self.total_docs:,} docs, {self.total_tokens:,} raw kept tokens")
            print(
                f"DocManifestDataLoader effective epoch: {self.usable_docs:,} docs -> "
                f"{self.num_tokens:,} training tokens"
            )
            print(f"DocManifestDataLoader manifest: {self.manifest_path}")
            if self.shuffle_docs:
                print(f"DocManifestDataLoader doc shuffling: enabled (seed={self.shuffle_seed})")

    def _load_shard(self, shard_idx):
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached

        shard = self.shards[shard_idx]
        lengths = np.load(shard["lengths_path"], mmap_mode="r")
        offsets = np.zeros(len(lengths), dtype=np.uint64)
        if len(lengths) > 1:
            np.cumsum(lengths[:-1], dtype=np.uint64, out=offsets[1:])
        tokens = np.memmap(shard["bin_path"], dtype=np.uint32, mode="r")
        cached = (tokens, lengths, offsets)
        self._shard_cache[shard_idx] = cached
        if len(self._shard_cache) > self._cache_size:
            self._shard_cache.popitem(last=False)
        return cached

    def _locate_doc(self, global_doc_idx):
        shard_idx = int(np.searchsorted(self.doc_offsets, global_doc_idx, side="right") - 1)
        local_doc_idx = int(global_doc_idx - self.doc_offsets[shard_idx])
        return shard_idx, local_doc_idx

    def _span_start(self, global_doc_idx, doc_len):
        max_start = doc_len - (self.T + 1)
        if max_start <= 0:
            return 0
        mixed = (
            (int(global_doc_idx) + 1) * 0x9E3779B185EBCA87
            + (int(self.epoch) + 1) * 0xC2B2AE3D27D4EB4F
        ) & 0xFFFFFFFFFFFFFFFF
        return mixed % (max_start + 1)

    def _refresh_doc_order(self):
        if not self.shuffle_docs:
            return
        rng = np.random.default_rng(self.shuffle_seed + int(self.epoch))
        self.doc_order = rng.permutation(self.usable_docs).astype(np.int64)

    def _doc_window(self, global_doc_idx):
        shard_idx, local_doc_idx = self._locate_doc(global_doc_idx)
        tokens, lengths, offsets = self._load_shard(shard_idx)
        doc_len = int(lengths[local_doc_idx])
        start = self._span_start(global_doc_idx, doc_len)
        doc_offset = int(offsets[local_doc_idx])
        buf = tokens[doc_offset + start:doc_offset + start + self.T + 1]
        if len(buf) != self.T + 1:
            raise RuntimeError(
                f"Short doc window for global_doc_idx={global_doc_idx}: "
                f"doc_len={doc_len}, start={start}, got={len(buf)}"
            )
        return np.asarray(buf, dtype=np.int64)

    def next_batch(self):
        batch_docs = []
        for i in range(self.B):
            doc_idx = self.current_position + i
            if self.shuffle_docs:
                doc_idx = int(self.doc_order[doc_idx])
            batch_docs.append(self._doc_window(doc_idx))

        buf = torch.from_numpy(np.stack(batch_docs, axis=0))
        if torch.cuda.is_available():
            buf = buf.pin_memory()
        x = buf[:, :-1]
        y = buf[:, 1:]

        self.current_position += self.docs_per_global_step
        if self.current_position + self.B > self.usable_docs:
            self.current_position = self.rank * self.B
            self.epoch += 1
            self._refresh_doc_order()
            if self.rank == 0:
                print(f"\n*** Epoch {self.epoch} completed ***\n")
        return x, y

    def get_position(self):
        return int(self.current_position)

    def set_position(self, position):
        self.current_position = int(position)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._refresh_doc_order()

    def start_from_beginning(self):
        self.current_position = self.rank * self.B

    def resume_from_checkpoint_position(self, position, drawn_tokens=None):
        # drawn_tokens accepted for signature parity with WeightedMultiLoader; this
        # loader derives its epoch from doc-order position, not a token tally.
        self.current_position = int(position) + self.rank * self.B

    def steps_from_position(self, position):
        return int(max(0, position) // self.docs_per_global_step)


def build_train_loader(
    data_path,
    batch_size,
    block_size,
    rank,
    world_size,
    doc_shuffle=0,
    doc_shuffle_seed=1337,
    train_sources=None,
    train_tokens=0,
):
    if train_sources:
        srcs = parse_train_sources(train_sources)
        return WeightedMultiLoader(
            srcs, batch_size, block_size, rank, world_size,
            seed=int(doc_shuffle_seed), train_tokens=train_tokens,
        )
    if data_path.endswith(".json"):
        return DocManifestDataLoader(
            data_path,
            batch_size,
            block_size,
            rank,
            world_size,
            shuffle_docs=bool(doc_shuffle),
            shuffle_seed=int(doc_shuffle_seed),
        )
    return DataLoader(data_path, batch_size, block_size, rank, world_size)


# Import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ArgonneConfig, ArgonneModel
from transformers import AutoTokenizer


def get_base_model(model):
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def apply_fp8_training(model, include_lm_head=True):
    """Argonne-3.5: swap eligible nn.Linear -> torchao Float8Linear (tensorwise DYNAMIC scaling) so the
    Linear matmuls run in FP8 on H200 tensor cores (~1.25x throughput, quality-neutral per the recipe
    search). Dynamic scaling adds NO persistent params/buffers, so checkpoints + state_dict stay 1:1
    with plain nn.Linear (resume/export work). Skips any Linear whose dims aren't /16 (scaled_mm req).
    Call BEFORE DDP/compile, with master weights in fp32. Returns (n_converted, n_skipped, lm_status)."""
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    try:
        from torchao.float8 import Float8LinearRecipeName            # torchao >= ~0.10
    except ImportError:
        from torchao.float8.config import Float8LinearRecipeName     # older torchao
    cfg = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.TENSORWISE)
    stat = {"c": 0, "s": 0, "lm": "absent"}

    def flt(mod, fqn):
        is_lm = "lm_head" in fqn
        if not isinstance(mod, nn.Linear):
            return False
        if is_lm and not include_lm_head:
            stat["s"] += 1; stat["lm"] = "skipped(flag_off)"; return False
        if (mod.in_features % 16) or (mod.out_features % 16):
            stat["s"] += 1
            if is_lm:
                stat["lm"] = f"SKIPPED(dim_not_/16: out={mod.out_features}) -> NO fp8 lm_head speedup!"
            return False
        stat["c"] += 1
        if is_lm:
            stat["lm"] = "converted"
        return True

    convert_to_float8_training(model, module_filter_fn=flt, config=cfg)
    return stat["c"], stat["s"], stat["lm"]


def generate_text(model, tokenizer, device, prompt="Long long time ago", max_new_tokens=100):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_length = input_ids.shape[1] + max_new_tokens
        gen_model = get_base_model(model)
        with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
            output = gen_model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.8, top_p=0.95)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return generated_text


def _nonfinite_params(base_model, limit=5):
    """GATE 1 (pre-write): does the LIVE model contain NaN/Inf?

    This is the gate that makes latest-only retention safe. model.py deliberately turns a NaN
    loss into a zero-gradient no-op and keeps training, so a diverged run looks healthy from the
    loss curve alone -- and without this check the next save would write the corrupt weights and
    the prune would then delete the last GOOD checkpoint. GPU-side reduction, well under a second.
    """
    bad = []
    for name, prm in base_model.named_parameters():
        if not torch.isfinite(prm.detach()).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _verify_written_checkpoint(path, expect_step, expect_tensors):
    """GATE 2 (post-write, pre-commit): re-open the bytes we just wrote and prove they are a
    complete, loadable checkpoint. Returns None if good, else a reason string.

    Catches the truncated/partial write -- a slice killed mid-torch.save previously left a file
    that LOOKED like a checkpoint. mmap keeps this cheap; reading a sample of tensors forces real
    disk I/O so a file with a valid header but missing tail is still caught.
    """
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:  # unreadable, truncated, or not a zip at all
        return f"unreadable ({type(e).__name__}: {e})"
    for k in ("global_step", "tokens_processed", "model_state_dict", "optimizer_state_dict"):
        if k not in ck:
            return f"missing key {k!r}"
    if ck["global_step"] != expect_step:
        return f"global_step {ck['global_step']} != {expect_step}"
    sd = ck["model_state_dict"]
    if len(sd) != expect_tensors:
        return f"{len(sd)} tensors on disk, expected {expect_tensors}"
    names = list(sd)
    # Sample the ends, the biggest tensor, and 5 evenly spaced tensors through the stack. The
    # ends+largest alone collapse to just the two tied embedding tensors on this arch, which
    # would leave every transformer layer unread; spreading the sample forces real disk I/O
    # across the whole file so a valid-header/bad-tail write cannot slip through. The .tmp is
    # still in page cache at this point, so this costs well under a second.
    idx = {0, len(names) - 1}
    idx.update(round(i * (len(names) - 1) / 6) for i in range(1, 6))
    sample = {names[i] for i in idx}
    sample.add(max(names, key=lambda k: sd[k].numel()))
    for n in sorted(sample):
        try:
            if not torch.isfinite(sd[n].float()).all():
                return f"non-finite values in {n}"
        except Exception as e:
            return f"could not read {n} ({type(e).__name__}: {e})"
    return None


def save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, loss, data_position, checkpoint_dir):
    """Save ONE checkpoint, verify it, and only then delete the previous one.

    Order is the whole point (owner directive 2026-08-14): latest-only retention is only safe if
    the new checkpoint is proven good BEFORE the old one is removed. Any failure leaves the
    previous checkpoint untouched and returns None -- a run that cannot save is far better than a
    run whose only checkpoint is corrupt.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_model = get_base_model(model)

    bad = _nonfinite_params(base_model)
    if bad:
        print(f"REFUSING TO SAVE at step {global_step}: non-finite parameters in {bad}"
              f"{' (and more)' if len(bad) >= 5 else ''}. The previous checkpoint is left intact.",
              flush=True)
        return None

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    tmp_path = checkpoint_path + ".tmp"
    checkpoint = {
        'global_step': global_step,
        'tokens_processed': tokens_processed,
        'loss': loss,
        'data_position': data_position,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    n_tensors = len(checkpoint['model_state_dict'])
    # write to .tmp so a mid-write kill can never leave a plausible-looking checkpoint_step_N.pt
    torch.save(checkpoint, tmp_path)

    why = _verify_written_checkpoint(tmp_path, global_step, n_tensors)
    if why is not None:
        print(f"REFUSING TO COMMIT checkpoint at step {global_step}: {why}. "
              f"Discarding {tmp_path}; the previous checkpoint is left intact.", flush=True)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None

    os.replace(tmp_path, checkpoint_path)   # atomic: the file appears complete or not at all

    latest_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    latest_tmp_path = latest_path + ".tmp"
    try:
        if os.path.lexists(latest_tmp_path):
            os.remove(latest_tmp_path)
        os.symlink(os.path.basename(checkpoint_path), latest_tmp_path)
        os.replace(latest_tmp_path, latest_path)
    except OSError:
        pass

    # ONLY NOW is it safe to delete the previous checkpoint.
    print(f"[retention] checkpoint at step {global_step} verified; pruning older checkpoints",
          flush=True)
    prune_old_checkpoints(checkpoint_dir, keep_path=checkpoint_path)
    return checkpoint_path


def cleanup_stale_tmp_checkpoints(checkpoint_dir, min_age_s=3600):
    """Remove abandoned .tmp checkpoint writes left behind by a slice killed mid-save.

    save_checkpoint writes to checkpoint_step_<N>.pt.tmp and renames it into place only after it
    verifies. If SLURM kills the slice during that write (preemption, node failure, wall clock)
    the .tmp survives -- and nothing else ever removes it, because prune_old_checkpoints
    deliberately ignores .tmp so it can never delete an in-flight write. Each file is the full
    checkpoint (~25 GB at 2.06B: fp32 weights + AdamW moments) and the name carries the step, so
    they do not overwrite each other; a handful of preempted slices silently costs hundreds of GB
    across a multi-week chain.

    Only files older than min_age_s are removed, so this can never race a save that is genuinely
    in progress -- a 25 GB torch.save takes minutes, not an hour.
    """
    try:
        now = time.time()
        for p in glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt.tmp")):
            try:
                age = now - os.path.getmtime(p)
                if age < min_age_s:
                    print(f"[retention] leaving {os.path.basename(p)} alone "
                          f"({age/60:.1f} min old -- may be an active write)", flush=True)
                    continue
                gib = os.path.getsize(p) / 2**30
                os.remove(p)
                print(f"[retention] removed abandoned partial write {os.path.basename(p)} "
                      f"({gib:.1f} GiB freed, {age/3600:.1f} h old)", flush=True)
            except OSError as e:
                print(f"[retention] could not remove {p}: {e}", flush=True)
    except Exception as e:  # cleanup must never take down a slice before it even starts
        print(f"[retention] stale-tmp cleanup skipped: {e}", flush=True)


def get_latest_checkpoint_path(checkpoint_dir):
    latest_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    if os.path.exists(latest_path):
        return latest_path

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    if not checkpoints:
        return None

    steps = [int(f.split("_step_")[-1].replace(".pt", "")) for f in checkpoints]
    latest_step = max(steps)
    return os.path.join(checkpoint_dir, f"checkpoint_step_{latest_step}.pt")


def write_completion_marker(marker_path, global_step, tokens_processed, final_model_dir):
    marker_dir = os.path.dirname(marker_path)
    if marker_dir:
        os.makedirs(marker_dir, exist_ok=True)
    tmp_path = marker_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"completed_at_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.write(f"global_step={global_step}\n")
        f.write(f"tokens_processed={tokens_processed}\n")
        f.write(f"final_model_dir={final_model_dir}\n")
    os.replace(tmp_path, marker_path)


def main():
    if IS_MAIN:
        print("=" * 60)
        print("Argonne Model Training (DDP)")
        print("=" * 60)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Using device: {DEVICE}, World size: {WORLD_SIZE}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    VOCAB_SIZE = len(tokenizer)
    if args.fp8 == 1 and args.fp8_lm_head == 1 and (VOCAB_SIZE % 16 != 0):
        # torch._scaled_mm needs both GEMM dims %16; len(tokenizer)=151669 is not, so the tied lm_head
        # would silently stay bf16 (only ~1.18x instead of the recipe's 1.25x). Pad vocab to a multiple
        # of 128 (tensor-core aligned): padding rows are unused ids and the final HF export trims
        # embeddings back to len(tokenizer). NOTE: this changes the embedding shape, so argonne3.5 fp8
        # checkpoints are NOT resume-compatible with unpadded (len-151669) checkpoints — fresh run only.
        padded = ((VOCAB_SIZE + 127) // 128) * 128
        if IS_MAIN:
            print(f"FP8 lm_head: padding vocab {VOCAB_SIZE} -> {padded} (mult of 128; export trims back)")
        VOCAB_SIZE = padded
    if IS_MAIN:
        print(f"Vocab size: {VOCAB_SIZE}, EOS token ID: {tokenizer.eos_token_id}")

    # Create model
    config = ArgonneConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=args.block_size,
        rope_theta=ROPE_THETA,
        use_flash_attention=args.flash_attention == 1,
        qk_norm=ENABLE_QK_NORM,
        v_norm=ENABLE_V_NORM,
        sandwich_norm=ENABLE_SANDWICH_NORM,
        z_loss_weight=Z_LOSS_WEIGHT,
        mtp_horizon=MTP_HORIZON if ENABLE_MTP else 1,
        mtp_loss_weight=MTP_LOSS_WEIGHT if ENABLE_MTP else 0.0,
        interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
        local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
        # argonne4.5 (no-ops unless A45 is on)
        attn_pattern=ATTN_PATTERN,
        sliding_window_size=SLIDING_WINDOW_SIZE,
        nope_global=NOPE_GLOBAL,
        attn_gate=ATTN_GATE,
        mlp_type=MLP_TYPE,
        mtp_module_layers=MTP_MODULE_LAYERS,
        doc_mask=DOC_MASK,
        logit_softcap=LOGIT_SOFTCAP,
        loss_chunk_size=args.loss_chunk_size,
        tie_word_embeddings=True,
    )
    config._keep_in_fp32_modules = []
    model = ArgonneModel(config)
    model = model.to(DEVICE)
    # Model stays in fp32 — autocast handles bf16/fp16 for forward pass
    # This keeps optimizer states in fp32 for proper precision

    # Gradient checkpointing (before DDP and compile)
    if args.gradient_checkpointing == 1:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if IS_MAIN:
                print("Gradient checkpointing enabled")
        # Selective checkpointing: checkpoint every Nth layer (stride>1 => store more activations,
        # recompute less => faster + more HBM). stride=1 keeps prior behavior (checkpoint all).
        if args.checkpoint_stride > 1 and hasattr(model, 'checkpoint_stride'):
            model.checkpoint_stride = int(args.checkpoint_stride)
            if IS_MAIN:
                print(f"Selective activation checkpointing: stride={args.checkpoint_stride} "
                      f"(checkpoint all EXCEPT store every {args.checkpoint_stride}th layer)")

    # Argonne-3.5 FP8 (torchao float8): convert Linear matmuls to FP8 BEFORE DDP/compile.
    if args.fp8 == 1:
        if args.torch_compile != 1 and IS_MAIN:
            print("WARNING: --fp8 without --torch_compile — FP8 scaling won't fuse; expect NO speedup.")
        embed_w = model.get_input_embeddings().weight            # capture the tied weight pre-conversion
        n_conv, n_skip, lm_status = apply_fp8_training(model, include_lm_head=(args.fp8_lm_head == 1))
        # The tie must survive (torchao reuses the weight Parameter). Fail loudly if it silently broke.
        assert model.get_output_embeddings().weight is embed_w, (
            "FP8 conversion broke the embedding<->lm_head tie — do not train (would be a different model)")
        if args.fp8_lm_head == 1 and lm_status != "converted" and IS_MAIN:
            print(f"WARNING: --fp8_lm_head 1 requested but lm_head {lm_status} — expect ~1.18x not 1.25x. "
                  f"Pad vocab to a multiple of 16 to enable it.")
        if IS_MAIN:
            print(f"FP8 training ON (torchao tensorwise): converted {n_conv} Linear, skipped {n_skip}; "
                  f"lm_head={lm_status}; embedding tie preserved.")

    # Wrap with DDP
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], gradient_as_bucket_view=True)
        if IS_MAIN:
            print(f"Using {WORLD_SIZE} GPUs with DistributedDataParallel")

    # torch.compile
    if args.torch_compile == 1:
        if IS_MAIN:
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.torch_compile_mode)

    if IS_MAIN:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Mixed precision: {'autocast ' + args.precision if USE_AUTOCAST else 'fp32 (no autocast)'}")

    # Create data loader (flat llm.c .bin OR doc-manifest .json, chosen by extension)
    train_loader = build_train_loader(
        args.data_path,
        args.batch_size,
        args.block_size,
        RANK,
        WORLD_SIZE,
        doc_shuffle=args.doc_shuffle,
        doc_shuffle_seed=args.doc_shuffle_seed,
        train_sources=args.train_sources,
        train_tokens=args.train_tokens,
    )
    val_loader = None
    if IS_MAIN and args.val_data_path:
        # Validation is always a flat held-out llm.c .bin (the val loop below reads
        # a contiguous slice via .tokens / .current_position); manifests are train-only.
        val_loader = DataLoader(
            args.val_data_path,
            args.val_batch_size,
            args.block_size,
            rank=0,
            world_size=1,
            start_token_offset=0,
        )

    # Estimate steps for scheduler. The doc-manifest loader exposes .num_tokens (one
    # T-window per doc per epoch = usable_docs*T); the flat loader exposes .tokens.
    num_tokens = train_loader.num_tokens if hasattr(train_loader, "num_tokens") else len(train_loader.tokens)
    estimated_steps = int((num_tokens * args.max_epochs) / ACTUAL_TOTAL_BATCH)
    if IS_MAIN:
        print(f"Training for {args.max_epochs} epoch(s) ~= {estimated_steps} steps ({num_tokens * args.max_epochs:,} tokens)")

    # Create optimizer. fused=True -> single fused CUDA kernel for the AdamW update over the
    # fp32 master params (same math as the default foreach path; numerically-equivalent), fewer
    # kernel launches / less memory traffic. Master weights are plain fp32 CUDA Parameters even
    # under torchao tensorwise fp8, so fused AdamW is supported.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )

    # Scheduler with warmup (cosine or WSD)
    min_lr = args.lr * args.min_lr_ratio
    min_lr_scale = min_lr / args.lr

    # Argonne-3.5: resolve the WSD cooldown length. A fraction (--cooldown_frac) is preferred
    # because estimated_steps is recomputed identically on every resume, so cooldown_start below
    # is stable across the wall-time-sliced training and the anneal always finishes at the run's
    # true end. Falls back to the fixed --cooldown step count when the fraction is 0.
    cooldown_steps = int(args.cooldown_frac * estimated_steps) if args.cooldown_frac > 0 else args.cooldown

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)

        if args.schedule == "cosine":
            progress = (step - args.warmup_steps) / max(1, estimated_steps - args.warmup_steps)
            return max(min_lr_scale, 0.5 * (1.0 + np.cos(np.pi * progress)))

        if cooldown_steps <= 0:
            return 1.0

        cooldown_start = max(args.warmup_steps, estimated_steps - cooldown_steps)
        if step < cooldown_start:
            return 1.0

        cooldown_progress = min(1.0, (step - cooldown_start) / max(1, cooldown_steps))
        return 1.0 - cooldown_progress * (1.0 - min_lr_scale)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    # A previous slice killed mid-save leaves a multi-GB .pt.tmp that nothing else collects.
    if IS_MAIN:
        cleanup_stale_tmp_checkpoints(args.checkpoint_dir)

    resume_from = args.resume_from or get_latest_checkpoint_path(args.checkpoint_dir)

    if resume_from and os.path.exists(resume_from):
        if IS_MAIN:
            print(f"\n=== Resuming from checkpoint: {resume_from} ===")
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        base_model = get_base_model(model)
        base_model.load_state_dict(checkpoint['model_state_dict'])

        if args.reset_schedule == 1:
            if IS_MAIN:
                print("Reset schedule mode: fresh optimizer, scheduler, step counter, data position")
                print(f"Previous training: {checkpoint['tokens_processed']:,} tokens, step {checkpoint['global_step']}")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.weight_decay,
                fused=True,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            global_step = 0
            tokens_processed = 0
            is_resumed = False
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            else:
                for _ in range(checkpoint['global_step']):
                    scheduler.step()
            global_step = checkpoint['global_step']
            tokens_processed = checkpoint['tokens_processed']
            data_position = checkpoint.get('data_position', 0)
            resumed_epoch = int(tokens_processed // num_tokens) if num_tokens > 0 else 0
            if hasattr(train_loader, "resume_from_checkpoint_position"):
                # Doc-manifest loader: data_position is a DOC index (rank-0's cursor).
                # Restore the per-epoch shuffle order first (so the resumed doc order
                # reproduces the original epoch) then place the per-rank cursor.
                if hasattr(train_loader, "set_epoch"):
                    train_loader.set_epoch(resumed_epoch)
                # Pass the checkpoint's tokens_processed as the authoritative progress
                # count: it is tracked independently of the loader and is exact, so the
                # sampler's epoch survives a micro-batch change on resume.
                train_loader.resume_from_checkpoint_position(
                    data_position, drawn_tokens=tokens_processed
                )
            else:
                train_loader.set_position(data_position + RANK * args.batch_size * args.block_size)
                train_loader.epoch = resumed_epoch
            if IS_MAIN:
                print(f"Resumed from step {global_step}, tokens: {tokens_processed:,}, epoch: {train_loader.epoch}, LR: {scheduler.get_last_lr()[0]:.2e}")
            is_resumed = True
    else:
        is_resumed = False

    # Training loop
    if IS_MAIN:
        print("\nStarting training...")
        print(f"GPUs: {WORLD_SIZE}, Batch size per GPU: {args.batch_size}")
        print(f"Sequence length: {args.block_size}")
        print(f"Total batch size: {ACTUAL_TOTAL_BATCH} tokens (requested: {args.total_batch_size})")
        print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
        print(f"Training for {args.max_epochs} epoch(s) (estimated ~{estimated_steps} steps)")
        print(f"Seed: base={BASE_SEED}, rank_seed={RUN_SEED}")
        print(
            f"QK norm: {ENABLE_QK_NORM} | v_norm: {ENABLE_V_NORM} | "
            f"sandwich_norm: {ENABLE_SANDWICH_NORM} | z_loss_weight: {Z_LOSS_WEIGHT} | "
            f"rope_theta: {ROPE_THETA} | mtp: {ENABLE_MTP} (horizon={MTP_HORIZON}, weight={MTP_LOSS_WEIGHT}) | "
            f"interleaved_local_attention: {ENABLE_INTERLEAVED_LOCAL_ATTENTION} (window={LOCAL_ATTENTION_WINDOW}) | "
            f"logit_softcap: {LOGIT_SOFTCAP}"
        )
        print(f"LR: {args.lr}, Warmup: {args.warmup_steps}, Min LR Ratio: {args.min_lr_ratio}, Precision: {args.precision}, TorchCompile: {args.torch_compile}")
        print(f"Schedule: {args.schedule}, Cooldown: {cooldown_steps} steps (frac={args.cooldown_frac}, ~{(cooldown_steps/max(1,estimated_steps)*100):.1f}% of run), Grad clip: {args.grad_clip}")
        print(f"Checkpoint interval: {args.checkpoint_interval} seconds")
        print(f"Validation data: {args.val_data_path if args.val_data_path else 'disabled (no held-out file provided)'}")
        if args.wall_time > 0:
            print(f"Wall time: {args.wall_time}s, will save checkpoint at {WALL_TIME_SAVE}s")
        if args.reset_schedule == 1:
            print("Mode: continued pretraining (fresh schedule)")
        print("-" * 60)

    if not is_resumed:
        global_step = 0
        tokens_processed = 0
    last_checkpoint_time = time.time()
    training_start_time = time.time()
    def _eval_val():
        # Val loss on the underlying (non-DDP) module at a SMALL fixed batch (val_batch_size) so the
        # logit tensor never OOMs regardless of the train batch. Used for the end eval + periodic eval.
        if val_loader is None:
            return float("nan")
        base_eval = get_base_model(model)
        was_training = base_eval.training
        base_eval.eval()
        losses = []
        with torch.no_grad():
            orig = val_loader.current_position
            val_loader.current_position = 0
            nb = min(100, max(1, min(2_000_000, len(val_loader.tokens)) // (args.val_batch_size * args.block_size)))
            for _ in range(nb):
                vx, vy = val_loader.next_batch()
                vx = vx.to(DEVICE, non_blocking=True); vy = vy.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                    vout = base_eval(vx, labels=vy)
                losses.append(vout.loss.item())
            val_loader.current_position = orig
        if was_training:
            base_eval.train()
        return float(np.mean(losses)) if losses else float("nan")

    train_losses = []
    export_refused = False   # set on rank 0 if the final checkpoint fails its integrity gates
    completed_max_epochs = train_loader.epoch >= args.max_epochs

    pbar = None
    if IS_MAIN:
        initial_steps = 0
        if is_resumed:
            # Use token-based progress so resumed runs remain accurate even if batch config changed.
            initial_steps = min(estimated_steps, int(tokens_processed // ACTUAL_TOTAL_BATCH))
        pbar = tqdm(total=estimated_steps, initial=initial_steps, desc="Training", unit="step", disable=False)

    model.train()

    if completed_max_epochs and IS_MAIN:
        print(f"\nCheckpoint is already at {train_loader.epoch} epoch(s); finalizing without more training.")

    while not completed_max_epochs:
        start_time = time.time()
        optimizer.zero_grad()
        # GPU-resident accumulator. Calling .item() inside the micro-step loop forces a
        # device sync per micro-batch, which stalls CPU run-ahead right after backward and
        # serialises the pipeline. At GRAD_ACCUM_STEPS=11 the old code did it TWICE per
        # micro-step = 22 syncs per optimizer step; this does one.
        step_loss_total = torch.zeros((), device=DEVICE, dtype=torch.float32)

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if WORLD_SIZE > 1 and micro_step < GRAD_ACCUM_STEPS - 1:
                with model.no_sync():
                    with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                        outputs = model(x, labels=y)
                        micro_loss = outputs.loss
                        loss = micro_loss / GRAD_ACCUM_STEPS
                    loss.backward()
            else:
                with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                    outputs = model(x, labels=y)
                    micro_loss = outputs.loss
                    loss = micro_loss / GRAD_ACCUM_STEPS
                loss.backward()

            tokens_processed += args.batch_size * args.block_size * WORLD_SIZE
            step_loss_total += micro_loss.detach().float()   # stays on GPU, no sync

        # the single sync per optimizer step
        step_loss = (step_loss_total / GRAD_ACCUM_STEPS).item()
        train_losses.append(step_loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        if pbar:
            pbar.update(1)

        current_lr = optimizer.param_groups[0]['lr']

        if IS_MAIN and (global_step in (1, 2, 3, 5, 10, 20, 50) or global_step % 200 == 0):
            _mem_res = torch.cuda.max_memory_reserved() / 1e9
            _mem_tot = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1e9
            print(f"  [HBM] step {global_step}: peak reserved {_mem_res:.1f}/{_mem_tot:.1f} GB ({100*_mem_res/_mem_tot:.1f}%) | micro_batch={args.batch_size} block={args.block_size} grad_ckpt={args.gradient_checkpointing} fp8={args.fp8}", flush=True)

        if IS_MAIN and global_step % 50 == 0:
            perplexity = np.exp(step_loss)
            print(f"Step {global_step} | Loss: {step_loss:.4f} | PPL: {perplexity:.2f} | Tokens: {tokens_processed:,} | LR: {current_lr:.2e}")
            if pbar:
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{current_lr:.2e}", "tokens": f"{tokens_processed/1e6:.2f}M"})

        # Periodic validation (translate-test / learning curve). Gated (default off). All ranks reach
        # here together (global_step is synced); rank 0 evals while the others wait at the barrier.
        if args.periodic_val_every > 0 and global_step % args.periodic_val_every == 0:
            if IS_MAIN and val_loader is not None:
                _pv = _eval_val()
                print(f"PERIODIC_VAL step={global_step} tokens={tokens_processed} val_loss={_pv:.4f} lr={current_lr:.3e}", flush=True)
            if WORLD_SIZE > 1:
                dist.barrier()

        # Synchronized checkpoint decision
        should_checkpoint = torch.tensor([0], device=DEVICE)
        if IS_MAIN:
            current_time = time.time()
            if current_time - last_checkpoint_time >= args.checkpoint_interval:
                should_checkpoint[0] = 1
        if WORLD_SIZE > 1:
            dist.broadcast(should_checkpoint, src=0)

        if should_checkpoint[0] == 1:
            if IS_MAIN:
                print("\n" + "=" * 60)
                print("Saving checkpoint...")
                data_position = train_loader.get_position()
                checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, step_loss, data_position, args.checkpoint_dir)
                if checkpoint_path is None:
                    # save_checkpoint already printed the reason it refused.
                    print("Checkpoint REFUSED at this step; continuing to train on the previous one.")
                else:
                    print(f"Checkpoint saved: {checkpoint_path}")

                # Pure telemetry, and it runs on rank 0 while every other rank is already
                # waiting on the barrier below -- so an exception here would not just lose a
                # sample, it would strand the whole slice until SLURM killed it at the wall
                # clock, AFTER a good checkpoint had been written. Same treatment the post-save
                # validation eval already gets. This path had never executed before 2026-08-14.
                print("\nGenerating sample text...")
                try:
                    generated = generate_text(model, tokenizer, DEVICE, prompt="Long long time ago")
                    print(f"Generated: {generated}")
                except Exception as exc:  # noqa: BLE001 - never fail a slice on post-save sampling
                    print(f"WARNING: sample generation failed and was skipped "
                          f"({type(exc).__name__}: {exc})")
                    torch.cuda.empty_cache()
                print("=" * 60 + "\n")

            if WORLD_SIZE > 1:
                dist.barrier()
            last_checkpoint_time = time.time()

        # Synchronized wall-time / deadline check -- save ONE checkpoint, then exit cleanly (the
        # weekend.sh chain resubmits on that clean exit = save-THEN-submit). Two triggers, whichever
        # fires first: (1) --save_deadline_epoch = ABSOLUTE wall clock (job_start + slice - lead) =>
        # fires a fixed time before the SLURM kill regardless of compile/startup drift = PRIMARY;
        # (2) WALL_TIME_SAVE = training-ELAPSED margin = backup.
        if WALL_TIME_SAVE > 0 or args.save_deadline_epoch > 0:
            should_wall_stop = torch.tensor([0], device=DEVICE)
            if IS_MAIN:
                elapsed = time.time() - training_start_time
                if WALL_TIME_SAVE > 0 and elapsed >= WALL_TIME_SAVE:
                    should_wall_stop[0] = 1
                if args.save_deadline_epoch > 0 and time.time() >= args.save_deadline_epoch:
                    should_wall_stop[0] = 1
            if WORLD_SIZE > 1:
                dist.broadcast(should_wall_stop, src=0)

            if should_wall_stop[0] == 1:
                if IS_MAIN:
                    print(f"\nApproaching wall limit (deadline_epoch={args.save_deadline_epoch}, wall_time={args.wall_time}s). Saving checkpoint and exiting...")
                    data_position = train_loader.get_position()
                    checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, step_loss, data_position, args.checkpoint_dir)
                    if checkpoint_path is None:
                        print("Wall-time checkpoint REFUSED; exiting on the previous good checkpoint.")
                    else:
                        print(f"Wall time checkpoint saved: {checkpoint_path}")
                if WORLD_SIZE > 1:
                    dist.barrier()
                break

        # Synchronized epoch completion check
        should_stop = torch.tensor([0], device=DEVICE)
        if train_loader.epoch >= args.max_epochs:
            should_stop[0] = 1
        if WORLD_SIZE > 1:
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)

        if should_stop[0] == 1:
            if IS_MAIN:
                print(f"\nCompleted {args.max_epochs} epoch(s) at step {global_step}. Finalizing...")
            completed_max_epochs = True
            break

    if pbar:
        pbar.close()

    if IS_MAIN:
        print("-" * 60)
        elapsed_time = time.time() - training_start_time
        print(f"Training completed in {elapsed_time:.1f} seconds!")

    # Evaluate on validation (rank 0 only)
    if IS_MAIN:
        val_losses = []
        if val_loader is not None:
            print("\nEvaluating on validation...")
            _vl = _eval_val()
            if not np.isnan(_vl):
                val_losses = [_vl]
        else:
            print("\nValidation skipped: no held-out validation file was provided.")

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else float("nan")
        val_loss_str = f"{val_loss:.4f}" if val_losses else "n/a"
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss_str}")

        if completed_max_epochs:
            print("\nSaving final checkpoint...")
            data_position = train_loader.get_position()
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, global_step, tokens_processed, train_loss, data_position, args.checkpoint_dir)
            if checkpoint_path is None:
                # The gate refused these weights. Exporting them to final_model_dir would publish
                # exactly the corrupt model the gate exists to stop, so skip the export. Do NOT
                # raise here: this block is rank-0 only, and bailing out would strand the other
                # ranks on the barrier below until SLURM killed the job at the wall clock. Flag it
                # and exit nonzero after every rank has cleaned up.
                export_refused = True
                print("Final checkpoint REFUSED — skipping HF export. The last good checkpoint in "
                      f"{args.checkpoint_dir} is the deliverable; investigate before exporting.")
            else:
                print(f"Final checkpoint saved: {checkpoint_path}")

                final_model_dir = args.final_model_dir or os.path.join(args.checkpoint_dir, "final_model_complete")
                os.makedirs(final_model_dir, exist_ok=True)
                save_model = get_base_model(model)
                if args.fp8 == 1:
                    # Export a clean nn.Linear model so the shipped HF checkpoint needs no torchao at
                    # inference. Dynamic tensorwise FP8 keeps weights in high precision → state_dict is 1:1.
                    clean = ArgonneModel(config)
                    missing, unexpected = clean.load_state_dict(save_model.state_dict(), strict=False)
                    assert not missing, (
                        f"FP8 export: clean model missing weights {missing[:8]} — refusing to export garbage")
                    if hasattr(clean, "tie_weights"):
                        clean.tie_weights()
                    if IS_MAIN:
                        print(f"FP8 export: rebuilt clean nn.Linear model ({len(missing)} missing / {len(unexpected)} extra keys)")
                    save_model = clean.to(DEVICE)

                actual_vocab = len(tokenizer)
                embed = save_model.get_input_embeddings()
                if embed.weight.shape[0] > actual_vocab:
                    print(f"Trimming embeddings from {embed.weight.shape[0]} to {actual_vocab}")
                    embed.weight = nn.Parameter(embed.weight[:actual_vocab])
                    lm_head = save_model.get_output_embeddings()
                    if lm_head is not None:
                        lm_head.weight = nn.Parameter(lm_head.weight[:actual_vocab])
                    save_model.config.vocab_size = actual_vocab

                save_model.save_pretrained(final_model_dir)
                tokenizer.save_pretrained(final_model_dir)
                config.save_pretrained(final_model_dir)
                print(f"Final model + tokenizer + config saved to: {final_model_dir}")

                if args.completion_marker:
                    write_completion_marker(args.completion_marker, global_step, tokens_processed, final_model_dir)
                    print(f"Completion marker written to: {args.completion_marker}")
        else:
            print("\nFinal checkpoint/model export skipped because training stopped before completing max_epochs.")

        elapsed_time = time.time() - training_start_time
        print("\n" + "=" * 60)
        print(f"SUMMARY: train_loss={train_loss:.4f} val_loss={val_loss_str} tokens_per_sec={tokens_processed/elapsed_time:.2f} steps={global_step}")
        print("=" * 60)

    if WORLD_SIZE > 1:
        dist.barrier()

    cleanup_distributed()

    # Every rank has now cleaned up, so it is safe to fail the job. Nonzero tells the resubmit
    # chain NOT to treat a refused final checkpoint as a successful finish.
    if export_refused:
        sys.exit(1)

def prune_old_checkpoints(checkpoint_dir, keep_path):
    """LATEST-ONLY retention (2026-08-05 owner directive, global CLAUDE.md).

    Keep `keep_path` (the checkpoint just written) and delete every other checkpoint_step_*.pt in
    THIS dir -- i.e. this run's own earlier checkpoints. At 12-35 GB apiece a non-rotating dir eats
    hundreds of GB, and one checkpoint is all a resume needs.

    HISTORY, because this spot previously said the opposite: a keep-last-3 prune added 2026-07-28
    was removed 2026-07-29 after it silently deleted 14 of 17 checkpoints and read as data loss.
    The owner reversed the policy on 2026-08-05: latest-only is now the default. What was actually
    wrong then was silence and aggressiveness, so this version PRINTS every deletion.

    Deliberately narrow, so it can only ever remove this run's superseded checkpoints:
      * only `checkpoint_step_<int>.pt` directly in `checkpoint_dir` (glob, not a walk)
      * never `keep_path`, never a symlink (so `checkpoint_last.pt` is untouched), never a `.tmp`
      * never the numerically-highest step present, even if `keep_path` somehow is not it
      * per-file try/except: a failed unlink must not abort training after a good save
    """
    try:
        keep = {os.path.realpath(keep_path)}
        found = []
        for p in glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt")):
            m = re.fullmatch(r"checkpoint_step_(\d+)\.pt", os.path.basename(p))
            if m and not os.path.islink(p):
                found.append((int(m.group(1)), p))
        if len(found) <= 1:
            return
        # Belt and braces: whatever has the highest step number also stays.
        keep.add(os.path.realpath(max(found)[1]))
        for _, p in sorted(found):
            if os.path.realpath(p) in keep:
                continue
            try:
                gib = os.path.getsize(p) / 2**30
                os.remove(p)
                print(f"[retention] removed superseded checkpoint {os.path.basename(p)} "
                      f"({gib:.1f} GiB freed); keeping {os.path.basename(keep_path)}", flush=True)
            except OSError as e:
                print(f"[retention] could not remove {p}: {e}", flush=True)
    except Exception as e:  # retention must never take down a run that just saved successfully
        print(f"[retention] prune skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
