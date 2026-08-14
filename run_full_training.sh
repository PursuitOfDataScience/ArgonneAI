#!/bin/bash
#SBATCH --job-name=software
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --exclude=midway3-0423,midway3-0385,midway3-[0298,0377-0378,0602-0606]  # 0385 added 2026-08-02: HwPowerBrake, SM pinned ~360MHz of 1785 (20%), silently 5x slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# CPU RAM. A FRESH slice peaks only ~16 GiB (measured 2026-07-21). But a RESUMING slice torch.loads
# the ~11.6 GB checkpoint to CPU on EACH of the 3 DDP ranks (pretrain.py:825 map_location='cpu') =>
# ~40-44 GiB peak. 56G is sized for that resume transient (down from 64G); sizing to the 16 GiB fresh
# peak would OOM every resume slice. TODO: measure a resume slice's MaxRSS and trim toward 1.3x it;
# a bigger cut (=> ~24G) needs resume to load via map_location=cuda (drops the CPU transient).
#SBATCH --mem=56G
#SBATCH --gres=gpu:3
#SBATCH --constraint=H100
# NOTE: the header constraint only applies to a MANUAL `sbatch run_full_training.sh`. Every slice
# submitted by weekend.sh/night.sh or by the self-resubmit path passes --constraint explicitly,
# derived from PRETRAIN_CARD (H100|H200). See pretrain_card() below.
#SBATCH --time=07:00:00
#SBATCH --output=report/1-train.out
#SBATCH --error=report/1-train.err

# =============================================================================
# run_full_training.sh (Argonne 4.0) -- ONE base-pretraining GPU slice.
#
# SAME per-slice WORKER + workflow as argonne3.5 (weekend.sh / night.sh -> this):
#   * weekend.sh -> submits this with AUTO_RESUBMIT=1  (continuous self-chain)
#   * night.sh   -> submits this once at 23:00 with AUTO_RESUBMIT=0 (one slice)
#   Both pass RESUBMIT_ON_FAILURE=1 so a crashed slice (ECC/NCCL/OOM/node death)
#   resubmits a FRESH slice that resumes from the last checkpoint and EXCLUDES the
#   failed node, up to FAILURE_RETRY_MAX times.
#
# Marker-gated stages (flow IDENTICAL to 3.5), each AUTO-RESUMING via its own
# get_latest_checkpoint_path() -- no --resume_from needed from the shell:
#   1. pretrain.py          (BASE: 50/30/20 edu/math/code weighted mix) -> .pretrain_complete
#   2. continue_pretrain.py (MIDTRAIN phase A: reasoning anneal, block 1024) -> .continue_pretrain_complete
#   3. continue_pretrain.py (MIDTRAIN phase B: ctx-extension to block 13568, models/argonne4_midtrain)
#                            -- gated on .midtrain_armed (default OFF, same as 3.5 before arming)
# The pretrain->continue transition is fully automatic (marker-gated); the ctx-extension
# stage stays gated on .midtrain_armed so the chain cleanly STOPS after the reasoning anneal
# until you arm it (arm once the block-13568 batch is smoke-tested).
#
# ARGONNE4.0 vs 3.5 (workflow unchanged; only model/data/sizing differ):
#   * ARCH  : 1.04B (hidden1536/32L/6h/2kv/head_dim256/inter4096) -- set in the .py constants,
#             NOT here. head_dim derived (1536/6=256); every fp8 GEMM dim /16-divisible.
#   * DATA  : pretrain reads the validated 50% edu / 30% math / 20% code WEIGHTED mixture via
#             pretrain.py --train_sources (per-micro-batch weighted sampling, DDP-synced,
#             resumable) -- NO pre-built blend bin. Reasoning anneal reuses 3.5's flat corpora.
#   * SAVE  : the ~11 GB checkpoint flushes ~3x faster than 3.5's ~32 GiB, so the .py's
#             pre-kill save margin was cut 300s->150s (in pretrain.py / continue_pretrain.py),
#             handing ~150s/slice back to training (owner directive 2026-07-20).
#   * fp8   : both stages run --fp8 1 (pads vocab 151669->151680) so pretrain->continue->midtrain
#             resume shapes agree. models/argonne4_pretrain + models/argonne4_midtrain are FRESH dirs.
# =============================================================================

set -eo pipefail

# --- Auto-resubmit plumbing (clean wall-time chain; used by weekend.sh) -------
# Absolute repo root, valid on every compute node (shared home FS). Honors an
# inherited REPO_ROOT (weekend.sh/night.sh export it, self-resubmits carry it via
# --export=ALL) so a moved repo works end-to-end; the fallback is the argonne4.0
# worktree. NEVER hardcode a throwaway path in an untracked .sh (lesson 2026-07-17).
REPO_ROOT="${REPO_ROOT:-/home/youzhi/ArgonneAI-4.0}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/report}"
LOG_BASENAME="${LOG_BASENAME:-train}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-0}"
AUTO_RESUBMIT_TIME="${AUTO_RESUBMIT_TIME:-07:00:00}"

# --- Failure-retry plumbing (transient infra: GPU ECC, node death, NCCL) ------
RESUBMIT_ON_FAILURE="${RESUBMIT_ON_FAILURE:-0}"
FAILURE_RETRY_COUNT="${FAILURE_RETRY_COUNT:-0}"
FAILURE_RETRY_MAX="${FAILURE_RETRY_MAX:-5}"
EXTRA_EXCLUDE="${EXTRA_EXCLUDE:-}"   # comma-list of failed nodes, accumulated across retries
# A command-line --exclude OVERRIDES (does not merge with) the #SBATCH directive above,
# so a failure-retry must restate the full base list. Keep in sync with #SBATCH --exclude.
BASE_EXCLUDE="midway3-0423,midway3-0385,midway3-[0298,0377-0378,0602-0606]"   # 0385: HwPowerBrake, ~20% clock

# --- Argonne-4.0 checkpoint dir = models/argonne4_pretrain (FRESH; fp8 pads vocab 151669->151680, so
# it must hold NO 151669 checkpoints -- a brand-new dir is safe). The `argonne4_` PREFIX denotes these
# as Argonne-4.0 checkpoints and keeps them fully distinct from Argonne-3.5's models/pretrain (which
# stays as-is and is being written live right now). Set CKPT_DIR_OVERRIDE to redirect.
CHECKPOINT_ROOT=/project/rcc/youzhi/models
CKPT_DIR="${CKPT_DIR_OVERRIDE:-${CHECKPOINT_ROOT}/argonne45_pretrain}"

find_next_train_log_index() {
  local max_index
  max_index=$(
    find "$REPORT_DIR" -maxdepth 1 -type f \( -name "[0-9]*-${LOG_BASENAME}.out" -o -name "[0-9]*-${LOG_BASENAME}.err" \) -printf '%f\n' 2>/dev/null \
      | sed -E "s/^([0-9]+)-${LOG_BASENAME}\.(out|err)$/\1/" \
      | sort -n \
      | tail -n 1
  )

  if [[ -n "$max_index" ]]; then
    echo $((max_index + 1))
  else
    echo 1
  fi
}

RESUBMIT_DONE=0
stage_name=""

# --- Day/night-aware slice schedule (America/Chicago) -------------------------------------------
# DAY 07:00-23:00: 1h slices (be polite -- release the node hourly so daytime users can grab it),
#   save-once (huge ckpt interval); the last day slice is capped so it ends ~23:00. NIGHT 23:00-07:00:
#   ONE straight slice to 07:00 (cluster is quiet -> fewer resubmits + no per-slice recompile tax),
#   with HOURLY periodic checkpoints (3600s) as the long-run safety net. Both save ~SAVE_LEAD_SECONDS
#   before the slice's SLURM limit, then resubmit on the clean exit. Sets SCHED_SECONDS + SCHED_CKPT_INTERVAL.
compute_slice_schedule() {
  local h m s sod until_night until_morning
  h=$(TZ='America/Chicago' date +%-H); m=$(TZ='America/Chicago' date +%-M); s=$(TZ='America/Chicago' date +%-S)
  sod=$(( 10#$h*3600 + 10#$m*60 + 10#$s ))
  # 2026-07-29 (owner directive: "it should be like pretraining using 1 hour per job"): the NIGHT
  # branch used to request ONE straight slice to 07:00, which is how the first post-pretrain anneal
  # slice ended up holding 3 GPUs for 5h46m. Both branches are now 1h, matching the pretrain cadence
  # and releasing the node hourly. Set SLICE_MODE=daynight to restore the old long-night behaviour.
  if [[ "${SLICE_MODE:-hourly}" != "daynight" ]]; then
    SCHED_SECONDS=3600
    SCHED_CKPT_INTERVAL=999999      # save-once; the in-process wall save handles it
  elif (( sod >= 25200 && sod < 82800 )); then          # DAY
    until_night=$(( 82800 - sod ))
    if (( until_night < 3600 )); then SCHED_SECONDS=$until_night; else SCHED_SECONDS=3600; fi
    SCHED_CKPT_INTERVAL=999999
  else                                                  # NIGHT
    if (( sod >= 82800 )); then until_morning=$(( (86400 - sod) + 25200 )); else until_morning=$(( 25200 - sod )); fi
    SCHED_SECONDS=$until_morning
    SCHED_CKPT_INTERVAL=3600
  fi
  if (( SCHED_SECONDS < 1200 )); then SCHED_SECONDS=1200; fi   # MIN 20min: a shorter slice can't finish compile(~5min)+save(12GB)+resubmit before its SLURM kill -- a 300s sliver stalled the chain at the 23:00 boundary 2026-07-21
}
fmt_hms() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

# Clean-chain resubmit (weekend.sh AUTO_RESUBMIT=1): pre-submit the next slice on the
# SLURM timeout signal / clean exit. Forwards CKPT_DIR_OVERRIDE so the whole chain stays
# pinned to the same fresh 4.0 dir, and EXPLICITLY resets FAILURE_RETRY_COUNT=0,EXTRA_EXCLUDE=
# (a clean slice must reset the per-consecutive-crash-streak budget; with --export=ALL a value
# leaked from a prior failure-retry would otherwise make the cap count lifetime crashes).
submit_next_slice() {
  local current_index next_index next_available next_job_id
  current_index="${LOG_INDEX:-}"
  if [[ "$current_index" =~ ^[0-9]+$ ]]; then
    next_index=$((current_index + 1))
  else
    next_index=$(find_next_train_log_index)
  fi
  next_available=$(find_next_train_log_index)
  if ((next_index < next_available)); then
    next_index="$next_available"
  fi
  compute_slice_schedule   # day/night-aware: SCHED_SECONDS + SCHED_CKPT_INTERVAL from Chicago time NOW
  local sched_hms; sched_hms=$(fmt_hms "$SCHED_SECONDS")
  echo "  [schedule] next slice = ${sched_hms} (ckpt_interval=${SCHED_CKPT_INTERVAL}s) @ Chicago $(TZ='America/Chicago' date '+%H:%M %Z')"
  # Card routing: pretrain slices -> H100 (== header, no-op). Post-pretrain (midtrain A/B) -> the
  # time-of-day card (H100 day / H200 night) from midtrain_card_now (or a pinned MIDTRAIN_CARD_MODE).
  local next_constraint="$SBATCH_CONSTRAINT"
  [[ -f "$PRETRAIN_DONE_MARKER" ]] && next_constraint="$(midtrain_card_now)"
  next_job_id=$(
    sbatch \
      --parsable \
      --time="$sched_hms" \
      --signal=B:USR1@60 \
      --constraint="$next_constraint" \
      --output="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.out" \
      --error="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.err" \
      `# Slurm 20.11 stores no StdOut/StdErr in accounting (added in 24.05) and` \
      `# slurmctld forgets a finished job after MinJobAge=120s, so after two` \
      `# minutes nothing records where this slice's log went. Comment IS stored` \
      `# permanently (AccountingStoreJobComment=Yes), so slurmpast can recover the` \
      `# path from sacct months later instead of guessing it from file mtimes.` \
      --comment="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.err" \
      --export=ALL,AUTO_RESUBMIT=1,RESUBMIT_ON_FAILURE="${RESUBMIT_ON_FAILURE}",FAILURE_RETRY_COUNT=0,EXTRA_EXCLUDE=,FAILURE_RETRY_MAX="${FAILURE_RETRY_MAX}",WALL_TIME_OVERRIDE="${SCHED_SECONDS}",AUTO_RESUBMIT_TIME="${sched_hms}",LOG_INDEX="${next_index}",REPORT_DIR="${REPORT_DIR}",LOG_BASENAME="${LOG_BASENAME}",CKPT_DIR_OVERRIDE="${CKPT_DIR}",CHECKPOINT_INTERVAL="${SCHED_CKPT_INTERVAL}",REPO_ROOT="${REPO_ROOT}",MIDTRAIN_CARD_MODE="${MIDTRAIN_CARD_MODE}",MIDTRAIN_BATCH_H200="${MIDTRAIN_BATCH_H200}",MIDTRAIN_ACCUM_H200="${MIDTRAIN_ACCUM_H200}",MIDTRAIN_BATCH_H100="${MIDTRAIN_BATCH_H100}",MIDTRAIN_ACCUM_H100="${MIDTRAIN_ACCUM_H100}",A45_TRAIN_TOKENS="${A45_TRAIN_TOKENS}",PRETRAIN_CARD="${PRETRAIN_CARD}" \
      "${REPO_ROOT}/run_full_training.sh"
  )
  echo "Submitted next training slice as job ${next_job_id} with log index ${next_index} (constraint=${next_constraint})"
}

# Crash resubmit: resume from the latest checkpoint (auto-detected by the .py), exclude the
# failed node, increment+forward the retry counter so the cap holds across the retry chain,
# and PRESERVE AUTO_RESUBMIT so a retried weekend-chain slice still chains on the next success.
submit_failure_retry_slice() {
  local next_index new_count failed_node new_extra exclude_list next_job_id
  next_index=$(find_next_train_log_index)
  new_count=$((FAILURE_RETRY_COUNT + 1))
  failed_node="${SLURMD_NODENAME:-$(hostname -s 2>/dev/null)}"
  new_extra="$EXTRA_EXCLUDE"
  if [[ -n "$failed_node" ]]; then
    if [[ -n "$new_extra" ]]; then new_extra="${new_extra},${failed_node}"; else new_extra="$failed_node"; fi
  fi
  exclude_list="$BASE_EXCLUDE"
  if [[ -n "$new_extra" ]]; then exclude_list="${exclude_list},${new_extra}"; fi
  compute_slice_schedule   # crash-retry also sizes to the CURRENT day/night window (mid-night crash -> remaining night, not a fresh 8h)
  local sched_hms; sched_hms=$(fmt_hms "$SCHED_SECONDS")
  # Card routing (same as clean resubmit): pretrain -> H100; post-pretrain midtrain -> the time-of-day
  # card from midtrain_card_now (H100 day / H200 night) or a pinned MIDTRAIN_CARD_MODE.
  local retry_constraint="$SBATCH_CONSTRAINT"
  [[ -f "$PRETRAIN_DONE_MARKER" ]] && retry_constraint="$(midtrain_card_now)"
  next_job_id=$(
    sbatch \
      --parsable \
      --time="$sched_hms" \
      --signal=B:USR1@60 \
      --exclude="$exclude_list" \
      --constraint="$retry_constraint" \
      --output="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.out" \
      --error="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.err" \
      `# See the clean-resubmit path: the only durable record of the log path.` \
      --comment="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.err" \
      --export=ALL,AUTO_RESUBMIT="${AUTO_RESUBMIT}",RESUBMIT_ON_FAILURE=1,FAILURE_RETRY_COUNT="${new_count}",FAILURE_RETRY_MAX="${FAILURE_RETRY_MAX}",EXTRA_EXCLUDE="${new_extra}",WALL_TIME_OVERRIDE="${SCHED_SECONDS}",AUTO_RESUBMIT_TIME="${sched_hms}",LOG_INDEX="${next_index}",REPORT_DIR="${REPORT_DIR}",LOG_BASENAME="${LOG_BASENAME}",CKPT_DIR_OVERRIDE="${CKPT_DIR}",CHECKPOINT_INTERVAL="${SCHED_CKPT_INTERVAL}",REPO_ROOT="${REPO_ROOT}",MIDTRAIN_CARD_MODE="${MIDTRAIN_CARD_MODE}",MIDTRAIN_BATCH_H200="${MIDTRAIN_BATCH_H200}",MIDTRAIN_ACCUM_H200="${MIDTRAIN_ACCUM_H200}",MIDTRAIN_BATCH_H100="${MIDTRAIN_BATCH_H100}",MIDTRAIN_ACCUM_H100="${MIDTRAIN_ACCUM_H100}",A45_TRAIN_TOKENS="${A45_TRAIN_TOKENS}",PRETRAIN_CARD="${PRETRAIN_CARD}" \
      "${REPO_ROOT}/run_full_training.sh"
  )
  echo "Training slice crashed on node '${failed_node}'; resubmitted failure-retry ${new_count}/${FAILURE_RETRY_MAX} as job ${next_job_id} (log index ${next_index}, constraint=${retry_constraint}); excluding: ${exclude_list}"
}

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persistent torch.compile (Inductor) cache under the checkpoint dir. With 1h slices the model
# recompiles every slice; a warm cache turns each recompile into a fast cache-hit, cutting the
# per-slice compile/startup tax. Reused across slices + resumes (all land on 94GB H100 NVL).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CKPT_DIR}/inductor_cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

cd "$REPO_ROOT"
mkdir -p "$REPORT_DIR"
mkdir -p "$CKPT_DIR"

# Tokenizer (Qwen3-0.6B-Base; kept for KD/soup compatibility -- same as 3.5).
PRETRAIN_TOKENIZER=${PRETRAIN_TOKENIZER:-/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base}

# --- Argonne-4.0 PRETRAIN data = the validated 50/30/20 (edu/math/code) WEIGHTED mixture ---
# Sampled per micro-batch by weight via pretrain.py --train_sources (DDP-synced, resumable);
# NO pre-built blend bin (the campaign showed sequential blend bins misorder; the weighted
# mixer realizes the exact ratio decoupled from raw source sizes). Every path/weight/budget
# is env-overridable.
#   DATA BUILT (2026-07-20): the SCALE bins below total ~38.6B tok (edu 20.6B / math 10B / code 8B),
#   built by build_a4_data.py from the full docbin (FineWeb-Edu fully tokenized -- all 218 shards).
#   Solid first run (~1.5x Chinchilla for 1B at 1x/source); raise A45_TRAIN_TOKENS for a more
#   overtrained run (<=~4x per-source repetition is ~free, Muennighoff). The old proxy dir
#   (argonne4_experiments/data, ~3.6B) remains the smoke fallback; the worker warns if the corpus is small.
A45_DATA_DIR=${A45_DATA_DIR:-/project/rcc/youzhi/data/argonne4_pretrain}
A45_EDU=${A45_EDU:-${A45_DATA_DIR}/edu_flat.bin}
A45_MATH=${A45_MATH:-${A45_DATA_DIR}/finemath_flat.bin}
A45_CODE=${A45_CODE:-${A45_DATA_DIR}/code_flat.bin}
A45_W_EDU=${A45_W_EDU:-50}
A45_W_MATH=${A45_W_MATH:-30}
A45_W_CODE=${A45_W_CODE:-20}
PRETRAIN_SOURCES=${PRETRAIN_SOURCES:-${A45_EDU}:${A45_W_EDU},${A45_MATH}:${A45_W_MATH},${A45_CODE}:${A45_W_CODE}}
# Token budget that sets the WSD schedule length (estimated_steps = budget/effective_batch, so
# the cooldown lands at the true end). 0 = default to the combined source size (~one weighted
# pass -- fine for a proxy/smoke run; set to the real budget for a full run).
A45_TRAIN_TOKENS=${A45_TRAIN_TOKENS:-0}
SOURCE_SEED=${A45_SOURCE_SEED:-1337}

# --- MIDTRAIN phase A (reasoning anneal, block 1024) reuses 3.5's flat corpora (Qwen3-tokenized,
# tokenizer-compatible). continue_pretrain.py reads a flat .bin densely; the cross-source
# interleave is baked in at flatten time. Seeded from the pretrain FINAL weights (fresh WSD).
CONTINUE_TOKENIZER=${CONTINUE_TOKENIZER:-$PRETRAIN_TOKENIZER}
CONTINUE_DATA=${CONTINUE_DATA:-/project/rcc/youzhi/data/reasoning_anneal/reasoning_anneal_flat.bin}
FINAL_MODEL_DIR=${FINAL_MODEL_DIR:-${CKPT_DIR}/final_model_complete}
PRETRAIN_DONE_MARKER=${PRETRAIN_DONE_MARKER:-${CKPT_DIR}/.pretrain_complete}
CONTINUE_STARTED_MARKER=${CONTINUE_STARTED_MARKER:-${CKPT_DIR}/.continue_pretrain_started}
CONTINUE_DONE_MARKER=${CONTINUE_DONE_MARKER:-${CKPT_DIR}/.continue_pretrain_complete}

# Number of GPUs (change --gres above to match)
# --- Which card the PRETRAIN stage runs on -------------------------------------------------------
# a4.0 hardcoded H100 for pretrain. a4.5 makes it selectable because H200 is 1.5x the HBM (141 vs 94 GB)
# and faster, so it can carry a larger micro-batch at chunk=0.
#   H100 (default): proven, and the 3-GPU job always lands on a 94GB NVL card (the one 80GB node is gpu:2).
#   H200          : pass --h200 to the launcher, or PRETRAIN_CARD=H200. Only midway3-0600/0601 survive the
#                   node-exclusion policy and they are frequently 8/8 booked by other users' multi-day
#                   jobs -- expect to queue. Everything else in the recipe is unchanged.
# The EFFECTIVE BATCH is held at 540,672 on BOTH cards by trading micro-batch against grad-accum,
# so the LR-6e-4 recipe stays valid whichever card a slice lands on (same trick a4.0 used for midtrain).
PRETRAIN_CARD=${PRETRAIN_CARD:-H100}
case "$PRETRAIN_CARD" in
  H100|H200) SBATCH_CONSTRAINT="$PRETRAIN_CARD" ;;
  any)       SBATCH_CONSTRAINT="H100|H200" ;;   # scheduler takes whichever frees first
  *) echo "FATAL: PRETRAIN_CARD must be H100, H200 or any" >&2; exit 4 ;;
esac

NGPUS=3

# Wall time in seconds = the SLURM slice length. The .py saves ONE checkpoint + exits cleanly a fixed
# lead before the SLURM kill; this script then resubmits on that CLEAN EXIT (save-THEN-submit, so the
# save never races the hard kill). Default 1h; weekend.sh passes WALL_TIME_OVERRIDE.
WALL_TIME=${WALL_TIME_OVERRIDE:-$((1 * 3600))}
# Absolute save deadline (SLURM-clock-relative -> immune to compile/startup drift): the .py saves +
# exits when wall-clock time reaches (this slice's job start + WALL_TIME - lead). Each slice stamps
# its own start, so the deadline stays correct across the self-resubmit chain.
SLICE_START_EPOCH=$(date +%s)
SAVE_LEAD_SECONDS=${SAVE_LEAD_SECONDS:-150}
SAVE_DEADLINE_EPOCH=$(( SLICE_START_EPOCH + WALL_TIME - SAVE_LEAD_SECONDS ))
# Periodic checkpoint interval (s). Default 1800 = a 30-min safety net (night.sh's long slice keeps it).
# weekend.sh overrides this HUGE (save-once): then the ONLY save is the deadline save near slice end.
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1800}

# --- PRETRAIN (stage 1) training config ---
# argonne4.5 batch. INVERTED from a4.0: chunked CE is OFF, so the micro-batch must be SMALL enough
# that the full fp32 logit tensor fits (16*1024 rows * 151,680 * 4B = 9.9 GiB). a4.0 ran batch 170,
# which would need 105 GiB un-chunked -- that is precisely why it had to chunk, and why it paid 23%.
# Effective batch 16*3*1024*11 = 540,672, within 3% of the LR-6e-4-validated 524,288.
# (a4.0 rationale kept below for provenance:)
# HARDWARE = H100 (--constraint=H100 above). Test-partition H100s (2026-07-21): the gpu:4 nodes
# (0372/0385/0423/0426) are 94GB NVL; the ONE 80GB node (0432) is gpu:2, so a 3-GPU job CANNOT land
# there -- the pretrain ALWAYS runs on a 94GB card. Measured (block1024/fp8/ck1): batch 170 -> 70.6 GiB
# = 73% of 94GB (safe headroom). tok/s is FLAT with batch (1B is COMPUTE-BOUND at 100% util), so batch
# 170 is the campaign-VALIDATED point: grad_accum=1 => effective 170*3*1024 = 522,240 tok/step ==
# validated 524288 (LR 6e-4 DIRECTLY validated). ~78.8k tok/s on H100. NOTE: H100 is ~28% SLOWER than
# H200 (108k tok/s -- power-limited ~400W vs 700W); set --constraint=H200 for the faster, HBM-scarcer card.
# chunk=0 needs a micro-batch small enough that the FULL logit tensor fits, and that limit depends
# on the CARD -- which with PRETRAIN_CARD=any is not known until the slice is already running.
# So detect HBM here and pick micro/accum to hold the EFFECTIVE BATCH at 540,672 on every card,
# which keeps the LR-6e-4 recipe valid no matter where a slice lands:
#   H200  141GB -> 22 x  8   (22*1024 = 22,528 rows)
#   H100   94GB -> 16 x 11   (16,384 rows)  <- the config all 51 probe arms measured
#   H100   80GB -> 11 x 16   (11,264 rows)
# A detection failure falls back to the safest (smallest) micro-batch, never the largest.
_hbm_mb=""
_hbm_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) || true
if [[ "$_hbm_mb" =~ ^[0-9]+$ ]] && (( _hbm_mb >= 120000 )); then
  _CARD_SEEN="H200-141G"; _MB=22; _GA=8
elif [[ "$_hbm_mb" =~ ^[0-9]+$ ]] && (( _hbm_mb >= 90000 )); then
  _CARD_SEEN="H100-94G";  _MB=16; _GA=11
else
  _CARD_SEEN="H100-80G/unknown"; _MB=11; _GA=16
fi
BLOCK_SIZE=1024      # RESTORED -- my card-detection edit deleted this, giving
                     # TOTAL_BATCH_SIZE=0 and an instant crash on the first slice.
BATCH_SIZE=${A45_BATCH:-$_MB}
GRAD_ACCUM=${A45_GRAD_ACCUM:-$_GA}
echo "[a4.5] card detected: ${_CARD_SEEN} (${_hbm_mb:-?} MB) -> micro_batch ${BATCH_SIZE} x accum ${GRAD_ACCUM}"
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))
PRETRAIN_LR=${A45_LR:-6e-4}   # campaign-1 validated at sigma=0.028; a4.5 probes could NOT resolve LR (sigma 0.130)
PRETRAIN_WARMUP=${A45_WARMUP:-8000}  # unresolved by probes (0.2 sigma); keep the inherited value
# SUPERSEDED for a4.5 -- chunk is now 0. The a4.0 tuning below optimised the NUMBER of chunks;
# exp e01->e03 showed going to ZERO chunks is worth far more (+22.9%) than any chunk size.
# Chunked-CE chunk size. MEASURED 2026-07-24 on an 80GB H100 (1 GPU, fresh-init probe of
# this exact config, iso effective batch 174,080 tok/optimizer-step/GPU):
#    chunk  4096 (old) -> 8.492 s/step @ 76.3/84.9 GB (89.8%)   baseline
#    chunk 10240       -> 7.743 s/step @ 79.1/84.9 GB (93.1%)   -8.8%   <- deployed
#    chunk 21760 @mb85 -> 7.927 s/step @ 99.2% (too tight, and NOISY: allocator pressure)
#    chunk  2048       -> +4.1% SLOWER (halving the chunk is the wrong direction)
# Why: _chunked_lm_loss runs eager (@torch.compiler.disable) AND wraps each chunk in
# torch.utils.checkpoint, so the lm_head is recomputed in backward once per chunk.
# Cost tracks the NUMBER of chunks/step (174,080 rows: 4096->43 chunks, 10240->17).
# Training-neutral: identical loss, per-chunk softcap applied either way (model.py:816);
# only the tiling changes. The live 3-GPU run sits ~3.4 GB below this 1-GPU probe, so
# expect ~89% HBM there -- full card with room for the checkpoint-save transient.
PRETRAIN_CHUNK=${A45_CHUNK:-0}        # 0 = compiled CE. exp e01->e03: 32,205 -> 39,593 tok/s (+22.9%)
PRETRAIN_GRAD_CKPT=${A45_GRAD_CKPT:-1}
PRETRAIN_MAX_EPOCHS=${PRETRAIN_MAX_EPOCHS:-1}

# --- Adaptive selective checkpointing (detected per-card at slice launch) -------------------------
# This cluster's H100s are MIXED 80GB/94GB under one indistinguishable --constraint=H100. The batch
# script runs ON the allocated node, so read the card's HBM here and pick --checkpoint_stride:
#   80GB card -> stride 1  (full checkpointing; the only config that fits -- current/safe behavior)
#   >=90GB card -> stride from A45_CKPT_STRIDE_94G (default 2 = store 12 of 24 layers). The >=90000
#                  MB test catches BOTH the 94GB H100 NVL and the 141GB H200. a4.0 used 16,
#                tuned for its 32-layer arch at chunk 10240; a4.5 MEASURED stride 2 as the optimum
#                at chunk=0 (exp e04: 42,016 tok/s @ 84% HBM, vs 39,593 at stride 1).
# checkpoint_stride is NUMERICALLY IDENTICAL (only recompute-vs-store) -> never changes the trained
# model, loss, or resume; only per-slice speed. Any detection failure keeps the safe stride 1.
# NOTE: the 94GB default (16) is conservative from 80GB-card data; verify/raise it on a real 94GB card.
PRETRAIN_CKPT_STRIDE=${A45_CKPT_STRIDE:-2}   # e03->e04: 39,593 -> 42,016 tok/s (+6.1%), HBM 65->84%
# MUST be set -e-safe: a failed/transient nvidia-smi must fall back to the safe stride, NEVER abort the
# slice. Under `set -eo pipefail` a bare `_gpu_mb=$(nvidia-smi|head)` propagates nvidia-smi's non-zero
# exit and kills the whole run (broke the chain on 0426, 2026-07-22). `|| true` + empty-guard fixes it.
_gpu_mb=""
_gpu_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) || true
if [[ "$_gpu_mb" =~ ^[0-9]+$ ]] && (( _gpu_mb >= 90000 )); then PRETRAIN_CKPT_STRIDE=${A45_CKPT_STRIDE_94G:-2}; fi
echo "[adaptive-ckpt] allocated GPU HBM=${_gpu_mb:-unknown}MB -> checkpoint_stride=${PRETRAIN_CKPT_STRIDE}"

# --- MIDTRAIN phase A (continue_pretrain, reasoning anneal, block 1024) ---
# Fresh WSD phase seeded from the pretrain final (reset_schedule=1 on the first slice). Effective
# batch matched to stage 1 so the pretrain->continue regime is seamless. Chunked CE frees the transient.
# ANNEAL throughput (MEASURED 2026-07-25, 1x80GB-H100, block 1024, iso 196,608 tok/GPU/step):
#   mb64 accum3 chunk4096 : 9.549 s/step @43.5/84.9 GB (51.2%)  <- old: 48 eager CE passes AND
#                                                                  half the card sitting idle
#   mb64 accum3 chunk16384: 8.590 (-10.0%) @71.8 (84.5%)   12 passes -- partial measure only
#   mb64 accum3 chunk0    : OOM (retained logits ~4x the bf16 tensor once fp32 CE intermediates
#                                are counted, not ~2x -- mb64 needs ~80GB for CE alone)
#   mb32 accum6 chunk0    : 6.041 (-36.7%) @77.2 (90.9%)  <== DEPLOYED  +58.1% tok/s
#   mb16 accum12 chunk0   : 6.36  (-33.4%)  slower: 12 micro-steps vs 6 = accumulation penalty
# chunk=0 puts CE inside the COMPILED graph with no per-chunk lm_head recompute. Note the gap
# between "fewer eager passes" (-10%) and "no eager passes" (-36.7%): the penalty is mostly a
# FIXED cost of leaving the compiled graph, so partial measures barely help.
# NOTE: after .pretrain_complete the launcher switches the card constraint to the day/night
# H100<->H200 selector, so the anneal alternates cards. There is no CONTINUE_BATCH_H100/H200
# split, so this config must be safe on the SMALLER card: mb32 = 90.9% on 80GB H100, ~51% on
# an H200 (under-filled there, but correct on both). Effective batch unchanged:
# 32*3*1024*6 = 589,824 (was 64*3*1024*3).
CONTINUE_BATCH=${CONTINUE_BATCH:-32}
CONTINUE_CHUNK=${CONTINUE_CHUNK:-0}   # 0 = OFF = compiled CE path
CONTINUE_GRAD_ACCUM=${CONTINUE_GRAD_ACCUM:-6}
CONTINUE_TOTAL_BATCH=$((CONTINUE_BATCH * NGPUS * BLOCK_SIZE * CONTINUE_GRAD_ACCUM))
CONTINUE_LR=${CONTINUE_LR:-2e-4}
# WSD cooldown, in STEPS (continue_pretrain.py:383 -- `--cooldown 0` means lr_lambda returns 1.0
# FOREVER, i.e. NO decay at all, not "auto"). This stage ran its whole 30,630-step epoch at a flat
# 2e-4 and ended at full LR on 2026-07-31 because it was set to 0 -- the same bug fixed on the 3.5
# side and never ported here. 15% of the 30,630-step epoch (matches the pretrain cooldown_frac 0.15).
# Inert for the completed a4 anneal (gated by .continue_pretrain_complete); correct if ever re-run.
CONTINUE_COOLDOWN=${CONTINUE_COOLDOWN:-4600}
CONTINUE_GRAD_CKPT=${CONTINUE_GRAD_CKPT:-1}
# Selective activation checkpointing, wired 2026-07-25 (model.py already supported it;
# continue_pretrain.py did not until today). stride 1 = prior behavior (checkpoint every
# layer). S>=2 stores ceil(32/S) layers -> more HBM, less recompute. Bit-identical.
CONTINUE_CKPT_STRIDE=${CONTINUE_CKPT_STRIDE:-1}

# --- MIDTRAIN phase B = CONTEXT-EXTENSION reasoning anneal (auto-chains AFTER continue; GATED) ---
# Same script + fp8 recipe as phase A; the only differences are the checkpoint dir (models/argonne4_midtrain),
# the EXTENDED context (block 13568), and a DISJOINT same-composite reasoning slice. Gated on
# MIDTRAIN_ARMED_MARKER (default OFF -> chain cleanly stops after phase A until you arm it). The
# block-13568 per-card batch is MEASURED (2026-07-23 probes): H200 microbatch 24, H100 microbatch 12.
MIDTRAIN_CKPT_DIR="${MIDTRAIN_CKPT_DIR_OVERRIDE:-${CHECKPOINT_ROOT}/argonne45_midtrain}"
MIDTRAIN_TOKENIZER=${MIDTRAIN_TOKENIZER:-$PRETRAIN_TOKENIZER}
MIDTRAIN_DATA=${MIDTRAIN_DATA:-/project/rcc/youzhi/data/reasoning_anneal/reasoning_midtrain_flat.bin}
MIDTRAIN_ARMED_MARKER=${MIDTRAIN_ARMED_MARKER:-${MIDTRAIN_CKPT_DIR}/.midtrain_armed}
MIDTRAIN_FINAL_MODEL_DIR=${MIDTRAIN_FINAL_MODEL_DIR:-${MIDTRAIN_CKPT_DIR}/final_model_complete}
MIDTRAIN_LR=${MIDTRAIN_LR:-1e-4}
# WSD cooldown in STEPS -- see the CONTINUE_COOLDOWN note; this was also `--cooldown 0` = no decay.
# THIS is the stage that matters: phase B is the LAST training stage, so its final checkpoint is what
# gets gated/harvested, and it must land at low LR (min_lr_ratio 0.1 -> 1e-4 decaying to 1e-5).
# Sized from the real corpus: reasoning_midtrain_flat.bin = 6,022,107,418 tok / 976,896 tok per step
# (batch 2 x 3 GPUs x 13568 x accum 12, identical on H100 and H200) = 6,164 steps; 15% ~= 900.
MIDTRAIN_COOLDOWN=${MIDTRAIN_COOLDOWN:-900}
MIDTRAIN_BLOCK=${MIDTRAIN_BLOCK:-13568}
MIDTRAIN_FP8=${MIDTRAIN_FP8:-1}
MIDTRAIN_CHUNK=${MIDTRAIN_CHUNK:-0}   # 0 = OFF = compiled CE path
MIDTRAIN_GRAD_CKPT=${MIDTRAIN_GRAD_CKPT:-1}
MIDTRAIN_CKPT_STRIDE=${MIDTRAIN_CKPT_STRIDE:-1}
# --- Midtrain card selection: TIME-OF-DAY auto-switch (America/Chicago); post-pretrain phases A+B ONLY ---
# MIDTRAIN_CARD_MODE = auto (default) | H100 | H200:
#   auto  -> DAY 07:00-23:00 = H100 (be polite; H200s are in daytime demand),
#            NIGHT 23:00-07:00 = H200 (grab the scarce fast card while the cluster is quiet).
#   H100/H200 -> pin that card for EVERY midtrain slice (manual override; e.g. weekend.sh --h100).
# The card is chosen at RESUBMIT time (submit_*_slice -> --constraint via midtrain_card_now); pretrain
# slices ALWAYS stay H100 and are never touched. Batch/accum FOLLOW the card so the effective batch is
# IDENTICAL (976,896 tok) either way -> the LR/recipe never changes across a day/night switch:
#   H200 -> microbatch 24 / accum 1  (measured 2026-07-23: 86% of 141GB @ block 13568)
#   H100 -> microbatch 12 / accum 2  (measured ~68GB; fits ANY H100 incl 80GB)
# 2026-07-29 (owner directive: "by default it's always h100s"): default flipped auto -> H100.
# `auto` routed post-pretrain slices by wall-clock (H100 day / H200 night), so when a4's pretrain
# completed at 01:14 the FIRST anneal slice was submitted as H200 and took 3 GPUs on midway3-0601 --
# the scarce card that 3.5's phase B is PINNED to (only 0600/0601 are un-excluded). Pinning H100
# costs nothing in optimization: MIDTRAIN_BATCH/ACCUM are identical on both cards (2/12), and the
# #SBATCH header is already H100 -- `auto` only ever overrode it at RESUBMIT time. Set
# MIDTRAIN_CARD_MODE=H200 explicitly for a night run when 3.5 is not competing.
MIDTRAIN_CARD_MODE=${MIDTRAIN_CARD_MODE:-H100}
# MIDTRAIN throughput (MEASURED 2026-07-25, 1x80GB-H100, block 13568, iso 325,632 tok/GPU/step):
#   mb12 accum2 chunk4096 : 25.731 s/step @76.3/84.9 GB (89.9%)   <- old (80 eager CE passes/step)
#   mb12 accum2 chunk32768: OOM   (no headroom to ENLARGE the chunk on an 80GB card)
#   mb6  accum4 chunk0    : OOM
#   mb4  accum6 chunk0    : OOM
#   mb3  accum8 chunk0    : 19.969 (-22.4%) @83.5 (98.3%)  FASTEST but undeployable
#   mb2  accum12 chunk0   : 20.142 (-21.7%) @64.8 (76.3%)  <== DEPLOYED (0.9% slower, 22pt safer)
# chunk=0 puts CE in the COMPILED graph (no per-chunk lm_head recompute). At long context
# chunk=0 INCREASES logit memory (retained logits scale with rows), hence mb6/mb4 OOM.
# THREE card tiers, because --constraint=H100 lands on an 80 GB OR a 94 GB card indistinguishably
# and those need DIFFERENT micro-batches. Sizing comes from the MEASURED linear model in
# reasoning/context_window_limits.md (a4 @ block 13,568: 18.5 GiB fixed + 27.9 GiB/sequence; it
# predicted the batch-2 figure of 74.3 GiB exactly, so it is trustworthy for batch 1 and 3):
#     batch 1 -> 46.4 GiB   batch 2 -> 74.3 GiB (measured)   batch 3 -> 102.2 GiB
#   80 GB H100 (79.0 GiB usable): batch 2 = 74.3 is only ~1.7 GiB under the 76 GiB safe fill, and
#     that figure is 1-GPU -- 3-GPU DDP adds gradient buckets on top, so it OOMs. -> batch 1 (59%).
#   94 GB H100 (93.6 GiB): batch 2 -> 79% fill, ~82% with DDP buckets. Well filled and safe.
#   H200 (131 GiB):        batch 3 -> 78% fill, and 8 micro-steps instead of 12 (the accumulation
#     penalty is real: mb16/accum12 measured SLOWER than mb32/accum6 at block 1024).
# EFFECTIVE BATCH IS IDENTICAL ON ALL THREE (976,896 tok) so the card a slice lands on never changes
# the optimization: 1*3*13568*24 = 2*3*13568*12 = 3*3*13568*8 = 976,896.
MIDTRAIN_BATCH_H200=${MIDTRAIN_BATCH_H200:-3};    MIDTRAIN_ACCUM_H200=${MIDTRAIN_ACCUM_H200:-8}
MIDTRAIN_BATCH_H100=${MIDTRAIN_BATCH_H100:-2};    MIDTRAIN_ACCUM_H100=${MIDTRAIN_ACCUM_H100:-12}
MIDTRAIN_BATCH_H100_80=${MIDTRAIN_BATCH_H100_80:-1}; MIDTRAIN_ACCUM_H100_80=${MIDTRAIN_ACCUM_H100_80:-24}
# THIS slice's phase-B batch is sized to the ACTUAL allocated card (via _gpu_mb detected above for
# adaptive-ckpt) so it always fits the card the slice landed on -- immune to a stale MIDTRAIN_BATCH that
# --export=ALL would otherwise carry across a day/night card switch (assigned UNCONDITIONALLY, so any
# inherited value is overwritten). Detection failure -> the 80 GB batch, which fits ANY card.
# MIDTRAIN_BATCH_OVERRIDE (explicit) wins for manual tuning.
# Thresholds: H200 >= 130000 MB; 94 GB H100 >= 90000 MB (midway3-0426 reads 95830); else 80 GB.
if [[ -n "${MIDTRAIN_BATCH_OVERRIDE:-}" ]]; then
  MIDTRAIN_BATCH="$MIDTRAIN_BATCH_OVERRIDE"; MIDTRAIN_GRAD_ACCUM="${MIDTRAIN_GRAD_ACCUM_OVERRIDE:-1}"
elif [[ "$_gpu_mb" =~ ^[0-9]+$ ]] && (( _gpu_mb >= 130000 )); then
  MIDTRAIN_BATCH="$MIDTRAIN_BATCH_H200"; MIDTRAIN_GRAD_ACCUM="$MIDTRAIN_ACCUM_H200"
elif [[ "$_gpu_mb" =~ ^[0-9]+$ ]] && (( _gpu_mb >= 90000 )); then
  MIDTRAIN_BATCH="$MIDTRAIN_BATCH_H100"; MIDTRAIN_GRAD_ACCUM="$MIDTRAIN_ACCUM_H100"
else
  MIDTRAIN_BATCH="$MIDTRAIN_BATCH_H100_80"; MIDTRAIN_GRAD_ACCUM="$MIDTRAIN_ACCUM_H100_80"
fi
MIDTRAIN_TOTAL_BATCH=$((MIDTRAIN_BATCH * NGPUS * MIDTRAIN_BLOCK * MIDTRAIN_GRAD_ACCUM))

# Card a midtrain slice submitted NOW should target (--constraint at resubmit time). Resolves
# MIDTRAIN_CARD_MODE against America/Chicago wall-clock. set -eo pipefail-safe: returns 0; if the clock
# can't be read it defaults to H100 (the safe/polite daytime card). DAY 07:00-23:00 -> H100 else H200.
midtrain_card_now() {
  case "$MIDTRAIN_CARD_MODE" in
    H100|H200) printf '%s' "$MIDTRAIN_CARD_MODE"; return 0 ;;
  esac
  local sod
  sod=$(TZ='America/Chicago' date '+%-H %-M %-S' 2>/dev/null | awk '{print $1*3600+$2*60+$3}') || sod=""
  if [[ "$sod" =~ ^[0-9]+$ ]] && (( sod >= 82800 || sod < 25200 )); then printf 'H200'; else printf 'H100'; fi
}

# Data-readiness guard: warn loudly (do NOT hard-fail -- proxy bins are fine for a smoke run).
combined_gb=0
for src_spec in ${PRETRAIN_SOURCES//,/ }; do
  src_path="${src_spec%:*}"
  if [[ -f "$src_path" ]]; then
    sz=$(stat -c %s "$src_path" 2>/dev/null || echo 0)
    combined_gb=$((combined_gb + sz / 1000000000))
  else
    echo "WARNING: train source not found: $src_path"
  fi
done
if ((combined_gb < 40)); then
  echo "############################################################################"
  echo "# NOTE: combined train corpus ~= ${combined_gb} GB (< ~40 GB). This looks like the"
  echo "#       PROXY data -- OK for a SMOKE test, NOT a full argonne4 run. For a real run,"
  echo "#       override A45_EDU/A45_MATH/A45_CODE with scale-tokenized corpora + set A45_TRAIN_TOKENS."
  echo "############################################################################"
fi

echo "Argonne-4.5 recipe: 2.06B (hidden2560/24L/10h/2kv/inter7040) FP8 lr=${PRETRAIN_LR} grad_clip=0.4 warmup=${PRETRAIN_WARMUP} cooldown_frac=0.15 chunked_CE=${PRETRAIN_CHUNK}"
echo "Pretrain sources:   ${PRETRAIN_SOURCES}"
echo "                    train_tokens=${A45_TRAIN_TOKENS} (0=combined size) effective_batch=${TOTAL_BATCH_SIZE} tok/step"
echo "Checkpoint dir:     ${CKPT_DIR}  (fresh fp8 run; vocab padded 151669->151680)"

write_stage_marker() {
  local marker_path="$1"
  local marker_text="$2"
  local marker_dir tmp_path
  marker_dir="$(dirname "$marker_path")"
  tmp_path="${marker_path}.tmp"
  mkdir -p "$marker_dir"
  {
    printf '%s\n' "$marker_text"
    printf 'written_at_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$tmp_path"
  mv "$tmp_path" "$marker_path"
}

hf_model_dir_exists() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" && -f "${model_dir}/tokenizer_config.json" ]]
}

pretrain_complete() {
  [[ -f "$PRETRAIN_DONE_MARKER" ]] || hf_model_dir_exists "$FINAL_MODEL_DIR"
}

continue_complete() {
  [[ -f "$CONTINUE_DONE_MARKER" ]]
}

# --- Midtrain (ctx-extension) stage gates (same design as 3.5) ---
midtrain_armed() {
  [[ -f "$MIDTRAIN_ARMED_MARKER" ]]
}

midtrain_complete() {
  hf_model_dir_exists "$MIDTRAIN_FINAL_MODEL_DIR"
}

# There is still work to chain if pretrain isn't done, or (defensively) continue isn't done,
# or midtrain is armed but not yet complete. Drives both the clean-exit chain and the USR1
# timeout pre-submit, uniformly across all stages.
pipeline_incomplete() {
  if ! pretrain_complete; then return 0; fi
  if ! continue_complete; then return 0; fi
  if midtrain_armed && ! midtrain_complete; then return 0; fi
  return 1
}

maybe_submit_next_slice() {
  if [[ "$AUTO_RESUBMIT" != "1" || "$RESUBMIT_DONE" == "1" ]]; then
    return
  fi
  if pipeline_incomplete; then
    submit_next_slice
    RESUBMIT_DONE=1
  fi
}

handle_timeout_warning() {
  echo "Received Slurm timeout warning signal; pre-submitting next slice."
  set +e
  maybe_submit_next_slice
  set -e
}

trap 'handle_timeout_warning' USR1

if [[ ! -f "$PRETRAIN_DONE_MARKER" ]] && hf_model_dir_exists "$FINAL_MODEL_DIR"; then
  write_stage_marker "$PRETRAIN_DONE_MARKER" "stage=pretrain"
fi

train_status=0

if ! pretrain_complete; then
  stage_name="pretrain"
  echo "Stage: pretrain.py (BASE, weighted 50/30/20 mixture)"
  echo "Sources: ${PRETRAIN_SOURCES}"
  echo "Logs:  ${REPORT_DIR}/${LOG_INDEX:-?}-${LOG_BASENAME}.out"

  set +e
  # argonne4.5 arch (2,063,667,712) is selected by the A45 flag at the top of pretrain.py.
  # Assert it is ON before burning a slice -- silently training the 1.04B a4.0 arch into the
  # argonne45_pretrain dir would be discovered days later.
  if ! grep -qE "^A45 = True" pretrain.py; then
    echo "FATAL: pretrain.py has A45 = False; this worker trains argonne4.5. Set A45 = True." >&2
    exit 3
  fi
  torchrun --nproc_per_node="$NGPUS" pretrain.py \
    --tokenizer_path "$PRETRAIN_TOKENIZER" \
    --train_sources "$PRETRAIN_SOURCES" \
    --train_tokens "$A45_TRAIN_TOKENS" \
    --doc_shuffle_seed "$SOURCE_SEED" \
    --checkpoint_dir "$CKPT_DIR" \
    --fp8 1 \
    --fp8_lm_head 1 \
    --lr "$PRETRAIN_LR" \
    --batch_size "$BATCH_SIZE" \
    --total_batch_size "$TOTAL_BATCH_SIZE" \
    --block_size "$BLOCK_SIZE" \
    --loss_chunk_size "$PRETRAIN_CHUNK" \
    --precision bf16 \
    --flash_attention 1 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --grad_clip 0.4 \
    --warmup_steps "$PRETRAIN_WARMUP" \
    --schedule wsd \
    --cooldown_frac 0.15 \
    --min_lr_ratio 0.1 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --save_deadline_epoch "$SAVE_DEADLINE_EPOCH" \
    --max_epochs "$PRETRAIN_MAX_EPOCHS" \
    --torch_compile 1 \
    --gradient_checkpointing "$PRETRAIN_GRAD_CKPT" \
    --checkpoint_stride "$PRETRAIN_CKPT_STRIDE" \
    --wall_time "$WALL_TIME" \
    --final_model_dir "$FINAL_MODEL_DIR" \
    --completion_marker "$PRETRAIN_DONE_MARKER"
  train_status=$?
  set -e
elif ! continue_complete; then
  stage_name="continue_pretrain"
  if [[ -f "$CONTINUE_STARTED_MARKER" ]]; then
    RESET_SCHEDULE=0
  else
    RESET_SCHEDULE=1
  fi
  echo "Stage: continue_pretrain.py (MIDTRAIN phase A: reasoning anneal, block 1024)"
  echo "Data:  ${CONTINUE_DATA}"
  echo "Reset data cursor (first continue slice): ${RESET_SCHEDULE}"
  echo "Logs:  ${REPORT_DIR}/${LOG_INDEX:-?}-${LOG_BASENAME}.out"

  set +e
  torchrun --nproc_per_node="$NGPUS" continue_pretrain.py \
    --tokenizer_path "$CONTINUE_TOKENIZER" \
    --data_path "$CONTINUE_DATA" \
    --checkpoint_dir "$CKPT_DIR" \
    --fp8 1 \
    --fp8_lm_head 1 \
    --lr "$CONTINUE_LR" \
    --batch_size "$CONTINUE_BATCH" \
    --total_batch_size "$CONTINUE_TOTAL_BATCH" \
    --block_size "$BLOCK_SIZE" \
    --loss_chunk_size "$CONTINUE_CHUNK" \
    --precision bf16 \
    --flash_attention 1 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --grad_clip 0.4 \
    --warmup_steps 0 \
    --schedule wsd \
    --cooldown "$CONTINUE_COOLDOWN" \
    --min_lr_ratio 0.1 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --save_deadline_epoch "$SAVE_DEADLINE_EPOCH" \
    --max_epochs 1 \
    --torch_compile 1 \
    --gradient_checkpointing "$CONTINUE_GRAD_CKPT" \
    --checkpoint_stride "$CONTINUE_CKPT_STRIDE" \
    --wall_time "$WALL_TIME" \
    --reset_schedule "$RESET_SCHEDULE" \
    --final_model_dir "$FINAL_MODEL_DIR" \
    --completion_marker "$CONTINUE_DONE_MARKER" \
    --started_marker "$CONTINUE_STARTED_MARKER"
  train_status=$?
  set -e

  if [[ "$train_status" -eq 0 && ! -f "$CONTINUE_STARTED_MARKER" ]]; then
    write_stage_marker "$CONTINUE_STARTED_MARKER" "stage=continue_pretrain_started"
  fi
elif midtrain_armed && ! midtrain_complete; then
  stage_name="midtrain"
  # MIDTRAIN phase B = CONTEXT EXTENSION to block ${MIDTRAIN_BLOCK}. Same script + fp8 recipe as
  # phase A; FIRST slice seeds from the pretrain-phase FINAL checkpoint (--resume_from + reset,
  # restart data cursor) and SAVES to models/argonne4_midtrain; later slices auto-resume there.
  mkdir -p "$MIDTRAIN_CKPT_DIR"
  if [[ -f "${MIDTRAIN_CKPT_DIR}/checkpoint_last.pt" ]]; then
    MIDTRAIN_SEED=(); MIDTRAIN_RESET=0
  else
    MIDTRAIN_SEED=(--resume_from "${CKPT_DIR}/checkpoint_last.pt"); MIDTRAIN_RESET=1
  fi
  echo "Stage: continue_pretrain.py (MIDTRAIN phase B = ctx-extension block=${MIDTRAIN_BLOCK} -> ${MIDTRAIN_CKPT_DIR})"
  echo "Data:  ${MIDTRAIN_DATA}"
  echo "Seed:  ${MIDTRAIN_SEED[*]:-<auto-resume ${MIDTRAIN_CKPT_DIR}>}  reset_schedule=${MIDTRAIN_RESET}  batch=${MIDTRAIN_BATCH} fp8=${MIDTRAIN_FP8}"
  echo "Logs:  ${REPORT_DIR}/${LOG_INDEX:-?}-${LOG_BASENAME}.out"

  set +e
  torchrun --nproc_per_node="$NGPUS" continue_pretrain.py \
    --tokenizer_path "$MIDTRAIN_TOKENIZER" \
    --data_path "$MIDTRAIN_DATA" \
    --checkpoint_dir "$MIDTRAIN_CKPT_DIR" \
    "${MIDTRAIN_SEED[@]}" \
    --fp8 "$MIDTRAIN_FP8" \
    --fp8_lm_head "$MIDTRAIN_FP8" \
    --lr "${MIDTRAIN_LR:-1e-4}" \
    --batch_size "$MIDTRAIN_BATCH" \
    --total_batch_size "$MIDTRAIN_TOTAL_BATCH" \
    --block_size "$MIDTRAIN_BLOCK" \
    --loss_chunk_size "$MIDTRAIN_CHUNK" \
    --precision bf16 \
    --flash_attention 1 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --grad_clip 0.4 \
    --warmup_steps 0 \
    --schedule wsd \
    --cooldown "$MIDTRAIN_COOLDOWN" \
    --min_lr_ratio 0.1 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --save_deadline_epoch "$SAVE_DEADLINE_EPOCH" \
    --max_epochs 1 \
    --torch_compile 1 \
    --gradient_checkpointing "$MIDTRAIN_GRAD_CKPT" \
    --checkpoint_stride "$MIDTRAIN_CKPT_STRIDE" \
    --wall_time "$WALL_TIME" \
    --reset_schedule "$MIDTRAIN_RESET" \
    --final_model_dir "$MIDTRAIN_FINAL_MODEL_DIR" \
    --completion_marker "${MIDTRAIN_CKPT_DIR}/.midtrain_complete" \
    --started_marker "${MIDTRAIN_CKPT_DIR}/.midtrain_started"
  train_status=$?
  set -e
else
  echo "All configured training stages are complete."
fi

# Clean wall-time / stage-complete exit -> chain the next slice (only when AUTO_RESUBMIT=1,
# i.e. weekend.sh; night.sh runs AUTO_RESUBMIT=0 = single slice).
if [[ "$train_status" -eq 0 ]]; then
  maybe_submit_next_slice
fi

# Crash (non-zero) -> bounded failure-retry, if enabled and we haven't already pre-submitted
# a slice via the USR1 timeout path.
if [[ "$train_status" -ne 0 && "$RESUBMIT_ON_FAILURE" == "1" && "$RESUBMIT_DONE" != "1" ]]; then
  set +e
  if (( FAILURE_RETRY_COUNT < FAILURE_RETRY_MAX )); then
    submit_failure_retry_slice
    RESUBMIT_DONE=1
  else
    echo "Training crashed (exit ${train_status}); failure-retry budget exhausted (${FAILURE_RETRY_COUNT}/${FAILURE_RETRY_MAX}); not resubmitting."
  fi
  set -e
fi

exit "$train_status"
