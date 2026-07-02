#!/bin/bash
#SBATCH --job-name=midtrain
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --exclude=midway3-0423,midway3-[0298,0377-0378,0603-0606]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:3
#SBATCH --constraint=H200
#SBATCH --time=07:00:00
#SBATCH --output=report/0-midtrain.out
#SBATCH --error=report/0-midtrain.err
#SBATCH --open-mode=truncate

set -eo pipefail

# --- Auto-resubmit plumbing (used by weekend.sh long chains) ---
REPO_ROOT="/home/youzhi/ArgonneAI"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/report}"
LOG_BASENAME="${LOG_BASENAME:-midtrain}"
LOG_INDEX="${LOG_INDEX:-0}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-0}"
AUTO_RESUBMIT_TIME="${AUTO_RESUBMIT_TIME:-07:00:00}"

# --- Failure-retry plumbing (transient infra failures: GPU ECC, node death, NCCL) ---
# A clean wall-time slice exits 0 (midtraining.py saves a checkpoint and returns);
# only a real crash (uncorrectable ECC, OOM, NCCL abort, killed process) exits
# non-zero. When RESUBMIT_ON_FAILURE=1, such a non-zero exit resubmits a FRESH
# slice that resumes from the latest checkpoint instead of letting the workflow
# stop. The node that just died is added to the SLURM exclude list so we are not
# rescheduled onto the same faulty hardware. A retry cap stops infinite crash
# loops (e.g. a genuine code bug that fails identically every time).
RESUBMIT_ON_FAILURE="${RESUBMIT_ON_FAILURE:-0}"
FAILURE_RETRY_COUNT="${FAILURE_RETRY_COUNT:-0}"
FAILURE_RETRY_MAX="${FAILURE_RETRY_MAX:-5}"
EXTRA_EXCLUDE="${EXTRA_EXCLUDE:-}"   # comma-list of failed nodes, accumulated across retries
# Mirror of the #SBATCH --exclude directive (top of file). A command-line
# --exclude OVERRIDES (does not merge with) the directive, so a failure-retry
# must re-state the full base list here, then append the failed node(s).
BASE_EXCLUDE="midway3-0423,midway3-[0298,0377-0378,0603-0606]"

find_next_log_index() {
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
    echo 0
  fi
}

RESUBMIT_DONE=0

submit_next_slice() {
  local current_index next_index next_available next_job_id
  current_index="${LOG_INDEX:-}"
  if [[ "$current_index" =~ ^[0-9]+$ ]]; then
    next_index=$((current_index + 1))
  else
    next_index=$(find_next_log_index)
  fi
  next_available=$(find_next_log_index)
  if ((next_index < next_available)); then
    next_index="$next_available"
  fi
  next_job_id=$(
    sbatch \
      --parsable \
      --time="$AUTO_RESUBMIT_TIME" \
      --signal=B:USR1@180 \
      --output="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.out" \
      --error="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.err" \
      --export=ALL,AUTO_RESUBMIT=1,RESUBMIT_ON_FAILURE="${RESUBMIT_ON_FAILURE}",FAILURE_RETRY_MAX="${FAILURE_RETRY_MAX}",WALL_TIME_OVERRIDE="${WALL_TIME}",AUTO_RESUBMIT_TIME="${AUTO_RESUBMIT_TIME}",LOG_INDEX="${next_index}",REPORT_DIR="${REPORT_DIR}",LOG_BASENAME="${LOG_BASENAME}" \
      "${REPO_ROOT}/midtraining.sh"
  )
  echo "Submitted next training slice as job ${next_job_id} with log index ${next_index}"
}

# Resubmit a fresh slice after a crash. Resumes from the latest checkpoint (the
# resume scan below auto-detects it), excludes the failed node, increments and
# forwards the retry counter so the cap is honored across the retry chain, and
# preserves AUTO_RESUBMIT so a retried weekend-chain slice still chains on success.
submit_failure_retry_slice() {
  local next_index new_count failed_node new_extra exclude_list next_job_id
  next_index=$(find_next_log_index)
  new_count=$((FAILURE_RETRY_COUNT + 1))
  failed_node="${SLURMD_NODENAME:-$(hostname -s 2>/dev/null)}"
  new_extra="$EXTRA_EXCLUDE"
  if [[ -n "$failed_node" ]]; then
    if [[ -n "$new_extra" ]]; then new_extra="${new_extra},${failed_node}"; else new_extra="$failed_node"; fi
  fi
  exclude_list="$BASE_EXCLUDE"
  if [[ -n "$new_extra" ]]; then exclude_list="${exclude_list},${new_extra}"; fi
  next_job_id=$(
    sbatch \
      --parsable \
      --time="$AUTO_RESUBMIT_TIME" \
      --signal=B:USR1@180 \
      --exclude="$exclude_list" \
      --output="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.out" \
      --error="${REPORT_DIR}/${next_index}-${LOG_BASENAME}.err" \
      --export=ALL,AUTO_RESUBMIT="${AUTO_RESUBMIT}",RESUBMIT_ON_FAILURE=1,FAILURE_RETRY_COUNT="${new_count}",FAILURE_RETRY_MAX="${FAILURE_RETRY_MAX}",EXTRA_EXCLUDE="${new_extra}",WALL_TIME_OVERRIDE="${WALL_TIME}",AUTO_RESUBMIT_TIME="${AUTO_RESUBMIT_TIME}",LOG_INDEX="${next_index}",REPORT_DIR="${REPORT_DIR}",LOG_BASENAME="${LOG_BASENAME}" \
      "${REPO_ROOT}/midtraining.sh"
  )
  echo "Training slice crashed on node '${failed_node}'; resubmitted failure-retry ${new_count}/${FAILURE_RETRY_MAX} as job ${next_job_id} (log index ${next_index}); excluding: ${exclude_list}"
}

handle_timeout_warning() {
  echo "Received Slurm timeout warning signal; pre-submitting next slice."
  set +e
  if [[ "$AUTO_RESUBMIT" == "1" && "$RESUBMIT_DONE" != "1" ]]; then
    submit_next_slice
    RESUBMIT_DONE=1
  fi
  set -e
}

trap 'handle_timeout_warning' USR1
# --- End auto-resubmit ---

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/youzhi/ArgonneAI
mkdir -p report

TOKENIZER=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base
DATA=${DATA_OVERRIDE:-/project/rcc/youzhi/data/phase2_longctx/longmino_pool_16k_32k_qwen3_docbin_manifest.json}
CHECKPOINT_ROOT=/project/rcc/youzhi/models
# Base model: /project/rcc/youzhi/models/pretrain/final_model_complete
# Use the most recent pretrain checkpoint as the seed checkpoint for the first midtraining launch.
INIT_CHECKPOINT=${INIT_CHECKPOINT_OVERRIDE:-${CHECKPOINT_ROOT}/pretrain/checkpoint_step_329148.pt}
# Midtraining checkpoints (and final_model) live under models/midtrain/.
CHECKPOINT_DIR=${CHECKPOINT_DIR_OVERRIDE:-${CHECKPOINT_ROOT}/midtrain}

NGPUS=3
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-1}
BLOCK_SIZE=${BLOCK_SIZE_OVERRIDE:-13568}
ROPE_THETA=${ROPE_THETA_OVERRIDE:-10000}
# Doc interleaving. 0 = read docs in manifest order (fine for a single-source
# corpus). 1 = globally shuffle docs each epoch -- REQUIRED for an INTERMIX
# manifest (general + math shards) so the two sources are truly mixed instead of
# trained sequentially (general-then-math would re-cause catastrophic forgetting).
DOC_SHUFFLE=${DOC_SHUFFLE_OVERRIDE:-0}
GRAD_ACCUM=${GRAD_ACCUM_OVERRIDE:-1}
DIST_STRATEGY=${DIST_STRATEGY_OVERRIDE:-fsdp}
FSDP_SHARDING=${FSDP_SHARDING_OVERRIDE:-shard_grad_op}
FSDP_MIXED_PRECISION=${FSDP_MIXED_PRECISION_OVERRIDE:-1}
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))
TARGET_MIDTRAINING_TOKENS=${TARGET_MIDTRAINING_TOKENS_OVERRIDE:-16000000000}
# Save + exit 5 minutes before the 7-hour SLURM limit.
WALL_TIME=${WALL_TIME_OVERRIDE:-$((7 * 3600))}
# Cooldown is off by default. Set COOLDOWN_OVERRIDE=1 to turn it on.
COOLDOWN=${COOLDOWN_OVERRIDE:-0}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL_OVERRIDE:-1800}
SAMPLE_MAX_NEW_TOKENS=${SAMPLE_MAX_NEW_TOKENS_OVERRIDE:-1024}
SAMPLE_DO_SAMPLE=${SAMPLE_DO_SAMPLE_OVERRIDE:-1}
SAMPLE_TEMPERATURE=${SAMPLE_TEMPERATURE_OVERRIDE:-0.8}
SAMPLE_TOP_P=${SAMPLE_TOP_P_OVERRIDE:-0.9}
SAMPLE_REPETITION_PENALTY=${SAMPLE_REPETITION_PENALTY_OVERRIDE:-1.3}
SAMPLE_NO_REPEAT_NGRAM_SIZE=${SAMPLE_NO_REPEAT_NGRAM_SIZE_OVERRIDE:-4}
SAMPLE_SEED=${SAMPLE_SEED_OVERRIDE:-444}
FINAL_MODEL_DIR_NAME=${FINAL_MODEL_DIR_NAME_OVERRIDE:-final_model_complete}

# ---------------------------------------------------------------------------
# Optional PHASE 2: automatically continue midtraining on a SECOND dataset once
# Phase 1 (the default longmino pool above) has finished its token target.
#
# Why this lives here (not in weekend.sh/night.sh): the dataset, seed checkpoint
# and token target are all chosen in THIS worker script.
#
# Completion signal: midtraining.py writes "$CHECKPOINT_DIR/$FINAL_MODEL_DIR_NAME"
# ONLY when the token target is reached (never on a wall-time slice exit). So the
# presence of that dir == that phase is done.
#
# The phase switch is re-evaluated in TWO places, so it works for both runners:
#   * In-slice (the loop below): the moment Phase 1 writes its marker we loop
#     back here and continue Phase 2 in the SAME GPU allocation, as long as
#     wall-time remains. This is what makes a single, non-resubmitting slice
#     (night.sh, AUTO_RESUBMIT=0) flip to Phase 2 instead of just quitting with
#     hours of GPU time unused.
#   * Across slices (weekend.sh's AUTO_RESUBMIT=1 chain): if a phase is still
#     running at the wall-time stop we break and the resubmitted slice resumes
#     it, re-entering this same switch on the next invocation.
#
# Fully opt-in: if PHASE2_DATA is empty, NOTHING below changes and the run is a
# normal single-phase run.
# ---------------------------------------------------------------------------
PHASE2_DATA="${PHASE2_DATA:-}"
# Don't START a fresh in-slice phase with less than this much wall time left
# (model load + a ~36 GB checkpoint write need headroom). The first phase of a
# slice always runs; this only gates whether to chain ANOTHER phase after one
# finishes early. Ignored when WALL_TIME<=0 (no limit).
PHASE_MIN_SECONDS="${PHASE_MIN_SECONDS_OVERRIDE:-1800}"

# Phase-1 defaults, captured before the loop so each iteration can reset to them
# before the PHASE 2 switch (re)decides what to run.
BASE_DATA="$DATA"
BASE_CHECKPOINT_DIR="$CHECKPOINT_DIR"
BASE_INIT_CHECKPOINT="$INIT_CHECKPOINT"
BASE_TARGET_MIDTRAINING_TOKENS="$TARGET_MIDTRAINING_TOKENS"

SLICE_START=$(date +%s)
phase_iter=0
train_status=0

while true; do
phase_iter=$((phase_iter + 1))

# Reset to Phase-1 defaults; the PHASE 2 switch overrides them when Phase 1 done.
DATA="$BASE_DATA"
CHECKPOINT_DIR="$BASE_CHECKPOINT_DIR"
INIT_CHECKPOINT="$BASE_INIT_CHECKPOINT"
TARGET_MIDTRAINING_TOKENS="$BASE_TARGET_MIDTRAINING_TOKENS"
# Marker written by midtraining.py ONLY when THIS phase reaches its token target.
CURRENT_DONE_MARKER="$CHECKPOINT_DIR/$FINAL_MODEL_DIR_NAME"

# Single-phase completion gate: once the phase has written its final-model
# marker (token target or max_epochs reached), there is nothing left to train.
# Without this exit, an AUTO_RESUBMIT=1 chain keeps launching slices that each
# train ~1 step past the finished epoch, save a 36 GB checkpoint, re-write the
# final artifacts, and resubmit -- forever. (The two-phase path already has its
# own PHASE2_DONE_MARKER exit below.)
if [ -z "$PHASE2_DATA" ] && [ -e "$CURRENT_DONE_MARKER" ]; then
  echo "Single-phase run already complete: $CURRENT_DONE_MARKER exists -- nothing to do; not resubmitting."
  exit 0
fi

if [ -n "$PHASE2_DATA" ]; then
  PHASE1_CKPT_DIR="$CHECKPOINT_ROOT/midtrain"
  PHASE1_DONE_MARKER="$PHASE1_CKPT_DIR/$FINAL_MODEL_DIR_NAME"
  PHASE2_CHECKPOINT_DIR="${PHASE2_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/midtrain_finemath}"
  PHASE2_DONE_MARKER="$PHASE2_CHECKPOINT_DIR/$FINAL_MODEL_DIR_NAME"
  PHASE2_TARGET_TOKENS="${PHASE2_TARGET_TOKENS:-10000000000}"   # ~10B FineMath tokens

  if [ -e "$PHASE2_DONE_MARKER" ]; then
    # Both phases finished -> stop cleanly (the resubmit code at the bottom never
    # runs because we exit here before torchrun).
    echo "PHASE 2 already complete: $PHASE2_DONE_MARKER -- nothing to do; not resubmitting."
    exit 0
  fi

  if [ -e "$PHASE1_DONE_MARKER" ]; then
    if [ ! -f "$PHASE2_DATA" ]; then
      # Phase 1 is done but the Phase-2 manifest isn't ready: stop rather than
      # loop on no-op finalize slices. (Tokenize FineMath, then re-launch.)
      echo "PHASE 1 done but PHASE2_DATA manifest not found: $PHASE2_DATA -- stopping; not resubmitting."
      exit 0
    fi
    echo "PHASE 1 complete -> switching to PHASE 2 dataset: $PHASE2_DATA"
    DATA="$PHASE2_DATA"
    CHECKPOINT_DIR="$PHASE2_CHECKPOINT_DIR"
    TARGET_MIDTRAINING_TOKENS="$PHASE2_TARGET_TOKENS"
    CURRENT_DONE_MARKER="$PHASE2_DONE_MARKER"
    mkdir -p "$CHECKPOINT_DIR"
    # First Phase-2 slice seeds from Phase 1's latest checkpoint; subsequent
    # slices resume from Phase-2 checkpoints found in CHECKPOINT_DIR (the resume
    # scan below now points at PHASE2_CHECKPOINT_DIR).
    phase1_seed=$(find "$PHASE1_CKPT_DIR" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -printf '%f\n' | sort -V | tail -n 1)
    if [ -n "$phase1_seed" ]; then
      INIT_CHECKPOINT="$PHASE1_CKPT_DIR/$phase1_seed"
    fi
  else
    echo "PHASE 2 enabled (PHASE2_DATA set) but PHASE 1 not finished yet -> continuing PHASE 1."
  fi
fi

# Per-phase wall-time budget: keep the WHOLE slice inside the original WALL_TIME
# so the in-process 600s pre-kill save keeps its margin no matter how many phases
# we chain here. Each chained phase gets only the slice time still remaining.
PHASE_WALL_TIME="$WALL_TIME"
if [ "$WALL_TIME" -gt 0 ]; then
  now_epoch=$(date +%s)
  elapsed=$((now_epoch - SLICE_START))
  PHASE_WALL_TIME=$((WALL_TIME - elapsed))
  if [ "$phase_iter" -gt 1 ] && [ "$PHASE_WALL_TIME" -lt "$PHASE_MIN_SECONDS" ]; then
    echo "Only ${PHASE_WALL_TIME}s of wall time left (< ${PHASE_MIN_SECONDS}s); not starting another phase this slice."
    break
  fi
fi

RESUME_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
  latest_step_checkpoint=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -printf '%f\n' | sort -V | tail -n 1)
  if [ -n "$latest_step_checkpoint" ]; then
    RESUME_CHECKPOINT="$CHECKPOINT_DIR/$latest_step_checkpoint"
  elif [ -e "$CHECKPOINT_DIR/checkpoint_last.pt" ]; then
    RESUME_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_last.pt"
  fi
fi

echo "Launching midtraining (slice phase #${phase_iter}) with:"
echo "  NGPUS=$NGPUS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BLOCK_SIZE=$BLOCK_SIZE"
echo "  ROPE_THETA=$ROPE_THETA"
echo "  DOC_SHUFFLE=$DOC_SHUFFLE"
echo "  DATA=$DATA"
echo "  GRAD_ACCUM=$GRAD_ACCUM"
echo "  DIST_STRATEGY=$DIST_STRATEGY"
echo "  FSDP_SHARDING=$FSDP_SHARDING"
echo "  FSDP_MIXED_PRECISION=$FSDP_MIXED_PRECISION"
echo "  TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
echo "  TARGET_MIDTRAINING_TOKENS=$TARGET_MIDTRAINING_TOKENS"
if [ "$WALL_TIME" -gt 0 ]; then
  echo "  WALL_TIME=$PHASE_WALL_TIME (remaining of slice budget $WALL_TIME)"
fi
echo "  CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "  COOLDOWN=$COOLDOWN"
echo "  FINAL_MODEL_DIR_NAME=$FINAL_MODEL_DIR_NAME"
echo "  SAMPLE_MAX_NEW_TOKENS=$SAMPLE_MAX_NEW_TOKENS"
echo "  SAMPLE_DO_SAMPLE=$SAMPLE_DO_SAMPLE"
echo "  SAMPLE_TEMPERATURE=$SAMPLE_TEMPERATURE"
echo "  SAMPLE_TOP_P=$SAMPLE_TOP_P"
echo "  SAMPLE_REPETITION_PENALTY=$SAMPLE_REPETITION_PENALTY"
echo "  SAMPLE_NO_REPEAT_NGRAM_SIZE=$SAMPLE_NO_REPEAT_NGRAM_SIZE"
echo "  SAMPLE_SEED=$SAMPLE_SEED"
echo "  LOG_INDEX=${LOG_INDEX}"
echo "  AUTO_RESUBMIT=${AUTO_RESUBMIT}"
if [ -n "$RESUME_CHECKPOINT" ]; then
  echo "  RESUME_FROM=$RESUME_CHECKPOINT"
else
  echo "  RESUME_FROM=<none>"
  echo "  INIT_CHECKPOINT=$INIT_CHECKPOINT"
fi

train_status=0
set +e
torchrun --nproc_per_node=$NGPUS midtraining.py \
  --tokenizer_path $TOKENIZER \
  --data_path $DATA \
  --checkpoint_dir $CHECKPOINT_DIR \
  --init_checkpoint_path $INIT_CHECKPOINT \
  --lr 3e-4 \
  --batch_size $BATCH_SIZE \
  --total_batch_size $TOTAL_BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --rope_theta $ROPE_THETA \
  --doc_shuffle $DOC_SHUFFLE \
  --precision bf16 \
  --flash_attention 1 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --warmup_steps 1000 \
  --schedule wsd \
  --cooldown $COOLDOWN \
  --min_lr_ratio 0.1 \
  --checkpoint_interval $CHECKPOINT_INTERVAL \
  --max_epochs 1 \
  --torch_compile 1 \
  --gradient_checkpointing 1 \
  --target_midtraining_tokens $TARGET_MIDTRAINING_TOKENS \
  --wall_time $PHASE_WALL_TIME \
  --sample_prompt "Long long time ago" \
  --sample_max_new_tokens $SAMPLE_MAX_NEW_TOKENS \
  --sample_do_sample $SAMPLE_DO_SAMPLE \
  --sample_temperature $SAMPLE_TEMPERATURE \
  --sample_top_p $SAMPLE_TOP_P \
  --sample_repetition_penalty $SAMPLE_REPETITION_PENALTY \
  --sample_no_repeat_ngram_size $SAMPLE_NO_REPEAT_NGRAM_SIZE \
  --sample_seed $SAMPLE_SEED \
  --final_model_dir_name $FINAL_MODEL_DIR_NAME \
  --distributed_strategy $DIST_STRATEGY \
  --fsdp_sharding_strategy $FSDP_SHARDING \
  --fsdp_mixed_precision $FSDP_MIXED_PRECISION \
  ${RESUME_CHECKPOINT:+--resume_from $RESUME_CHECKPOINT}
train_status=$?
set -e

if [ "$train_status" -ne 0 ]; then
  echo "midtraining.py exited non-zero (${train_status}); stopping phase loop."
  break
fi

# Continue IN-SLICE only if this phase actually reached its token target (marker
# written) and another phase may follow. If the marker is absent the phase
# stopped on wall time -- leave the rest to the next slice (weekend.sh resubmit
# or the next night.sh invocation), which resumes it from its checkpoints.
if [ -z "$PHASE2_DATA" ]; then
  break   # single-phase run: nothing to chain in-slice
fi
if [ ! -e "$CURRENT_DONE_MARKER" ]; then
  echo "Phase did not reach its token target this slice (wall-time stop); the next slice will resume it."
  break
fi
echo "Phase complete ($CURRENT_DONE_MARKER) -> re-evaluating for the next phase within this slice."
done

if [[ "$train_status" -eq 0 && "$AUTO_RESUBMIT" == "1" && "$RESUBMIT_DONE" != "1" ]]; then
  # Don't chain another slice if this slice just finished the (single-phase)
  # run -- the completion marker now exists and the next slice would only hit
  # the completion gate above and die. Saves one no-op GPU allocation.
  if [[ -z "$PHASE2_DATA" && -e "$CURRENT_DONE_MARKER" ]]; then
    echo "Phase completed this slice ($CURRENT_DONE_MARKER); ending the chain without resubmitting."
  else
    submit_next_slice
    RESUBMIT_DONE=1
  fi
fi

# A crash (non-zero exit) is NOT a clean wall-time stop. Rather than letting the
# workflow die here, resubmit a fresh slice that resumes from the last checkpoint
# -- bounded by FAILURE_RETRY_MAX so a deterministic bug can't loop forever.
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
