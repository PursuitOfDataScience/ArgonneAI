#!/bin/bash
# =============================================================================
# night.sh -- run ONE midtraining slice scheduled for 23:00 (11pm) tonight.
#
# Mechanism (the "placeholder" trick): a tiny 1-CPU/1G placeholder job is queued
# NOW. Its only job is to sleep until 23:00 tonight, then submit the real GPU
# midtraining.sh slice (AUTO_RESUBMIT=0 -> no chaining on a CLEAN slice end).
# The 8h slice (23:00 -> 07:00) keeps the same 7am finish as before.
# Why a placeholder instead of `at`/cron: it lives in the SLURM queue, survives
# login-node restarts, and lets the scheduler hold the slot. The placeholder is
# tried across PLACEHOLDER_PARTITIONS until one accepts it.
#
# Crash resilience: the slice is submitted with RESUBMIT_ON_FAILURE=1, so if the
# GPU/node dies mid-slice (e.g. an uncorrectable ECC error, as in report/98) the
# slice does NOT just stop the workflow -- midtraining.sh resubmits a fresh slice
# that resumes from the last checkpoint and EXCLUDES the failed node, up to
# FAILURE_RETRY_MAX (default 5) times. A clean wall-time end (exit 0) still does
# not chain, preserving the single-slice intent. AUTO_RESUBMIT_TIME is set to the
# 8h GPU_SLICE_TIME so each retry gets a full window.
#
# Contrast with weekend.sh: weekend.sh launches a continuous self-resubmitting
# chain; night.sh fires exactly one GPU slice per invocation, at 23:00 (11pm).
#
# INTERMIX single-phase training: seeds the healthy pretrain base and trains on a
# doc-shuffled FineWeb+FineMath mix -> models/midtrain (same override vars as
# weekend.sh, forwarded through the placeholder). Replaces the old longmino->
# FineMath two-phase chain (which forgot world knowledge; see §13).
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/youzhi/ArgonneAI"
RUN_SCRIPT="${REPO_ROOT}/midtraining.sh"
REPORT_DIR="${REPO_ROOT}/report"
GPU_SLICE_TIME="${GPU_SLICE_TIME:-08:00:00}"
WALL_TIME_SECONDS="${WALL_TIME_SECONDS:-28800}"

PLACEHOLDER_ACCOUNT="${PLACEHOLDER_ACCOUNT:-rcc-staff}"
PLACEHOLDER_PARTITIONS="${PLACEHOLDER_PARTITIONS:-caslake amd}"
PLACEHOLDER_CPUS="${PLACEHOLDER_CPUS:-1}"
PLACEHOLDER_MEM="${PLACEHOLDER_MEM:-1G}"

# --- INTERMIX single-phase overrides (match weekend.sh; see §13) -------------
# Defaults match weekend.sh. Embedded as literals into the placeholder's wrapped
# submit below, then forwarded to the GPU slice via its --export list.
DATA_OVERRIDE="${DATA_OVERRIDE:-/project/rcc/youzhi/data/intermix/intermix_manifest.json}"
DOC_SHUFFLE_OVERRIDE="${DOC_SHUFFLE_OVERRIDE:-1}"
ROPE_THETA_OVERRIDE="${ROPE_THETA_OVERRIDE:-1000000}"
PHASE2_DATA=""   # single-phase intermix; no sequential FineMath phase

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "ERROR: run script not found: $RUN_SCRIPT" >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"

to_slurm_time() {
  local total="$1"
  local days=$((total / 86400))
  local rem=$((total % 86400))
  local hh=$((rem / 3600))
  local mm=$(((rem % 3600) / 60))
  local ss=$((rem % 60))
  if ((days > 0)); then
    printf "%d-%02d:%02d:%02d" "$days" "$hh" "$mm" "$ss"
  else
    printf "%02d:%02d:%02d" "$hh" "$mm" "$ss"
  fi
}

now_epoch=$(date +%s)
target_epoch=$(date -d "today 23:00:00" +%s)
wait_seconds=$((target_epoch - now_epoch))
if ((wait_seconds < 1)); then
  target_epoch=$(date -d "today 23:00:00 +1 day" +%s)
  wait_seconds=$((target_epoch - now_epoch))
fi

placeholder_seconds=$((wait_seconds + 900))
if ((placeholder_seconds < 60)); then
  placeholder_seconds=60
fi
placeholder_time=$(to_slurm_time "$placeholder_seconds")

printf -v WRAP_CMD '%s\n' \
  "set -euo pipefail" \
  "TARGET_EPOCH=${target_epoch}" \
  "if [ \$(date +%s) -ge \$TARGET_EPOCH ]; then" \
  "  TARGET_EPOCH=\$(date -d 'today 23:00:00' +%s)" \
  "fi" \
  "scontrol update JobId=\"\$SLURM_JOB_ID\" EndTime=\$(date -d \"@\${TARGET_EPOCH}\" '+%Y-%m-%dT%H:%M:%S')" \
  "while true; do" \
  "  now_epoch=\$(date +%s)" \
  "  remaining=\$((TARGET_EPOCH - now_epoch))" \
  "  if [ \"\$remaining\" -le 0 ]; then" \
  "    break" \
  "  fi" \
  "  if [ \"\$remaining\" -gt 60 ]; then" \
  "    sleep 60" \
  "  else" \
  "    sleep \"\$remaining\"" \
  "  fi" \
  "done" \
  "cd ${REPO_ROOT}" \
  "REPORT_DIR=${REPORT_DIR}" \
  "LOG_BASENAME=midtrain" \
  "BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE:-1}" \
  "BLOCK_SIZE_OVERRIDE=${BLOCK_SIZE_OVERRIDE:-}" \
  "GPU_SLICE_TIME=${GPU_SLICE_TIME}" \
  "WALL_TIME_SECONDS=${WALL_TIME_SECONDS}" \
  "DATA_OVERRIDE=${DATA_OVERRIDE}" \
  "DOC_SHUFFLE_OVERRIDE=${DOC_SHUFFLE_OVERRIDE}" \
  "ROPE_THETA_OVERRIDE=${ROPE_THETA_OVERRIDE}" \
  "mkdir -p \"\$REPORT_DIR\"" \
  "max_index=\$(find \"\$REPORT_DIR\" -maxdepth 1 -type f \\( -name '[0-9]*-midtrain.out' -o -name '[0-9]*-midtrain.err' \\) -printf '%f\n' 2>/dev/null | sed -E 's/^([0-9]+)-midtrain\\.(out|err)\$/\\1/' | sort -n | tail -n 1)" \
  "if [ -n \"\$max_index\" ]; then" \
  "  LOG_INDEX=\$((max_index + 1))" \
  "else" \
  "  LOG_INDEX=0" \
  "fi" \
  "EXTRA_EXPORT=" \
  "if [ -n \"\$BATCH_SIZE_OVERRIDE\" ]; then EXTRA_EXPORT=\"\${EXTRA_EXPORT},BATCH_SIZE_OVERRIDE=\${BATCH_SIZE_OVERRIDE}\"; fi" \
  "if [ -n \"\$BLOCK_SIZE_OVERRIDE\" ]; then EXTRA_EXPORT=\"\${EXTRA_EXPORT},BLOCK_SIZE_OVERRIDE=\${BLOCK_SIZE_OVERRIDE}\"; fi" \
  "EXTRA_EXPORT=\"\${EXTRA_EXPORT},DATA_OVERRIDE=\${DATA_OVERRIDE},DOC_SHUFFLE_OVERRIDE=\${DOC_SHUFFLE_OVERRIDE},ROPE_THETA_OVERRIDE=\${ROPE_THETA_OVERRIDE},PHASE2_DATA=\"" \
  "echo \"Submitting ${RUN_SCRIPT} with logs \${REPORT_DIR}/\${LOG_INDEX}-midtrain.out\"" \
  "sbatch --time=\"\${GPU_SLICE_TIME}\" --output=\"\${REPORT_DIR}/\${LOG_INDEX}-midtrain.out\" --error=\"\${REPORT_DIR}/\${LOG_INDEX}-midtrain.err\" --export=ALL,AUTO_RESUBMIT=0,RESUBMIT_ON_FAILURE=1,AUTO_RESUBMIT_TIME=\"\${GPU_SLICE_TIME}\",WALL_TIME_OVERRIDE=\"\${WALL_TIME_SECONDS}\",LOG_INDEX=\"\${LOG_INDEX}\",REPORT_DIR=\"\${REPORT_DIR}\",LOG_BASENAME=midtrain\${EXTRA_EXPORT} ${RUN_SCRIPT}" \
  "exit 0"

SBATCH_CMD=(
  :
)

partition_candidates_raw="${PLACEHOLDER_PARTITIONS//,/ }"
read -r -a PARTITION_CANDIDATES <<< "$partition_candidates_raw"
if [[ "${#PARTITION_CANDIDATES[@]}" -eq 0 ]]; then
  echo "ERROR: PLACEHOLDER_PARTITIONS is empty." >&2
  exit 2
fi

build_sbatch_cmd() {
  local partition="$1"
  SBATCH_CMD=(
    sbatch
    --parsable
    --job-name=test
    --account="$PLACEHOLDER_ACCOUNT"
    --partition="$partition"
    --cpus-per-task="$PLACEHOLDER_CPUS"
    --mem="$PLACEHOLDER_MEM"
    --time="$placeholder_time"
    --output=/dev/null
    --error=/dev/null
    --wrap "$WRAP_CMD"
  )
}

echo "Current time:  $(date '+%F %T %Z')"
echo "Target time:   $(date -d "@${target_epoch}" '+%F %T %Z')"
echo "Wait seconds:  ${wait_seconds}"
echo "Placeholder:   account=${PLACEHOLDER_ACCOUNT} partitions='${PARTITION_CANDIDATES[*]}' cpus=${PLACEHOLDER_CPUS} mem=${PLACEHOLDER_MEM} time=${placeholder_time}"
echo "GPU job:       ${RUN_SCRIPT} (${GPU_SLICE_TIME} slice; runs midtraining; logs use next report/N-midtrain.* index)"

if ((DRY_RUN == 1)); then
  echo
  echo "Dry run commands (first valid partition wins at runtime):"
  for partition in "${PARTITION_CANDIDATES[@]}"; do
    build_sbatch_cmd "$partition"
    printf '  %q ' "${SBATCH_CMD[@]}"
    echo
  done
  exit 0
fi

job_id=""
selected_partition=""
last_error=""
for partition in "${PARTITION_CANDIDATES[@]}"; do
  build_sbatch_cmd "$partition"
  echo "Trying placeholder submit on partition: $partition"
  if submit_output="$("${SBATCH_CMD[@]}" 2>&1)"; then
    job_id="$(echo "$submit_output" | tail -n1 | awk -F';' '{print $1}')"
    selected_partition="$partition"
    break
  fi
  echo "Partition $partition rejected; trying next."
  last_error="$submit_output"
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit placeholder on all candidate partitions: ${PARTITION_CANDIDATES[*]}" >&2
  if [[ -n "$last_error" ]]; then
    echo "$last_error" >&2
  fi
  exit 1
fi

echo "Submitted placeholder job: ${job_id} (partition=${selected_partition})"
echo "It will submit ${RUN_SCRIPT} at (or just after) 23:00 (11pm)."
