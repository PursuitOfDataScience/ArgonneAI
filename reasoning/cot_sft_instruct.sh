#!/bin/bash

#SBATCH --job-name=argonne-cot-sft-think
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --output=/home/youzhi/ArgonneAI/report/cot-sft-think-%j.out
#SBATCH --error=/home/youzhi/ArgonneAI/report/cot-sft-think-%j.err
#SBATCH --open-mode=truncate

# cot-sft.py is HF-Trainer based and only supports step-based save cadence
# (--save_steps), not wall-clock. Sized so a 30-min wall-clock slice lands
# at roughly the next save boundary even on slow jobs.
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --exclude=midway3-0423,midway3-[0298,0377-0378,0603-0606]
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL

set -eo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

cd /home/youzhi/ArgonneAI
mkdir -p report

# ---- Defaults (override at submit time with sbatch --export=ALL,KEY=val) ----
# Start from the SFT+DPO instruct model (HF safetensors, not .pt). cot-sft.py
# loads HF dirs via config.json + model*.safetensors.
MODEL_PATH="${MODEL_PATH:-/project/rcc/youzhi/models/instruct/dpo_ckpts}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
# Reasoning corpus. The chat_template.jinja (Qwen3-style) parses <think>...</think>
# out of assistant content into a separate reasoning_content field, so any
# dataset whose assistant turn already contains <think>...</think> will work
# out of the box. Pre-stage the data with arrow / load_from_disk and point
# DATA_PATH at the resulting directory (or a .jsonl with a `messages` column).
DATA_PATH="${DATA_PATH:-/project/rcc/youzhi/data/Bespoke-Stratos-17k}"
OUTPUT_DIR="${OUTPUT_DIR:-/project/rcc/youzhi/models/instruct/think_ckpts}"

# argonne-3.0-instruct is already at 13568 ctx (RoPE extrapolated from base),
# so MODEL_CONTEXT_LENGTH matches MAX_SEQ_LENGTH. cot-sft.py rebuilds every
# layer's rotary_emb to this size and updates config.max_position_embeddings.
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-13568}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-13568}"
ROPE_THETA="${ROPE_THETA:-1000000.0}"

# Reasoning data is long; size BATCH_SIZE down to keep 1x H200 in the ~90%
# HBM band. Effective batch = BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE.
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
SEED="${SEED:-42}"

# Step-based save cadence. ~500 steps at the defaults is roughly 20-30 min
# wall-clock; keep 4 intermediate checkpoints on disk.
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
# cot-sft.py supports exit-after-save (StopAfterCheckpointSaveCallback); the
# wrapper resubmits itself with auto-resume on each save.
EXIT_AFTER_CHECKPOINT_SAVE="${EXIT_AFTER_CHECKPOINT_SAVE:-1}"
# Slice step limit (with exit-after-save) acts as a hard safety net: force a
# checkpoint save and exit after this many steps even if save_steps hasn't
# been hit. Slightly below the SLURM --time wall budget.
SLICE_STEPS="${SLICE_STEPS:-0}"

# Skip / forward quality probes to cot-sft.py. They're useful for spotting
# the model learning <think>...</think> but are slow (4096 max_new_tokens).
# Disable by default; set QUALITY_STEPS>0 to enable (e.g. 1000 to probe
# every 1000 steps).
QUALITY_STEPS="${QUALITY_STEPS:-0}"
MAX_NEW_TOKENS_QUALITY="${MAX_NEW_TOKENS_QUALITY:-1024}"
QUALITY_DO_SAMPLE="${QUALITY_DO_SAMPLE:-0}"
QUALITY_TEMPERATURE="${QUALITY_TEMPERATURE:-0.7}"
QUALITY_TOP_P="${QUALITY_TOP_P:-0.9}"
QUALITY_TOP_K="${QUALITY_TOP_K:-40}"
QUALITY_SEED_THINK="${QUALITY_SEED_THINK:-1}"
QUALITY_NO_REPEAT_NGRAM="${QUALITY_NO_REPEAT_NGRAM:-4}"
QUALITY_REPETITION_PENALTY="${QUALITY_REPETITION_PENALTY:-1.3}"
QUALITY_FORCE_MCQ_POSTPROCESS="${QUALITY_FORCE_MCQ_POSTPROCESS:-0}"
QUALITY_FORCE_NON_MCQ_POSTPROCESS="${QUALITY_FORCE_NON_MCQ_POSTPROCESS:-1}"
QUALITY_LOG_RAW="${QUALITY_LOG_RAW:-0}"
QUALITY_QUESTIONS_FILE="${QUALITY_QUESTIONS_FILE:-/home/youzhi/ArgonneAI/cot_experiments/root_cause/quality_questions_sample.txt}"
RUN_BEFORE_QUALITY="${RUN_BEFORE_QUALITY:-0}"
RUN_AFTER_QUALITY="${RUN_AFTER_QUALITY:-0}"

# Reasoning-data preprocessing knobs (passed to cot-sft.py).
PRESERVE_RAW_REASONING="${PRESERVE_RAW_REASONING:-1}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-0}"
MAX_REASONING_ROWS="${MAX_REASONING_ROWS:-0}"

PRECISION="${PRECISION:-bf16}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# ---- Validation ----
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: $MODEL_PATH/config.json missing" >&2
  exit 2
fi
if ! compgen -G "$MODEL_PATH/model*.safetensors" >/dev/null \
   && [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "ERROR: no safetensors weights found under MODEL_PATH: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$TOKENIZER_PATH" ]] || [[ ! -e "$TOKENIZER_PATH/tokenizer_config.json" ]]; then
  echo "ERROR: tokenizer files missing under TOKENIZER_PATH: $TOKENIZER_PATH" >&2
  exit 2
fi
if [[ ! -e "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH does not exist: $DATA_PATH" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

# ---- Resume support ----
RESUME_FLAG=""
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  if [[ "$RESUME_FROM_CHECKPOINT" == "auto" || "$RESUME_FROM_CHECKPOINT" == "latest" ]]; then
    latest=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -n "$latest" ]]; then
      echo "Auto-resuming from: $latest"
      RESUME_FROM_CHECKPOINT="$latest"
    else
      echo "No intermediate checkpoint under $OUTPUT_DIR -- starting fresh."
      RESUME_FROM_CHECKPOINT=""
    fi
  fi
  RESUME_FLAG="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

SKIP_FINAL_SAVE_FLAG=""
if [[ "$SKIP_FINAL_SAVE" == "1" ]]; then
  SKIP_FINAL_SAVE_FLAG="--skip_final_save"
fi

# ---- Echo config ----
echo "============================================================"
echo "Argonne 3.0-instruct -> 3.0-think CoT SFT"
echo "============================================================"
echo "  MODEL_PATH          = $MODEL_PATH"
echo "  TOKENIZER_PATH      = $TOKENIZER_PATH"
echo "  DATA_PATH           = $DATA_PATH"
echo "  OUTPUT_DIR          = $OUTPUT_DIR"
echo "  MAX_SEQ_LENGTH      = $MAX_SEQ_LENGTH"
echo "  MODEL_CONTEXT_LENGTH= $MODEL_CONTEXT_LENGTH"
echo "  ROPE_THETA          = $ROPE_THETA"
echo "  BATCH_SIZE          = $BATCH_SIZE | GRAD_ACCUM = $GRAD_ACCUM"
echo "  LR                  = $LR | WARMUP_STEPS = $WARMUP_STEPS"
echo "  NUM_EPOCHS          = $NUM_EPOCHS | MAX_STEPS = $MAX_STEPS"
echo "  SAVE_STRATEGY       = $SAVE_STRATEGY | SAVE_STEPS = $SAVE_STEPS"
echo "  SAVE_TOTAL_LIMIT    = $SAVE_TOTAL_LIMIT"
echo "  EXIT_AFTER_SAVE     = $EXIT_AFTER_CHECKPOINT_SAVE"
echo "  SLICE_STEPS         = $SLICE_STEPS"
echo "  QUALITY_STEPS       = $QUALITY_STEPS"
echo "  PRESERVE_RAW_REASONING = $PRESERVE_RAW_REASONING"
echo "  RESUME_FROM         = ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "  SEED                = $SEED"
echo "============================================================"

# ---- Run CoT SFT (HF Trainer + torchrun for DDP-readiness) ----
EXIT_FLAG=""
if [[ "$EXIT_AFTER_CHECKPOINT_SAVE" == "1" ]]; then
  EXIT_FLAG="--exit_after_checkpoint_save"
fi
SLICE_FLAG=""
if [[ "${SLICE_STEPS:-0}" -gt 0 ]]; then
  SLICE_FLAG="--slice_steps $SLICE_STEPS"
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" /home/youzhi/ArgonneAI/reasoning/cot-sft.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --model_def /home/youzhi/ArgonneAI/model.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --model_context_length "$MODEL_CONTEXT_LENGTH" \
  --preserve_raw_reasoning "$PRESERVE_RAW_REASONING" \
  --max_think_tokens "$MAX_THINK_TOKENS" \
  --max_reasoning_rows "$MAX_REASONING_ROWS" \
  --rope_theta "$ROPE_THETA" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --num_epochs "$NUM_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --warmup_steps "$WARMUP_STEPS" \
  --quality_steps "$QUALITY_STEPS" \
  --max_new_tokens_quality "$MAX_NEW_TOKENS_QUALITY" \
  --quality_do_sample "$QUALITY_DO_SAMPLE" \
  --quality_temperature "$QUALITY_TEMPERATURE" \
  --quality_top_p "$QUALITY_TOP_P" \
  --quality_top_k "$QUALITY_TOP_K" \
  --quality_seed_think "$QUALITY_SEED_THINK" \
  --quality_no_repeat_ngram "$QUALITY_NO_REPEAT_NGRAM" \
  --quality_repetition_penalty "$QUALITY_REPETITION_PENALTY" \
  --quality_force_mcq_postprocess "$QUALITY_FORCE_MCQ_POSTPROCESS" \
  --quality_force_non_mcq_postprocess "$QUALITY_FORCE_NON_MCQ_POSTPROCESS" \
  --quality_log_raw "$QUALITY_LOG_RAW" \
  --quality_questions_file "$QUALITY_QUESTIONS_FILE" \
  --run_before_quality "$RUN_BEFORE_QUALITY" \
  --run_after_quality "$RUN_AFTER_QUALITY" \
  --save_strategy "$SAVE_STRATEGY" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --seed "$SEED" \
  --precision "$PRECISION" \
  $SKIP_FINAL_SAVE_FLAG \
  $RESUME_FLAG \
  $EXIT_FLAG \
  $SLICE_FLAG
train_status=$?

echo "cot-sft.py exited with status $train_status"

# ---- Decide whether to resubmit ----
# If the run succeeded with --exit_after_checkpoint_save, find the latest
# intermediate checkpoint and resubmit with auto-resume.
if [[ $train_status -eq 0 && "$EXIT_AFTER_CHECKPOINT_SAVE" == "1" ]]; then
  # If the final model + completion marker already exist, training is done.
  if [[ -f "$OUTPUT_DIR/.cot_sft_complete" && -f "$OUTPUT_DIR/config.json" && -f "$OUTPUT_DIR/model.safetensors" ]]; then
    echo "Found $OUTPUT_DIR/.cot_sft_complete -- CoT SFT is finished. Not resubmitting."
    exit 0
  fi
  LATEST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' 2>/dev/null | sort -V | tail -n 1 || true)
  # Detect "training complete" by checking if HF Trainer wrote the final
  # model + tokenizer at the output root and the latest checkpoint is
  # beyond the total expected steps. Without an explicit completion marker
  # from cot-sft.py, the safest check is: the final save files exist AND
  # there are no more pending steps to run.
  if [[ -n "$LATEST_CKPT" ]]; then
    CURRENT_STEP=$(basename "$LATEST_CKPT" | sed 's/checkpoint-//')
    echo "Latest checkpoint: $LATEST_CKPT (step $CURRENT_STEP of max $MAX_STEPS)"
    echo "Resubmitting with auto-resume..."

    RESUME_FROM_CHECKPOINT=auto \
    MODEL_PATH="$MODEL_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    DATA_PATH="$DATA_PATH" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" \
    MODEL_CONTEXT_LENGTH="$MODEL_CONTEXT_LENGTH" \
    ROPE_THETA="$ROPE_THETA" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    LR="$LR" \
    NUM_EPOCHS="$NUM_EPOCHS" \
    WARMUP_STEPS="$WARMUP_STEPS" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STRATEGY="$SAVE_STRATEGY" \
    SAVE_STEPS="$SAVE_STEPS" \
    SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
    EXIT_AFTER_CHECKPOINT_SAVE="$EXIT_AFTER_CHECKPOINT_SAVE" \
    SLICE_STEPS="$SLICE_STEPS" \
    QUALITY_STEPS="$QUALITY_STEPS" \
    MAX_NEW_TOKENS_QUALITY="$MAX_NEW_TOKENS_QUALITY" \
    QUALITY_DO_SAMPLE="$QUALITY_DO_SAMPLE" \
    QUALITY_TEMPERATURE="$QUALITY_TEMPERATURE" \
    QUALITY_TOP_P="$QUALITY_TOP_P" \
    QUALITY_TOP_K="$QUALITY_TOP_K" \
    QUALITY_SEED_THINK="$QUALITY_SEED_THINK" \
    QUALITY_NO_REPEAT_NGRAM="$QUALITY_NO_REPEAT_NGRAM" \
    QUALITY_REPETITION_PENALTY="$QUALITY_REPETITION_PENALTY" \
    QUALITY_FORCE_MCQ_POSTPROCESS="$QUALITY_FORCE_MCQ_POSTPROCESS" \
    QUALITY_FORCE_NON_MCQ_POSTPROCESS="$QUALITY_FORCE_NON_MCQ_POSTPROCESS" \
    QUALITY_LOG_RAW="$QUALITY_LOG_RAW" \
    QUALITY_QUESTIONS_FILE="$QUALITY_QUESTIONS_FILE" \
    RUN_BEFORE_QUALITY="$RUN_BEFORE_QUALITY" \
    RUN_AFTER_QUALITY="$RUN_AFTER_QUALITY" \
    PRESERVE_RAW_REASONING="$PRESERVE_RAW_REASONING" \
    MAX_THINK_TOKENS="$MAX_THINK_TOKENS" \
    MAX_REASONING_ROWS="$MAX_REASONING_ROWS" \
    PRECISION="$PRECISION" \
    SEED="$SEED" \
    sbatch "$0"

    echo "Resubmit requested."
    exit 0
  else
    echo "No checkpoint found under $OUTPUT_DIR -- assuming training complete."
    exit 0
  fi
fi

exit "$train_status"
