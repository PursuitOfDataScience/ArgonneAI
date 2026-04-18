#!/bin/bash

#SBATCH --job-name=node-evaluation
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --output=/project/rcc/youzhi/report/cot-sft.out
#SBATCH --error=/project/rcc/youzhi/report/cot-sft.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=youzhi@rcc.uchicago.edu

set -eo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

cd /home/youzhi/ArgonneAI
mkdir -p cot_experiments/runs /project/rcc/youzhi/report

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/project/rcc/youzhi/llm.c/checkpoints/midtraining_ctx13568_t10000}"
FINAL_LONG_MODEL_DIR="${FINAL_LONG_MODEL_DIR:-${CHECKPOINT_DIR}/final_model_long}"

: "${MODEL_PATH:=$FINAL_LONG_MODEL_DIR}"
: "${TOKENIZER_PATH:=$MODEL_PATH}"
: "${DATA_PATH:=/project/rcc/youzhi/data/PursuitOfDataScience_0.5M-thinking}"
: "${OUTPUT_DIR:=/project/rcc/youzhi/llm.c/checkpoints/final_model_long_think}"

: "${MAX_SEQ_LENGTH:=13568}"
: "${MODEL_CONTEXT_LENGTH:=13568}"
: "${PRESERVE_RAW_REASONING:=1}"
: "${MAX_THINK_TOKENS:=0}"
: "${MAX_REASONING_ROWS:=0}"
: "${ROPE_THETA:=10000.0}"
: "${LR:=2e-5}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=4}"
: "${NUM_EPOCHS:=1}"
: "${MAX_STEPS:=-1}"
: "${WARMUP_STEPS:=80}"
: "${QUALITY_STEPS:=500}"
: "${MAX_NEW_TOKENS_QUALITY:=4096}"
: "${QUALITY_DO_SAMPLE:=1}"
: "${QUALITY_TEMPERATURE:=0.7}"
: "${QUALITY_TOP_P:=0.9}"
: "${QUALITY_TOP_K:=40}"
: "${QUALITY_SEED_THINK:=1}"
: "${QUALITY_NO_REPEAT_NGRAM:=4}"
: "${QUALITY_REPETITION_PENALTY:=1.3}"
: "${QUALITY_FORCE_MCQ_POSTPROCESS:=0}"
: "${QUALITY_FORCE_NON_MCQ_POSTPROCESS:=0}"
: "${QUALITY_LOG_RAW:=0}"
: "${QUALITY_QUESTIONS_FILE:=/home/youzhi/ArgonneAI/cot_experiments/root_cause/quality_questions_sample.txt}"
: "${RUN_BEFORE_QUALITY:=0}"
: "${RUN_AFTER_QUALITY:=1}"
: "${SAVE_STRATEGY:=steps}"
: "${SAVE_STEPS:=$QUALITY_STEPS}"
: "${SAVE_TOTAL_LIMIT:=2}"
: "${SKIP_FINAL_SAVE:=0}"
: "${RESUME_FROM_CHECKPOINT:=}"
: "${SEED:=48}"
: "${PRECISION:=bf16}"
: "${NPROC_PER_NODE:=1}"

if [[ "${SKIP_FINAL_SAVE}" == "1" ]]; then
  SKIP_FINAL_SAVE_FLAG="--skip_final_save"
else
  SKIP_FINAL_SAVE_FLAG=""
fi

if [[ "$MODEL_PATH" == "latest" || "$MODEL_PATH" == "latest_checkpoint" ]]; then
  latest_checkpoint=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -printf '%p\n' | sort -V | tail -n 1)
  if [[ -z "$latest_checkpoint" ]]; then
    echo "ERROR: no checkpoint_step_*.pt files found under CHECKPOINT_DIR: $CHECKPOINT_DIR" >&2
    exit 2
  fi
  MODEL_PATH="$latest_checkpoint"
fi

if [[ -z "$RESUME_FROM_CHECKPOINT" || "$RESUME_FROM_CHECKPOINT" == "latest" || "$RESUME_FROM_CHECKPOINT" == "latest_checkpoint" ]]; then
  latest_output_checkpoint=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "${latest_output_checkpoint:-}" ]]; then
    RESUME_FROM_CHECKPOINT="$latest_output_checkpoint"
  else
    RESUME_FROM_CHECKPOINT=""
  fi
fi

if [[ -f "$MODEL_PATH" && "$MODEL_PATH" == *.pt ]]; then
  if [[ ! -d "$TOKENIZER_PATH" || ! -e "$TOKENIZER_PATH/tokenizer_config.json" ]]; then
    echo "ERROR: .pt MODEL_PATH requires TOKENIZER_PATH with tokenizer files: $TOKENIZER_PATH" >&2
    exit 2
  fi
elif [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist yet: $MODEL_PATH" >&2
  echo "This job is intended to run after the midtraining final_model_long save completes." >&2
  exit 2
else
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: config.json missing under MODEL_PATH: $MODEL_PATH" >&2
    exit 2
  fi
  if ! compgen -G "$MODEL_PATH/model*.safetensors" >/dev/null && [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "ERROR: no safetensors weights found under MODEL_PATH: $MODEL_PATH" >&2
    exit 2
  fi
fi
if [[ ! -d "$TOKENIZER_PATH" || ! -e "$TOKENIZER_PATH/tokenizer_config.json" ]]; then
  echo "ERROR: tokenizer files missing under TOKENIZER_PATH: $TOKENIZER_PATH" >&2
  exit 2
fi
if [[ ! -e "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH does not exist: $DATA_PATH" >&2
  exit 2
fi

RUN_ID="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/youzhi/ArgonneAI/cot_experiments/runs/final_long_cot_${RUN_ID}}"
mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/run_config.txt" <<EOF
model_path=$MODEL_PATH
tokenizer_path=$TOKENIZER_PATH
data_path=$DATA_PATH
output_dir=$OUTPUT_DIR
max_seq_length=$MAX_SEQ_LENGTH
model_context_length=$MODEL_CONTEXT_LENGTH
preserve_raw_reasoning=$PRESERVE_RAW_REASONING
max_think_tokens=$MAX_THINK_TOKENS
max_reasoning_rows=$MAX_REASONING_ROWS
rope_theta=$ROPE_THETA
lr=$LR
batch_size=$BATCH_SIZE
grad_accum=$GRAD_ACCUM
num_epochs=$NUM_EPOCHS
max_steps=$MAX_STEPS
warmup_steps=$WARMUP_STEPS
quality_steps=$QUALITY_STEPS
max_new_tokens_quality=$MAX_NEW_TOKENS_QUALITY
quality_do_sample=$QUALITY_DO_SAMPLE
quality_temperature=$QUALITY_TEMPERATURE
quality_top_p=$QUALITY_TOP_P
quality_top_k=$QUALITY_TOP_K
quality_seed_think=$QUALITY_SEED_THINK
quality_no_repeat_ngram=$QUALITY_NO_REPEAT_NGRAM
quality_repetition_penalty=$QUALITY_REPETITION_PENALTY
quality_force_mcq_postprocess=$QUALITY_FORCE_MCQ_POSTPROCESS
quality_force_non_mcq_postprocess=$QUALITY_FORCE_NON_MCQ_POSTPROCESS
quality_log_raw=$QUALITY_LOG_RAW
quality_questions_file=$QUALITY_QUESTIONS_FILE
run_before_quality=$RUN_BEFORE_QUALITY
run_after_quality=$RUN_AFTER_QUALITY
save_strategy=$SAVE_STRATEGY
save_steps=$SAVE_STEPS
save_total_limit=$SAVE_TOTAL_LIMIT
skip_final_save=$SKIP_FINAL_SAVE
resume_from_checkpoint=$RESUME_FROM_CHECKPOINT
precision=$PRECISION
seed=$SEED
dependency_job_id=${SLURM_JOB_DEPENDENCY:-}
EOF

echo "Launching CoT SFT for final long model:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  TOKENIZER_PATH=$TOKENIZER_PATH"
echo "  DATA_PATH=$DATA_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RUN_ROOT=$RUN_ROOT"
echo "  MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "  MODEL_CONTEXT_LENGTH=$MODEL_CONTEXT_LENGTH"
echo "  MAX_STEPS=$MAX_STEPS"
echo "  SAVE_STRATEGY=$SAVE_STRATEGY"
echo "  SAVE_STEPS=$SAVE_STEPS"
echo "  SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
echo "  RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "  SKIP_FINAL_SAVE=$SKIP_FINAL_SAVE"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" /home/youzhi/ArgonneAI/cot-sft.py \
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
  ${SKIP_FINAL_SAVE_FLAG} \
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}
