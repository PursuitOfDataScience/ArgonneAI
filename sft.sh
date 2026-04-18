#!/bin/bash

#SBATCH --job-name=node-evaluation
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --output=/home/youzhi/ArgonneAI/report/sft-long-%j.out
#SBATCH --error=/home/youzhi/ArgonneAI/report/sft-long-%j.err
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
mkdir -p report

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/project/rcc/youzhi/llm.c/checkpoints/midtraining_ctx13568_t10000}"
FINAL_LONG_MODEL_DIR="${FINAL_LONG_MODEL_DIR:-${CHECKPOINT_DIR}/final_model_long}"

: "${MODEL_PATH:=$FINAL_LONG_MODEL_DIR}"
: "${DATA_PATH:=/project/rcc/youzhi/data/HuggingFaceH4_ultrachat_200k/train_sft}"
: "${OUTPUT_DIR:=/project/rcc/youzhi/llm.c/checkpoints/final_model_sft_long}"
: "${MAX_SEQ_LENGTH:=13568}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=4}"
: "${LR:=2e-5}"
: "${NUM_EPOCHS:=1}"
: "${WARMUP_STEPS:=100}"
: "${QUALITY_EVERY:=500}"
: "${MAX_NEW_TOKENS_QUALITY:=512}"
: "${RUN_BEFORE_QUALITY:=1}"
: "${RUN_AFTER_QUALITY:=1}"
: "${MAX_STEPS:=-1}"
: "${SKIP_FINAL_SAVE:=0}"
: "${SEED:=42}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist yet: $MODEL_PATH" >&2
  echo "This job is intended to run after midtraining saves final_model_long." >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: config.json missing under MODEL_PATH: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "ERROR: model.safetensors missing under MODEL_PATH: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -e "$MODEL_PATH/tokenizer_config.json" ]]; then
  echo "ERROR: tokenizer files missing under MODEL_PATH: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH does not exist: $DATA_PATH" >&2
  exit 2
fi

if [[ "${SKIP_FINAL_SAVE}" == "1" ]]; then
  SKIP_FINAL_SAVE_FLAG="--skip_final_save"
else
  SKIP_FINAL_SAVE_FLAG=""
fi

echo "Launching long-context SFT:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  DATA_PATH=$DATA_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  GRAD_ACCUM=$GRAD_ACCUM"
echo "  NUM_EPOCHS=$NUM_EPOCHS"
echo "  MAX_STEPS=$MAX_STEPS"
echo "  QUALITY_EVERY=$QUALITY_EVERY"
echo "  SKIP_FINAL_SAVE=$SKIP_FINAL_SAVE"

python /home/youzhi/ArgonneAI/sft.py \
  --argonne_root /home/youzhi/ArgonneAI \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --num_epochs "$NUM_EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --quality_every "$QUALITY_EVERY" \
  --max_new_tokens_quality "$MAX_NEW_TOKENS_QUALITY" \
  --run_before_quality "$RUN_BEFORE_QUALITY" \
  --run_after_quality "$RUN_AFTER_QUALITY" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  ${SKIP_FINAL_SAVE_FLAG}
