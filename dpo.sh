#!/bin/bash

#SBATCH --job-name=argonne-dpo-long
#SBATCH --account=rcc-staff
#SBATCH --partition=test

##SBATCH --account=ssd
##SBATCH --qos=ssd
##SBATCH --partition=ssd-gpu

#SBATCH --output=/project/rcc/youzhi/report/argonne-dpo-long.out
#SBATCH --error=/project/rcc/youzhi/report/argonne-dpo-long.err
#SBATCH --open-mode=truncate

#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --exclude=midway3-0423
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL

set -eo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

cd /home/youzhi/ArgonneAI

# Defaults can be overridden at submit time with sbatch --export=ALL,VAR=value,...
ARGONNE_ROOT="${ARGONNE_ROOT:-/home/youzhi/ArgonneAI}"
MODEL_PATH="${MODEL_PATH:-/project/rcc/youzhi/llm.c/checkpoints/final_model_sft_long}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-$MODEL_PATH}"
DATA_PATH="${DATA_PATH:-/project/rcc/youzhi/data/KatoHF_chatbot_arena_binarized}"
OUTPUT_DIR="${OUTPUT_DIR:-/project/rcc/youzhi/llm.c/checkpoints/final_model_dpo_long}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-13568}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:-0}"
BETA="${BETA:-0.2}"
SCORE_MODE="${SCORE_MODE:-avg}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
CHOSEN_SFT_WEIGHT="${CHOSEN_SFT_WEIGHT:-0.1}"
DATASET_RECIPE="${DATASET_RECIPE:-chat_refine_strict}"
TRAIN_LAST_BLOCKS="${TRAIN_LAST_BLOCKS:-0}"
TRAIN_EMBED_AND_HEAD="${TRAIN_EMBED_AND_HEAD:-0}"
SAVE_FINAL="${SAVE_FINAL:-1}"
QUALITY_EVERY="${QUALITY_EVERY:-7}"
SEED="${SEED:-42}"

echo "Experiment config:"
echo "  DATA_PATH=$DATA_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  DATASET_RECIPE=$DATASET_RECIPE"
echo "  MAX_STEPS=$MAX_STEPS"
echo "  QUALITY_EVERY=$QUALITY_EVERY"

python /home/youzhi/ArgonneAI/dpo.py \
  --argonne_root "$ARGONNE_ROOT" \
  --model_path "$MODEL_PATH" \
  --reference_model_path "$REFERENCE_MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --num_epochs "$NUM_EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --max_steps "$MAX_STEPS" \
  --beta "$BETA" \
  --score_mode "$SCORE_MODE" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --chosen_sft_weight "$CHOSEN_SFT_WEIGHT" \
  --dataset_recipe "$DATASET_RECIPE" \
  --train_last_blocks "$TRAIN_LAST_BLOCKS" \
  --train_embed_and_head "$TRAIN_EMBED_AND_HEAD" \
  --save_final "$SAVE_FINAL" \
  --quality_every "$QUALITY_EVERY" \
  --seed "$SEED"
