#!/bin/bash
#SBATCH --job-name=node-evaluation
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --exclude=midway3-0423,midway3-0315,midway3-0600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --constraint=H200
#SBATCH --time=7-00:00:00
#SBATCH --output=report/4-midtraining.out
#SBATCH --error=report/4-midtraining.err
#SBATCH --open-mode=truncate

set -eo pipefail

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/youzhi/ArgonneAI
mkdir -p report

TOKENIZER=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base
DATA=${DATA_OVERRIDE:-/project/rcc/youzhi/data/phase2_longctx/longmino_pool_16k_32k_qwen3_docbin_manifest.json}
INIT_CHECKPOINT=/project/rcc/youzhi/llm.c/checkpoints/checkpoint_step_425975.pt
CHECKPOINT_DIR=/project/rcc/youzhi/llm.c/checkpoints/midtraining_ctx13568_t10000

NGPUS=2
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-4}
BLOCK_SIZE=${BLOCK_SIZE_OVERRIDE:-13568}
ROPE_THETA=${ROPE_THETA_OVERRIDE:-10000}
GRAD_ACCUM=${GRAD_ACCUM_OVERRIDE:-1}
DIST_STRATEGY=${DIST_STRATEGY_OVERRIDE:-fsdp}
FSDP_SHARDING=${FSDP_SHARDING_OVERRIDE:-shard_grad_op}
FSDP_MIXED_PRECISION=${FSDP_MIXED_PRECISION_OVERRIDE:-1}
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))
TARGET_MIDTRAINING_TOKENS=${TARGET_MIDTRAINING_TOKENS_OVERRIDE:-16000000000}
WALL_TIME=${WALL_TIME_OVERRIDE:-0}
COOLDOWN=${COOLDOWN_OVERRIDE:-30000}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL_OVERRIDE:-1800}
SAMPLE_MAX_NEW_TOKENS=${SAMPLE_MAX_NEW_TOKENS_OVERRIDE:-4096}
SAMPLE_DO_SAMPLE=${SAMPLE_DO_SAMPLE_OVERRIDE:-1}
SAMPLE_TEMPERATURE=${SAMPLE_TEMPERATURE_OVERRIDE:-0.8}
SAMPLE_TOP_P=${SAMPLE_TOP_P_OVERRIDE:-0.9}
SAMPLE_REPETITION_PENALTY=${SAMPLE_REPETITION_PENALTY_OVERRIDE:-1.3}
SAMPLE_NO_REPEAT_NGRAM_SIZE=${SAMPLE_NO_REPEAT_NGRAM_SIZE_OVERRIDE:-4}
SAMPLE_SEED=${SAMPLE_SEED_OVERRIDE:-444}
FINAL_MODEL_DIR_NAME=${FINAL_MODEL_DIR_NAME_OVERRIDE:-final_model_long}

RESUME_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
  latest_step_checkpoint=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -printf '%f\n' | sort -V | tail -n 1)
  if [ -n "$latest_step_checkpoint" ]; then
    RESUME_CHECKPOINT="$CHECKPOINT_DIR/$latest_step_checkpoint"
  elif [ -e "$CHECKPOINT_DIR/checkpoint_last.pt" ]; then
    RESUME_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_last.pt"
  fi
fi

echo "Launching midtraining with:"
echo "  NGPUS=$NGPUS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BLOCK_SIZE=$BLOCK_SIZE"
echo "  ROPE_THETA=$ROPE_THETA"
echo "  DATA=$DATA"
echo "  GRAD_ACCUM=$GRAD_ACCUM"
echo "  DIST_STRATEGY=$DIST_STRATEGY"
echo "  FSDP_SHARDING=$FSDP_SHARDING"
echo "  FSDP_MIXED_PRECISION=$FSDP_MIXED_PRECISION"
echo "  TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
echo "  TARGET_MIDTRAINING_TOKENS=$TARGET_MIDTRAINING_TOKENS"
if [ "$WALL_TIME" -gt 0 ]; then
  echo "  WALL_TIME=$WALL_TIME"
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
if [ -n "$RESUME_CHECKPOINT" ]; then
  echo "  RESUME_FROM=$RESUME_CHECKPOINT"
else
  echo "  RESUME_FROM=<none>"
  echo "  INIT_CHECKPOINT=$INIT_CHECKPOINT"
fi

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
  --wall_time $WALL_TIME \
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
