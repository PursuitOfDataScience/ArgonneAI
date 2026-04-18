#!/bin/bash
#SBATCH --job-name=node-evaluation
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --exclude=midway3-0423,midway3-0315
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --constraint=H200
#SBATCH --time=7-00:00:00
#SBATCH --output=report/1-train.out
#SBATCH --error=report/1-train.err

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/youzhi/ArgonneAI

# Paths
TOKENIZER=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base
DATA=/project/rcc/youzhi/fineweb-binary-qwen3/train.bin
VAL_DATA=/home/youzhi/ArgonneAI/nextrun/val.bin
CKPT_DIR=/project/rcc/youzhi/argonne3.0/checkpoints

# Number of GPUs (change --gres above to match)
NGPUS=2

# Wall time in seconds (0 = disabled)
# Example: 12 hours = 43200, 7 days = 604800
WALL_TIME=0

# Training config
BATCH_SIZE=19
BLOCK_SIZE=1024
GRAD_ACCUM=26
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))
COOLDOWN_STEPS=4000

torchrun --nproc_per_node=$NGPUS pretrain.py \
  --tokenizer_path $TOKENIZER \
  --data_path $DATA \
  --val_data_path $VAL_DATA \
  --checkpoint_dir $CKPT_DIR \
  --lr 6e-4 \
  --batch_size $BATCH_SIZE \
  --total_batch_size $TOTAL_BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --precision bf16 \
  --flash_attention 1 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 0.4 \
  --warmup_steps 2000 \
  --schedule wsd \
  --cooldown $COOLDOWN_STEPS \
  --min_lr_ratio 0.1 \
  --checkpoint_interval 1800 \
  --max_epochs 1 \
  --torch_compile 1 \
  --gradient_checkpointing 1 \
  --wall_time $WALL_TIME
