#!/bin/bash
#SBATCH --job-name=node-evaluation
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --exclude=midway3-0423,midway3-0602
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint=H200
#SBATCH --time=7-00:00:00
#SBATCH --output=report/7-train.out
#SBATCH --error=report/7-train.err

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/youzhi/ArgonneAI

NGPUS=2

BATCH_SIZE=8
BLOCK_SIZE=1024
GRAD_ACCUM=61
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))

# New data path
TOKENIZER=/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/tokenizer
DATA=/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/train.bin

# Same checkpoint dir — picks up latest checkpoint from previous run
CKPT_DIR=/project/rcc/youzhi/argonne3.0/checkpoints

WALL_TIME=0
RESET_SCHEDULE=0

# Use --reset_schedule 1 only once when switching to a new dataset.
# After the first checkpoint on that dataset, switch back to 0 so resumes keep
# the dataset-local cursor instead of restarting from the beginning again.
torchrun --nproc_per_node=$NGPUS continue_pretrain.py \
  --tokenizer_path $TOKENIZER \
  --data_path $DATA \
  --checkpoint_dir $CKPT_DIR \
  --lr 4e-4 \
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
  --cooldown 4000 \
  --min_lr_ratio 0.1 \
  --checkpoint_interval 1800 \
  --max_epochs 1 \
  --torch_compile 1 \
  --gradient_checkpointing 1 \
  --wall_time $WALL_TIME \
  --reset_schedule $RESET_SCHEDULE
