#!/bin/bash
#SBATCH --job-name=attres-9
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --time=00:20:00
#SBATCH --output=report/9-train.out
#SBATCH --error=report/9-train.err

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /home/youzhi/ArgonneAI/att_res

TOKENIZER=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base
DATA=/project/rcc/youzhi/fineweb-binary-qwen3/train.bin
CKPT_DIR=/project/rcc/youzhi/llm.c/test
NGPUS=2

BATCH_SIZE=8
BLOCK_SIZE=512
GRAD_ACCUM=1
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))

torchrun --nproc_per_node=$NGPUS train_llm_c.py \
  --tokenizer_path $TOKENIZER \
  --data_path $DATA \
  --checkpoint_dir $CKPT_DIR \
  --lr 1e-4 \
  --batch_size $BATCH_SIZE \
  --total_batch_size $TOTAL_BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --precision bf16 \
  --flash_attention 1 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --grad_clip 1.0 \
  --warmup_steps 1000 \
  --min_lr_ratio 0.1 \
  --checkpoint_interval 999999 \
  --max_epochs 1 \
  --torch_compile 1 \
  --torch_compile_mode default \
  --gradient_checkpointing 1 \
  --use_attn_res 1 \
  --attn_res_block_size 4 \
  --reset_schedule 1
