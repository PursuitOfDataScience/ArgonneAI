#!/bin/bash
#SBATCH --job-name=preprocess-cc
#SBATCH --account=rcc-staff
#SBATCH --partition=amd
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=report/preprocess.out
#SBATCH --error=report/preprocess.err

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1

cd /home/youzhi/ArgonneAI

python3 preprocess_data.py \
  --tokenizer_path /project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base \
  --data_dir /project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21 \
  --output_dir /project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary \
  --text_column text \
  --workers 16