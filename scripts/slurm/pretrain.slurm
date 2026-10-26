#!/bin/bash
# Pretrain the 124M model on one full A100. Resubmitting resumes from the latest checkpoint.
#   sbatch scripts/slurm/pretrain.slurm
#SBATCH --job-name=pretrain-124m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@300          # 5 min warning before the time limit -> save and exit
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
export PYTHONUNBUFFERED=1           # print() lines appear in the log immediately, in order
# Compute nodes have no internet: tiktoken must find GPT-2's vocab files in this cache
# (filled once on the login node by setup_scratch.sh), and Hugging Face must not try to connect.
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_MODE=offline           # compute nodes have no internet; `wandb sync` later

srun python -m rlhf.train.pretrain --config configs/pretrain.yaml
