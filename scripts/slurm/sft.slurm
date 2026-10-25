#!/bin/bash
# SFT on one MIG slice (20 GB is plenty for 124M fine-tuning). Extra key=value arguments
# override the config, e.g. an LR-sweep run:
#   sbatch scripts/slurm/sft.slurm lr=1e-5 max_steps=200 run_name=sft_lr1e-5 write_results=false
# Resubmitting with the same arguments resumes from the latest checkpoint.
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_MODE=offline

srun python -m rlhf.train.sft --config configs/sft.yaml "$@"
