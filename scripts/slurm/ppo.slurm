#!/bin/bash
# PPO on a full A100 (4 models + rollouts). Extra key=value arguments
# override the config, e.g. an ablation-A run (no KL penalty):
#   sbatch scripts/slurm/ppo.slurm beta=0 run_name=ppo_beta0
# Resubmitting with the same arguments resumes from the latest checkpoint.
#SBATCH --job-name=ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
export PYTHONUNBUFFERED=1           # print() lines appear in the log immediately, in order
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_MODE=offline

srun python -m rlhf.train.ppo --config configs/ppo.yaml "$@"
