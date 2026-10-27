#!/bin/bash
# Final-eval step 1: greedy summaries + RM scores + KL for every finished policy.
# Extra arguments go to rlhf.eval.summarize, e.g. the over-optimization milestones:
#   sbatch scripts/slurm/summarize.slurm --policies ppo_iter0100=checkpoints/ppo_default/iter_0100.pt
# Policies whose output already exists are skipped, so resubmitting is safe.
#SBATCH --job-name=summarize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
export PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

srun python -m rlhf.eval.summarize "$@"
