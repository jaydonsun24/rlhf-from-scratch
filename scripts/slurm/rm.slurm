#!/bin/bash
# Reward model training; any GPU works (a MIG slice is plenty). Extra key=value arguments
# override the config, e.g. an LR-sweep run:
#   sbatch scripts/slurm/rm.slurm lr=1e-5 max_steps=300 run_name=rm_lr1e-5 final_eval_pairs=5000 write_results=false save_checkpoints=false
# Resubmitting with the same arguments resumes from the latest checkpoint.
#SBATCH --job-name=rm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_MODE=offline

srun python -m rlhf.train.reward --config configs/rm.yaml "$@"
