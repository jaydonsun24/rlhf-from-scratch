#!/bin/bash
# Phase 1 GPU checks on a full A100: the CUDA-only tests, then the official KV-cache
# benchmark (writes results/generation.json). Takes a few minutes once it starts;
# the queue for a full A100 may take much longer.
#   sbatch scripts/slurm/benchmark.slurm
#SBATCH --job-name=benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
export PYTHONUNBUFFERED=1           # print() lines appear in the log immediately, in order
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

nvidia-smi
pytest -q tests/test_cuda.py
python -m rlhf.model.generate --model gpt124m --batch 64 --prompt-len 128 --new-tokens 256
