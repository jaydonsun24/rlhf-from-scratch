#!/bin/bash
# Tokenize the downloaded FineWeb-Edu parquet files into uint16 shards.
# CPU only (no GPU, no internet needed). Run the download on the login node first:
#   python -m rlhf.data.fineweb step=download
#   sbatch scripts/slurm/tokenize.slurm
#SBATCH --job-name=tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail

module purge
module load anaconda3/2026.7
conda activate /scratch/network/$USER/envs/rlhf
# Compute nodes have no internet: tiktoken must find GPT-2's vocab files in this cache
# (filled once on the login node by setup_scratch.sh), and Hugging Face must not try to connect.
export TIKTOKEN_CACHE_DIR=/scratch/network/$USER/tiktoken_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

python -m rlhf.data.fineweb step=tokenize num_workers=$SLURM_CPUS_PER_TASK
