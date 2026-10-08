#!/bin/bash
# One-time Adroit setup: keep big outputs on scratch (100 GB) instead of home (10 GB).
# Creates the folders on scratch and symlinks them into the repo, so code can keep
# using relative paths like data/fineweb. Run from the repo root on the login node:
#   bash scripts/setup_scratch.sh
set -euo pipefail

SCRATCH=/scratch/network/$USER/rlhf
for dir in data checkpoints runs logs wandb; do
    mkdir -p "$SCRATCH/$dir"
    if [ -e "$dir" ] && [ ! -L "$dir" ]; then
        echo "ERROR: $dir already exists as a real folder; move it to $SCRATCH/$dir first" >&2
        exit 1
    fi
    ln -sfn "$SCRATCH/$dir" "$dir"
    echo "$dir -> $SCRATCH/$dir"
done
