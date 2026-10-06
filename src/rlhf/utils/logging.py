"""Metric logging to a local JSONL file (always) and Weights & Biases (optional).

The JSONL file is the source of truth for plots, so nothing depends on W&B.
Only the main process (rank 0) should create a MetricsLogger.
"""

import json
import platform
import time
from pathlib import Path

import torch

from rlhf.utils.checkpoint import git_hash
from rlhf.utils.config import REPO_ROOT


class MetricsLogger:
    def __init__(self, run_dir: Path, run_name: str, group: str, config: dict, use_wandb: bool):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if use_wandb:
            import wandb  # imported lazily so tests and laptop runs don't need a W&B login

            # On the cluster, set WANDB_MODE=offline in the Slurm script and `wandb sync` later.
            # resume="allow" + a fixed id makes a resumed job continue the same W&B run.
            self.wandb = wandb.init(
                project="rlhf-from-scratch",
                group=group,
                name=run_name,
                id=run_name,
                config=config,
                dir=str(run_dir),
                resume="allow",
            )

    def log(self, metrics: dict, step: int) -> None:
        record = {"step": step, "time": time.time(), **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def hardware_info() -> dict:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        n = torch.cuda.device_count()
    else:
        gpu, n = "none", 0
    return {"gpu": gpu, "num_gpus": n, "platform": platform.platform()}


def write_results(name: str, metrics: dict) -> Path:
    """Write final numbers to results/<name>.json. The README only quotes numbers from here."""
    path = REPO_ROOT / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **metrics,
        "git_hash": git_hash(),
        "hardware": hardware_info(),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
