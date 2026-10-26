"""Save and resume training state. One format for every phase (docs/08-infrastructure.md).

A checkpoint holds everything needed to continue as if the job never stopped:
model weights, optimizer state (Adam's running averages), the step counter
(which also fixes the LR schedule, since LR is a pure function of step),
the data loader's position, and every RNG state.

Rolling checkpoints are named ckpt_<step>.pt and only the newest `keep_last`
are kept. Any other name (e.g. PPO's iter_0100.pt) is a milestone and is never deleted.
"""

import os
import random
import signal
import subprocess
from pathlib import Path

import numpy as np
import torch

from rlhf.utils.distributed import unwrap_model


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states must be CPU byte tensors. Checkpoints are loaded with map_location=<gpu>,
    # which moves EVERY tensor in them to the GPU, these included, so move them back.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _optimizer_state(optimizer):
    # PPO has two optimizers (policy and value), so allow a dict of them.
    if isinstance(optimizer, dict):
        return {name: opt.state_dict() for name, opt in optimizer.items()}
    return optimizer.state_dict()


def save_checkpoint(
    run_dir: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer,
    config: dict,
    phase: str,
    data_state: dict | None = None,
    extra: dict | None = None,
    name: str | None = None,
    keep_last: int = 3,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / (name or f"ckpt_{step:08d}.pt")

    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": _optimizer_state(optimizer),
        "step": step,
        "data_state": data_state or {},
        "rng": get_rng_state(),
        "config": config,
        "git_hash": git_hash(),
        "phase": phase,
        "extra": extra or {},
    }

    # Atomic save: write to a temp file, then rename. A rename is all-or-nothing,
    # so a job killed mid-save never leaves a half-written "latest" checkpoint.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

    prune_checkpoints(run_dir, keep_last)
    return path


def delete_rolling_checkpoints(run_dir: Path) -> None:
    """Once a run has finished and written its results, the rolling ckpt_*.pt files (which only
    exist to resume a killed job) are dead weight: ~1.5-3 GB each against a 100 GB quota.
    Milestones and final.pt are kept."""
    for path in Path(run_dir).glob("ckpt_*.pt"):
        path.unlink()


def prune_checkpoints(run_dir: Path, keep_last: int) -> None:
    rolling = sorted(Path(run_dir).glob("ckpt_*.pt"))
    for old in rolling[:-keep_last]:
        old.unlink()


def latest_checkpoint(run_dir: Path) -> Path | None:
    rolling = sorted(Path(run_dir).glob("ckpt_*.pt"))  # zero-padded steps sort correctly
    return rolling[-1] if rolling else None


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    # weights_only=False because we also store numpy/python RNG state.
    # Safe here: we only ever load checkpoints we wrote ourselves.
    return torch.load(path, map_location=map_location, weights_only=False)


def restore(state: dict, model: torch.nn.Module, optimizer=None) -> int:
    """Load a checkpoint's weights, optimizer and RNG state. Returns the step to resume from."""
    unwrap_model(model).load_state_dict(state["model"])
    if optimizer is not None:
        if isinstance(optimizer, dict):
            for name, opt in optimizer.items():
                opt.load_state_dict(state["optimizer"][name])
        else:
            optimizer.load_state_dict(state["optimizer"])
    set_rng_state(state["rng"])
    return state["step"]


class PreemptionFlag:
    """Catch Slurm's warning signal so the training loop can save and exit cleanly.

    The Slurm scripts use `#SBATCH --signal=USR1@300`: five minutes before the
    time limit, Slurm sends SIGUSR1. We only set a flag here (doing real work
    inside a signal handler is unsafe); the loop checks `flag.set` every step.
    """

    def __init__(self):
        self.set = False
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        self.set = True
