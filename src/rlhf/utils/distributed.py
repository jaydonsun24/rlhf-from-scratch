"""Device selection and (optional) multi-GPU setup.

The same script runs three ways:
  python -m rlhf.train.X ...                  -> one process, cuda > mps > cpu
  torchrun --nproc_per_node=1 -m rlhf.train.X -> one process, DDP with world_size 1
  torchrun --nproc_per_node=N -m rlhf.train.X -> N processes, one GPU each

torchrun tells each process who it is through environment variables
(RANK, LOCAL_RANK, WORLD_SIZE). If they're missing, we're not under torchrun.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int  # global process id, 0..world_size-1
    local_rank: int  # process id on this machine; picks the GPU
    world_size: int  # total number of processes
    device: torch.device

    @property
    def is_main(self) -> bool:
        # Only rank 0 logs, writes checkpoints and prints.
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed() -> DistInfo:
    if "RANK" not in os.environ:
        return DistInfo(rank=0, local_rank=0, world_size=1, device=get_device())

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo")  # CPU fallback, for testing torchrun locally
        device = torch.device("cpu")
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP (`.module`) and torch.compile (`._orig_mod`) wrappers.

    Saving the unwrapped model means checkpoint keys never carry
    `module.` or `_orig_mod.` prefixes, so any checkpoint loads into a plain model.
    """
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        else:
            return model
