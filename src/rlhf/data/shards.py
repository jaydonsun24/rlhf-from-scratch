"""Stream (x, y) training batches from the tokenized FineWeb shards (docs/02-pretraining.md).

Each batch is one contiguous run of B*T + 1 tokens, cut into B rows of T:
    x = tokens[:-1]   the input
    y = tokens[1:]    the target: the token that comes next at every position

With several GPUs (DDP), rank r reads the chunk at offset r*B*T and every rank jumps
ahead by world_size*B*T, so the GPUs never see the same tokens.

The loader's whole position is two numbers (which shard, where in it), saved in every
checkpoint so a resumed run continues at exactly the next token.
"""

import json
from pathlib import Path

import numpy as np
import torch


class ShardLoader:
    def __init__(self, data_dir, split, batch_size, seq_len, rank=0, world_size=1, seed=0):
        meta = json.loads((Path(data_dir) / "meta.json").read_text())
        ids = meta["val_shards"] if split == "val" else meta["train_shards"]
        prefix = "val" if split == "val" else "train"
        self.files = [Path(data_dir) / f"{prefix}_{i:03d}.npy" for i in ids]
        if split == "train":
            # Shuffle shard order once, with a fixed seed: every run (and every rank) agrees.
            order = np.random.default_rng(seed).permutation(len(self.files))
            self.files = [self.files[i] for i in order]
        self.B, self.T = batch_size, seq_len
        self.rank, self.world_size = rank, world_size
        self.shard_idx = 0
        self.pos = rank * batch_size * seq_len
        self.epoch = 0
        self._load()

    def _load(self):
        # mmap: read from disk on demand instead of loading the 200 MB shard into RAM
        self.tokens = np.load(self.files[self.shard_idx], mmap_mode="r")

    def _next_shard(self):
        self.shard_idx += 1
        if self.shard_idx == len(self.files):  # used every shard: start another pass
            self.shard_idx = 0
            self.epoch += 1
        self.pos = self.rank * self.B * self.T
        self._load()

    def next_batch(self):
        """Returns x, y: (B, T) int64 tensors on the CPU."""
        need = self.B * self.T + 1
        if self.pos + need > len(self.tokens):  # not enough left in this shard
            self._next_shard()
        chunk = np.asarray(self.tokens[self.pos : self.pos + need], dtype=np.int64)
        buf = torch.from_numpy(chunk)
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.pos += self.B * self.T * self.world_size
        return x, y

    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.shard_idx, self.pos, self.epoch = state["shard_idx"], state["pos"], state["epoch"]
        self._load()
