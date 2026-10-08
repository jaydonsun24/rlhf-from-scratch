"""Download FineWeb-Edu and tokenize it into uint16 shards (docs/02-pretraining.md).

Two steps, because on Adroit only the login node has internet and only compute
nodes should do heavy work:

  1. download (login node):    python -m rlhf.data.fineweb step=download
  2. tokenize (CPU Slurm job): sbatch scripts/slurm/tokenize.slurm

Output in data/fineweb/:
  val_000.npy               the first 100M tokens (validation, never trained on)
  train_001.npy, ...        100M tokens each
  meta.json                 counts, tokenizer, which source files were used

Every document is prefixed with <|endoftext|> and then concatenated, so the model
sees where one document ends and the next begins. Tokens are stored as uint16
(2 bytes each): every GPT-2 id is < 65,536, so this halves the size vs. int32.
"""

import json
import math
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rlhf.data.tokenizer import EOT_ID, get_tokenizer
from rlhf.utils.config import load_config

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"  # 14 parquet files, ~10B GPT-2 tokens, 28.5 GB
TOKENS_PER_FILE = 0.7e9  # rough: 10B tokens / 14 files. Only used to pick how many to download.


@dataclass
class FineWebConfig:
    step: str = "all"  # "download", "tokenize", or "all"
    raw_dir: str = "data/fineweb_raw"  # downloaded parquet files
    out_dir: str = "data/fineweb"  # tokenized shards
    tokens: int = 5_000_000_000  # training tokens (validation shard is extra)
    shard_size: int = 100_000_000
    num_files: int = 0  # parquet files to download; 0 = enough for `tokens` plus margin
    num_workers: int = 0  # tokenizer processes; 0 = all CPUs Slurm gave us


def files_needed(cfg: FineWebConfig) -> int:
    if cfg.num_files > 0:
        return cfg.num_files
    total = cfg.tokens + cfg.shard_size  # + validation shard
    return min(14, math.ceil(total / TOKENS_PER_FILE) + 1)  # +1 file of margin


def parquet_names(n: int) -> list[str]:
    return [f"{SUBSET}/{i:03d}_00000.parquet" for i in range(n)]


def download(cfg: FineWebConfig) -> None:
    from huggingface_hub import snapshot_download

    names = parquet_names(files_needed(cfg))
    print(f"downloading {len(names)} files (~{len(names) * 2.15:.0f} GB) to {cfg.raw_dir}")
    snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=names, local_dir=cfg.raw_dir)


def tokenize_doc(text: str) -> np.ndarray:
    """One document -> [EOT, token, token, ...] as uint16."""
    tokens = [EOT_ID] + get_tokenizer().encode_ordinary(text)
    return np.array(tokens, dtype=np.uint16)


def iter_texts(parquet_files: list[Path]):
    from datasets import load_dataset

    # streaming=True reads the files row by row instead of loading 17 GB into memory
    ds = load_dataset(
        "parquet", data_files=[str(p) for p in parquet_files], split="train", streaming=True
    )
    for row in ds:
        yield row["text"]


def tokenize(cfg: FineWebConfig) -> dict:
    raw_files = sorted(Path(cfg.raw_dir).rglob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"no parquet files in {cfg.raw_dir}; run step=download first")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = cfg.num_workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    target = cfg.shard_size + cfg.tokens  # val shard + train tokens
    print(f"tokenizing {len(raw_files)} files with {workers} processes, target {target:,} tokens")

    buffer = np.empty(cfg.shard_size, dtype=np.uint16)  # the shard being filled
    filled = 0  # tokens in the current buffer
    shard_index = 0
    total = 0

    def write_shard(n_tokens: int) -> None:
        nonlocal shard_index
        name = f"{'val' if shard_index == 0 else 'train'}_{shard_index:03d}.npy"
        np.save(out_dir / name, buffer[:n_tokens])
        print(f"  wrote {name} ({n_tokens:,} tokens)", flush=True)
        shard_index += 1

    with Pool(workers) as pool:
        for doc in pool.imap(tokenize_doc, iter_texts(raw_files), chunksize=16):
            while len(doc) > 0 and total < target:
                take = min(len(doc), cfg.shard_size - filled, target - total)
                buffer[filled : filled + take] = doc[:take]
                filled += take
                total += take
                doc = doc[take:]  # whatever didn't fit continues in the next shard
                if filled == cfg.shard_size:
                    write_shard(filled)
                    filled = 0
            if total >= target:
                break
    if filled > 0:
        write_shard(filled)  # final partial shard
    if total < target:
        print(f"WARNING: ran out of data at {total:,} tokens; download more files (num_files=...)")

    meta = {
        "source": f"{REPO_ID}/{SUBSET}",
        "files": [p.name for p in raw_files],
        "tokenizer": "gpt2 (tiktoken)",
        "dtype": "uint16",
        "shard_size": cfg.shard_size,
        "num_shards": shard_index,
        "val_shards": [0],
        "train_shards": list(range(1, shard_index)),
        "total_tokens": total,
        "train_tokens": total - min(total, cfg.shard_size),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {shard_index} shards, {total:,} tokens -> {out_dir}")
    return meta


if __name__ == "__main__":
    cfg = load_config(FineWebConfig)
    if cfg.step in ("download", "all"):
        download(cfg)
    if cfg.step in ("tokenize", "all"):
        tokenize(cfg)
