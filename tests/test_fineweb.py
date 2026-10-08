"""Tokenization pipeline on a tiny fake parquet file (no download needed)."""

import json

import numpy as np
import pandas as pd

from rlhf.data.fineweb import FineWebConfig, tokenize
from rlhf.data.tokenizer import EOT_ID, get_tokenizer

DOCS = [f"Document number {i} talks about topic {i % 7}. " * 5 for i in range(200)]


def run_tokenize(tmp_path, tokens, shard_size):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"text": DOCS}).to_parquet(raw / "000_00000.parquet")
    cfg = FineWebConfig(
        step="tokenize",
        raw_dir=str(raw),
        out_dir=str(tmp_path / "out"),
        tokens=tokens,
        shard_size=shard_size,
        num_workers=2,
    )
    return tokenize(cfg), tmp_path / "out"


def test_shards_and_meta(tmp_path):
    meta, out = run_tokenize(tmp_path, tokens=3000, shard_size=1000)
    assert meta["num_shards"] == 4  # 1 val + 3 train, each exactly 1000 tokens
    assert meta["total_tokens"] == 4000 and meta["train_tokens"] == 3000
    assert sorted(p.name for p in out.glob("*.npy")) == [
        "train_001.npy",
        "train_002.npy",
        "train_003.npy",
        "val_000.npy",
    ]
    assert json.loads((out / "meta.json").read_text())["val_shards"] == [0]
    shard = np.load(out / "train_001.npy")
    assert shard.dtype == np.uint16 and len(shard) == 1000


def test_tokens_round_trip_and_documents_start_with_eot(tmp_path):
    _, out = run_tokenize(tmp_path, tokens=3000, shard_size=1000)
    tokens = np.concatenate([np.load(out / f) for f in ["val_000.npy", "train_001.npy"]])
    assert tokens[0] == EOT_ID  # the very first token is a document separator
    # Splitting on EOT recovers the original documents, in order.
    text = get_tokenizer().decode(tokens.tolist())
    docs = [d for d in text.split("<|endoftext|>") if d]
    assert docs[0] == DOCS[0] and docs[1] == DOCS[1]


def test_running_out_of_data_writes_partial_shard(tmp_path):
    meta, _ = run_tokenize(tmp_path, tokens=10_000_000, shard_size=1000)
    assert meta["total_tokens"] < 10_000_000  # only ~200 short docs available
    assert meta["num_shards"] >= 2
