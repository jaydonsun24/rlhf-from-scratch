"""Tests for data/shards.py, using small shards of consecutive numbers so every check is exact."""

import json

import numpy as np
import torch

from rlhf.data.shards import ShardLoader


def make_shards(tmp_path, n_train=3, shard_len=100):
    # shard i holds i*1000, i*1000+1, ...: you can tell from any token which shard it came from
    np.save(tmp_path / "val_000.npy", np.arange(shard_len, dtype=np.uint16))
    for i in range(1, n_train + 1):
        np.save(tmp_path / f"train_{i:03d}.npy", np.arange(shard_len, dtype=np.uint16) + i * 1000)
    meta = {"val_shards": [0], "train_shards": list(range(1, n_train + 1))}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_y_is_x_shifted_by_one(tmp_path):
    loader = ShardLoader(make_shards(tmp_path), "val", batch_size=2, seq_len=4)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (2, 4) and x.dtype == torch.int64
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert y.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]  # every target is the next token
    x2, _ = loader.next_batch()
    assert x2[0, 0].item() == 8  # the next batch starts right where this one ended


def test_moves_to_next_shard_and_wraps_around(tmp_path):
    loader = ShardLoader(
        make_shards(tmp_path), "train", batch_size=4, seq_len=10
    )  # 41 tokens/batch
    shards_seen = []
    for _ in range(8):  # 2 batches fit per 100-token shard, so 8 batches = 3 shards + wrap
        x, _ = loader.next_batch()
        shards_seen.append(x[0, 0].item() // 1000)
    assert sorted(set(shards_seen)) == [1, 2, 3]
    assert loader.epoch == 1  # wrapped around to a second pass


def test_resume_continues_at_the_exact_next_token(tmp_path):
    data = make_shards(tmp_path)
    loader = ShardLoader(data, "train", batch_size=2, seq_len=8, seed=5)
    for _ in range(7):  # crosses into a later shard
        loader.next_batch()
    state = loader.state_dict()
    expected = [loader.next_batch()[0] for _ in range(3)]

    resumed = ShardLoader(data, "train", batch_size=2, seq_len=8, seed=5)
    resumed.load_state_dict(state)
    for exp in expected:
        assert torch.equal(resumed.next_batch()[0], exp)


def test_shard_order_shuffled_by_seed_and_reproducible(tmp_path):
    data = make_shards(tmp_path, n_train=6)

    def order(seed):
        return [p.name for p in ShardLoader(data, "train", 1, 4, seed=seed).files]

    assert order(1) == order(1)  # same seed, same order
    assert sorted(order(1)) == sorted(order(2))  # every shard appears exactly once
    assert any(order(s) != order(1) for s in range(2, 10))  # different seeds differ


def test_ddp_ranks_read_different_tokens(tmp_path):
    data = make_shards(tmp_path)
    rank0 = ShardLoader(data, "val", batch_size=1, seq_len=4, rank=0, world_size=2)
    rank1 = ShardLoader(data, "val", batch_size=1, seq_len=4, rank=1, world_size=2)
    assert rank0.next_batch()[0].tolist() == [[0, 1, 2, 3]]
    assert rank1.next_batch()[0].tolist() == [[4, 5, 6, 7]]
    assert rank0.next_batch()[0].tolist() == [[8, 9, 10, 11]]  # jumps past rank 1's chunk
