"""End-to-end tests for train/pretrain.py on a tiny model and fake data (CPU, a few seconds)."""

import json

import numpy as np
import torch

from rlhf.train import pretrain
from rlhf.utils.checkpoint import load_checkpoint
from rlhf.utils.distributed import DistInfo


def fake_data(tmp_path, learnable=False):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()

    def tokens():
        if learnable:  # a repeating 0..49 pattern: the next token is always predictable
            return (np.arange(5000) % 50).astype(np.uint16)
        return rng.integers(0, 50257, 5000, dtype=np.uint16)

    np.save(data / "val_000.npy", tokens())
    for i in (1, 2):
        np.save(data / f"train_{i:03d}.npy", tokens())
    (data / "meta.json").write_text(json.dumps({"val_shards": [0], "train_shards": [1, 2]}))
    return data


def tiny_config(tmp_path, run_name):
    return pretrain.PretrainConfig(
        model="tiny",
        data_dir=str(tmp_path / "data"),
        run_name=run_name,
        checkpoint_root=str(tmp_path / "checkpoints"),
        runs_root=str(tmp_path / "runs"),
        micro_batch_size=2,
        seq_len=16,
        total_batch_tokens=64,  # 2 micro-batches per step
        max_steps=8,
        warmup_steps=2,
        log_every=1,
        val_every=4,
        val_batches=2,
        sample_every=0,
        ckpt_every=0,
        compile=False,
        use_wandb=False,
        write_results=False,
    )


class FakeFlag:
    set = False


def run(monkeypatch, cfg, preempt_after=None):
    """Run pretrain.main on the CPU, optionally faking Slurm's warning signal after N steps."""
    cpu = DistInfo(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    monkeypatch.setattr(pretrain, "setup_distributed", lambda: cpu)
    flag = FakeFlag()
    monkeypatch.setattr(pretrain, "PreemptionFlag", lambda: flag)
    real_step, calls = pretrain.train_step, []

    def counting_step(*args, **kwargs):
        out = real_step(*args, **kwargs)
        calls.append(1)
        if preempt_after is not None and len(calls) == preempt_after:
            flag.set = True  # what the SIGUSR1 handler does 5 minutes before the time limit
        return out

    monkeypatch.setattr(pretrain, "train_step", counting_step)
    pretrain.main(cfg)
    return len(calls)


def test_preempt_and_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    fake_data(tmp_path)
    straight = tiny_config(tmp_path, "straight")
    resumed = tiny_config(tmp_path, "resumed")

    assert run(monkeypatch, straight) == 8
    assert run(monkeypatch, resumed, preempt_after=5) == 5  # "killed" after step 5
    ckpts = sorted(p.name for p in (tmp_path / "checkpoints" / "resumed").glob("*.pt"))
    assert ckpts == ["ckpt_00000005.pt"]  # saved on the way out
    assert run(monkeypatch, resumed) == 3  # resubmitted: only steps 6-8 remain

    a = load_checkpoint(tmp_path / "checkpoints" / "straight" / "final.pt")
    b = load_checkpoint(tmp_path / "checkpoints" / "resumed" / "final.pt")
    assert a["step"] == b["step"] == 8
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name  # bit-for-bit identical


def test_logs_metrics_and_loss_goes_down(tmp_path, monkeypatch):
    fake_data(tmp_path, learnable=True)
    cfg = tiny_config(tmp_path, "logs")
    cfg.max_steps, cfg.max_lr = 30, 3e-3
    run(monkeypatch, cfg)
    lines = (tmp_path / "runs" / "logs" / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    train = [r for r in records if "loss" in r]
    assert train[0]["step"] == 1 and train[-1]["step"] == 30
    # Random tokens can't be learned (loss would sit near ln(50257) = 10.8); a pattern can.
    assert train[-1]["loss"] < train[0]["loss"] - 2
    assert any("val_loss" in r for r in records)
    assert (tmp_path / "runs" / "logs" / "config.yaml").exists()
