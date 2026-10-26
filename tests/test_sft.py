"""End-to-end tests for train/sft.py: tiny model, fake TL;DR data, CPU."""

import json

import torch

from rlhf.data.tldr import prepare
from rlhf.model.gpt import GPT
from rlhf.train import sft
from rlhf.utils.checkpoint import load_checkpoint, save_checkpoint
from rlhf.utils.config import load_model_config


def fake_tldr(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split, n in (("train", 64), ("valid", 1200), ("test", 600)):
        rows = [
            {
                "id": f"{split}{i}",
                "subreddit": "aww",
                "title": f"Cat {i}",
                "post": f"My cat number {i} sat on the mat all day.",
                "summary": f"cat {i} sat",
            }
            for i in range(n)
        ]
        (raw / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    prepare(str(raw), str(tmp_path / "tldr"))


def tiny_config(tmp_path, run_name, **overrides):
    cfg = sft.SFTConfig(
        model="tiny",
        init_from="",
        data_dir=str(tmp_path / "tldr"),
        run_name=run_name,
        checkpoint_root=str(tmp_path / "checkpoints"),
        runs_root=str(tmp_path / "runs"),
        batch_size=8,
        micro_batch_size=4,
        max_steps=6,
        lr=1e-3,
        log_every=1,
        eval_every=3,
        eval_examples=8,
        eval_batch_size=4,
        ckpt_every=0,
        use_wandb=False,
        write_results=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class FakeFlag:
    set = False


def run(monkeypatch, cfg, preempt_after=None):
    monkeypatch.setattr(sft, "get_device", lambda: torch.device("cpu"))
    flag = FakeFlag()
    monkeypatch.setattr(sft, "PreemptionFlag", lambda: flag)
    real_step, calls = sft.train_step, []

    def counting_step(*args, **kwargs):
        out = real_step(*args, **kwargs)
        calls.append(1)
        if preempt_after is not None and len(calls) == preempt_after:
            flag.set = True
        return out

    monkeypatch.setattr(sft, "train_step", counting_step)
    sft.main(cfg)
    return len(calls)


def test_preempt_and_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    fake_tldr(tmp_path)
    assert run(monkeypatch, tiny_config(tmp_path, "straight")) == 6
    assert run(monkeypatch, tiny_config(tmp_path, "resumed"), preempt_after=4) == 4
    assert run(monkeypatch, tiny_config(tmp_path, "resumed")) == 2
    a = load_checkpoint(tmp_path / "checkpoints" / "straight" / "final.pt")
    b = load_checkpoint(tmp_path / "checkpoints" / "resumed" / "final.pt")
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name


def test_eval_outputs_and_epoch_wraparound(tmp_path, monkeypatch):
    fake_tldr(tmp_path)
    cfg = tiny_config(tmp_path, "eval", max_steps=10)  # 10 x 8 = 80 > 64 examples: wraps once
    run(monkeypatch, cfg)
    summary = json.loads((tmp_path / "runs" / "eval" / "summary.json").read_text())
    for key in ("val_loss", "rouge1", "rouge2", "rougeL", "eot_rate", "mean_length"):
        assert key in summary
    assert summary["n_eval"] == 8 and summary["examples_seen"] == 80
    assert 0 <= summary["eot_rate"] <= 1 and 1 <= summary["mean_length"] <= 53
    assert "**Human:** cat" in (tmp_path / "runs" / "eval" / "samples.md").read_text()
    final = load_checkpoint(tmp_path / "checkpoints" / "eval" / "final.pt")
    assert final["data_state"]["epoch"] == 1  # reshuffled into a second pass


def test_starts_from_the_pretrained_checkpoint(tmp_path):
    pretrained = GPT(load_model_config("tiny"))
    save_checkpoint(
        tmp_path,
        step=1,
        model=pretrained,
        optimizer=torch.optim.SGD(pretrained.parameters(), lr=0.1),
        config={},
        phase="pretrain",
        name="final.pt",
    )
    cfg = sft.SFTConfig(model="tiny", init_from=str(tmp_path / "final.pt"))
    loaded = sft.load_initial_model(cfg, torch.device("cpu"))
    for (name, p), q in zip(
        loaded.state_dict().items(), pretrained.state_dict().values(), strict=True
    ):
        assert torch.equal(p, q), name


def test_sweep_runs_save_no_checkpoints_and_finished_runs_drop_rolling_ones(tmp_path, monkeypatch):
    fake_tldr(tmp_path)
    run(monkeypatch, tiny_config(tmp_path, "sweep", save_checkpoints=False))
    assert not (tmp_path / "checkpoints" / "sweep").exists()
    assert (tmp_path / "runs" / "sweep" / "summary.json").exists()  # the numbers survive
    run(monkeypatch, tiny_config(tmp_path, "full", ckpt_every=2))
    names = sorted(p.name for p in (tmp_path / "checkpoints" / "full").glob("*.pt"))
    assert names == ["final.pt"]  # rolling ckpt_*.pt deleted once the run finished
