"""End-to-end tests for train/reward.py: tiny model, fake preference + TL;DR data, CPU."""

import json

import pandas as pd
import pytest
import torch

from rlhf.data import preferences, tldr
from rlhf.train import reward
from rlhf.utils.checkpoint import load_checkpoint


def fake_data(tmp_path):
    raw = tmp_path / "raw_tldr"
    raw.mkdir()
    for split, n in (("train", 50), ("valid", 1100), ("test", 550)):
        rows = [
            {
                "id": f"{split}{i}",
                "subreddit": "aww",
                "title": f"Cat {i}",
                "post": f"My cat {i} sat on the mat.",
                "summary": f"cat {i} sat",
            }
            for i in range(n)
        ]
        (raw / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    tldr.prepare(str(raw), str(tmp_path / "tldr"))

    raw_prefs = tmp_path / "raw_prefs"
    for split, n in (("train", 96), ("validation", 40)):
        (raw_prefs / split).mkdir(parents=True)
        rows = [
            {
                "info": {
                    "id": f"p{i}",
                    "post": f"My dog {i} ran far.",
                    "title": f"Dog {i}",
                    "subreddit": "aww",
                },
                "summaries": [
                    {"text": f" dog {i} ran", "policy": "a"},
                    {"text": " something else entirely", "policy": "b"},
                ],
                "choice": i % 2,
                "extra": {"confidence": [1, 4, 6, 9][i % 4]},
            }
            for i in range(n)
        ]
        pd.DataFrame(rows).to_parquet(raw_prefs / split / "0000.parquet")
    preferences.prepare(str(raw_prefs), str(tmp_path / "prefs"))


def tiny_config(tmp_path, run_name, **overrides):
    cfg = reward.RMConfig(
        model="tiny",
        init_from="",
        data_dir=str(tmp_path / "prefs"),
        tldr_dir=str(tmp_path / "tldr"),
        run_name=run_name,
        checkpoint_root=str(tmp_path / "checkpoints"),
        runs_root=str(tmp_path / "runs"),
        batch_size=8,
        micro_batch_size=4,
        max_steps=6,
        lr=1e-3,
        log_every=1,
        eval_every=3,
        eval_batch_size=8,
        ckpt_every=0,
        norm_examples=50,
        length_samples=8,
        use_wandb=False,
        write_results=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class FakeFlag:
    set = False


def run(monkeypatch, cfg, preempt_after=None):
    monkeypatch.setattr(reward, "get_device", lambda: torch.device("cpu"))
    flag = FakeFlag()
    monkeypatch.setattr(reward, "PreemptionFlag", lambda: flag)
    real_step, calls = reward.train_step, []

    def counting_step(*args, **kwargs):
        out = real_step(*args, **kwargs)
        calls.append(1)
        if preempt_after is not None and len(calls) == preempt_after:
            flag.set = True
        return out

    monkeypatch.setattr(reward, "train_step", counting_step)
    reward.main(cfg)
    return len(calls)


def test_preempt_and_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    fake_data(tmp_path)
    assert run(monkeypatch, tiny_config(tmp_path, "straight")) == 6
    assert run(monkeypatch, tiny_config(tmp_path, "resumed"), preempt_after=4) == 4
    assert run(monkeypatch, tiny_config(tmp_path, "resumed")) == 2
    a = load_checkpoint(tmp_path / "checkpoints" / "straight" / "final.pt")
    b = load_checkpoint(tmp_path / "checkpoints" / "resumed" / "final.pt")
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name


def test_normalization_and_outputs(tmp_path, monkeypatch):
    fake_data(tmp_path)
    run(monkeypatch, tiny_config(tmp_path, "rm"))
    s = json.loads((tmp_path / "runs" / "rm" / "summary.json").read_text())
    # After shifting the head bias, the reference summaries average reward ~0.
    assert abs(s["reference_reward_mean_after_norm"]) < 1e-4
    for key in ("accuracy", "n", "length_spearman", "sft_sample_reward_mean", "reward_offset"):
        assert key in s
    assert s["n"] == 40 and 0 <= s["accuracy"] <= 1
    assert "accuracy_certainty_4" in s and "accuracy_certainty_1" in s
    final = load_checkpoint(tmp_path / "checkpoints" / "rm" / "final.pt")
    assert final["extra"]["reward_offset"] == pytest.approx(s["reward_offset"])
    dists = json.loads((tmp_path / "runs" / "rm" / "reward_distributions.json").read_text())
    assert len(dists["sft_samples"]) == 8


def test_spearman():
    assert reward.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert reward.spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)
    assert reward.spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)  # ranks, not values
