"""End-to-end tests for train/dpo.py: tiny model, fake preference + TL;DR data, CPU."""

import json
import math

import pandas as pd
import pytest
import torch

from rlhf.data import preferences, tldr
from rlhf.model.reward_model import RewardModel
from rlhf.train import dpo
from rlhf.utils.checkpoint import load_checkpoint, save_checkpoint
from rlhf.utils.config import load_model_config

pytestmark = pytest.mark.slow  # end-to-end runs: skipped by CI's `-m "not slow"`


def fake_data(tmp_path):
    raw = tmp_path / "raw_tldr"
    raw.mkdir()
    for split, n in (("train", 20), ("valid", 1100), ("test", 550)):
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
                    {"text": " something else", "policy": "b"},
                ],
                "choice": i % 2,
                "extra": {"confidence": 9},
            }
            for i in range(n)
        ]
        pd.DataFrame(rows).to_parquet(raw_prefs / split / "0000.parquet")
    preferences.prepare(str(raw_prefs), str(tmp_path / "prefs"))
    rm = RewardModel(load_model_config("tiny"))
    save_checkpoint(
        tmp_path / "rm",
        step=1,
        model=rm,
        optimizer=torch.optim.SGD(rm.parameters(), lr=0.1),
        config={},
        phase="rm",
        name="final.pt",
    )


def tiny_config(tmp_path, run_name, **overrides):
    cfg = dpo.DPOConfig(
        model="tiny",
        init_from="",
        rm_checkpoint=str(tmp_path / "rm" / "final.pt"),
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
        eval_examples=8,
        kl_prompts=4,
        use_wandb=False,
        write_results=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class FakeFlag:
    set = False


def run(monkeypatch, cfg, preempt_after=None):
    monkeypatch.setattr(dpo, "get_device", lambda: torch.device("cpu"))
    flag = FakeFlag()
    monkeypatch.setattr(dpo, "PreemptionFlag", lambda: flag)
    real_step, calls = dpo.train_step, []

    def counting_step(*args, **kwargs):
        out = real_step(*args, **kwargs)
        calls.append(1)
        if preempt_after is not None and len(calls) == preempt_after:
            flag.set = True
        return out

    monkeypatch.setattr(dpo, "train_step", counting_step)
    dpo.main(cfg)
    return len(calls)


def test_first_step_loss_is_ln2_and_outputs(tmp_path, monkeypatch):
    """At step 1 the policy IS the reference, so every log-ratio is 0 and the loss is ln 2."""
    fake_data(tmp_path)
    run(monkeypatch, tiny_config(tmp_path, "d"))
    records = [
        json.loads(line)
        for line in (tmp_path / "runs" / "d" / "metrics.jsonl").read_text().splitlines()
    ]
    first = next(r for r in records if r["step"] == 1 and "loss" in r)
    assert abs(first["loss"] - math.log(2)) < 1e-4
    assert abs(first["reward_margin"]) < 1e-4
    s = json.loads((tmp_path / "runs" / "d" / "summary.json").read_text())
    for key in (
        "heldout_pref_accuracy",
        "dpo_eot_rate",
        "sft_eot_rate",
        "dpo_mean_length",
        "dpo_rm_score",
        "sft_rm_score",
        "kl_from_sft",
    ):
        assert key in s
    assert (tmp_path / "runs" / "d" / "samples.md").exists()


def test_reference_stays_frozen_and_kl_is_zero_before_training(tmp_path, monkeypatch):
    fake_data(tmp_path)
    run(monkeypatch, tiny_config(tmp_path, "zero", max_steps=1, lr=0.0))  # lr 0: policy never moves
    s = json.loads((tmp_path / "runs" / "zero" / "summary.json").read_text())
    assert abs(s["kl_from_sft"]) < 1e-4  # policy == reference -> no drift
    assert s["dpo_rm_score"] == s["sft_rm_score"]


def test_preempt_and_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    fake_data(tmp_path)
    assert run(monkeypatch, tiny_config(tmp_path, "straight")) == 6
    assert run(monkeypatch, tiny_config(tmp_path, "resumed"), preempt_after=4) == 4
    assert run(monkeypatch, tiny_config(tmp_path, "resumed")) == 2
    a = load_checkpoint(tmp_path / "checkpoints" / "straight" / "final.pt")
    b = load_checkpoint(tmp_path / "checkpoints" / "resumed" / "final.pt")
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
