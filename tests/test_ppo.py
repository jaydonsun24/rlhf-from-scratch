"""End-to-end tests for train/ppo.py: tiny random models, fake TL;DR prompts, CPU."""

import json

import pytest
import torch

from rlhf.data import tldr
from rlhf.train import ppo
from rlhf.utils.checkpoint import load_checkpoint
from rlhf.utils.config import load_model_config

pytestmark = pytest.mark.slow  # end-to-end runs: skipped by CI's `-m "not slow"`


def fake_tldr(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split, n in (("train", 40), ("valid", 1100), ("test", 550)):
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


def tiny_config(tmp_path, run_name, **overrides):
    cfg = ppo.PPOConfig(
        model="tiny",
        sft_checkpoint="",
        rm_checkpoint="",
        tldr_dir=str(tmp_path / "tldr"),
        run_name=run_name,
        checkpoint_root=str(tmp_path / "checkpoints"),
        runs_root=str(tmp_path / "runs"),
        rollout_batch=8,
        max_iterations=3,
        ppo_epochs=2,
        minibatch_size=4,
        micro_batch_size=2,
        lr=1e-4,
        warmup_iterations=1,
        scoring_batch=4,
        sample_every=1,
        ckpt_every=0,
        milestone_every=2,
        eval_examples=4,
        kl_prompts=4,
        eval_batch_size=4,
        use_wandb=False,
        write_results=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class FakeFlag:
    set = False


def run(monkeypatch, cfg, preempt_after=None):
    monkeypatch.setattr(ppo, "get_device", lambda: torch.device("cpu"))
    flag = FakeFlag()
    monkeypatch.setattr(ppo, "PreemptionFlag", lambda: flag)
    real_update, calls = ppo.ppo_update, []

    def counting_update(*args, **kwargs):
        out = real_update(*args, **kwargs)
        calls.append(1)
        if preempt_after is not None and len(calls) == preempt_after:
            flag.set = True
        return out

    monkeypatch.setattr(ppo, "ppo_update", counting_update)
    ppo.main(cfg)
    return len(calls)


def metrics(tmp_path, run_name):
    lines = (tmp_path / "runs" / run_name / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_sanity_checks_at_the_start(tmp_path, monkeypatch):
    """At iteration 1 the policy IS the reference (KL 0), and before the first
    update the policy hasn't changed (ratio 1, approx_kl 0)."""
    fake_tldr(tmp_path)
    run(monkeypatch, tiny_config(tmp_path, "sane"))
    first = next(r for r in metrics(tmp_path, "sane") if r["step"] == 1 and "kl" in r)
    assert abs(first["kl"]) < 1e-4
    assert first["first_microbatch_approx_kl"] < 1e-8
    assert abs(first["first_microbatch_ratio_max"] - 1.0) < 1e-4
    for key in (
        "score",
        "rlhf_reward",
        "entropy",
        "clipfrac",
        "vf_loss",
        "explained_variance",
        "no_eos_fraction",
    ):
        assert key in first
    s = json.loads((tmp_path / "runs" / "sane" / "summary.json").read_text())
    for key in ("ppo_eot_rate", "ppo_mean_length", "ppo_rm_score", "sft_rm_score", "kl_from_sft"):
        assert key in s
    names = sorted(p.name for p in (tmp_path / "checkpoints" / "sane").glob("*.pt"))
    assert "iter_0002.pt" in names and "final.pt" in names  # milestones for the over-opt curve


def test_eos_trick_penalizes_summaries_that_never_stop(tmp_path, monkeypatch):
    fake_tldr(tmp_path)
    cfg = tiny_config(tmp_path, "eos")
    torch.manual_seed(0)
    models = ppo.load_models(cfg, load_model_config("tiny"), torch.device("cpu"))
    prompts = [tldr.TLDRDataset("train", cfg.tldr_dir)[i][0] for i in range(8)]
    roll = ppo.rollout(*models, prompts, cfg, torch.device("cpu"), False)
    never_stopped = ~roll["has_eot"]
    assert never_stopped.any()  # a random tiny model rarely emits EOT within 53 tokens
    assert (roll["score"][never_stopped] == -1.0).all()
    assert torch.equal(roll["score"][~never_stopped], roll["raw_score"][~never_stopped])
    # shapes: everything per-token is response-only (B, R)
    B, R = roll["tokens"].shape
    for key in ("logp_old", "logp_ref", "values", "entropy", "response_mask"):
        assert roll[key].shape == (B, R), key


def test_beta_zero_means_no_kl_penalty(tmp_path, monkeypatch):
    fake_tldr(tmp_path)
    run(monkeypatch, tiny_config(tmp_path, "b0", beta=0.0, max_iterations=2))
    assert all(
        r["non_score_reward"] == 0 for r in metrics(tmp_path, "b0") if "non_score_reward" in r
    )


def test_preempt_and_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    fake_tldr(tmp_path)
    assert run(monkeypatch, tiny_config(tmp_path, "straight", milestone_every=0)) == 3
    assert (
        run(monkeypatch, tiny_config(tmp_path, "resumed", milestone_every=0), preempt_after=2) == 2
    )
    assert run(monkeypatch, tiny_config(tmp_path, "resumed", milestone_every=0)) == 1
    a = load_checkpoint(tmp_path / "checkpoints" / "straight" / "final.pt")
    b = load_checkpoint(tmp_path / "checkpoints" / "resumed" / "final.pt")
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
