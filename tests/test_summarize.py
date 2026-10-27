"""eval/summarize.py end to end on tiny random models, fake TL;DR data, CPU."""

import json

import torch

from rlhf.data import tldr
from rlhf.eval import summarize
from rlhf.model.gpt import GPT
from rlhf.model.reward_model import RewardModel
from rlhf.utils.config import load_model_config


def fake_tldr(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split, n in (("train", 10), ("valid", 1000), ("test", 500)):
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


def test_summarize_writes_aligned_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize, "get_device", lambda: torch.device("cpu"))
    fake_tldr(tmp_path)
    cfg = load_model_config("tiny")
    torch.manual_seed(0)
    for name, model in (("sft", GPT(cfg)), ("rm", RewardModel(cfg))):
        torch.save({"model": model.state_dict()}, tmp_path / f"{name}.pt")
    args = summarize.parse_args(
        [
            "--policies",
            f"sft={tmp_path / 'sft.pt'}",
            f"missing={tmp_path / 'nope.pt'}",
            "--model",
            "tiny",
            "--sft-checkpoint",
            str(tmp_path / "sft.pt"),
            "--rm-checkpoint",
            str(tmp_path / "rm.pt"),
            "--tldr-dir",
            str(tmp_path / "tldr"),
            "--out-dir",
            str(tmp_path / "out"),
            "--n",
            "4",
            "--kl-prompts",
            "2",
            "--eval-batch-size",
            "2",
        ]
    )
    summarize.main(args)

    rows = [json.loads(line) for line in (tmp_path / "out" / "sft.jsonl").open()]
    subset = json.loads((tmp_path / "tldr" / "test_subset.json").read_text())["indices"][:4]
    assert [r["index"] for r in rows] == subset
    for r in rows:
        assert r["post"].startswith("SUBREDDIT: r/aww") and not r["post"].endswith("TL;DR:")
        assert r["reference"].startswith("cat ") and r["reference_length"] >= 2
        assert 1 <= r["length"] <= tldr.RESPONSE_CAP
    meta = json.loads((tmp_path / "out" / "sft.meta.json").read_text())
    assert meta["n"] == 4 and abs(meta["kl_from_sft"]) < 1e-4  # SFT vs itself
    assert not (tmp_path / "out" / "missing.jsonl").exists()  # missing checkpoint skipped
