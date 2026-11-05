"""The over-optimization table and the reward-hacking gallery, on small fake summaries."""

import json

import pytest
from test_judge import FakeClient, cfg_for, prefers_word, write_summaries

from rlhf.eval.gallery import ROWS, build_gallery, excerpt, split_post
from rlhf.eval.overopt import (
    COLUMNS,
    curve_points,
    curve_rows,
    judge_win_rates,
    one_order_score,
    write_csv,
)
from rlhf.eval.plots import plot_overopt, read_curve


def test_curve_points_cover_every_run_from_sft_to_final():
    points = curve_points()
    assert len(points) == 4 * 6  # SFT + 4 snapshots + final, per beta
    default = [p["policy"] for p in points if p["beta"] == 0.05]
    assert default[0] == "sft" and default[1] == "ppo_default_iter0100" and default[-1] == "ppo"
    assert len({p["policy"] for p in points}) == 21  # SFT is shared


def test_one_order_score():
    assert one_order_score("A", policy_first=True) == 1.0
    assert one_order_score("B", policy_first=True) == 0.0
    assert one_order_score("B", policy_first=False) == 1.0  # the policy was shown as B
    assert one_order_score("A", policy_first=False) == 0.0
    assert one_order_score(None, policy_first=True) is None


def test_judge_win_rates_follow_the_summary_not_the_position(tmp_path):
    write_summaries(tmp_path, "good", ["good one"] * 8, ["ref"] * 8)
    write_summaries(tmp_path, "bad", ["meh one"] * 8, ["good ref"] * 8)
    cfg = cfg_for(tmp_path)
    wins = judge_win_rates(["good", "bad"], cfg, n_posts=8, client=FakeClient(prefers_word("good")))
    assert wins["good"]["judge_win_rate"] == 1.0 and wins["good"]["judge_n"] == 8
    assert wins["bad"]["judge_win_rate"] == 0.0


def test_judge_win_rates_only_use_the_first_n_posts(tmp_path):
    write_summaries(tmp_path, "ppo", ["x"] * 8, ["ref"] * 8)
    client = FakeClient(lambda p: "A")
    wins = judge_win_rates(["ppo"], cfg_for(tmp_path), n_posts=3, client=client)
    assert client.calls == 3 and wins["ppo"]["judge_n"] == 3


def test_judge_win_rates_count_refusals_as_unscored(tmp_path):
    write_summaries(tmp_path, "ppo", ["x"], ["ref"])
    client = FakeClient(lambda p: "A")
    real_create = client.create

    def refuse(**kw):
        out = real_create(**kw)
        out.stop_reason, out.content = "refusal", []
        return out

    client.beta.messages.create = refuse
    wins = judge_win_rates(["ppo"], cfg_for(tmp_path), n_posts=1, client=client)
    assert wins["ppo"] == {"judge_n": 0, "judge_unscored": 1}


def write_metas(d):
    d.mkdir(exist_ok=True)
    for i, policy in enumerate(dict.fromkeys(p["policy"] for p in curve_points())):
        meta = {
            "n": 500,
            "kl_from_sft": float(i * i),
            "rm_score": 0.1 * i,
            "mean_length": 30.0 + i,
            "rouge2": 0.1,
            "rougeL": 0.2,
        }
        (d / f"{policy}.meta.json").write_text(json.dumps(meta))


def test_curve_csv_without_judge_then_plot(tmp_path):
    summaries = tmp_path / "summaries"
    write_metas(summaries)
    rows = curve_rows(summaries)
    assert rows[1]["sqrt_kl"] == pytest.approx(1.0) and rows[2]["sqrt_kl"] == pytest.approx(2.0)

    path = tmp_path / "overopt.csv"
    write_csv(rows, path)
    curve = read_curve(path)
    assert list(curve) == [0.0, 0.01, 0.05, 0.2]
    assert [int(r["iteration"]) for r in curve[0.05]] == [0, 100, 200, 300, 400, 488]
    assert curve[0.0][0]["judge_win_rate"] == ""
    assert set(curve[0.0][0]) == set(COLUMNS)

    out = plot_overopt(path, tmp_path / "fig.png")
    assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"


def test_curve_csv_with_judge_then_plot(tmp_path):
    summaries = tmp_path / "summaries"
    write_metas(summaries)
    policies = {p["policy"] for p in curve_points()}
    wins = {
        name: {
            "judge_win_rate": 0.5,
            "judge_ci_low": 0.4,
            "judge_ci_high": 0.6,
            "judge_n": 200,
            "judge_unscored": 0,
        }
        for name in policies
    }
    path = tmp_path / "overopt.csv"
    write_csv(curve_rows(summaries, wins, "some-model"), path)
    assert read_curve(path)[0.2][-1]["judge_model"] == "some-model"
    out = plot_overopt(path, tmp_path / "fig.png")
    assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"


def test_split_post_and_excerpt():
    subreddit, title, body = split_post("SUBREDDIT: r/aww\n\nTITLE: A cat\n\nPOST: one\n\ntwo")
    assert (subreddit, title, body) == ("r/aww", "A cat", "one\n\ntwo")
    assert excerpt("one\n\n two") == "one two"
    assert excerpt("aaa bbb ccc", limit=8) == "aaa bbb ..."


def test_gallery_copies_summaries_and_counts_distinct(tmp_path):
    n = max(ROWS) + 1
    references = [f"human {i}" for i in range(n)]
    write_summaries(tmp_path, "sft", [f"sft {i}" for i in range(n)], references)
    write_summaries(tmp_path, "ppo", [f"a | b {i}" for i in range(n)], references)
    write_summaries(tmp_path, "ppo_beta0", ["same text"] * n, references)
    text = build_gallery(tmp_path / "summaries")
    k = ROWS[0]
    assert f"sft {k}" in text and f"human {k}" in text and "same text" in text
    assert f"a \\| b {k}" in text  # a pipe would break the markdown table
    assert f"**1 distinct summaries across {n} posts**" in text
    assert text.count("\n## ") == len(ROWS)
