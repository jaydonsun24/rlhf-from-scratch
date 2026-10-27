"""Phase 7: the LLM judge, bootstrap CIs, and judge/RM-human agreement. No real API calls:
a fake client stands in for Claude."""

import json
from types import SimpleNamespace

import pytest

from rlhf.eval import judge
from rlhf.eval.agreement import judge_agreement, rm_agreement
from rlhf.eval.bootstrap import bootstrap_ci
from rlhf.eval.judge import (
    JudgeCache,
    JudgeConfig,
    aggregate,
    pair_score,
    parse_preference,
    run_prompts,
)

# ---------- parsing ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Comparison: A is tighter.\nPreferred: A", "A"),
        ("comparison: ...\npreferred: b", "B"),
        ("Comparison: ...\n**Preferred:** B", "B"),
        ('Preferred: "A"', "A"),
        ("Preferred: A\n...on reflection...\nPreferred: B", "B"),  # last answer wins
        ("I like both of them.", None),  # garbage -> parse failure
        ("Preferred: Both", None),
        ("", None),
    ],
)
def test_parse_preference(text, expected):
    assert parse_preference(text) == expected


# ---------- both orders -> win / tie / loss ----------


def test_pair_score():
    assert pair_score("A", "B") == 1.0  # `a` preferred in both orders
    assert pair_score("B", "A") == 0.0  # dispreferred in both
    assert pair_score("A", "A") == 0.5  # judge just picked the first slot: a tie
    assert pair_score("B", "B") == 0.5
    assert pair_score(None, "B") is None


def test_aggregate_counts_ties_as_half():
    out = aggregate([1.0, 1.0, 0.5, 0.0, None], n_resamples=200)
    assert (out["wins"], out["ties"], out["losses"], out["unscored"]) == (2, 1, 1, 1)
    assert out["win_rate"] == pytest.approx(2.5 / 4)
    assert out["order_disagreement_rate"] == pytest.approx(1 / 4)
    lo, hi = out["ci95"]
    assert lo <= out["win_rate"] <= hi


# ---------- bootstrap ----------


def test_bootstrap_ci_contains_true_rate():
    scores = [1.0] * 300 + [0.0] * 200  # 60/40
    lo, hi = bootstrap_ci(scores)
    assert lo < 0.6 < hi
    assert 0.03 < (hi - lo) / 2 < 0.06  # ~ +-4 points at n = 500


def test_bootstrap_ci_constant_and_seeded():
    assert bootstrap_ci([0.5] * 20) == pytest.approx((0.5, 0.5))
    assert bootstrap_ci([0, 1, 1, 0, 1], seed=3) == bootstrap_ci([0, 1, 1, 0, 1], seed=3)


# ---------- calling the judge (fake client) ----------


class FakeClient:
    """Answers like Claude would. `rule(prompt) -> 'A' | 'B'`."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        prompt = kw["messages"][0]["content"]
        text = f"Comparison: fine.\nPreferred: {self.rule(prompt)}"
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=text),
            ],
            stop_reason="end_turn",
            model=kw["model"],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def cfg_for(tmp_path, **kw):
    base = {
        "summaries_dir": str(tmp_path / "summaries"),
        "out_dir": str(tmp_path / "judge"),
        "cache_path": str(tmp_path / "cache.jsonl"),
        "yes": True,
        "write_results": False,
        "workers": 2,
        "bootstrap_resamples": 200,
    }
    return JudgeConfig(**{**base, **kw})


def test_cache_means_never_paying_twice(tmp_path):
    cfg = cfg_for(tmp_path)
    client = FakeClient(lambda p: "A")
    prompts = ["p1", "p2", "p2", "p3"]  # duplicates are only asked once
    recs, stats = run_prompts(prompts, cfg, JudgeCache(cfg.cache_path), client)
    assert client.calls == 3 and stats["new"] == 3
    assert all(r["text"].endswith("Preferred: A") for r in recs)
    _, stats = run_prompts(prompts, cfg, JudgeCache(cfg.cache_path), client)  # reload from disk
    assert client.calls == 3 and stats["new"] == 0 and stats["cached"] == 4


def test_max_calls_cap(tmp_path):
    cfg = cfg_for(tmp_path, max_calls=2)
    client = FakeClient(lambda p: "A")
    with pytest.raises(SystemExit):
        run_prompts(["a", "b", "c"], cfg, JudgeCache(cfg.cache_path), client)
    assert client.calls == 0  # refused up front, nothing spent


def write_summaries(tmp_path, name, summaries, references):
    d = tmp_path / "summaries"
    d.mkdir(exist_ok=True)
    rows = [
        {
            "index": i,
            "id": f"t{i}",
            "post": f"SUBREDDIT: r/aww\n\nTITLE: cat {i}\n\nPOST: my cat {i}",
            "reference": ref,
            "summary": s,
            "length": len(s.split()) + 1,
            "reference_length": len(ref.split()) + 1,
            "eot": True,
            "rm_summary": 0.0,
            "rm_reference": 0.0,
        }
        for i, (s, ref) in enumerate(zip(summaries, references, strict=True))
    ]
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def prefers_word(word):
    """A fake judge that picks whichever summary contains `word`."""

    def rule(prompt):
        summary_a = prompt.split("Summary A:")[1].split("Summary B:")[0]
        return "A" if word in summary_a else "B"

    return rule


def test_compare_win_rate_and_outputs(tmp_path):
    # post 3: the reference is the good one, so ppo loses it in both orders
    write_summaries(tmp_path, "ppo", ["good one", "good two", "bad three"], ["r", "r", "good r"])
    cfg = cfg_for(tmp_path, a="ppo", b="reference")
    s = judge.compare(cfg, FakeClient(prefers_word("good")))
    assert (s["wins"], s["ties"], s["losses"]) == (2, 0, 1)
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["order_disagreement_rate"] == 0
    pairs = [json.loads(line) for line in (tmp_path / "judge" / "ppo_vs_reference.jsonl").open()]
    assert [p["score"] for p in pairs] == [1.0, 1.0, 0.0]


def test_pure_position_bias_is_all_ties(tmp_path):
    write_summaries(tmp_path, "sft", ["x", "y"], ["r1", "r2"])
    s = judge.compare(cfg_for(tmp_path, a="sft"), FakeClient(lambda p: "A"))
    assert s["win_rate"] == 0.5 and s["ties"] == 2 and s["order_disagreement_rate"] == 1.0


def test_head_to_head(tmp_path):
    write_summaries(tmp_path, "ppo", ["good a", "good b"], ["r", "r"])
    write_summaries(tmp_path, "sft", ["meh a", "meh b"], ["r", "r"])
    s = judge.compare(cfg_for(tmp_path, a="ppo", b="sft"), FakeClient(prefers_word("good")))
    assert s["win_rate"] == 1.0


def test_refusals_are_unscored_and_counted(tmp_path):
    write_summaries(tmp_path, "ppo", ["good"], ["ref"])
    client = FakeClient(lambda p: "A")
    real_create = client.create

    def refuse(**kw):
        out = real_create(**kw)
        out.stop_reason, out.content = "refusal", []
        return out

    client.beta.messages.create = refuse
    s = judge.compare(cfg_for(tmp_path, a="ppo"), client)
    assert s["refusals"] == 2 and s["unscored"] == 1 and s["win_rate"] is None


# ---------- agreement with my labels ----------


def test_agreement_definitions():
    assert judge_agreement("policy", 1.0) == 1.0
    assert judge_agreement("reference", 1.0) == 0.0
    assert judge_agreement("reference", 0.5) == 0.5  # judge split across orders
    assert rm_agreement("policy", rm_summary=0.3, rm_reference=-0.1) == 1.0
    assert rm_agreement("policy", rm_summary=-0.3, rm_reference=0.1) == 0.0
