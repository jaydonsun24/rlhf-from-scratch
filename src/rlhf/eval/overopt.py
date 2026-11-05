"""Over-optimization curve: as a policy drifts from SFT, does judged quality follow its
reward-model score, or do the two come apart?

    python -m rlhf.eval.overopt            # reward-model side only, no API calls
    python -m rlhf.eval.overopt --judge    # adds the judge's win rate (asks before spending)
    python -m rlhf.eval.plots overopt      # assets/overoptimization.png

One row per checkpoint per beta run: SFT (every run's starting point), the snapshots at
iterations 100-400, and the final policy. KL, reward-model score and length come from the meta
files written by eval/summarize.py.

To save money the judge sees 200 posts in one random order each instead of both orders, so a
post scores 1 or 0, never a tie. A post gets the same order at every checkpoint, so position bias
can't make one checkpoint look better than another. Prompts already answered by a two-order
comparison in eval/judge.py come from the cache and cost nothing.

Writes results/overopt.csv.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from rlhf.eval.bootstrap import bootstrap_ci
from rlhf.eval.judge import (
    JudgeCache,
    JudgeConfig,
    build_prompt,
    load_summaries,
    parse_preference,
    run_prompts,
)
from rlhf.utils.config import load_config

RUNS = [  # (beta, run name, file name of the final policy's summaries)
    (0.0, "ppo_beta0", "ppo_beta0"),
    (0.01, "ppo_beta0.01", "ppo_beta0.01"),
    (0.05, "ppo_default", "ppo"),
    (0.2, "ppo_beta0.2", "ppo_beta0.2"),
]
SNAPSHOTS = (100, 200, 300, 400)
FINAL_ITERATION = 488
START = "sft"
COLUMNS = [
    "beta",
    "iteration",
    "policy",
    "kl_from_sft",
    "sqrt_kl",
    "rm_score",
    "mean_length",
    "rouge2",
    "rougeL",
    "n_posts",
    "judge_win_rate",
    "judge_ci_low",
    "judge_ci_high",
    "judge_n",
    "judge_unscored",
    "judge_model",
]


def curve_points() -> list[dict]:
    """Every (beta, iteration, policy) on the curve, in plotting order."""
    points = []
    for beta, run, final in RUNS:
        points.append({"beta": beta, "iteration": 0, "policy": START})
        for it in SNAPSHOTS:
            points.append({"beta": beta, "iteration": it, "policy": f"{run}_iter{it:04d}"})
        points.append({"beta": beta, "iteration": FINAL_ITERATION, "policy": final})
    return points


def one_order_score(pref: str | None, policy_first: bool) -> float | None:
    """1 if the judge picked the policy's summary, 0 if it picked the human's, None if no answer."""
    if pref is None:
        return None
    return float((pref == "A") == policy_first)


def judge_win_rates(
    policies: list[str], cfg: JudgeConfig, n_posts: int, seed: int = 0, client=None
) -> dict:
    """{policy: judge columns} for each policy against the human TL;DRs, one order per post."""
    template = Path(cfg.prompt_file).read_text()
    rows = {name: load_summaries(cfg.summaries_dir, name)[:n_posts] for name in policies}
    indices = [r["index"] for r in rows[policies[0]]]
    policy_first = np.random.default_rng(seed).random(len(indices)) < 0.5

    prompts = []  # len(policies) x len(indices), in policy order
    for name in policies:
        if [r["index"] for r in rows[name]] != indices:
            raise SystemExit(f"{name} covers different posts than {policies[0]}")
        for row, first in zip(rows[name], policy_first, strict=True):
            pair = (
                (row["summary"], row["reference"]) if first else (row["reference"], row["summary"])
            )
            prompts.append(build_prompt(template, row["post"], *pair))

    records, stats = run_prompts(prompts, cfg, JudgeCache(cfg.cache_path), client)
    print(f"judge calls: {stats}")

    wins = {}
    for p, name in enumerate(policies):
        chunk = records[p * len(indices) : (p + 1) * len(indices)]
        scores = []
        for rec, first in zip(chunk, policy_first, strict=True):
            answered = rec is not None and rec["stop_reason"] != "refusal"
            pref = parse_preference(rec["text"]) if answered else None
            scores.append(one_order_score(pref, bool(first)))
        valid = [s for s in scores if s is not None]
        wins[name] = {"judge_n": len(valid), "judge_unscored": len(scores) - len(valid)}
        if valid:
            low, high = bootstrap_ci(valid, cfg.bootstrap_resamples, seed=cfg.seed)
            wins[name].update(
                judge_win_rate=float(np.mean(valid)), judge_ci_low=low, judge_ci_high=high
            )
    return wins


def curve_rows(summaries_dir, wins: dict | None = None, judge_model: str = "") -> list[dict]:
    rows = []
    for point in curve_points():
        with open(Path(summaries_dir) / f"{point['policy']}.meta.json") as f:
            meta = json.load(f)
        row = {
            **point,
            "kl_from_sft": meta["kl_from_sft"],
            "sqrt_kl": math.sqrt(max(meta["kl_from_sft"], 0.0)),  # a KL estimate can dip below 0
            "rm_score": meta["rm_score"],
            "mean_length": meta["mean_length"],
            "rouge2": meta["rouge2"],
            "rougeL": meta["rougeL"],
            "n_posts": meta["n"],
        }
        if wins is not None:
            row.update(wins[point["policy"]], judge_model=judge_model)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    cfg = load_config(JudgeConfig, ["--config", args.judge_config, f"max_calls={args.max_calls}"])
    wins = None
    if args.judge:
        policies = list(dict.fromkeys(p["policy"] for p in curve_points()))  # SFT only once
        wins = judge_win_rates(policies, cfg, args.posts, args.seed)
    write_csv(curve_rows(cfg.summaries_dir, wins, cfg.model if wins else ""), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reward-model score and judge win rate vs. KL")
    p.add_argument("--judge", action="store_true", help="also get judge win rates (API calls)")
    p.add_argument("--judge-config", default="configs/judge.yaml")
    p.add_argument("--posts", type=int, default=200)
    p.add_argument("--max-calls", type=int, default=4400)  # 21 policies x 200 posts = 4,200
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/overopt.csv")
    main(p.parse_args())
