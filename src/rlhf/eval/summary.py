"""The final results table: every finished policy on the same 500 held-out test posts.

    python -m rlhf.eval.summary        # writes results/summary.csv

Copied from eval/summarize.py's outputs in runs/eval/summaries/ (greedy summaries). PPO is trained
to raise the reward-model score, so that column alone says nothing about summary quality.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rlhf.eval.judge import load_summaries

POLICIES = [  # (label, file name in the summaries folder, beta)
    ("SFT", "sft", ""),
    ("DPO (β = 0.1)", "dpo", 0.1),
    ("PPO (β = 0.2)", "ppo_beta0.2", 0.2),
    ("PPO (β = 0.05)", "ppo", 0.05),
    ("PPO (β = 0.01)", "ppo_beta0.01", 0.01),
    ("PPO, no KL penalty (β = 0)", "ppo_beta0", 0.0),
]
COLUMNS = [
    "model",
    "beta",
    "n_posts",
    "rm_score",
    "kl_from_sft",
    "mean_length",
    "eot_rate",
    "rouge1",
    "rouge2",
    "rougeL",
    "distinct_summaries",
]


def summary_rows(summaries_dir) -> list[dict]:
    summaries_dir = Path(summaries_dir)
    human = None
    rows = []
    for label, name, beta in POLICIES:
        meta = json.loads((summaries_dir / f"{name}.meta.json").read_text())
        posts = load_summaries(summaries_dir, name)
        if human is None:
            human = {
                "model": "Human TL;DR",
                "n_posts": len(posts),
                "rm_score": meta["rm_score_reference"],
                "mean_length": float(np.mean([r["reference_length"] for r in posts])),
                "distinct_summaries": len({r["reference"] for r in posts}),
            }
        row = {"model": label, "beta": beta, "n_posts": meta["n"]}
        row.update({k: meta[k] for k in COLUMNS[3:10]})
        row["distinct_summaries"] = len({r["summary"] for r in posts})
        rows.append(row)
    return [human, *rows]


def main(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(summary_rows(args.summaries_dir))
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Final results table")
    p.add_argument("--summaries-dir", default="runs/eval/summaries")
    p.add_argument("--out", default="results/summary.csv")
    main(p.parse_args())
