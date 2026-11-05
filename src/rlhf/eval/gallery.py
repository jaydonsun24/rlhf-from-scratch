"""Reward-hacking gallery: the same held-out posts summarized by a human, by SFT, and by PPO with
and without the KL penalty.

    python -m rlhf.eval.gallery        # writes results/hacking_examples.md

Every summary and score is copied verbatim from runs/eval/summaries/ (eval/summarize.py).
"""

import argparse
from pathlib import Path

import numpy as np

from rlhf.eval.judge import load_summaries

POLICIES = [
    ("SFT", "sft"),
    ("PPO (β = 0.05)", "ppo"),
    ("PPO, no KL penalty (β = 0)", "ppo_beta0"),
]
# Everyday, non-personal topics, chosen from the post titles before reading their summaries
ROWS = (11, 84, 231, 293, 365)
EXCERPT_CHARS = 350


def split_post(post: str) -> tuple[str, str, str]:
    """(subreddit, title, body) from 'SUBREDDIT: r/x\\n\\nTITLE: t\\n\\nPOST: body'."""
    subreddit, title, body = post.split("\n\n", 2)
    return (
        subreddit.removeprefix("SUBREDDIT: "),
        title.removeprefix("TITLE: "),
        body.removeprefix("POST: "),
    )


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def table_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def build_gallery(summaries_dir, rows=ROWS) -> str:
    data = {name: load_summaries(summaries_dir, name) for _, name in POLICIES}
    first = data[POLICIES[0][1]]
    no_penalty = data["ppo_beta0"]
    distinct = len({r["summary"] for r in no_penalty})

    lines = [
        "# Reward-hacking examples",
        "",
        (
            "Greedy summaries of held-out Reddit posts from the TL;DR test split, copied verbatim "
            "from the model outputs. The score is what the reward model gave each summary. Posts "
            "are shortened here; the models saw them in full. These posts were chosen by topic "
            "(everyday, non-personal) from their titles, before reading the models' summaries of them."
        ),
        "",
        (
            f"With no KL penalty, PPO wrote only **{distinct} distinct summaries across "
            f"{len(no_penalty)} posts**. It found text the reward model scores highly and repeats "
            "it whatever the post says."
        ),
        "",
        f"| Source | Mean reward-model score ({len(first)} posts) |",
        "|---|---|",
        f"| Human TL;DR | {np.mean([r['rm_reference'] for r in first]):.2f} |",
    ]
    for label, name in POLICIES:
        lines.append(f"| {label} | {np.mean([r['rm_summary'] for r in data[name]]):.2f} |")

    for k in rows:
        subreddit, title, body = split_post(first[k]["post"])
        lines += [
            "",
            f"## {subreddit}: {table_cell(title)}",
            "",
            f"> {excerpt(body)}",
            "",
            "| Source | Score | Summary |",
            "|---|---|---|",
            f"| Human TL;DR | {first[k]['rm_reference']:.2f} | {table_cell(first[k]['reference'])} |",
        ]
        for label, name in POLICIES:
            row = data[name][k]
            lines.append(f"| {label} | {row['rm_summary']:.2f} | {table_cell(row['summary'])} |")
    return "\n".join(lines) + "\n"


def main(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_gallery(args.summaries_dir))
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Side-by-side reward-hacking examples")
    p.add_argument("--summaries-dir", default="runs/eval/summaries")
    p.add_argument("--out", default="results/hacking_examples.md")
    main(p.parse_args())
