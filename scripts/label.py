"""Hand-label summary pairs, blind, to check the LLM judge against a human (me).

    python scripts/label.py            # resumes where you left off; q to stop any time

Shows a Reddit post and two summaries in random order: one written by a model (SFT, DPO or
PPO), one the human TL;DR. Which is which is hidden. Type 1 or 2 for the better summary,
s to skip, q to quit. Labels go to data/human_labels.jsonl; eval/agreement.py then measures
how often the judge (and the reward model) agree with them.
"""

import argparse
import json
import random
import textwrap
from pathlib import Path


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def queue(summaries_dir, policies, per_policy, seed):
    """A fixed, shuffled list of (policy, row) to label: same order on every run."""
    rng = random.Random(seed)
    items = []
    for name in policies:
        path = Path(summaries_dir) / f"{name}.jsonl"
        if not path.exists():
            print(f"(no {path}, skipping {name})")
            continue
        rows = [r for r in load_rows(path) if r["summary"].strip() != r["reference"].strip()]
        items += [(name, r) for r in rng.sample(rows, min(per_policy, len(rows)))]
    rng.shuffle(items)
    return items


def show(post, first, second, done, total):
    print("\n" + "=" * 80)
    print(f"[{done + 1}/{total}]\n")
    print(textwrap.fill(post, 80, replace_whitespace=False))
    print("\n--- 1 " + "-" * 74)
    print(textwrap.fill(first, 80))
    print("\n--- 2 " + "-" * 74)
    print(textwrap.fill(second, 80))
    print()


def main(args):
    out = Path(args.out)
    labeled = set()
    if out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            labeled.add((rec["policy"], rec["index"]))

    items = queue(args.summaries_dir, args.policies, args.per_policy, args.seed)
    rng = random.Random(args.seed + 1)
    sides = [rng.random() < 0.5 for _ in items]  # True = model summary shown first
    total = len(items)
    out.parent.mkdir(parents=True, exist_ok=True)
    for done, ((policy, row), model_first) in enumerate(zip(items, sides, strict=True)):
        if (policy, row["index"]) in labeled:
            continue
        first, second = (
            (row["summary"], row["reference"])
            if model_first
            else (row["reference"], row["summary"])
        )
        show(row["post"], first, second, done, total)
        while (key := input("better summary? [1/2, s=skip, q=quit] ").strip().lower()) not in {
            "1",
            "2",
            "s",
            "q",
        }:
            pass
        if key == "q":
            break
        if key == "s":
            choice = "skip"
        else:
            picked_first = key == "1"
            choice = "policy" if picked_first == model_first else "reference"
        with open(out, "a") as f:
            f.write(
                json.dumps(
                    {
                        "policy": policy,
                        "index": row["index"],
                        "model_shown_first": model_first,
                        "choice": choice,
                    }
                )
                + "\n"
            )
    n = len(out.read_text().splitlines()) if out.exists() else 0
    print(f"\n{n} labels in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Blind A/B labeling for judge validation")
    p.add_argument("--summaries-dir", default="runs/eval/summaries")
    p.add_argument("--policies", nargs="+", default=["sft", "dpo", "ppo"])
    p.add_argument("--per-policy", type=int, default=50)
    p.add_argument("--out", default="data/human_labels.jsonl")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
