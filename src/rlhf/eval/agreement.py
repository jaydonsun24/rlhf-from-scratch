"""How often do the LLM judge and the reward model agree with a human (my labels)?

    python -m rlhf.eval.agreement      # after scripts/label.py and the judge runs vs. reference

For each hand-labeled pair (model summary vs. human TL;DR):
  judge: it answered twice (both orders), so its agreement on a pair is the fraction of the
         two answers that picked what I picked (1, 0.5 or 0)
  RM:    agrees if it gave the summary I picked the higher score (1 or 0)

Skipped labels are left out. Writes results/judge_agreement.json with n and 95% bootstrap CIs.
The judge's numbers are only trustworthy if this agreement is reasonably high.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rlhf.eval.bootstrap import bootstrap_ci
from rlhf.utils.logging import write_results


def judge_agreement(choice: str, pair_score: float) -> float:
    """pair_score = fraction of the judge's two answers that picked the policy's summary."""
    return pair_score if choice == "policy" else 1.0 - pair_score


def rm_agreement(choice: str, rm_summary: float, rm_reference: float) -> float:
    rm_pick = "policy" if rm_summary > rm_reference else "reference"
    return float(rm_pick == choice)


def summarize(values) -> dict:
    if not values:
        return {"n": 0, "agreement": None, "ci95": None}
    return {
        "n": len(values),
        "agreement": float(np.mean(values)),
        "ci95": list(bootstrap_ci(values)),
    }


def main(args):
    labels = [json.loads(line) for line in Path(args.labels).read_text().splitlines()]
    labels = [lab for lab in labels if lab["choice"] != "skip"]

    judged, rows = {}, {}
    for policy in {lab["policy"] for lab in labels}:
        path = Path(args.judge_dir) / f"{policy}_vs_reference.jsonl"
        with open(path) as f:
            judged[policy] = {p["index"]: p for p in map(json.loads, f)}
        with open(Path(args.summaries_dir) / f"{policy}.jsonl") as f:
            rows[policy] = {r["index"]: r for r in map(json.loads, f)}

    judge_vals, rm_vals = [], []
    by_policy = defaultdict(lambda: {"judge": [], "rm": []})
    no_verdict = 0
    for lab in labels:
        pair = judged[lab["policy"]][lab["index"]]
        row = rows[lab["policy"]][lab["index"]]
        r = rm_agreement(lab["choice"], row["rm_summary"], row["rm_reference"])
        rm_vals.append(r)
        by_policy[lab["policy"]]["rm"].append(r)
        if pair["score"] is None:  # judge refused / unparseable on this pair
            no_verdict += 1
            continue
        j = judge_agreement(lab["choice"], pair["score"])
        judge_vals.append(j)
        by_policy[lab["policy"]]["judge"].append(j)

    result = {
        "n_labels": len(labels),
        "judge": summarize(judge_vals),
        "reward_model": summarize(rm_vals),
        "judge_no_verdict": no_verdict,
        "human_picked_policy_rate": float(np.mean([lab["choice"] == "policy" for lab in labels])),
        "by_policy": {
            p: {"judge": summarize(v["judge"]), "reward_model": summarize(v["rm"])}
            for p, v in sorted(by_policy.items())
        },
    }
    write_results("judge_agreement", result)
    print(json.dumps({k: result[k] for k in ("n_labels", "judge", "reward_model")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Judge-human and RM-human agreement")
    p.add_argument("--labels", default="data/human_labels.jsonl")
    p.add_argument("--judge-dir", default="runs/eval/judge")
    p.add_argument("--summaries-dir", default="runs/eval/summaries")
    main(p.parse_args())
