"""Human preference comparisons for the reward model (docs/04-reward-model.md).

    python -m rlhf.data.preferences      # once: data/raw/prefs/*.parquet -> data/prefs/

Source: OpenAI's summarize_from_feedback "comparisons" (Stiennon et al. 2020). Each row is
a Reddit post, two summaries, and which one a human labeler preferred. Validation also
contains CNN/DailyMail news articles; those are filtered out (Reddit only).

Each example becomes (prompt, chosen, rejected, certainty), in the same token format as SFT:
the prompt template from data/tldr.py, and " " + summary + EOT as responses.

`certainty` is how sure the labeler was. They rated 1-9 (1 = sure summary 0 is better,
9 = sure summary 1 is; there is no 5), so |rating - 5| gives 1 (barely) to 4 (definitely).
-1 = not recorded (about half of train).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rlhf.data.tldr import build_prompt, build_response, collate

RAW_DIR = "data/raw/prefs/comparisons"
OUT_DIR = "data/prefs"
SPLITS = {"train": "train", "valid": "validation"}
# Max summary length (tokens, incl. EOT) for RM data. Higher than SFT/PPO's 53 on purpose:
# unlike the SFT set, these summaries weren't pre-filtered (9.6% of pairs exceed 53, incl.
# 34% of one RL policy's outputs), and those long summaries are what teach the RM how much
# credit length deserves. 80 keeps ~99.4% of pairs. PPO still generates at most 53.
PREF_RESPONSE_CAP = 80


def certainty(extra) -> int:
    rating = None if extra is None else extra.get("confidence")
    return -1 if rating is None or pd.isna(rating) else int(abs(int(rating) - 5))


def convert(df: pd.DataFrame) -> tuple[dict, dict]:
    prompts, chosen, rejected, cert, ids = [], [], [], [], []
    counts = {"raw": len(df), "not_reddit": 0, "over_cap": 0}
    for row in df.itertuples():
        info = row.info
        if info.get("subreddit") is None:  # news articles in validation
            counts["not_reddit"] += 1
            continue
        good = build_response(row.summaries[row.choice]["text"])
        bad = build_response(row.summaries[1 - row.choice]["text"])
        if max(len(good), len(bad)) > PREF_RESPONSE_CAP:
            counts["over_cap"] += 1
            continue
        prompts.append(build_prompt(info["subreddit"], info["title"] or "", info["post"]))
        chosen.append(good)
        rejected.append(bad)
        cert.append(certainty(row.extra))
        ids.append(info["id"])
    counts["kept"] = len(ids)

    def flat(seqs):
        return np.concatenate(seqs).astype(np.uint16), np.cumsum([0] + [len(s) for s in seqs])

    arrays = {}
    for name, seqs in (("prompts", prompts), ("chosen", chosen), ("rejected", rejected)):
        arrays[name], arrays[f"{name}_offsets"] = flat(seqs)
    arrays["certainty"] = np.array(cert, dtype=np.int8)
    arrays["ids"] = np.array(ids, dtype=str)
    return arrays, counts


def prepare(raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "openai/summarize_from_feedback (comparisons)",
        "response_cap": PREF_RESPONSE_CAP,
    }
    for name, raw_split in SPLITS.items():
        files = sorted(Path(raw_dir, raw_split).glob("*.parquet"))
        arrays, counts = convert(pd.concat(pd.read_parquet(f) for f in files))
        np.savez(out / f"{name}.npz", **arrays)
        meta[name] = counts
        print(f"{name}: {counts}", flush=True)
    # A fixed 2,000-comparison slice of validation for quick checks during training (LR sweep).
    n_valid = meta["valid"]["kept"]
    subset = sorted(
        np.random.default_rng(seed).choice(n_valid, size=min(2000, n_valid), replace=False).tolist()
    )
    (out / "val_subset.json").write_text(json.dumps({"indices": subset, "seed": seed}))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class PrefsDataset:
    def __init__(self, split: str, data_dir: str = OUT_DIR):
        d = np.load(Path(data_dir) / f"{split}.npz")
        self.d = {k: d[k] for k in d.files}
        self.ids = list(self.d["ids"])

    def __len__(self):
        return len(self.ids)

    def _get(self, name, i):
        off = self.d[f"{name}_offsets"]
        return self.d[name][off[i] : off[i + 1]].tolist()

    def __getitem__(self, i):
        """-> (prompt, chosen, rejected, certainty)"""
        return (
            self._get("prompts", i),
            self._get("chosen", i),
            self._get("rejected", i),
            int(self.d["certainty"][i]),
        )


def collate_pairs(examples) -> dict:
    """B comparisons -> one batch of 2B sequences in the shared layout:
    rows 0..B-1 are (prompt, chosen), rows B..2B-1 are (prompt, rejected).
    Running both halves through the model together is one forward pass instead of two."""
    pairs = [(p, c) for p, c, _, _ in examples] + [(p, r) for p, _, r, _ in examples]
    batch = collate(pairs)
    batch["n_pairs"] = len(examples)
    return batch


if __name__ == "__main__":
    prepare()
