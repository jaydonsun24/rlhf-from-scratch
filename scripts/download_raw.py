"""Download the raw files for the later phases (login node only: it has internet).

    python scripts/download_raw.py

  data/raw/tldr/       SFT posts + human summaries (train/valid/test.jsonl)      docs/03
  data/raw/prefs/      human preference comparisons (parquet)                    docs/04
  data/raw/hellaswag/  HellaSwag validation set (parquet)                        docs/02

Each phase's own data script (tldr.py, preferences.py, eval/hellaswag.py) reads these
local files, so nothing on a compute node ever needs the internet.
Note: HellaSwag's original GitHub repo was taken down (DMCA, 2026-09-14); the Hugging Face
copy Rowan/hellaswag is used instead.
"""

from huggingface_hub import snapshot_download

DOWNLOADS = [
    # (repo, revision, files to fetch, local folder)
    ("vwxyzjn/summarize_from_feedback_tldr_3_filtered", None, ["*.jsonl"], "data/raw/tldr"),
    ("openai/summarize_from_feedback", "refs/convert/parquet", ["comparisons/*"], "data/raw/prefs"),
    ("Rowan/hellaswag", None, ["data/validation-*.parquet"], "data/raw/hellaswag"),
]

if __name__ == "__main__":
    for repo, revision, patterns, local_dir in DOWNLOADS:
        print(f"downloading {repo} -> {local_dir}", flush=True)
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=local_dir,
        )
    print("all downloads done")
