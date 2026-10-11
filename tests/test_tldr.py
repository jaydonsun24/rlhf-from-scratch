"""Tests for data/tldr.py: prompt building, truncation, and the shared batch layout."""

import json

from rlhf.data.tldr import (
    MAX_PROMPT_TOKENS,
    TLDRDataset,
    build_prompt,
    build_response,
    collate,
    prepare,
)
from rlhf.data.tokenizer import EOT_ID, PAD_ID, get_tokenizer


def test_prompt_template():
    text = get_tokenizer().decode(build_prompt("relationships", "Help", "My post."))
    assert text == "SUBREDDIT: r/relationships\n\nTITLE: Help\n\nPOST: My post.\n\nTL;DR:"


def test_response_is_space_summary_eot():
    response = build_response("I learned a lesson.")
    assert response[-1] == EOT_ID
    assert get_tokenizer().decode(response[:-1]) == " I learned a lesson."


def test_long_post_truncated_at_paragraph_boundary():
    paragraphs = [
        f"Paragraph {i} has some words in it that go on for a while." * 3 for i in range(80)
    ]
    post = "\n".join(paragraphs)  # ~2,000+ tokens
    tokens = build_prompt("test", "Long", post)
    assert len(tokens) <= MAX_PROMPT_TOKENS
    text = get_tokenizer().decode(tokens)
    assert text.endswith("\n\nTL;DR:")  # the suffix is never cut
    kept_post = text.split("POST: ")[1].removesuffix("\n\nTL;DR:")
    assert kept_post.split("\n")[-1] in paragraphs  # last kept paragraph is a whole paragraph
    assert kept_post.startswith(paragraphs[0])  # cuts from the end, not the start


def test_collate_layout():
    # Row 0: prompt 3, response 2 (incl. EOT). Row 1: prompt 1, response 4.
    batch = collate([([11, 12, 13], [21, EOT_ID]), ([14], [22, 23, 24, EOT_ID])])
    P = batch["prompt_len"]
    assert P == 3 and batch["input_ids"].shape == (2, 7)  # P + longest response (4)
    assert batch["input_ids"].tolist() == [
        [11, 12, 13, 21, EOT_ID, PAD_ID, PAD_ID],  # right-padded after its response
        [PAD_ID, PAD_ID, 14, 22, 23, 24, EOT_ID],  # left-padded before its prompt
    ]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1]]
    assert batch["response_mask"].tolist() == [[0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 4, 4], [0, 0, 0, 1, 2, 3, 4]]
    assert (batch["input_ids"][:, P:] != PAD_ID).any(dim=1).all()  # responses start at column P


def test_collate_prompts_only_for_generation():
    batch = collate([([1, 2], []), ([3], [])])
    assert batch["input_ids"].tolist() == [[1, 2], [PAD_ID, 3]]
    assert batch["response_mask"].sum() == 0


def test_prepare_and_dataset_round_trip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "valid", "test"):
        n = 1200 if split != "train" else 10
        rows = [
            {
                "id": f"{split}{i}",
                "subreddit": "aww",
                "title": f"T{i}",
                "post": f"Post {i}.",
                "summary": f"S{i}",
            }
            for i in range(n)
        ]
        (raw / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    meta = prepare(str(raw), str(tmp_path / "out"))
    assert meta["splits"]["train"] == {"raw": 10, "kept": 10, "dropped_over_cap": 0}
    ds = TLDRDataset("train", str(tmp_path / "out"))
    assert len(ds) == 10 and ds.ids[3] == "train3"
    prompt, response = ds[3]
    assert prompt == build_prompt("aww", "T3", "Post 3.") and response == build_response("S3")
    subset = json.loads((tmp_path / "out" / "val_subset.json").read_text())
    assert len(subset["indices"]) == 1000 and len(set(subset["indices"])) == 1000
