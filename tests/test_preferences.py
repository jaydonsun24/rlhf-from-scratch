"""Tests for data/preferences.py."""

import pandas as pd

from rlhf.data.preferences import certainty, collate_pairs, convert
from rlhf.data.tldr import build_prompt, build_response
from rlhf.data.tokenizer import get_tokenizer


def row(choice, subreddit="aww", conf=None, long=False):
    text = "word " * 100 if long else "a short summary"
    return {
        "info": {"id": "x", "post": "My post.", "title": "T", "subreddit": subreddit},
        "summaries": [
            {"text": " first " + text, "policy": "p"},
            {"text": " second " + text, "policy": "q"},
        ],
        "choice": choice,
        "extra": {"confidence": conf},
    }


def test_certainty_from_1_to_9_rating():
    assert [certainty({"confidence": c}) for c in (1, 4, 6, 9)] == [4, 1, 1, 4]
    assert certainty({"confidence": None}) == -1 and certainty(None) == -1


def test_convert_picks_chosen_filters_news_and_long():
    df = pd.DataFrame([row(1, conf=9), row(0), row(0, subreddit=None), row(1, long=True)])
    arrays, counts = convert(df)
    assert counts == {"raw": 4, "not_reddit": 1, "over_cap": 1, "kept": 2}
    enc = get_tokenizer()
    off = arrays["chosen_offsets"]
    first_chosen = enc.decode(arrays["chosen"][off[0] : off[1]].tolist())
    assert first_chosen.startswith(" second")  # choice=1 -> the second summary is "chosen"
    assert list(arrays["certainty"]) == [4, -1]


def test_collate_pairs_puts_chosen_then_rejected():
    p = build_prompt("aww", "T", "Post.")
    c, r = build_response("good one"), build_response("bad")
    batch = collate_pairs([(p, c, r, 2), (p, r, c, 1)])
    assert batch["n_pairs"] == 2 and batch["input_ids"].shape[0] == 4
    P = batch["prompt_len"]
    rows = batch["input_ids"][:, P:].tolist()
    assert rows[0][: len(c)] == c and rows[2][: len(r)] == r  # pair 0: chosen row 0, rejected row 2
    assert rows[1][: len(r)] == r and rows[3][: len(c)] == c  # pair 1: chosen row 1, rejected row 3
