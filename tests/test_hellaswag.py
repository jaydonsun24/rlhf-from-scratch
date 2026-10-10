"""Tests for eval/hellaswag.py, using a fake 'model' whose predictions we control exactly."""

import pandas as pd
import pytest
import torch

from rlhf.eval.hellaswag import ending_losses, evaluate_hellaswag, load_examples

V = 50304


class CountingModel(torch.nn.Module):
    """Always confidently predicts 'next token = this token + 1'. So endings that keep
    counting up from the context are 'obvious', and anything else looks surprising."""

    def forward(self, idx):
        logits = torch.full((*idx.shape, V), -10.0)
        logits.scatter_(-1, ((idx + 1) % V).unsqueeze(-1), 10.0)
        return logits


def test_obvious_ending_gets_lowest_loss():
    ctx = [10, 11, 12]
    endings = [[50, 7], [13, 14, 15], [99], [13, 20]]  # ending 1 continues the count
    losses = ending_losses(CountingModel(), ctx, endings, torch.device("cpu"))
    assert losses.shape == (4,)
    assert losses.argmin().item() == 1
    assert losses[1] < 0.01  # fully predicted
    assert losses[3] < losses[0]  # half right (13 fits, 20 doesn't) beats all wrong


def test_loss_is_averaged_per_token_not_summed():
    # Both endings are perfectly predicted; the longer one must NOT score worse for its length.
    losses = ending_losses(CountingModel(), [1, 2], [[3], [3, 4, 5, 6, 7]], torch.device("cpu"))
    assert torch.allclose(losses[0], losses[1], atol=1e-6)


def test_accuracy_on_three_hand_made_examples():
    examples = [
        {"ctx_tokens": [1, 2], "ending_tokens": [[3, 4], [9], [8, 8], [5]], "label": 0},
        {"ctx_tokens": [5], "ending_tokens": [[1], [2], [6, 7], [3]], "label": 2},
        {"ctx_tokens": [7, 8], "ending_tokens": [[1], [9], [2], [3]], "label": 3},  # model says 1
    ]
    out = evaluate_hellaswag(CountingModel(), examples, torch.device("cpu"))
    assert out["hellaswag_n"] == 3
    assert out["hellaswag_acc"] == pytest.approx(2 / 3)  # computed in float32, so not exactly 2/3


def test_load_examples(tmp_path, monkeypatch):
    (tmp_path / "hs").mkdir()
    pd.DataFrame(
        {
            "ctx": ["A man is sitting on a roof. he"],
            "endings": [["a", "b", "c", "d"]],
            "label": ["3"],
        }
    ).to_parquet(tmp_path / "hs" / "validation-0.parquet")
    monkeypatch.chdir(tmp_path)
    (ex,) = load_examples("hs/validation-*.parquet")
    assert ex["label"] == 3 and len(ex["ending_tokens"]) == 4
    assert ex["ending_tokens"][0] == [257]  # " a": the leading space is part of the token
