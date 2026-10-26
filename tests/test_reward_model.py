"""Tests for the reward model (Phase 4): model/reward_model.py and bradley_terry_loss in rl/losses.py.

RewardModel(config): a GPT backbone (attribute `backbone`) + a linear head (attribute `head`,
d_model -> 1). forward(input_ids, position_ids=None, attention_mask=None) returns a score at
EVERY position, shape (B, L). reward_at_eot(scores, attention_mask) picks each row's score at
its last real token (the EOT ending its summary), shape (B,).
"""

import math

import torch

from rlhf.data.tldr import collate
from rlhf.data.tokenizer import EOT_ID
from rlhf.model.gpt import GPT
from rlhf.model.reward_model import RewardModel, reward_at_eot
from rlhf.rl.losses import bradley_terry_loss
from rlhf.utils.config import load_model_config

LN2, LN3 = math.log(2), math.log(3)


# ---------- bradley_terry_loss(chosen_rewards, rejected_rewards) -> scalar ----------


def test_bt_equal_rewards_is_ln2():
    # sigmoid(0) = 1/2: the model is 50/50, loss = -ln(1/2) = ln 2
    r = torch.tensor([0.3, -1.0, 2.0])
    assert torch.allclose(bradley_terry_loss(r, r), torch.tensor(LN2))


def test_bt_hand_computed_and_averaged():
    # margin ln 3 -> P(chosen wins) = sigmoid(ln 3) = 3/4 -> loss -ln(3/4) = 0.2877
    # margin 0    -> loss ln 2 = 0.6931;  mean of the two = 0.4904
    chosen = torch.tensor([LN3, 1.0])
    rejected = torch.tensor([0.0, 1.0])
    expected = (-math.log(0.75) + LN2) / 2
    assert torch.allclose(bradley_terry_loss(chosen, rejected), torch.tensor(expected))


def test_bt_rewards_the_right_order():
    lo = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0]))  # chosen ahead: small loss
    hi = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([2.0]))  # chosen behind: big loss
    assert lo < LN2 < hi


def test_bt_gradient_pushes_chosen_up_and_rejected_down():
    chosen = torch.tensor([0.5], requires_grad=True)
    rejected = torch.tensor([0.5], requires_grad=True)
    bradley_terry_loss(chosen, rejected).backward()
    assert (
        chosen.grad.item() < 0 < rejected.grad.item()
    )  # gradient descent raises chosen, lowers rejected
    assert torch.allclose(
        chosen.grad, torch.tensor([-0.5])
    )  # d/dc of -log sigmoid(c - r) = -(1 - 1/2)


def test_bt_is_numerically_stable_for_huge_margins():
    loss = bradley_terry_loss(torch.tensor([-200.0]), torch.tensor([0.0]))  # badly wrong
    assert torch.isfinite(loss) and abs(loss.item() - 200.0) < 1e-3  # not inf


# ---------- RewardModel and reward_at_eot ----------


def tiny_rm():
    torch.manual_seed(0)
    return RewardModel(load_model_config("tiny"))


def test_reward_at_eot_picks_last_real_token():
    scores = torch.tensor([[1.0, 2.0, 3.0, 9.0, 9.0], [9.0, 9.0, 5.0, 6.0, 7.0]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])  # right-padded row, left-padded row
    assert reward_at_eot(scores, mask).tolist() == [3.0, 7.0]


def test_scores_every_position_from_the_backbone():
    rm = tiny_rm()
    idx = torch.randint(0, 50257, (2, 7))
    scores = rm(idx)
    assert scores.shape == (2, 7)
    expected = rm.head(rm.backbone.hidden_states(idx)).squeeze(-1)
    assert torch.allclose(scores, expected)


def test_head_init():
    rm = tiny_rm()
    d = load_model_config("tiny").d_model
    assert rm.head.weight.shape == (1, d)
    assert abs(rm.head.weight.std().item() - 1 / math.sqrt(d + 1)) < 0.02
    assert torch.equal(rm.head.bias, torch.zeros(1))


def test_reward_ignores_padding():
    """The same (prompt, summary) must get the same reward however much padding its batch adds."""
    rm = tiny_rm().eval()
    prompt, summary = [11, 12, 13, 14], [21, 22, EOT_ID]
    alone = collate([(prompt, summary)])
    padded = collate([(prompt, summary), ([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, EOT_ID])])
    with torch.no_grad():
        r_alone = reward_at_eot(
            rm(alone["input_ids"], alone["position_ids"], alone["attention_mask"]),
            alone["attention_mask"],
        )
        r_padded = reward_at_eot(
            rm(padded["input_ids"], padded["position_ids"], padded["attention_mask"]),
            padded["attention_mask"],
        )
    assert torch.allclose(r_alone[0], r_padded[0], atol=1e-4)


def test_backbone_loads_sft_weights():
    sft = GPT(load_model_config("tiny"))
    rm = tiny_rm()
    rm.backbone.load_state_dict(sft.state_dict())
    for a, b in zip(rm.backbone.parameters(), sft.parameters(), strict=True):
        assert torch.equal(a, b)
