"""Tests for rl/losses.py. Every loss in the project gets hand-checkable tests here.

sft_loss(logits, input_ids, response_mask):
    cross-entropy on RESPONSE tokens only (summary + EOT). Prompt and pad tokens add nothing.
    Remember the shift: logits at position t predict the token at position t + 1.
"""

import math

import torch
import torch.nn.functional as F

from rlhf.data.tldr import collate
from rlhf.data.tokenizer import EOT_ID
from rlhf.rl.losses import sft_loss

LN2 = math.log(2)


def test_uniform_prediction_costs_ln_vocab():
    # Logits all equal -> every token has probability 1/V -> loss ln(V) per token.
    V = 5
    logits = torch.zeros(2, 6, V)
    input_ids = torch.randint(0, V, (2, 6))
    response_mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0]])
    assert torch.allclose(sft_loss(logits, input_ids, response_mask), torch.tensor(math.log(V)))


def test_hand_computed_with_shift():
    # Sequence: [p0, p1 | r0, r1]  -> tokens 0, 1 are the prompt; 2, 1 are the response.
    #   logits[0] predicts token at 1 (prompt)   -> ignored
    #   logits[1] predicts token at 2 (= id 2)   -> counted. softmax([0,0,ln2]) gives id 2 prob 1/2 -> loss ln 2
    #   logits[2] predicts token at 3 (= id 1)   -> counted. same softmax gives id 1 prob 1/4 -> loss ln 4
    #   logits[3] predicts past the end          -> ignored
    # mean over the 2 counted tokens = (ln2 + ln4) / 2 = 1.5 * ln2
    input_ids = torch.tensor([[0, 1, 2, 1]])
    response_mask = torch.tensor([[0, 0, 1, 1]])
    row = torch.tensor([0.0, 0.0, LN2])
    logits = row.repeat(1, 4, 1)  # (1, 4, 3)
    assert torch.allclose(sft_loss(logits, input_ids, response_mask), torch.tensor(1.5 * LN2))


def test_prompt_targets_and_logits_do_not_matter():
    input_ids = torch.tensor([[3, 1, 4, 1, 2]])
    response_mask = torch.tensor([[0, 0, 0, 1, 1]])  # prompt = first 3 tokens
    logits = torch.randn(1, 5, 6)
    base = sft_loss(logits, input_ids, response_mask)

    changed_ids = input_ids.clone()
    changed_ids[0, 1] = 5  # a prompt token's target changes (the target of logits[0])
    assert torch.allclose(sft_loss(logits, changed_ids, response_mask), base)

    changed_logits = logits.clone()
    changed_logits[0, :2] = torch.randn(2, 6)  # predictions made AT prompt positions 0-1
    assert torch.allclose(sft_loss(changed_logits, input_ids, response_mask), base)


def test_matches_a_slow_loop_on_a_real_padded_batch():
    batch = collate([([11, 12, 13], [21, 22, EOT_ID]), ([14], [23, EOT_ID])])
    ids, mask = batch["input_ids"], batch["response_mask"]
    logits = torch.randn(*ids.shape, 50304)
    total, count = 0.0, 0
    for b in range(ids.shape[0]):
        for t in range(1, ids.shape[1]):
            if mask[b, t]:  # token t is a response token: predicted by logits at t - 1
                total += F.cross_entropy(logits[b, t - 1], ids[b, t]).item()
                count += 1
    assert count == 5  # 3 + 2 response tokens; pads and prompts never counted
    assert abs(sft_loss(logits, ids, mask).item() - total / count) < 1e-4


def test_gradient_only_reaches_positions_that_predict_response_tokens():
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    response_mask = torch.tensor([[0, 0, 1, 1, 0, 0]])  # response = tokens at 2, 3; then pads
    logits = torch.randn(1, 6, 5, requires_grad=True)
    sft_loss(logits, input_ids, response_mask).backward()
    has_grad = logits.grad.abs().sum(-1)[0] > 0
    assert has_grad.tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]  # positions 1, 2 predict 2, 3
