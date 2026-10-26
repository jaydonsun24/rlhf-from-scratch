"""Tests for rl/logprobs.py: log-probabilities of the tokens actually in a sequence.

token_logprobs(logits, input_ids, temperature=1.0) -> (B, L-1)
    entry [b, t] = log P(input_ids[b, t+1] | everything before it), under softmax(logits / T),
    with unused vocab ids masked out (same distribution as sampling: see model/generate.py).
    Same shift as sft_loss: logits at position t predict the token at t + 1.

sequence_logprobs(logits, input_ids, response_mask, temperature=1.0) -> (B,)
    sum of token_logprobs over the response tokens (summary + EOT) of each row.
"""

import math

import torch
import torch.nn.functional as F

from rlhf.data.tldr import collate
from rlhf.data.tokenizer import EOT_ID, N_REAL_TOKENS
from rlhf.rl.logprobs import sequence_logprobs, token_logprobs


def slow_token_logprobs(logits, input_ids, temperature=1.0):
    """The obvious loop, for comparison."""
    B, L, _ = logits.shape
    out = torch.zeros(B, L - 1)
    for b in range(B):
        for t in range(L - 1):
            dist = F.log_softmax(logits[b, t] / temperature, dim=-1)
            out[b, t] = dist[input_ids[b, t + 1]]
    return out


def test_shape_and_matches_slow_loop():
    logits = torch.randn(2, 6, 7)
    ids = torch.randint(0, 7, (2, 6))
    lp = token_logprobs(logits, ids)
    assert lp.shape == (2, 5)
    assert torch.allclose(lp, slow_token_logprobs(logits, ids), atol=1e-5)


def test_temperature_divides_logits():
    logits = torch.randn(2, 5, 9)
    ids = torch.randint(0, 9, (2, 5))
    assert torch.allclose(
        token_logprobs(logits, ids, temperature=0.7),
        slow_token_logprobs(logits, ids, 0.7),
        atol=1e-5,
    )


def test_unused_vocab_ids_are_masked():
    # Give the never-real ids (pad + vocab padding) huge logits. If they weren't masked, they'd
    # soak up almost all probability and every real token's log-prob would be very negative.
    V = 50304
    logits = torch.zeros(1, 3, V)
    logits[..., N_REAL_TOKENS:] = 50.0
    ids = torch.tensor([[1, 2, 3]])
    expected = -math.log(N_REAL_TOKENS)  # uniform over the 50,257 real tokens only
    assert torch.allclose(token_logprobs(logits, ids), torch.full((1, 2), expected), atol=1e-4)


def test_sequence_logprobs_sums_response_tokens_only():
    # Uniform logits over V = 4: every token has log-prob -ln 4.
    ids = torch.tensor([[0, 1, 2, 3, 1]])
    response_mask = torch.tensor([[0, 0, 1, 1, 1]])  # 3 response tokens
    logits = torch.zeros(1, 5, 4)
    assert torch.allclose(
        sequence_logprobs(logits, ids, response_mask), torch.tensor([-3 * math.log(4)])
    )


def test_sequence_logprobs_on_a_padded_batch():
    """Real batches have PAD tokens (id 50257) after each EOT. PAD is a masked-out vocab id, so
    its log-prob is -inf, and multiplying -inf by a 0 mask gives NaN, which would poison the loss.
    Non-response positions must be zeroed by selection (torch.where), not multiplication."""
    batch = collate([([11, 12, 13], [21, 22, EOT_ID]), ([14], [23, EOT_ID])])
    ids, mask = batch["input_ids"], batch["response_mask"]
    assert (ids == 50257).any()  # this batch really contains pad targets
    logits = torch.randn(*ids.shape, 50304)
    logits[..., N_REAL_TOKENS:] = float("-inf")  # the slow loop sees the same masked vocabulary
    slow = slow_token_logprobs(logits, ids)
    expected = torch.where(mask[:, 1:].bool(), slow, 0.0).sum(-1)  # response positions only
    out = sequence_logprobs(logits, ids, mask)
    assert torch.isfinite(out).all()  # no NaN / inf leaking from the pad positions
    assert torch.allclose(out, expected, atol=1e-4)


def test_gradient_only_through_response_tokens():
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 0, 0]])
    logits = torch.randn(1, 6, 5, requires_grad=True)
    sequence_logprobs(logits, ids, mask).sum().backward()
    has_grad = (logits.grad.abs().sum(-1)[0] > 0).tolist()
    assert has_grad == [False, True, True, False, False, False]
