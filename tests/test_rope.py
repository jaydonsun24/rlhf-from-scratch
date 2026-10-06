import math

import torch

from rlhf.model.layers import RotaryEmbedding, rotate_half


def test_rotate_half():
    # Split [1, 2 | 3, 4] into halves x1 = [1, 2], x2 = [3, 4]; result is [-x2, x1].
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_hand_computed_rotation():
    # head_dim = 2: one pair, rotating at 1 radian per position.
    # The arrow [1, 0] at position 1 is rotated by 1 radian -> [cos 1, sin 1].
    rope = RotaryEmbedding(head_dim=2, max_len=8)
    x = torch.tensor([1.0, 0.0]).view(1, 1, 1, 2)  # (B=1, H=1, T=1, D=2)
    out = rope(x, position_ids=torch.tensor([[1]]))
    expected = torch.tensor([math.cos(1), math.sin(1)])  # [0.5403, 0.8415]
    assert torch.allclose(out.view(2), expected, atol=1e-6)


def test_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=64, max_len=16)
    x = torch.randn(2, 4, 5, 64)  # (B, H, T, D)
    out = rope(x, position_ids=torch.zeros(2, 5, dtype=torch.long))
    assert torch.allclose(out, x, atol=1e-6)  # rotating by angle 0 changes nothing


def test_rotation_preserves_length():
    rope = RotaryEmbedding(head_dim=64, max_len=128)
    x = torch.randn(1, 2, 100, 64)
    pos = torch.arange(100).unsqueeze(0)  # (1, T)
    out = rope(x, pos)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_score_depends_only_on_relative_distance():
    """The whole point of RoPE: q.k depends on how far apart two tokens are, not where they are."""
    rope = RotaryEmbedding(head_dim=64, max_len=64)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)

    def score(q_pos, k_pos):
        q_rot = rope(q, torch.tensor([[q_pos]]))
        k_rot = rope(k, torch.tensor([[k_pos]]))
        return (q_rot * k_rot).sum()

    assert torch.allclose(score(3, 1), score(10, 8), atol=1e-4)  # both 2 apart
    assert torch.allclose(score(3, 1), score(40, 38), atol=1e-4)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-4)  # different distance


def test_explicit_position_ids_for_left_padding():
    # Row 0: [pad, pad, a, b], so a and b sit at indices 2, 3 but are positions 0, 1.
    # Rotating them with position_ids [0, 0, 0, 1] must match an unpadded row [a, b] at [0, 1].
    rope = RotaryEmbedding(head_dim=8, max_len=16)
    ab = torch.randn(1, 2, 2, 8)  # (B=1, H=2, T=2, D=8)
    padded = torch.cat([torch.zeros(1, 2, 2, 8), ab], dim=2)  # (1, 2, 4, 8)

    out_unpadded = rope(ab, torch.tensor([[0, 1]]))
    out_padded = rope(padded, torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(out_padded[:, :, 2:], out_unpadded, atol=1e-6)


def test_tables_are_buffers_not_parameters():
    rope = RotaryEmbedding(head_dim=64, max_len=1024)
    assert list(rope.parameters()) == []  # nothing to learn
    assert rope.cos.shape == (1024, 64) and rope.sin.shape == (1024, 64)


def test_keeps_input_dtype():
    rope = RotaryEmbedding(head_dim=16, max_len=8)
    x = torch.randn(1, 2, 3, 16, dtype=torch.bfloat16)
    out = rope(x, torch.arange(3).unsqueeze(0))
    assert out.dtype == torch.bfloat16
