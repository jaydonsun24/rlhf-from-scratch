import torch

from rlhf.model.layers import SwiGLU


def test_swiglu_shape_and_param_count():
    mlp = SwiGLU(768, 2048)
    x = torch.randn(2, 5, 768)  # (B, T, C)
    assert mlp(x).shape == (2, 5, 768)  # comes back to the same width

    n_params = sum(p.numel() for p in mlp.parameters())
    assert n_params == 3 * 768 * 2048  # three matrices, no biases
    assert n_params == 2 * 768 * (4 * 768)  # same size as GPT-2's 2-matrix MLP with 4x hidden


def test_swiglu_hand_computed():
    mlp = SwiGLU(2, 2)
    with torch.no_grad():
        mlp.gate.weight.copy_(torch.eye(2))  # gate(x) = x
        mlp.up.weight.copy_(2 * torch.eye(2))  # up(x) = 2x
        mlp.down.weight.copy_(torch.eye(2))  # down(h) = h
    x = torch.tensor([[1.0, -1.0]])
    # gate(x) = [1, -1]
    # silu(z) = z * sigmoid(z):  silu(1) = 0.7311,  silu(-1) = -0.2689
    # up(x)   = [2, -2]
    # product = [0.7311 * 2, -0.2689 * -2] = [1.4621, 0.5379]
    expected = torch.tensor([[1.4621, 0.5379]])
    assert torch.allclose(mlp(x), expected, atol=1e-4)


def test_swiglu_closed_gate_blocks_everything():
    mlp = SwiGLU(8, 16)
    with torch.no_grad():
        mlp.gate.weight.zero_()  # gate outputs 0, and silu(0) = 0
    out = mlp(torch.randn(3, 8))
    assert torch.equal(out, torch.zeros(3, 8))  # nothing gets through, whatever `up` says
