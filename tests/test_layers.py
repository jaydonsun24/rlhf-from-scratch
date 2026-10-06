import torch

from rlhf.model.layers import RMSNorm


def test_rmsnorm_hand_computed():
    # x = [3, 4]: mean of squares = (9 + 16) / 2 = 12.5, rms = sqrt(12.5) = 3.5355
    # so x / rms = [0.8485, 1.1314]
    norm = RMSNorm(2, eps=0.0)
    out = norm(torch.tensor([[[3.0, 4.0]]]))  # (B=1, T=1, C=2)
    assert torch.allclose(out, torch.tensor([[[0.8485, 1.1314]]]), atol=1e-4)


def test_rmsnorm_output_has_rms_one():
    norm = RMSNorm(768)
    x = torch.randn(2, 5, 768) * 50  # (B, T, C), deliberately large values
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(dim=-1).sqrt()  # (B, T): one value per token
    assert torch.allclose(rms, torch.ones(2, 5), atol=1e-3)


def test_rmsnorm_weight_is_learned_scale():
    norm = RMSNorm(4)
    assert torch.equal(norm.weight, torch.ones(4))  # starts as "do nothing extra"
    assert norm.weight.requires_grad
    x = torch.randn(3, 4)
    base = norm(x)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    assert torch.allclose(norm(x), 2 * base, atol=1e-6)


def test_rmsnorm_matches_pytorch_builtin():
    norm = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(4, 7, 64)
    expected = torch.nn.functional.rms_norm(x, (64,), norm.weight, eps=1e-5)
    assert torch.allclose(norm(x), expected, atol=1e-5)


def test_rmsnorm_keeps_input_dtype():
    norm = RMSNorm(16)
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16
