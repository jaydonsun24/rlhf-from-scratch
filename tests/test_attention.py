"""Tests for model/attention.py.

The module is imported as a whole (not `from ... import X`) so you can write it one
part at a time: tests for parts you haven't written yet fail, the others still run.
"""

import math

import torch

from rlhf.model import attention
from rlhf.model.config import ModelConfig
from rlhf.utils.config import load_model_config


def tiny_config() -> ModelConfig:
    return load_model_config("tiny")  # d_model 128, 4 heads -> head_dim 32


# ---------- Part A: build_attention_mask ----------


def test_mask_causal_only():
    mask = attention.build_attention_mask(None, T=3)
    expected = torch.tensor(
        [
            [True, False, False],  # token 0 sees only itself
            [True, True, False],  # token 1 sees 0, 1
            [True, True, True],  # token 2 sees 0, 1, 2
        ]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask.view(3, 3), expected)


def test_mask_with_left_padding():
    # One row: [pad, a, b]. Real tokens must never look at the pad.
    # The pad's own row would be all False, so it is allowed to see itself (the diagonal)
    # to keep softmax from dividing 0 by 0. Its output is garbage that nothing uses.
    mask = attention.build_attention_mask(torch.tensor([[0, 1, 1]]), T=3)
    expected = torch.tensor(
        [
            [True, False, False],  # pad: itself only (diagonal rule)
            [False, True, False],  # a: itself, not the pad
            [False, True, True],  # b: a and itself, not the pad
        ]
    )
    assert mask.shape == (1, 1, 3, 3)  # (B, 1, T, T): the 1 broadcasts over heads
    assert torch.equal(mask[0, 0], expected)


def test_mask_is_built_on_the_requested_device():
    # On the cluster everything lives on the GPU; a mask built on the CPU would crash
    # when combined with GPU tensors. "meta" is a fake device (shapes only), so this
    # test catches the bug on any machine, even without a GPU.
    mask = attention.build_attention_mask(None, T=4, device="meta")
    assert mask.device.type == "meta"
    padded = torch.ones(2, 4, dtype=torch.long, device="meta")
    mask = attention.build_attention_mask(padded, T=4, device="meta")
    assert mask.device.type == "meta"


# ---------- Part B: manual_attention ----------


def test_manual_attention_hand_computed():
    # head_dim D = 1, so the scale 1/sqrt(D) is 1.
    # token 0 can only see itself       -> output = v0 = 4
    # token 1 scores: q1*k0 = 0, q1*k1 = ln 3 -> softmax = [1/4, 3/4]
    #                 output = 1/4 * 4 + 3/4 * 8 = 7
    q = torch.tensor([1.0, 1.0]).view(1, 1, 2, 1)  # (B, H, T, D)
    k = torch.tensor([0.0, math.log(3)]).view(1, 1, 2, 1)
    v = torch.tensor([4.0, 8.0]).view(1, 1, 2, 1)
    mask = attention.build_attention_mask(None, T=2)
    out = attention.manual_attention(q, k, v, mask)
    assert torch.allclose(out.view(2), torch.tensor([4.0, 7.0]), atol=1e-5)


def test_manual_attention_scales_by_sqrt_head_dim():
    # Two keys; D = 4. Raw scores are [0, 4]; scaled by 1/sqrt(4) they become [0, 2].
    q = torch.tensor([1.0, 1.0, 1.0, 1.0]).view(1, 1, 1, 4)
    k = torch.stack([torch.zeros(4), torch.ones(4)]).view(1, 1, 2, 4)
    v = torch.tensor([[0.0] * 4, [1.0] * 4]).view(1, 1, 2, 4)
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)  # the single query may see both keys
    out = attention.manual_attention(q, k, v, mask)
    weight_on_key1 = math.exp(2) / (1 + math.exp(2))  # softmax([0, 2])[1] = 0.8808
    assert torch.allclose(out, torch.full((1, 1, 1, 4), weight_on_key1), atol=1e-5)


# ---------- Part C: CausalSelfAttention ----------


def make_pair(cfg):
    """Two attention modules with identical weights: one manual, one SDPA."""
    torch.manual_seed(0)
    manual = attention.CausalSelfAttention(cfg, impl="manual")
    sdpa = attention.CausalSelfAttention(cfg, impl="sdpa")
    sdpa.load_state_dict(manual.state_dict())
    return manual, sdpa


def test_shapes_and_param_count():
    cfg = tiny_config()
    attn = attention.CausalSelfAttention(cfg)
    x = torch.randn(2, 5, cfg.d_model)  # (B, T, C)
    pos = torch.arange(5).expand(2, 5)
    assert attn(x, pos).shape == (2, 5, cfg.d_model)
    n_params = sum(p.numel() for p in attn.parameters())
    assert n_params == 4 * cfg.d_model * cfg.d_model  # qkv (3 C x C) + out (C x C), no biases


def test_manual_matches_sdpa():
    cfg = tiny_config()
    manual, sdpa = make_pair(cfg)
    x = torch.randn(2, 7, cfg.d_model)
    pos = torch.arange(7).expand(2, 7)
    assert torch.allclose(manual(x, pos), sdpa(x, pos), atol=1e-5)


def test_manual_matches_sdpa_with_padding():
    cfg = tiny_config()
    manual, sdpa = make_pair(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])  # row 0 left-padded by 2
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out_m = manual(x, pos, attention_mask=mask)
    out_s = sdpa(x, pos, attention_mask=mask)
    real = mask.bool()
    assert torch.allclose(out_m[real], out_s[real], atol=1e-5)  # compare real tokens only


def test_causality():
    """Changing a later token must not change any earlier token's output."""
    cfg = tiny_config()
    attn = attention.CausalSelfAttention(cfg, impl="manual")
    x = torch.randn(1, 6, cfg.d_model)
    pos = torch.arange(6).unsqueeze(0)
    out1 = attn(x, pos)
    x2 = x.clone()
    x2[0, 4] = torch.randn(cfg.d_model)  # change token 4
    out2 = attn(x2, pos)
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-6)  # tokens 0-3 unaffected
    assert not torch.allclose(out1[0, 4], out2[0, 4])


def test_left_padding_does_not_change_real_tokens():
    """[pad, pad, a, b, c] must give the same outputs for a, b, c as plain [a, b, c]."""
    cfg = tiny_config()
    for impl in ("manual", "sdpa"):
        torch.manual_seed(0)
        attn = attention.CausalSelfAttention(cfg, impl=impl)
        abc = torch.randn(1, 3, cfg.d_model)
        padded = torch.cat([torch.randn(1, 2, cfg.d_model), abc], dim=1)  # pads hold junk
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        pos = (mask.cumsum(-1) - 1).clamp(min=0)  # [0, 0, 0, 1, 2]

        out_plain = attn(abc, torch.arange(3).unsqueeze(0))
        out_padded = attn(padded, pos, attention_mask=mask)
        assert torch.allclose(out_padded[:, 2:], out_plain, atol=1e-5), impl
        assert not torch.isnan(out_padded).any(), impl  # pad rows must not produce NaN
