"""Tests for model/gpt.py: the full transformer."""

import dataclasses
import math

import torch
import torch.nn.functional as F

from rlhf.model.gpt import GPT
from rlhf.utils.config import load_model_config


def tiny_model(**kwargs):
    torch.manual_seed(0)
    return GPT(load_model_config("tiny"), **kwargs)


def test_output_shapes():
    model = tiny_model()
    idx = torch.randint(0, 50257, (2, 9))  # (B, T) token ids
    logits = model(idx)
    assert logits.shape == (2, 9, 50304)  # one score per vocab entry, per position
    hidden = model.hidden_states(idx)
    assert hidden.shape == (2, 9, 128)  # (B, T, d_model)


def test_param_count_gpt124m():
    # Built on the "meta" device: shapes only, no memory, so this is instant.
    with torch.device("meta"):
        model = GPT(load_model_config("gpt124m"))
    n = sum(p.numel() for p in model.parameters())
    # embedding 50304*768 + 12 blocks * (attn 4*768^2 + mlp 3*768*2048 + 2 norms) + final norm
    assert n == 50304 * 768 + 12 * (4 * 768**2 + 3 * 768 * 2048 + 2 * 768) + 768
    assert n == 123_587_328


def test_weight_tying():
    model = tiny_model()
    v, c = model.config.vocab_size, model.config.d_model
    big = [p for p in model.parameters() if p.shape == (v, c)]
    assert len(big) == 1  # the embedding table is the ONLY (vocab, d_model) matrix
    idx = torch.randint(0, 50257, (1, 5))
    expected = model.hidden_states(idx) @ model.tok_emb.weight.T
    assert torch.allclose(model(idx), expected, atol=1e-5)


def test_last_only_matches_full_logits():
    model = tiny_model()
    idx = torch.randint(0, 50257, (2, 9))
    full = model(idx)
    last = model(idx, last_only=True)
    assert last.shape == (2, 1, 50304)
    assert torch.allclose(last, full[:, -1:], atol=1e-5)


def test_init_scales():
    model = tiny_model()
    n_layer = model.config.n_layer  # 2 for tiny
    block = model.blocks[0]
    assert abs(model.tok_emb.weight.std().item() - 0.02) < 0.002
    assert abs(block.attn.qkv.weight.std().item() - 0.02) < 0.002
    residual_std = 0.02 / math.sqrt(2 * n_layer)  # = 0.01 for 2 layers
    assert abs(block.attn.out.weight.std().item() - residual_std) < 0.001
    assert abs(block.mlp.down.weight.std().item() - residual_std) < 0.001
    assert torch.equal(block.attn_norm.weight, torch.ones(model.config.d_model))


def test_causality():
    model = tiny_model()
    idx = torch.randint(0, 50257, (1, 8))
    idx2 = idx.clone()
    idx2[0, 5] = (idx[0, 5] + 1) % 50257  # change token 5
    out1, out2 = model(idx), model(idx2)
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-5)  # positions 0-4 unaffected
    assert not torch.allclose(out1[0, 5], out2[0, 5])


def test_left_padding_invariance():
    """The same prompt with and without left padding must give the same last-token logits.

    This catches padding-mask bugs, which would otherwise silently wreck SFT, the reward
    model and PPO. (Wrong position ids would NOT fail it: RoPE only sees relative distances,
    and padding shifts every real token equally.)
    """
    model = tiny_model()
    prompt = torch.randint(0, 50257, (1, 6))
    pad = torch.full((1, 3), 50257)  # the pad token id
    padded = torch.cat([pad, prompt], dim=1)  # (1, 9)
    mask = torch.cat([torch.zeros(1, 3), torch.ones(1, 6)], dim=1).long()

    plain_logits = model(prompt)[:, -1]
    padded_logits = model(padded, attention_mask=mask)[:, -1]  # position ids derived from mask
    assert torch.allclose(plain_logits, padded_logits, atol=1e-4)


def test_manual_and_sdpa_models_match():
    manual = tiny_model(attn_impl="manual")
    sdpa = tiny_model(attn_impl="sdpa")
    # Without these, the test would pass even if attn_impl never reached the blocks
    # (both models would silently be SDPA and trivially agree).
    assert all(block.attn.impl == "manual" for block in manual.blocks)
    assert all(block.attn.impl == "sdpa" for block in sdpa.blocks)
    sdpa.load_state_dict(manual.state_dict())
    idx = torch.randint(0, 50257, (2, 7))
    assert torch.allclose(manual(idx), sdpa(idx), atol=1e-4)


def test_overfits_one_batch():
    """A correct model can memorize a single batch. If loss won't drop, something is broken."""
    # A 512-token vocab keeps this fast; the full 50K vocab makes it ~50x slower.
    torch.manual_seed(0)
    model = GPT(dataclasses.replace(load_model_config("tiny"), vocab_size=512))
    torch.manual_seed(1)
    data = torch.randint(0, 512, (4, 17))
    x, y = data[:, :-1], data[:, 1:]  # predict each next token
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
