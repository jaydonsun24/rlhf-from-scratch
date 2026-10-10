"""Tests for model/generate.py: KV cache and generation."""

import torch
import torch.nn.functional as F

from rlhf.data.tokenizer import EOT_ID, N_REAL_TOKENS, PAD_ID
from rlhf.model.generate import LayerKVCache, generate, mask_unused_vocab
from rlhf.model.gpt import GPT
from rlhf.utils.config import load_model_config


def tiny_model(attn_impl="sdpa"):
    torch.manual_seed(0)
    return GPT(load_model_config("tiny"), attn_impl=attn_impl).eval()


def left_padded_batch():
    """Two prompts of different lengths, left-padded to the same width."""
    torch.manual_seed(1)
    a = torch.randint(0, N_REAL_TOKENS, (7,))
    b = torch.randint(0, N_REAL_TOKENS, (4,))
    ids = torch.stack([a, torch.cat([torch.full((3,), PAD_ID), b])])  # (2, 7)
    mask = torch.tensor([[1] * 7, [0] * 3 + [1] * 4])
    return ids, mask, a, b


def test_mask_unused_vocab_never_samples_fake_tokens():
    logits = torch.randn(1_000, 50304) * 5
    logits[:, N_REAL_TOKENS:] += 100  # make the fake ids by far the most likely
    sampled = torch.multinomial(F.softmax(mask_unused_vocab(logits), dim=-1), 1)
    assert sampled.max().item() < N_REAL_TOKENS


def test_cache_update_appends_and_returns_everything():
    cache = LayerKVCache(max_len=10)
    k1, v1 = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8)  # (B, H, T=3, D)
    k, v = cache.update(k1, v1)
    assert k.shape == (2, 4, 3, 8) and cache.length == 3
    k2, v2 = torch.randn(2, 4, 1, 8), torch.randn(2, 4, 1, 8)  # one more token
    k, v = cache.update(k2, v2)
    assert k.shape == (2, 4, 4, 8) and cache.length == 4
    assert torch.equal(k, torch.cat([k1, k2], dim=2)) and torch.equal(v, torch.cat([v1, v2], dim=2))


def test_greedy_with_cache_matches_without_cache():
    """The key KV-cache test: identical tokens and log-probs over 50 steps."""
    for impl in ("sdpa", "manual"):
        model = tiny_model(impl)
        prompt = torch.randint(0, N_REAL_TOKENS, (2, 5))
        fast = generate(model, prompt, max_new_tokens=50, eos_id=None, use_cache=True)
        slow = generate(model, prompt, max_new_tokens=50, eos_id=None, use_cache=False)
        assert torch.equal(fast["tokens"], slow["tokens"]), impl
        assert torch.allclose(fast["logprobs"], slow["logprobs"], atol=1e-4), impl


def test_cache_matches_with_left_padding():
    model = tiny_model()
    ids, mask, _, _ = left_padded_batch()
    fast = generate(model, ids, mask, max_new_tokens=20, eos_id=None, use_cache=True)
    slow = generate(model, ids, mask, max_new_tokens=20, eos_id=None, use_cache=False)
    assert torch.equal(fast["tokens"], slow["tokens"])


def test_batched_generation_matches_one_prompt_at_a_time():
    """A padded row must generate exactly what that prompt generates on its own."""
    model = tiny_model()
    ids, mask, a, b = left_padded_batch()
    batched = generate(model, ids, mask, max_new_tokens=15, eos_id=None)["tokens"]
    alone_a = generate(model, a[None], max_new_tokens=15, eos_id=None)["tokens"]
    alone_b = generate(model, b[None], max_new_tokens=15, eos_id=None)["tokens"]
    assert torch.equal(batched[0], alone_a[0])
    assert torch.equal(batched[1], alone_b[0])


class ForceEOT(torch.nn.Module):
    """Wraps a model so the chosen batch rows always predict EOT (to test stopping)."""

    def __init__(self, model, rows):
        super().__init__()
        self.model, self.rows, self.blocks = model, rows, model.blocks

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs).clone()
        logits[self.rows, :, EOT_ID] += 1e4
        return logits


def test_stops_at_eos_then_pads():
    model = ForceEOT(tiny_model(), rows=[0, 1, 2])
    prompt = torch.randint(0, N_REAL_TOKENS, (3, 4))
    out = generate(model, prompt, max_new_tokens=6)
    assert (out["tokens"][:, 0] == EOT_ID).all()  # every row ends immediately
    assert out["tokens"].shape[1] == 1  # all rows done, so the loop stopped early
    assert out["response_mask"].tolist() == [[1], [1], [1]]  # EOT itself counts as a response token


def test_finished_row_gets_pad_while_others_continue():
    inner = tiny_model()
    prompt = torch.randint(0, N_REAL_TOKENS, (2, 4))
    out = generate(ForceEOT(inner, rows=[0]), prompt, max_new_tokens=5, eos_id=EOT_ID)
    assert out["tokens"][0].tolist() == [EOT_ID] + [PAD_ID] * 4  # row 0: EOT, then padding
    assert out["response_mask"][0].tolist() == [1, 0, 0, 0, 0]
    assert out["response_mask"][1].tolist() == [1, 1, 1, 1, 1]  # row 1 keeps going
    # Row 0 finishing must not disturb row 1: same tokens as generating row 1 alone.
    alone = generate(inner, prompt[1:], max_new_tokens=5, eos_id=None)["tokens"]
    assert torch.equal(out["tokens"][1], alone[0])


def test_sampled_logprobs_match_recomputed():
    """Log-probs returned by sampling must equal log_softmax(logits / T) at the sampled token."""
    model = tiny_model()
    prompt = torch.randint(0, N_REAL_TOKENS, (2, 5))
    gen = torch.Generator().manual_seed(0)
    out = generate(model, prompt, max_new_tokens=8, temperature=0.7, eos_id=None, generator=gen)
    full = torch.cat([prompt, out["tokens"]], dim=1)
    with torch.no_grad():
        logits = model(full)[:, prompt.shape[1] - 1 : -1]  # logits at p predict token p+1
    logp = F.log_softmax(mask_unused_vocab(logits) / 0.7, dim=-1)
    recomputed = logp.gather(-1, out["tokens"][..., None]).squeeze(-1)
    assert torch.allclose(out["logprobs"], recomputed, atol=1e-4)


def test_sampling_is_reproducible_with_a_seed():
    model = tiny_model()
    prompt = torch.randint(0, N_REAL_TOKENS, (2, 5))
    run1 = generate(
        model,
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        generator=torch.Generator().manual_seed(42),
    )
    run2 = generate(
        model,
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(run1["tokens"], run2["tokens"])
