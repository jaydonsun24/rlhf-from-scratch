"""GPU-only checks. Skipped automatically where there's no CUDA (laptop, GitHub Actions).

Run on Adroit inside a GPU session: pytest -q tests/test_cuda.py
These catch bugs CPU tests can't: tensors created on the wrong device, and bf16 issues.
"""

import pytest
import torch

from rlhf.data.tokenizer import N_REAL_TOKENS, PAD_ID
from rlhf.model.generate import generate
from rlhf.model.gpt import GPT
from rlhf.utils.config import load_model_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def tiny_on_gpu(attn_impl="sdpa"):
    torch.manual_seed(0)
    return GPT(load_model_config("tiny"), attn_impl=attn_impl).cuda().eval()


def test_manual_matches_sdpa_on_gpu_with_padding():
    manual, sdpa = tiny_on_gpu("manual"), tiny_on_gpu("sdpa")
    sdpa.load_state_dict(manual.state_dict())
    idx = torch.randint(0, N_REAL_TOKENS, (2, 9), device="cuda")
    idx[0, :3] = PAD_ID
    mask = torch.ones(2, 9, dtype=torch.long, device="cuda")
    mask[0, :3] = 0
    with torch.no_grad():
        out_m = manual(idx, attention_mask=mask)
        out_s = sdpa(idx, attention_mask=mask)
    assert torch.allclose(out_m[mask.bool()], out_s[mask.bool()], atol=1e-4)


def test_kv_cache_matches_no_cache_on_gpu():
    model = tiny_on_gpu()
    prompt = torch.randint(0, N_REAL_TOKENS, (2, 6), device="cuda")
    fast = generate(model, prompt, max_new_tokens=30, eos_id=None, use_cache=True)
    slow = generate(model, prompt, max_new_tokens=30, eos_id=None, use_cache=False)
    assert torch.equal(fast["tokens"], slow["tokens"])


def test_bf16_autocast_forward_and_backward():
    model = GPT(load_model_config("tiny")).cuda()
    idx = torch.randint(0, N_REAL_TOKENS, (2, 16), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), idx[:, 1:].reshape(-1)
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_resume_on_gpu_restores_rng_state(tmp_path):
    """Checkpoints are loaded with map_location=<gpu>, which moves the saved RNG states to
    the GPU too; restoring must still work (this crashed the first real resumed run)."""
    from rlhf.utils.checkpoint import latest_checkpoint, load_checkpoint, restore, save_checkpoint

    model = tiny_on_gpu()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(tmp_path, step=7, model=model, optimizer=opt, config={}, phase="test")
    expected = torch.rand(3)  # the next random numbers after saving
    state = load_checkpoint(latest_checkpoint(tmp_path), map_location=torch.device("cuda"))
    assert restore(state, tiny_on_gpu(), torch.optim.AdamW(model.parameters(), lr=1e-3)) == 7
    assert torch.equal(torch.rand(3), expected)  # RNG continues exactly where it was saved
