"""Tests for train/step.py: one optimizer step with gradient accumulation and clipping."""

import contextlib
import dataclasses

import torch
import torch.nn.functional as F

from rlhf.model.gpt import GPT
from rlhf.train.step import train_step
from rlhf.utils.config import load_model_config

VOCAB = 512  # small vocab keeps these fast


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(dataclasses.replace(load_model_config("tiny"), vocab_size=VOCAB))


def batch(n, seed=1, T=12):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, VOCAB, (n, T + 1), generator=g)
    return data[:, :-1], data[:, 1:]  # x, y


def test_accumulation_equals_one_big_batch():
    """4 micro-batches of 2 must give the same update as 1 batch of 8 (same data)."""
    x, y = batch(8)
    small, big = tiny_model(), tiny_model()
    # Plain SGD on purpose: its update is proportional to the gradient. Adam's first update is
    # ~lr * sign(gradient), so float rounding on near-zero gradients would flip some updates.
    opt_small = torch.optim.SGD(small.parameters(), lr=0.1)
    opt_big = torch.optim.SGD(big.parameters(), lr=0.1)
    micro = [(x[i : i + 2], y[i : i + 2]) for i in range(0, 8, 2)]

    out_small = train_step(small, opt_small, micro, lr=0.1, grad_clip=1e9)
    out_big = train_step(big, opt_big, [(x, y)], lr=0.1, grad_clip=1e9)

    assert abs(out_small["loss"] - out_big["loss"]) < 1e-5
    for p_small, p_big in zip(small.parameters(), big.parameters()):
        assert torch.allclose(p_small, p_big, atol=1e-5)


def test_returns_mean_loss_and_preclip_grad_norm():
    model = tiny_model()
    x, y = batch(4)
    # What the loss and gradient norm should be, computed independently first.
    reference = tiny_model()
    logits = reference(x)
    expected_loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    expected_loss.backward()
    expected_norm = torch.norm(torch.stack([p.grad.norm() for p in reference.parameters()]))

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = train_step(model, opt, [(x[:2], y[:2]), (x[2:], y[2:])], lr=0.1, grad_clip=1e-3)
    assert abs(out["loss"] - expected_loss.item()) < 1e-5
    assert abs(out["grad_norm"] - expected_norm.item()) < 1e-4  # BEFORE clipping, so > 1e-3


def test_clipping_limits_the_update():
    # With plain SGD at lr=1, the update is exactly -gradient. After clipping to 0.5,
    # the whole update vector must have length 0.5.
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    out = train_step(model, opt, [batch(4)], lr=1.0, grad_clip=0.5)
    assert out["grad_norm"] > 0.5  # it really was clipped
    update = torch.cat([(p - b).flatten() for p, b in zip(model.parameters(), before)])
    assert abs(update.norm().item() - 0.5) < 1e-4


def test_sets_lr_and_clears_gradients():
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=123.0)
    train_step(model, opt, [batch(2)], lr=3e-4, grad_clip=1.0)
    assert all(group["lr"] == 3e-4 for group in opt.param_groups)
    assert all(p.grad is None for p in model.parameters())  # ready for the next step


def test_ddp_syncs_gradients_only_on_last_micro_batch():
    """With DDP, gradients are averaged across GPUs on every backward() unless you're inside
    model.no_sync(). Syncing once per step (on the last micro-batch) is enough and much faster."""

    class FakeDDP(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
            self.no_sync_calls = 0

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        @contextlib.contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield

    model = FakeDDP(tiny_model())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    micro = [batch(2, seed=s) for s in range(4)]
    train_step(model, opt, micro, lr=1e-3, grad_clip=1.0)
    assert model.no_sync_calls == 3  # skip sync on micro-batches 1-3, sync on the 4th


def test_bf16_autocast_runs_on_cpu():
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = train_step(model, opt, [batch(2)], lr=1e-3, grad_clip=1.0, use_bf16=True)
    assert torch.isfinite(torch.tensor(out["loss"]))


def test_custom_loss_fn():
    """SFT, the reward model and DPO reuse train_step with their own loss: train_step calls
    loss_fn(model, micro_batch) for each micro-batch instead of the default next-token loss."""
    seen = []

    def loss_fn(model, micro_batch):
        seen.append(micro_batch["tag"])
        return model(micro_batch["x"]).float().pow(2).mean()  # any scalar works

    x1, _ = batch(2, seed=1)
    x2, _ = batch(2, seed=2)
    micro = [{"x": x1, "tag": "first"}, {"x": x2, "tag": "second"}]

    reference = tiny_model()  # same weights: compute the expected loss before any update
    with torch.no_grad():
        expected = (loss_fn(reference, micro[0]) + loss_fn(reference, micro[1])) / 2
    seen.clear()

    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = train_step(model, opt, micro, lr=0.1, grad_clip=1e9, loss_fn=loss_fn)
    assert seen == ["first", "second"]  # called once per micro-batch, in order
    assert abs(out["loss"] - expected.item()) < 1e-5  # averaged across micro-batches
