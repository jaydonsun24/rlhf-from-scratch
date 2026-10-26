"""Tests for PPO's math (Phase 6). All tensors are RESPONSE-ONLY, shape (B, R):
position t = the t-th generated token. `mask` is 1 on real response tokens (up to and
including EOT), 0 on the padding after it.

  rl/masked.py   masked_mean(x, mask), masked_whiten(x, mask)
  rl/rewards.py  compute_rewards(logp_old, logp_ref, score, mask, beta) -> (rewards, kl)
  rl/gae.py      compute_gae(rewards, values, mask, gamma, lam) -> (advantages, returns)
  rl/losses.py   ppo_policy_loss(logp_new, logp_old, advantages, mask, clip_eps) -> (loss, stats)
                 ppo_value_loss(values_new, values_old, returns, mask, clip_eps) -> (loss, stats)
"""

import pytest
import torch

from rlhf.rl.gae import compute_gae
from rlhf.rl.losses import ppo_policy_loss, ppo_value_loss
from rlhf.rl.masked import masked_mean, masked_whiten
from rlhf.rl.rewards import compute_rewards

INF = float("inf")


# ---------- 1. masked_mean / masked_whiten ----------


def test_masked_mean_ignores_masked_entries():
    x = torch.tensor([[1.0, 2.0, 3.0, 100.0], [4.0, -INF, 9.0, 9.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    assert masked_mean(x, mask).item() == pytest.approx((1 + 2 + 3 + 4) / 4)  # -inf never touched


def test_masked_whiten():
    x = torch.tensor([[1.0, 2.0, 3.0, 50.0], [4.0, 5.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    w = masked_whiten(x, mask)
    valid = w[mask.bool()]
    assert valid.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert (valid**2).mean().item() == pytest.approx(1.0, abs=1e-4)  # variance 1 over valid entries
    assert (w[~mask.bool()] == 0).all()  # padding stays 0


# ---------- 2. compute_rewards: per-token KL penalty + the RM score at the last token ----------


def test_rewards_hand_computed():
    #             t:   0     1     2     3 (pad)
    logp_old = torch.tensor([[-1.0, -2.0, -3.0, -INF]])  # pad position: -inf (masked vocab id)
    logp_ref = torch.tensor([[-1.5, -2.0, -2.0, -INF]])
    mask = torch.tensor([[1, 1, 1, 0]])
    score = torch.tensor([2.0])
    rewards, kl = compute_rewards(logp_old, logp_ref, score, mask, beta=0.1)
    # kl_t = logp_old - logp_ref = [0.5, 0, -1, (pad) 0]
    assert torch.allclose(kl, torch.tensor([[0.5, 0.0, -1.0, 0.0]]))
    # reward_t = -beta * kl_t, plus the score on the LAST REAL token (t = 2)
    assert torch.allclose(rewards, torch.tensor([[-0.05, 0.0, 0.1 + 2.0, 0.0]]))


def test_rewards_score_goes_on_each_rows_last_token():
    logp = torch.zeros(2, 5)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    rewards, _ = compute_rewards(logp, logp, torch.tensor([3.0, -1.0]), mask, beta=0.05)
    assert rewards.tolist() == [[0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0]]


# ---------- 3. compute_gae ----------


def test_gae_hand_computed_four_tokens():
    # delta_t = r_t + gamma * V_{t+1} - V_t   (V after the last token = 0)
    # A_t     = delta_t + gamma * lam * A_{t+1}
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    mask = torch.ones(1, 4)
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    # A3 = 1 - 0.5 = 0.5 ; A2 = 0 + 0.95*0.5 = 0.475 ; A1 = 0.95*0.475 ; A0 = 0.95*A1
    expected = torch.tensor([[0.4286875, 0.45125, 0.475, 0.5]])
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(ret, expected + values, atol=1e-6)  # returns = advantages + values


def test_gae_stops_at_each_rows_end():
    # Row ends after 2 tokens. The values at padded positions (9.0) must NOT leak in:
    # the value "after the last token" is 0.
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    values = torch.tensor([[0.2, 0.4, 9.0, 9.0]])
    mask = torch.tensor([[1, 1, 0, 0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    # A1 = 1 + 0 - 0.4 = 0.6 ; A0 = (0 + 0.4 - 0.2) + 0.95 * 0.6 = 0.77
    assert torch.allclose(adv[0, :2], torch.tensor([0.77, 0.6]), atol=1e-6)
    assert (adv[0, 2:] == 0).all()
    assert torch.allclose(ret[0, :2], torch.tensor([0.97, 1.0]), atol=1e-6)


def test_gae_lambda_extremes():
    rewards = torch.tensor([[0.1, 0.2, 0.3]])
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.ones(1, 3)
    # lam = 1: A_t = (sum of future rewards) - V_t   (pure Monte Carlo)
    adv1, _ = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv1, torch.tensor([[0.6 - 1.0, 0.5 - 2.0, 0.3 - 3.0]]), atol=1e-6)
    # lam = 0: A_t = delta_t = r_t + V_{t+1} - V_t   (one-step TD)
    adv0, _ = compute_gae(rewards, values, mask, gamma=1.0, lam=0.0)
    assert torch.allclose(adv0, torch.tensor([[0.1 + 2 - 1, 0.2 + 3 - 2, 0.3 + 0 - 3]]), atol=1e-6)


def test_gae_rows_are_independent():
    rewards = torch.tensor([[0.0, 1.0], [5.0, 5.0]])
    values = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    mask = torch.ones(2, 2)
    adv, _ = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    alone, _ = compute_gae(rewards[:1], values[:1], mask[:1], gamma=1.0, lam=0.95)
    assert torch.allclose(adv[:1], alone)


# ---------- 4. PPO losses ----------


def test_policy_loss_at_ratio_one():
    # Before any update logp_new == logp_old: ratio 1, nothing clipped, loss = -mean(A).
    logp = torch.tensor([[-1.0, -2.0, -3.0]])
    adv = torch.tensor([[1.0, -0.5, 2.0]])
    mask = torch.tensor([[1, 1, 0]])
    loss, stats = ppo_policy_loss(logp, logp, adv, mask, clip_eps=0.2)
    assert loss.item() == pytest.approx(-(1.0 - 0.5) / 2)
    assert stats["clipfrac"] == 0.0 and stats["approx_kl"] == pytest.approx(0.0)


def test_policy_loss_clipping_hand_computed():
    #   A = +1, ratio 1.5 : -A*r = -1.5, -A*clip(r) = -1.2 -> max = -1.2  (clipped: no extra credit)
    #   A = -1, ratio 0.5 : -A*r =  0.5, -A*clip(r) =  0.8 -> max =  0.8  (clipped)
    #   A = +1, ratio 0.5 : -A*r = -0.5, -A*clip(r) = -0.8 -> max = -0.5  (not clipped: pessimistic)
    logp_old = torch.zeros(1, 3)
    logp_new = torch.log(torch.tensor([[1.5, 0.5, 0.5]])).requires_grad_(True)
    adv = torch.tensor([[1.0, -1.0, 1.0]])
    loss, stats = ppo_policy_loss(logp_new, logp_old, adv, torch.ones(1, 3), clip_eps=0.2)
    assert loss.item() == pytest.approx((-1.2 + 0.8 - 0.5) / 3)
    assert stats["clipfrac"] == pytest.approx(2 / 3)
    loss.backward()
    assert logp_new.grad[0, 0] == 0 and logp_new.grad[0, 1] == 0  # clipped tokens get no gradient
    assert logp_new.grad[0, 2] != 0


def test_policy_loss_ignores_padding():
    logp_old = torch.tensor([[-1.0, -INF]])
    logp_new = torch.tensor([[-1.0, -INF]])  # pad positions can hold -inf
    loss, _ = ppo_policy_loss(
        logp_new, logp_old, torch.tensor([[2.0, 7.0]]), torch.tensor([[1, 0]]), 0.2
    )
    assert torch.isfinite(loss) and loss.item() == pytest.approx(-2.0)


def test_value_loss_hand_computed():
    # old 0, new 1, return 1: unclipped error 0; clipped value 0 + 0.2 -> error 0.64.
    # max(0, 0.64) = 0.64 -> times 0.5 = 0.32. The clip stops the value function jumping.
    loss, stats = ppo_value_loss(
        torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.ones(1, 1), 0.2
    )
    assert loss.item() == pytest.approx(0.32)
    assert stats["vf_clipfrac"] == pytest.approx(1.0)


def test_value_loss_no_change_is_half_mse():
    v = torch.tensor([[0.5, 2.0, 9.0]])
    ret = torch.tensor([[1.5, 1.0, 0.0]])
    loss, _ = ppo_value_loss(v, v, ret, torch.tensor([[1, 1, 0]]), 0.2)
    assert loss.item() == pytest.approx(0.5 * (1.0 + 1.0) / 2)


def test_approx_kl():
    logp_old = torch.zeros(1, 2)
    logp_new = torch.tensor([[0.2, -0.2]])
    _, stats = ppo_policy_loss(logp_new, logp_old, torch.zeros(1, 2), torch.ones(1, 2), 0.2)
    assert stats["approx_kl"] == pytest.approx(0.5 * (0.04 + 0.04) / 2)  # 0.5 * mean(logratio^2)
