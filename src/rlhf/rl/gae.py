import torch


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    values = values * mask
    rewards = rewards * mask
    B, R = rewards.shape
    next_adv = torch.zeros_like(rewards[:, 0])
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(R)):
        next_value = values[:, t + 1] if t + 1 < R else torch.zeros_like(rewards[:, 0])
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        next_adv = delta + gamma * lam * next_adv
        advantages[:, t] = next_adv
    returns = advantages + values
    return advantages, returns
