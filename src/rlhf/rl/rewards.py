import torch


def compute_rewards(logp_old, logp_ref, score, mask, beta):
    B = mask.shape[0]
    kl = torch.where(mask.bool(), logp_old - logp_ref, 0.0)
    rewards = -beta * kl
    rewards[torch.arange(B), mask.sum(dim=1) - 1] += score
    return rewards, kl
