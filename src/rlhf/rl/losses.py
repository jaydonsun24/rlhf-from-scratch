import torch
from torch import nn
from rlhf.rl.masked import masked_mean 

def sft_loss(logits, input_ids, response_mask):
    V = logits.shape[-1]
    logits = logits[:, :-1].reshape(-1, V).to(torch.float)
    input_ids = input_ids[:, 1:].reshape(-1)
    response_mask = response_mask[:, 1:].reshape(-1).to(torch.float)
    losses = nn.functional.cross_entropy(logits, input_ids, reduction="none") * response_mask
    avg_loss = losses.sum() / response_mask.sum()
    return avg_loss


def bradley_terry_loss(chosen_rewards, rejected_rewards):
    return -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    ratio_chosen = policy_chosen - ref_chosen
    ratio_rejected = policy_rejected - ref_rejected
    return -nn.functional.logsigmoid(beta * (ratio_chosen - ratio_rejected)).mean()


def ppo_policy_loss(logp_new, logp_old, advantages, mask, clip_eps=0.2):
    logratio = torch.where(mask.bool(), logp_new - logp_old, 0.0)
    ratio = torch.exp(logratio)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = masked_mean(torch.max(unclipped, clipped), mask)
    with torch.no_grad():
        clipfrac  = masked_mean((clipped > unclipped).float(), mask)
        approx_kl = masked_mean(0.5 * logratio ** 2, mask)
    return loss, {"clipfrac": clipfrac.item(), "approx_kl": approx_kl.item()}


def ppo_value_loss(values_new, values_old, returns, mask, clip_eps=0.2):
    values_clipped = values_old + torch.clamp(values_new - values_old, -clip_eps, clip_eps)
    err_unclipped = (values_new - returns) ** 2
    err_clipped = (values_clipped - returns) ** 2
    loss = 0.5 * masked_mean(torch.max(err_unclipped, err_clipped), mask)
    vf_clipfrac = masked_mean((err_clipped > err_unclipped).float(), mask).item()
    return loss, {"vf_clipfrac": vf_clipfrac}

