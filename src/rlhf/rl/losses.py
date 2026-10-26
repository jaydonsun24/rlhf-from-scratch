import torch
from torch import nn


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
