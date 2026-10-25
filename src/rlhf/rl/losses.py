import torch
from torch import nn

def sft_loss(logits, input_ids, response_mask):
    V = logits.shape[-1]
    logits = logits[:, :-1].reshape(-1, V).to(torch.float)
    input_ids = input_ids[:, 1:].reshape(-1)
    response_mask = response_mask[:, 1:].reshape(-1).to(torch.float)
    losses = nn.functional.cross_entropy(logits, input_ids, reduction = "none") * response_mask
    avg_loss = losses.sum() / response_mask.sum()
    return avg_loss


