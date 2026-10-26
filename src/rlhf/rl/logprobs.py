import torch
import torch.nn.functional as F

from rlhf.model.generate import mask_unused_vocab


def token_logprobs(logits, input_ids, temperature=1.0):
    logits = logits[:, :-1]
    input_ids = input_ids[:, 1:].unsqueeze(-1)
    logits = logits.float()
    tens = F.log_softmax(mask_unused_vocab(logits) / temperature, dim=-1)
    tens = tens.gather(-1, input_ids)
    tens = tens.squeeze(-1)
    return tens


def sequence_logprobs(logits, input_ids, response_mask, temperature=1.0):
    result = token_logprobs(logits, input_ids, temperature)
    response_mask = response_mask[:, 1:].bool()
    return torch.where(response_mask, result, 0.0).sum(-1)
