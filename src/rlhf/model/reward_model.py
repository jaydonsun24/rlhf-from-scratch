import math

import torch
from torch import nn

from rlhf.model.gpt import GPT


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = GPT(config)
        self.head = nn.Linear(config.d_model, 1)
        nn.init.normal_(
            self.head.weight,
            mean=0.0,
            std=(1 / math.sqrt(config.d_model + 1)),
        )
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        hidden = self.backbone.hidden_states(input_ids, position_ids, attention_mask)
        output = self.head(hidden)
        return output.squeeze(-1)


def reward_at_eot(scores, attention_mask):
    B, L = attention_mask.shape
    steps_from_end = attention_mask.flip(-1).argmax(-1)
    idx = (L - 1) - steps_from_end
    return scores[torch.arange(B), idx]
