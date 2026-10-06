import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        temp = x.to(torch.float)
        temp = temp * torch.rsqrt(torch.mean(temp.pow(2), dim=-1, keepdim=True) + self.eps)
        return (temp * self.weight).type_as(x)
