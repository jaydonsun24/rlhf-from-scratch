import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        temp = x.to(torch.float)
        temp = temp * torch.rsqrt(torch.mean(temp.pow(2), dim=-1, keepdim=True) + self.eps)
        return (temp * self.weight).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.up = nn.Linear(
            in_features = dim,
            out_features = hidden_dim,
            bias = False
        )
        self.gate = nn.Linear(
            in_features = dim,
            out_features = hidden_dim,
            bias = False
        )
        self.down = nn.Linear(
            in_features = hidden_dim,
            out_features = dim,
            bias = False
        )
    def forward(self, x):
        temp1 = self.up(x)
        temp2 = self.gate(x)
        temp3 = torch.mul(temp1, F.silu(temp2))
        return self.down(temp3)

