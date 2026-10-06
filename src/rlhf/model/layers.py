import torch
from torch import nn
from torch.nn import functional as F

# RMS Norm function so that weights don't get too big as they pass through layers
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        temp = x.to(torch.float)
        temp = temp * torch.rsqrt(torch.mean(temp.pow(2), dim=-1, keepdim=True) + self.eps)
        return (temp * self.weight).type_as(x)

#SwiGLU unit needed as a good activation function for hidden layers

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
        upped = self.up(x)
        gated = self.gate(x)
        hidden = torch.mul(upped, F.silu(gated))
        return self.down(hidden)

# rotate_half function needed for RoPE
def rotate_half(x):
    first_half, second_half = x.chunk(2, dim = -1)
    return torch.cat((-second_half, first_half), dim = -1)

# creating the Rotary Embedding 
class RotaryEmbedding (nn.Module):
    def __init__(self, head_dim, max_len, theta = 10000.0):
        super().__init__()
        speeds = 1 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, speeds)
        angles = torch.cat((angles, angles), dim = -1)
        self.register_buffer("cos", torch.cos(angles), persistent = False)
        self.register_buffer("sin", torch.sin(angles), persistent = False)

    def forward(self, x, position_ids):
        cos = self.cos[position_ids].unsqueeze(dim = 1)
        sin = self.sin[position_ids].unsqueeze(dim = 1)
        xf = x.float()
        xf = xf * cos  + rotate_half(xf) * sin
        return xf.type_as(x)





