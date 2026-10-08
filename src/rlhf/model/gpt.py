import math

import torch
from torch import nn

from rlhf.model.attention import CausalSelfAttention
from rlhf.model.layers import RMSNorm, SwiGLU


class Block(nn.Module):
    def __init__(self, config, attn_impl="sdpa"):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config, attn_impl)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, position_ids, attention_mask=None, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), position_ids, attention_mask, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, attn_impl="sdpa"):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, attn_impl) for i in range(config.n_layer))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        for c in self.modules():
            if isinstance(c, (nn.Linear, nn.Embedding)):
                nn.init.normal_(c.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
            )
            nn.init.normal_(
                block.mlp.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
            )

    def hidden_states(self, idx, position_ids=None, attention_mask=None, kv_cache=None):
        # kv_cache: None, or a list with one LayerKVCache per block (see generate.py)
        B, T = idx.shape
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(T, device=idx.device).expand(B, T)
            else:
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x = block(x, position_ids, attention_mask, layer_cache)
        return self.norm(x)

    def forward(self, idx, position_ids=None, attention_mask=None, kv_cache=None):
        x = self.hidden_states(idx, position_ids, attention_mask, kv_cache)
        return x @ self.tok_emb.weight.T
