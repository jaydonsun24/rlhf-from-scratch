import math

import torch
from torch import nn
from torch.nn import functional as F

from rlhf.model.layers import RotaryEmbedding


def build_attention_mask(attention_mask, T, device=None):
    causal = torch.tril(torch.ones(size=(T, T), dtype=torch.bool, device=device), diagonal=0)

    if attention_mask is None:
        causal = causal.view(1, 1, T, T)
        return causal
    else:
        keys_real = attention_mask.bool()[:, None, None, :]
        output = causal & keys_real
        diagonals = torch.eye(T, dtype=torch.bool, device=device)
        mask = output | diagonals
        return mask


# Manual_attention for practice and debugging if necessary
def manual_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1)
    scores = scores / (math.sqrt(q.shape[-1]))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    results = weights @ v
    return results


class CausalSelfAttention(nn.Module):
    def __init__(self, config, impl="sdpa"):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.impl = impl
        self.qkv = nn.Linear(
            in_features=config.d_model, out_features=3 * config.d_model, bias=False
        )
        self.rope = RotaryEmbedding(self.head_dim, config.context_len, config.rope_theta)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, position_ids, attention_mask=None, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # KV cache (generation only): store this step's keys/values and get back ALL of them
        # so far. Queries stay just the new token(s); keys/values cover the whole sequence.
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)  # (B, H, S, D), S = tokens seen so far
        S = k.shape[2]

        if S > T:
            # Decoding one new token against the cache. Every cached key is in its past,
            # so there's no causal triangle; only padded keys need hiding.
            # attention_mask must cover all S keys here: (B, S).
            assert T == 1, "with a filled KV cache, feed one token at a time"
            mask = torch.ones(B, 1, T, S, dtype=torch.bool, device=x.device)
            if attention_mask is not None:
                mask = mask & attention_mask.bool()[:, None, None, :]
            if self.impl == "manual":
                y = manual_attention(q, k, v, mask)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        elif self.impl == "manual":
            mask = build_attention_mask(attention_mask, T, device=x.device)
            y = manual_attention(q, k, v, mask)

        elif attention_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        else:
            mask = build_attention_mask(attention_mask, T, device=x.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)
