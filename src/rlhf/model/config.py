"""Model hyperparameters. Presets live in configs/model/*.yaml."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50304  # GPT-2's 50257 padded up to a multiple of 64 (faster matmuls)
    context_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12  # = n_head for MHA; < n_head enables GQA (optional)
    d_model: int = 768
    d_ff: int = 2048  # SwiGLU hidden size, ~(8/3) * d_model rounded to a multiple of 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0  # keep 0 everywhere; RLHF needs deterministic log-probs
    tie_embeddings: bool = True
