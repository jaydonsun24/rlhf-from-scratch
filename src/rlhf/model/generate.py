"""Text generation with a KV cache.

Generation is a loop: predict one token, append it, predict the next. Without a cache,
every step re-runs the model over the whole sequence so far. With a cache, each layer
keeps the keys and values it has already computed, so every step only processes the
one new token. Same output, much less work. PPO generates hundreds of thousands of
summaries, so this matters.

Batch layout (shared with SFT/RM/PPO): prompts are LEFT-padded, so every row's prompt
ends in the same column and new tokens are appended in the same column for all rows.

Run as a script to benchmark cache vs. no cache and write results/generation.json:
    python -m rlhf.model.generate --model gpt124m
"""

import argparse
import time

import torch
import torch.nn.functional as F

from rlhf.data.tokenizer import EOT_ID, N_REAL_TOKENS, PAD_ID


def mask_unused_vocab(logits: torch.Tensor) -> torch.Tensor:
    """(..., V) -> same shape, with ids >= 50257 (pad + vocab padding) set to -inf.

    Those ids are never real text, but the tied output layer still gives them scores.
    Every sampling step AND every log-prob computation (rl/logprobs.py) must use this,
    so sampling and log-probs always see the same distribution.
    """
    unused = torch.arange(logits.shape[-1], device=logits.device) >= N_REAL_TOKENS
    return logits.masked_fill(unused, float("-inf"))


class LayerKVCache:
    """One layer's saved keys and values, preallocated to the maximum length.

    Preallocating (instead of torch.cat-ing a growing tensor every step) means no new
    memory is allocated per token. The buffers are created on the first update so they
    get the right batch size, device and dtype (bf16 under autocast) automatically.
    Memory cost: 2 (k and v) * B * H * max_len * D numbers per layer.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.k = None  # (B, H, max_len, D) once allocated
        self.v = None
        self.length = 0  # how many positions are filled

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Write the new keys/values (B, H, T, D) after the cached ones; return all of them."""
        B, H, T, D = k.shape
        if self.k is None:
            self.k = torch.zeros(B, H, self.max_len, D, dtype=k.dtype, device=k.device)
            self.v = torch.zeros(B, H, self.max_len, D, dtype=v.dtype, device=v.device)
        end = self.length + T
        assert end <= self.max_len, "KV cache is full"
        self.k[:, :, self.length : end] = k
        self.v[:, :, self.length : end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]  # (B, H, S, D)


def make_kv_cache(model, max_len: int) -> list[LayerKVCache]:
    return [LayerKVCache(max_len) for _ in model.blocks]


def sample_next(logits, temperature, top_k=None, generator=None):
    """logits (B, V) -> (token ids (B,), their log-probs (B,)).

    temperature 0 = greedy (always the top token). Otherwise sample from
    softmax(logits / temperature). The returned log-prob is under that same distribution,
    which is the one PPO needs (it divides by the same temperature when recomputing).
    """
    logits = mask_unused_vocab(logits.float())
    if temperature == 0:
        tokens = logits.argmax(dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
    else:
        logits = logits / temperature
        if top_k is not None:  # keep only the k best tokens (for readable samples, not RLHF)
            kth_best = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth_best, float("-inf"))
        logprobs = F.log_softmax(logits, dim=-1)
        tokens = torch.multinomial(logprobs.exp(), num_samples=1, generator=generator).squeeze(-1)
    return tokens, logprobs.gather(-1, tokens[:, None]).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask=None,
    max_new_tokens=64,
    temperature=0.0,
    top_k=None,
    eos_id=EOT_ID,
    use_cache=True,
    generator=None,
):
    """Generate up to max_new_tokens after each left-padded prompt.

    input_ids, attention_mask: (B, P). attention_mask is 1 on real tokens, 0 on pads.
    Each row stops at its own EOS; after that it gets PAD tokens that are masked out.
    Pass eos_id=None to never stop early (used by the benchmark).

    Returns a dict:
      tokens        (B, N) generated ids, PAD after each row's EOS
      response_mask (B, N) 1 on real generated tokens up to and including EOS, 0 after
      logprobs      (B, N) log-prob of each generated token under the sampling distribution
                           (for debugging; PPO recomputes these itself)
    """
    B, P = input_ids.shape
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attn = attention_mask.long()

    # Prefill: run the whole prompt once (this also fills the cache).
    cache = make_kv_cache(model, P + max_new_tokens) if use_cache else None
    ids = input_ids
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(ids, position_ids, attn, cache, last_only=True)[:, -1]  # (B, V): next token

    done = torch.zeros(B, dtype=torch.bool, device=device)  # which rows have hit EOS
    tokens, response_mask, logprobs = [], [], []
    for _ in range(max_new_tokens):
        token, logprob = sample_next(logits, temperature, top_k, generator)
        token = torch.where(done, torch.full_like(token, PAD_ID), token)  # finished rows: pad
        tokens.append(token)
        response_mask.append(~done)
        logprobs.append(torch.where(done, torch.zeros_like(logprob), logprob))

        # The new token is real for rows that weren't finished before this step (EOS counts).
        attn = torch.cat([attn, (~done).long()[:, None]], dim=1)  # (B, S)
        if eos_id is not None:
            done = done | (token == eos_id)
            if done.all():
                break

        if use_cache:
            # Feed only the new token; its position = number of real tokens before it.
            new_pos = (attn.sum(dim=-1, keepdim=True) - 1).clamp(min=0)  # (B, 1)
            logits = model(token[:, None], new_pos, attn, cache, last_only=True)[:, -1]
        else:
            # No cache: re-run the model over the entire sequence so far.
            ids = torch.cat([ids, token[:, None]], dim=1)
            position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
            logits = model(ids, position_ids, attn, last_only=True)[:, -1]

    return {
        "tokens": torch.stack(tokens, dim=1),
        "response_mask": torch.stack(response_mask, dim=1).long(),
        "logprobs": torch.stack(logprobs, dim=1),
    }


def benchmark(
    model_name: str, batch: int, prompt_len: int, new_tokens: int, repeats: int, save: bool
):
    """Tokens/sec with vs. without the KV cache. Weights don't affect speed, so random init is fine."""
    from rlhf.model.gpt import GPT
    from rlhf.utils.config import load_model_config
    from rlhf.utils.distributed import get_device
    from rlhf.utils.logging import write_results

    device = get_device()
    model = GPT(load_model_config(model_name)).to(device).eval()
    prompt = torch.randint(0, N_REAL_TOKENS, (batch, prompt_len), device=device)
    use_bf16 = device.type == "cuda"

    def timed(use_cache):
        best = float("inf")
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                generate(model, prompt, max_new_tokens=new_tokens, eos_id=None, use_cache=use_cache)
            if device.type == "cuda":
                torch.cuda.synchronize()  # GPU work is asynchronous; wait before stopping the clock
            best = min(best, time.perf_counter() - start)
        return batch * new_tokens / best  # generated tokens per second

    generate(model, prompt, max_new_tokens=4, use_cache=True)  # warm-up
    with_cache, without_cache = timed(True), timed(False)
    result = {
        "model": model_name,
        "batch_size": batch,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "tokens_per_sec_with_cache": round(with_cache, 1),
        "tokens_per_sec_without_cache": round(without_cache, 1),
        "speedup": round(with_cache / without_cache, 2),
        "bf16": use_bf16,
    }
    print(result)
    if save:
        print("wrote", write_results("generation", result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark generation with vs. without KV cache")
    parser.add_argument("--model", default="gpt124m")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-save", action="store_true", help="print only; don't write results/")
    args = parser.parse_args()
    benchmark(
        args.model, args.batch, args.prompt_len, args.new_tokens, args.repeats, not args.no_save
    )
