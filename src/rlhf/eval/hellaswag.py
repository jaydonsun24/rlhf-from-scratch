"""HellaSwag: can the model pick the sensible ending of a sentence? (docs/02-pretraining.md)

Each example is a context ("A man is sitting on a roof. he") and 4 endings, one correct.
A language model "answers" by scoring how likely each ending is: we compute the average
per-token loss of the ending's tokens given the context, and predict the ending with the
LOWEST loss (= the one the model finds most natural). Averaging per token (not summing)
stops long endings from being penalized just for being long ("acc_norm"-style).

Random guessing scores 25%. GPT-2 124M scores ~29.6% with this method.

    python -m rlhf.eval.hellaswag --checkpoint checkpoints/pretrain/final.pt
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from rlhf.data.tokenizer import PAD_ID, get_tokenizer

DATA_GLOB = "data/raw/hellaswag/data/validation-*.parquet"  # from scripts/download_raw.py


def load_examples(path_glob: str = DATA_GLOB) -> list[dict]:
    """-> [{"ctx_tokens": [...], "ending_tokens": [[...] x4], "label": int}, ...]"""
    files = sorted(Path().glob(path_glob))
    if not files:
        raise FileNotFoundError(f"no HellaSwag files at {path_glob}; run scripts/download_raw.py")
    enc = get_tokenizer()
    df = pd.concat(pd.read_parquet(f) for f in files)
    return [
        {
            "ctx_tokens": enc.encode_ordinary(row.ctx),
            # leading space: in the text, the ending follows the context after a space
            "ending_tokens": [enc.encode_ordinary(" " + e) for e in row.endings],
            "label": int(row.label),
        }
        for row in df.itertuples()
    ]


@torch.no_grad()
def ending_losses(model, ctx_tokens, ending_tokens, device, use_bf16=False) -> torch.Tensor:
    """Average loss of each ending given the context. Returns (4,): lower = more likely.

    The 4 candidates go through the model as one batch of 4 rows, right-padded:
        [context][ending 0][pad pad]
        [context][ending 1 .......]
    Padding at the END needs no attention mask: a causal model never looks forward,
    so the pads can't affect the real tokens, and their losses are masked out.
    """
    rows = [ctx_tokens + end for end in ending_tokens]
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    ending_mask = torch.zeros(len(rows), width, dtype=torch.bool)  # True on ending tokens
    for i, (row, end) in enumerate(zip(rows, ending_tokens, strict=True)):
        ids[i, : len(row)] = torch.tensor(row)
        ending_mask[i, len(ctx_tokens) : len(row)] = True
    ids, ending_mask = ids.to(device), ending_mask.to(device)

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(ids)  # (4, W, V)
    # logits at position p predict token p+1, so compare logits[:, :-1] with ids[:, 1:]
    token_loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(len(rows), -1)  # (4, W-1)
    mask = ending_mask[:, 1:].float()  # which targets are ending tokens
    return (token_loss * mask).sum(dim=1) / mask.sum(dim=1)


@torch.no_grad()
def evaluate_hellaswag(model, examples, device, use_bf16=False, rank=0, world_size=1) -> dict:
    """Accuracy over `examples`. With several GPUs, each rank scores every world_size-th example."""
    was_training = model.training
    model.eval()
    correct = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for ex in examples[rank::world_size]:
        losses = ending_losses(model, ex["ctx_tokens"], ex["ending_tokens"], device, use_bf16)
        correct += int(losses.argmin().item() == ex["label"])
        total += 1
    if world_size > 1:
        torch.distributed.all_reduce(correct)
        torch.distributed.all_reduce(total)
    model.train(was_training)
    return {"hellaswag_acc": (correct / total).item(), "hellaswag_n": int(total.item())}


if __name__ == "__main__":
    from rlhf.model.config import ModelConfig
    from rlhf.model.gpt import GPT
    from rlhf.utils.checkpoint import load_checkpoint
    from rlhf.utils.config import load_model_config
    from rlhf.utils.distributed import get_device

    parser = argparse.ArgumentParser(description="HellaSwag accuracy of a pretraining checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all 10,042")
    args = parser.parse_args()

    device = get_device()
    state = load_checkpoint(args.checkpoint, map_location=device)
    model_name = state["config"]["model"]
    model = GPT(load_model_config(model_name) if model_name else ModelConfig()).to(device)
    model.load_state_dict(state["model"])
    examples = load_examples()
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(evaluate_hellaswag(model, examples, device, use_bf16=device.type == "cuda"))
