import contextlib

import torch
import torch.optim
from torch import nn


def train_step(model, optimizer, micro_batches, lr, grad_clip, use_bf16=False):
    loss_aggregate = 0
    for group in optimizer.param_groups:
        group["lr"] = lr

    for i, (x, y) in enumerate(micro_batches):
        last = i == len(micro_batches) - 1
        ctx = (
            model.no_sync() if hasattr(model, "no_sync") and not last else contextlib.nullcontext()
        )
        with ctx:
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(x).to(torch.float)
                V = logits.shape[-1]
                logits = logits.reshape(-1, V)
            y = y.reshape(-1)
            loss = nn.functional.cross_entropy(logits, y) / len(micro_batches)
            loss.backward()
        loss_aggregate += loss.item()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": loss_aggregate, "grad_norm": norm.item()}
