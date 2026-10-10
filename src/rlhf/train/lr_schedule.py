import math


def lr_at_step(step, max_lr, warmup_steps, max_steps, min_lr_ratio=0.1):
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= warmup_steps and step <= max_steps:
        p = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (max_lr - min_lr)
    else:
        return min_lr
