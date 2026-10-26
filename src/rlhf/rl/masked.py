import torch


def masked_mean(x, mask):
    output = torch.where(mask.bool(), x, 0).sum() / mask.sum()
    return output


def masked_whiten(x, mask):
    mean = masked_mean(x, mask)
    variance = masked_mean((x - mean) ** 2, mask)
    whitened = (x - mean) / torch.sqrt(variance + 1e-8)
    return torch.where(mask.bool(), whitened, 0.0)
