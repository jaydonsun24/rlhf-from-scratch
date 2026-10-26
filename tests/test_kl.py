import copy

import torch

from rlhf.eval.kl import estimate_kl
from rlhf.model.gpt import GPT
from rlhf.utils.config import load_model_config


def tiny():
    torch.manual_seed(0)
    return GPT(load_model_config("tiny"))


PROMPTS = [[11, 12, 13], [14, 15], [16, 17, 18, 19]]


def test_identical_models_have_zero_kl():
    policy = tiny()
    kl, per = estimate_kl(policy, copy.deepcopy(policy), PROMPTS, torch.device("cpu"))
    assert len(per) == 3 and abs(kl) < 1e-4


def test_different_models_have_positive_kl():
    policy, reference = tiny(), tiny()
    with torch.no_grad():
        for p in reference.parameters():
            p.add_(0.05 * torch.randn_like(p))  # perturb the reference
    gen = torch.Generator().manual_seed(0)
    kl, _ = estimate_kl(policy, reference, PROMPTS * 10, torch.device("cpu"), generator=gen)
    assert kl > 0  # samples come from the policy, so on average they're likelier under it
