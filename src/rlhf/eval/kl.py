"""How far has a fine-tuned policy drifted from the SFT model? (docs/07-evaluation.md)

KL(policy || reference) on a set of prompts, estimated by Monte Carlo:
  1. sample a summary from the policy (temperature T, the same as PPO's rollouts),
  2. add up, over its tokens, log pi(token) - log pi_ref(token), both computed from logits / T,
  3. average over prompts.

Units: nats per summary. 0 means "identical to SFT". This is the same quantity PPO penalizes
during training, so DPO, PPO and the over-optimization plot all use this one definition.
"""

import torch

from rlhf.data.tldr import RESPONSE_CAP, collate
from rlhf.model.generate import generate
from rlhf.rl.logprobs import sequence_logprobs


@torch.no_grad()
def estimate_kl(
    policy,
    reference,
    prompts,
    device,
    temperature=0.7,
    batch_size=32,
    use_bf16=False,
    generator=None,
):
    """prompts: list of prompt token lists. Returns (mean KL in nats, list of per-prompt values)."""
    policy.eval()
    reference.eval()
    per_prompt = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        batch = collate([(p, []) for p in chunk])
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = generate(
                policy,
                ids,
                mask,
                max_new_tokens=RESPONSE_CAP,
                temperature=temperature,
                generator=generator,
            )
        # Re-assemble [prompt][sampled summary] in the shared layout and score it under both models.
        full = collate(
            [
                (p, out["tokens"][i, : int(out["response_mask"][i].sum())].tolist())
                for i, p in enumerate(chunk)
            ]
        )
        full = {k: v.to(device) if torch.is_tensor(v) else v for k, v in full.items()}
        args = (full["input_ids"], full["position_ids"], full["attention_mask"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            lp = sequence_logprobs(
                policy(*args).float(), full["input_ids"], full["response_mask"], temperature
            )
            lp_ref = sequence_logprobs(
                reference(*args).float(), full["input_ids"], full["response_mask"], temperature
            )
        per_prompt += (lp - lp_ref).tolist()
    return sum(per_prompt) / len(per_prompt), per_prompt
