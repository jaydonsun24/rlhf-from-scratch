"""PPO: optimize the SFT policy against the reward model, with a per-token KL penalty to SFT
(docs/06-ppo.md, following Stiennon et al. 2020 and Huang et al. 2024).

    sbatch scripts/slurm/ppo.slurm                              # beta 0.05 (the default run)
    sbatch scripts/slurm/ppo.slurm beta=0 run_name=ppo_beta0    # ablation A: no KL penalty

Four models:
  policy      starts as SFT, trained              writes summaries
  reference   frozen SFT copy                     the KL anchor
  reward      frozen, normalized RM               scores whole summaries
  value       starts as the RM, trained           per-token forecasts for GAE

One iteration:
  1. rollout   the policy samples summaries for `rollout_batch` posts (T = 0.7, KV cache)
  2. score     log-probs under policy and reference, values, and the RM score at EOT;
               a summary that never emits EOT gets score -1 instead ("EOS trick")
  3. rewards   per-token -beta * KL, plus the score on the last token     (rl/rewards.py)
  4. GAE       advantages and returns; advantages whitened                (rl/gae.py, rl/masked.py)
  5. update    `ppo_epochs` passes over the rollout in minibatches: clipped policy loss
               and clipped value loss, each model with its own AdamW     (rl/losses.py)

Indexing (tested once, used everywhere): prompts are left-padded to length P, so response
token t sits at column P + t. Its log-prob and value come from position P + t - 1:
logits[:, P-1:] score input_ids[:, P:], and values[:, P-1:-1] line up with them.
Every per-token tensor below is response-only, shape (B, R).
"""

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rlhf.data.tldr import RESPONSE_CAP, TLDRDataset, collate
from rlhf.data.tokenizer import EOT_ID, get_tokenizer
from rlhf.eval.kl import estimate_kl
from rlhf.model.generate import generate, mask_unused_vocab
from rlhf.model.gpt import GPT
from rlhf.model.reward_model import RewardModel, reward_at_eot
from rlhf.rl.gae import compute_gae
from rlhf.rl.logprobs import token_logprobs
from rlhf.rl.losses import ppo_policy_loss, ppo_value_loss
from rlhf.rl.masked import masked_mean, masked_whiten
from rlhf.rl.rewards import compute_rewards
from rlhf.train.dpo import greedy_pairs
from rlhf.train.lr_schedule import lr_at_step
from rlhf.train.reward import score as rm_score
from rlhf.train.sft import generation_eval
from rlhf.train.step import train_step
from rlhf.utils.checkpoint import (
    PreemptionFlag,
    delete_rolling_checkpoints,
    latest_checkpoint,
    load_checkpoint,
    restore,
    save_checkpoint,
)
from rlhf.utils.config import load_config, load_model_config, save_config
from rlhf.utils.distributed import get_device
from rlhf.utils.logging import MetricsLogger, hardware_info, write_results
from rlhf.utils.seed import seed_everything


@dataclass
class PPOConfig:
    model: str = "gpt124m"
    sft_checkpoint: str = "checkpoints/sft/final.pt"  # policy + reference; empty = random (tests)
    rm_checkpoint: str = "checkpoints/rm/final.pt"  # reward model (normalized); empty = random
    tldr_dir: str = "data/tldr"
    run_name: str = "ppo_default"  # results/<run_name>.json
    checkpoint_root: str = "checkpoints"
    runs_root: str = "runs"
    seed: int = 1337
    # rollouts
    rollout_batch: int = 512  # prompts per iteration (use 64 while debugging)
    total_episodes: int = 250_000  # summaries generated over the whole run
    max_iterations: int = 0  # > 0 overrides total_episodes
    temperature: float = 0.7
    eos_trick: bool = True  # ablation B: False
    eos_penalty: float = -1.0
    # rewards and advantages
    beta: float = 0.05  # KL coefficient; ablation A: {0, 0.01, 0.05, 0.2}
    gamma: float = 1.0
    lam: float = 0.95
    whiten_advantages: bool = True  # ablation D: False
    # updates
    ppo_epochs: int = 4
    minibatch_size: int = 128  # sequences per optimizer step (4 minibatches per epoch)
    micro_batch_size: int = 16
    clip_eps: float = 0.2
    vf_coef: float = 0.1
    lr: float = 3e-6  # docs/06 sweep {3e-6, 1e-5}
    value_lr: float = 0.0  # 0 = same as lr
    warmup_iterations: int = 10
    min_lr_ratio: float = 0.1
    eps: float = 1e-5
    grad_clip: float = 1.0
    value_init: str = "rm"  # "rm" (default) or "sft" (ablation C: SFT backbone + fresh head)
    # bookkeeping
    scoring_batch: int = 32  # sequences per no-grad scoring forward
    log_every: int = 1
    sample_every: int = 50
    ckpt_every: int = 10  # rolling checkpoints for resume
    milestone_every: int = 100  # policy-only snapshots for the over-optimization curve
    eval_examples: int = 1000
    kl_prompts: int = 200
    eval_batch_size: int = 32
    bf16: bool = True
    use_wandb: bool = True
    write_results: bool = True
    save_checkpoints: bool = (
        True  # sweep runs: false (their numbers are in runs/<name>/summary.json)
    )


def load_models(cfg: PPOConfig, model_cfg, device):
    policy = GPT(model_cfg)
    if cfg.sft_checkpoint:
        policy.load_state_dict(load_checkpoint(cfg.sft_checkpoint, map_location="cpu")["model"])
    reference = copy.deepcopy(policy)
    reward_model = RewardModel(model_cfg)
    if cfg.rm_checkpoint:
        reward_model.load_state_dict(
            load_checkpoint(cfg.rm_checkpoint, map_location="cpu")["model"]
        )
    if cfg.value_init == "rm":
        value_model = copy.deepcopy(reward_model)  # already knows what a good summary looks like
    else:  # ablation C: SFT backbone with a fresh scalar head
        value_model = RewardModel(model_cfg)
        value_model.backbone.load_state_dict(policy.state_dict())
    for frozen in (reference, reward_model):
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
    return [m.to(device) for m in (policy, reference, reward_model, value_model)]


def response_logprobs(logits_from_p_minus_1, input_ids, P, temperature):
    """Log-probs of the response tokens. logits must already be sliced to [:, P-1:]."""
    return token_logprobs(logits_from_p_minus_1, input_ids[:, P - 1 :], temperature)  # (B, R)


@torch.no_grad()
def rollout(policy, reference, reward_model, value_model, prompts, cfg, device, use_bf16):
    """Steps 1-2: sample summaries, then score everything PPO needs (no gradients)."""
    policy.eval()
    value_model.eval()
    batch = collate([(p, []) for p in prompts])
    prompt_ids, prompt_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    P = prompt_ids.shape[1]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out = generate(
            policy,
            prompt_ids,
            prompt_mask,
            max_new_tokens=RESPONSE_CAP,
            temperature=cfg.temperature,
        )
    tokens, response_mask = out["tokens"], out["response_mask"]  # (B, R)
    input_ids = torch.cat([prompt_ids, tokens], dim=1)  # (B, P + R)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

    parts = {k: [] for k in ("logp_old", "logp_ref", "values", "score", "entropy")}
    for s in range(0, len(prompts), cfg.scoring_batch):
        rows = slice(s, s + cfg.scoring_batch)
        args = (input_ids[rows], position_ids[rows], attention_mask[rows])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pol_logits = policy(*args)[
                :, P - 1 :
            ].float()  # only the positions that predict responses
            ref_logits = reference(*args)[:, P - 1 :].float()
            values = value_model(*args)[:, P - 1 : -1].float()
            rm_scores = reward_model(*args).float()
        ids = input_ids[rows]
        parts["logp_old"].append(response_logprobs(pol_logits, ids, P, cfg.temperature))
        parts["logp_ref"].append(response_logprobs(ref_logits, ids, P, cfg.temperature))
        parts["values"].append(values)
        parts["score"].append(reward_at_eot(rm_scores, attention_mask[rows]))
        # Entropy of the sampling distribution at each response position (a health metric:
        # a collapsing entropy means the policy is becoming deterministic).
        probs = torch.softmax(mask_unused_vocab(pol_logits[:, :-1]) / cfg.temperature, dim=-1)
        parts["entropy"].append(torch.special.entr(probs).sum(-1))
    r = {k: torch.cat(v) for k, v in parts.items()}
    has_eot = (tokens == EOT_ID).any(dim=1)
    r["raw_score"] = r["score"]
    if cfg.eos_trick:  # never stopping is penalized, so rambling to the cap can't pay off
        r["score"] = torch.where(has_eot, r["score"], torch.full_like(r["score"], cfg.eos_penalty))
    r.update(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        response_mask=response_mask,
        tokens=tokens,
        has_eot=has_eot,
        prompt_len=P,
    )
    return r


def make_policy_loss_fn(cfg, stats):
    def loss_fn(model, mb):
        P = mb["prompt_len"]
        logits = model(mb["input_ids"], mb["position_ids"], mb["attention_mask"])[:, P - 1 :]
        logp_new = response_logprobs(logits.float(), mb["input_ids"], P, cfg.temperature)
        loss, st = ppo_policy_loss(
            logp_new, mb["logp_old"], mb["advantages"], mb["response_mask"], cfg.clip_eps
        )
        with torch.no_grad():
            mask = mb["response_mask"].bool()
            ratio = torch.exp(torch.where(mask, logp_new - mb["logp_old"], 0.0))
            st = {
                **st,
                "ratio_mean": masked_mean(ratio, mask).item(),
                "ratio_max": ratio[mask].max().item(),
            }
        stats.append(st)
        return loss

    return loss_fn


def make_value_loss_fn(cfg, stats):
    def loss_fn(model, mb):
        P = mb["prompt_len"]
        values = model(mb["input_ids"], mb["position_ids"], mb["attention_mask"])[
            :, P - 1 : -1
        ].float()
        loss, st = ppo_value_loss(
            values, mb["values_old"], mb["returns"], mb["response_mask"], cfg.clip_eps
        )
        stats.append({**st, "vf_loss": loss.item()})
        return cfg.vf_coef * loss

    return loss_fn


def ppo_update(policy, value_model, opt_policy, opt_value, roll, cfg, lr, use_bf16):
    """Step 5: several epochs over the rollout, in shuffled minibatches split into micro-batches."""
    policy.train()
    value_model.train()
    fields = (
        "input_ids",
        "position_ids",
        "attention_mask",
        "response_mask",
        "logp_old",
        "advantages",
        "values_old",
        "returns",
    )
    B = roll["input_ids"].shape[0]
    p_stats, v_stats = [], []
    for _ in range(cfg.ppo_epochs):
        order = torch.randperm(B, device=roll["input_ids"].device)
        for start in range(0, B, cfg.minibatch_size):
            idx = order[start : start + cfg.minibatch_size]
            micro = [
                {
                    **{f: roll[f][idx[m : m + cfg.micro_batch_size]] for f in fields},
                    "prompt_len": roll["prompt_len"],
                }
                for m in range(0, len(idx), cfg.micro_batch_size)
            ]
            train_step(
                policy,
                opt_policy,
                micro,
                lr,
                cfg.grad_clip,
                use_bf16,
                loss_fn=make_policy_loss_fn(cfg, p_stats),
            )
            train_step(
                value_model,
                opt_value,
                micro,
                cfg.value_lr or lr,
                cfg.grad_clip,
                use_bf16,
                loss_fn=make_value_loss_fn(cfg, v_stats),
            )
    avg = {
        k: float(np.mean([s[k] for s in p_stats])) for k in ("clipfrac", "approx_kl", "ratio_mean")
    }
    avg["ratio_max"] = float(np.max([s["ratio_max"] for s in p_stats]))
    avg.update({k: float(np.mean([s[k] for s in v_stats])) for k in ("vf_loss", "vf_clipfrac")})
    # Sanity check (docs/06): before the first update the policy hasn't changed, so the very
    # first micro-batch must have ratio ~1 and approx_kl ~0. Anything else is a logits/
    # temperature/indexing mismatch between rollout and training.
    avg["first_microbatch_approx_kl"] = p_stats[0]["approx_kl"]
    avg["first_microbatch_ratio_max"] = p_stats[0]["ratio_max"]
    return avg


def masked_var(x, mask):
    mean = masked_mean(x, mask)
    return masked_mean((x - mean) ** 2, mask)


def iteration_metrics(roll, rewards, kl, cfg) -> dict:
    mask = roll["response_mask"]
    kl_sum = kl.sum(dim=1)  # per summary, nats
    explained = 1 - masked_var(roll["returns"] - roll["values_old"], mask) / masked_var(
        roll["returns"], mask
    )
    return {
        "score": roll["raw_score"].mean().item(),  # the RM's opinion
        "score_with_eos_penalty": roll["score"].mean().item(),
        "kl": kl_sum.mean().item(),
        "non_score_reward": (-cfg.beta * kl_sum).mean().item(),
        "rlhf_reward": (roll["score"] - cfg.beta * kl_sum).mean().item(),
        "entropy": masked_mean(roll["entropy"], mask).item(),
        "response_length": mask.sum(dim=1).float().mean().item(),
        "no_eos_fraction": (~roll["has_eot"]).float().mean().item(),
        "explained_variance": explained.item(),
    }


def write_samples(roll, path, n=4):
    enc = get_tokenizer()
    P = roll["prompt_len"]
    lines = []
    for i in range(min(n, roll["input_ids"].shape[0])):
        prompt = [t for t in roll["input_ids"][i, :P].tolist() if t < 50257]
        n_resp = int(roll["response_mask"][i].sum())
        summary = enc.decode([t for t in roll["tokens"][i, :n_resp].tolist() if t != EOT_ID])
        lines.append(
            f"## sample {i} (RM score {roll['raw_score'][i].item():.3f})\n\n```\n{enc.decode(prompt)[-600:]}\n```\n\n"
            f"**Policy:** {summary}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(cfg: PPOConfig):
    device = get_device()
    use_bf16 = cfg.bf16 and device.type == "cuda"
    seed_everything(cfg.seed)
    model_cfg = load_model_config(cfg.model)
    train_prompts = TLDRDataset("train", cfg.tldr_dir)
    iterations = cfg.max_iterations or cfg.total_episodes // cfg.rollout_batch
    ckpt_dir, run_dir = Path(cfg.checkpoint_root) / cfg.run_name, Path(cfg.runs_root) / cfg.run_name

    policy, reference, reward_model, value_model = load_models(cfg, model_cfg, device)
    fused = device.type == "cuda"
    opt_policy = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, eps=cfg.eps, weight_decay=0.0, fused=fused
    )
    opt_value = torch.optim.AdamW(
        value_model.parameters(),
        lr=cfg.value_lr or cfg.lr,
        eps=cfg.eps,
        weight_decay=0.0,
        fused=fused,
    )
    optimizers = {"policy": opt_policy, "value": opt_value}

    it, epoch, pos, train_seconds = 0, 0, 0, 0.0
    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, map_location=device)
        it = restore(state, policy, optimizers)
        value_model.load_state_dict(state["extra"]["value_model"])
        epoch, pos = state["data_state"]["epoch"], state["data_state"]["pos"]
        train_seconds = state["extra"].get("train_seconds", 0.0)

    def order(ep):
        return np.random.default_rng(cfg.seed + ep).permutation(len(train_prompts))

    perm = order(epoch)
    print(
        f"PPO {cfg.model}: beta {cfg.beta}, {iterations} iterations x {cfg.rollout_batch} rollouts, lr {cfg.lr}"
    )
    print(f"resuming from {latest} at iteration {it}" if latest else "starting from SFT")
    save_config(cfg, run_dir / "config.yaml")
    logger = MetricsLogger(run_dir, cfg.run_name, "ppo", asdict(cfg), cfg.use_wandb)

    def checkpoint(name=None, milestone=False):
        if not cfg.save_checkpoints:
            return
        save_checkpoint(
            ckpt_dir,
            step=it,
            model=policy,
            optimizer={} if milestone else optimizers,
            config=asdict(cfg),
            phase="ppo",
            data_state={"epoch": epoch, "pos": pos},
            # milestones keep only the policy (for the over-optimization curve)
            extra={}
            if milestone
            else {"value_model": value_model.state_dict(), "train_seconds": train_seconds},
            name=name,
            keep_last=1,  # ~3 GB each (policy + value + 2 optimizers); only the newest is needed to resume
        )

    preempt = PreemptionFlag()
    while it < iterations:
        t0 = time.time()
        if pos + cfg.rollout_batch > len(perm):
            epoch, pos = epoch + 1, 0
            perm = order(epoch)
        prompts = [train_prompts[int(i)][0] for i in perm[pos : pos + cfg.rollout_batch]]
        pos += cfg.rollout_batch

        roll = rollout(policy, reference, reward_model, value_model, prompts, cfg, device, use_bf16)
        rewards, kl = compute_rewards(
            roll["logp_old"], roll["logp_ref"], roll["score"], roll["response_mask"], cfg.beta
        )
        advantages, returns = compute_gae(
            rewards, roll["values"], roll["response_mask"], cfg.gamma, cfg.lam
        )
        if cfg.whiten_advantages:
            advantages = masked_whiten(advantages, roll["response_mask"])
        roll.update(advantages=advantages, returns=returns, values_old=roll["values"])

        lr = lr_at_step(it, cfg.lr, cfg.warmup_iterations, iterations, cfg.min_lr_ratio)
        update = ppo_update(policy, value_model, opt_policy, opt_value, roll, cfg, lr, use_bf16)
        train_seconds += time.time() - t0
        it += 1

        metrics = {
            **iteration_metrics(roll, rewards, kl, cfg),
            **update,
            "lr": lr,
            "iteration_seconds": time.time() - t0,
        }
        if it % cfg.log_every == 0 or it == 1:
            print(
                f"iter {it}/{iterations} | score {metrics['score']:.3f} | kl {metrics['kl']:.2f} | "
                f"len {metrics['response_length']:.1f} | no_eos {metrics['no_eos_fraction']:.2f} | "
                f"clipfrac {metrics['clipfrac']:.3f} | approx_kl {metrics['approx_kl']:.4f} | "
                f"ev {metrics['explained_variance']:.2f} | {metrics['iteration_seconds']:.0f}s",
                flush=True,
            )
            logger.log(metrics, it)
        if cfg.sample_every and (it % cfg.sample_every == 0 or it == 1):
            write_samples(roll, run_dir / "samples" / f"iter_{it:04d}.md")
        if cfg.milestone_every and it % cfg.milestone_every == 0:
            checkpoint(name=f"iter_{it:04d}.pt", milestone=True)
        if (cfg.ckpt_every and it % cfg.ckpt_every == 0) or preempt.set:
            checkpoint()
        if preempt.set:
            print(f"preempted: saved at iteration {it}; resubmit to continue")
            return

    checkpoint()
    checkpoint(name="final.pt", milestone=True)

    # ---- final evaluation (same posts and definitions as SFT/DPO) ----
    tldr_val = TLDRDataset("valid", cfg.tldr_dir)
    eval_idx = json.loads((Path(cfg.tldr_dir) / "val_subset.json").read_text())["indices"][
        : cfg.eval_examples
    ]
    final = {
        f"ppo_{k}": v
        for k, v in generation_eval(
            policy, tldr_val, eval_idx, cfg, device, use_bf16, run_dir / "eval_samples.md"
        ).items()
    }
    for name, model in (("ppo", policy), ("sft", reference)):
        final[f"{name}_rm_score"] = float(
            np.mean(
                rm_score(
                    reward_model,
                    greedy_pairs(model, tldr_val, eval_idx, cfg, device, use_bf16),
                    device,
                    use_bf16,
                )
            )
        )
    rng = np.random.default_rng(cfg.seed)
    kl_idx = rng.choice(len(tldr_val), size=min(cfg.kl_prompts, len(tldr_val)), replace=False)
    final["kl_from_sft"], _ = estimate_kl(
        policy,
        reference,
        [tldr_val[int(i)][0] for i in kl_idx],
        device,
        cfg.temperature,
        use_bf16=use_bf16,
    )
    final["kl_prompts"] = len(kl_idx)
    logger.log({f"final_{k}": v for k, v in final.items() if isinstance(v, float)}, it)
    summary = {
        "model": cfg.model,
        "beta": cfg.beta,
        "lr": cfg.lr,
        "iterations": it,
        "episodes": it * cfg.rollout_batch,
        **final,
        "gpu_hours": round(train_seconds / 3600, 3),
        "hardware": hardware_info(),
        "config": asdict(cfg),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    if cfg.write_results:
        write_results(cfg.run_name, summary)
    delete_rolling_checkpoints(ckpt_dir)  # finished: keep final.pt (and PPO milestones) only
    logger.finish()


if __name__ == "__main__":
    main(load_config(PPOConfig))
