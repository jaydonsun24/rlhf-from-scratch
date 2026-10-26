"""Train the reward model on human preference comparisons (docs/04-reward-model.md).

    # Adroit: LR sweep (short runs), then the full epoch
    sbatch scripts/slurm/rm.slurm lr=3e-6 max_steps=300 run_name=rm_lr3e-6 write_results=false save_checkpoints=false
    sbatch scripts/slurm/rm.slurm lr=<best>

Starts from the SFT model (backbone) plus a fresh scalar head, and trains with the
Bradley-Terry loss so chosen summaries score higher than rejected ones. Uses the same
train_step as pretraining and SFT, given the reward-model loss via loss_fn.

After training:
  1. Normalize: shift the head's bias so human reference summaries average reward 0.
     PPO's reward scale (and its -1 penalty for never stopping) assumes this.
  2. Evaluate: accuracy on all held-out human comparisons, accuracy by how sure the labeler
     was, and length bias (Spearman correlation of reward vs. length on SFT samples).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rlhf.data.preferences import PrefsDataset, collate_pairs
from rlhf.data.tldr import RESPONSE_CAP, TLDRDataset, collate
from rlhf.model.generate import generate
from rlhf.model.gpt import GPT
from rlhf.model.reward_model import RewardModel, reward_at_eot
from rlhf.rl.losses import bradley_terry_loss
from rlhf.train.lr_schedule import lr_at_step
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
class RMConfig:
    model: str = "gpt124m"
    init_from: str = "checkpoints/sft/final.pt"  # SFT weights for the backbone; empty = random
    data_dir: str = "data/prefs"
    tldr_dir: str = "data/tldr"  # reference summaries (normalization) and prompts (length check)
    run_name: str = "rm"
    checkpoint_root: str = "checkpoints"
    runs_root: str = "runs"
    seed: int = 1337
    # 64 comparisons (128 sequences) per optimizer step
    batch_size: int = 64
    micro_batch_size: int = 8
    epochs: int = 1
    max_steps: int = 0  # > 0 overrides epochs (LR sweep)
    lr: float = 1e-5  # docs/04 sweep {3e-6, 1e-5, 3e-5}, pick by val accuracy
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    eps: float = 1e-5
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 200  # accuracy on data/prefs/val_subset.json (2,000 comparisons)
    eval_batch_size: int = 32  # comparisons per eval batch
    ckpt_every: int = 200
    # after training
    final_eval_pairs: int = 0  # 0 = all held-out comparisons (~84K)
    norm_examples: int = 5000  # SFT-train reference summaries used to center rewards
    length_samples: int = 1000  # SFT samples for the length-bias check
    bf16: bool = True
    use_wandb: bool = True
    write_results: bool = True
    save_checkpoints: bool = (
        True  # sweep runs: false (their numbers are in runs/<name>/summary.json)
    )


def rm_batch_loss(model, batch):
    """loss_fn for train_step: rows [0, n) are chosen summaries, rows [n, 2n) rejected."""
    scores = model(batch["input_ids"], batch["position_ids"], batch["attention_mask"])
    rewards = reward_at_eot(scores, batch["attention_mask"])  # (2n,)
    n = batch["n_pairs"]
    return bradley_terry_loss(rewards[:n], rewards[n:])


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def score(model, examples, device, use_bf16, batch_size=64) -> np.ndarray:
    """Reward for each (prompt, response) pair, in order."""
    model.eval()
    out = []
    for start in range(0, len(examples), batch_size):
        batch = to_device(collate(examples[start : start + batch_size]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            scores = model(batch["input_ids"], batch["position_ids"], batch["attention_mask"])
        out.append(reward_at_eot(scores.float(), batch["attention_mask"]).cpu())
    model.train()
    return torch.cat(out).numpy()


def pair_accuracy(model, prefs, indices, cfg, device, use_bf16) -> dict:
    """How often the RM gives the human-chosen summary the higher reward, overall and by certainty."""
    items = [prefs[i] for i in indices]
    chosen = score(
        model, [(p, c) for p, c, _, _ in items], device, use_bf16, cfg.eval_batch_size * 2
    )
    rejected = score(
        model, [(p, r) for p, _, r, _ in items], device, use_bf16, cfg.eval_batch_size * 2
    )
    correct = chosen > rejected
    cert = np.array([k for _, _, _, k in items])
    result = {
        "accuracy": float(correct.mean()),
        "n": len(items),
        "mean_margin": float((chosen - rejected).mean()),
    }
    for k in (1, 2, 3, 4):  # 1 = labeler barely preferred it ... 4 = labeler was certain
        if (cert == k).any():
            result[f"accuracy_certainty_{k}"] = float(correct[cert == k].mean())
            result[f"n_certainty_{k}"] = int((cert == k).sum())
    return result


def spearman(x, y) -> float:
    """Rank correlation: Pearson correlation of the ranks (ties broken by order; fine for continuous rewards)."""
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(cfg: RMConfig):
    device = get_device()
    use_bf16 = cfg.bf16 and device.type == "cuda"
    seed_everything(cfg.seed)
    train, valid = PrefsDataset("train", cfg.data_dir), PrefsDataset("valid", cfg.data_dir)
    val_subset = json.loads((Path(cfg.data_dir) / "val_subset.json").read_text())["indices"]
    assert cfg.batch_size % cfg.micro_batch_size == 0
    max_steps = cfg.max_steps or (len(train) // cfg.batch_size) * cfg.epochs
    warmup = max(1, int(cfg.warmup_ratio * max_steps))
    ckpt_dir, run_dir = Path(cfg.checkpoint_root) / cfg.run_name, Path(cfg.runs_root) / cfg.run_name

    model_cfg = load_model_config(cfg.model)
    model = RewardModel(model_cfg)
    if cfg.init_from:
        model.backbone.load_state_dict(load_checkpoint(cfg.init_from, map_location="cpu")["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )

    step, epoch, pos, train_seconds = 0, 0, 0, 0.0
    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, map_location=device)
        step = restore(state, model, optimizer)
        epoch, pos = state["data_state"]["epoch"], state["data_state"]["pos"]
        train_seconds = state["extra"].get("train_seconds", 0.0)

    def order(ep):
        return np.random.default_rng(cfg.seed + ep).permutation(len(train))

    perm = order(epoch)
    print(f"RM {cfg.model}: {len(train):,} comparisons, {max_steps} steps x {cfg.batch_size} pairs")
    print(
        f"resuming from {latest} at step {step}"
        if latest
        else f"backbone from {cfg.init_from or 'random init'}"
    )
    save_config(cfg, run_dir / "config.yaml")
    logger = MetricsLogger(run_dir, cfg.run_name, "rm", asdict(cfg), cfg.use_wandb)

    def checkpoint(name=None, extra=None):
        if not cfg.save_checkpoints:
            return
        save_checkpoint(
            ckpt_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            config=asdict(cfg),
            phase="rm",
            data_state={"epoch": epoch, "pos": pos},
            extra={"train_seconds": train_seconds, **(extra or {})},
            name=name,
        )

    preempt = PreemptionFlag()
    model.train()
    while step < max_steps:
        t0 = time.time()
        if pos + cfg.batch_size > len(perm):
            epoch, pos = epoch + 1, 0
            perm = order(epoch)
        idx = perm[pos : pos + cfg.batch_size]
        pos += cfg.batch_size
        micro = [
            to_device(
                collate_pairs([train[int(i)] for i in idx[m : m + cfg.micro_batch_size]]), device
            )
            for m in range(0, cfg.batch_size, cfg.micro_batch_size)
        ]
        lr = lr_at_step(step, cfg.lr, warmup, max_steps, cfg.min_lr_ratio)
        out = train_step(
            model, optimizer, micro, lr, cfg.grad_clip, use_bf16, loss_fn=rm_batch_loss
        )
        train_seconds += time.time() - t0
        step += 1

        if step % cfg.log_every == 0 or step == 1:
            print(
                f"step {step}/{max_steps} | loss {out['loss']:.4f} | lr {lr:.2e} | grad_norm {out['grad_norm']:.3f}",
                flush=True,
            )
            logger.log({**out, "lr": lr}, step)
        if cfg.eval_every and step % cfg.eval_every == 0 and step < max_steps:
            acc = pair_accuracy(model, valid, val_subset, cfg, device, use_bf16)
            print(f"step {step} | val_accuracy {acc['accuracy']:.4f}", flush=True)
            logger.log({"val_accuracy": acc["accuracy"]}, step)
        if (cfg.ckpt_every and step % cfg.ckpt_every == 0) or preempt.set:
            checkpoint()
        if preempt.set:
            print(f"preempted: saved at step {step}; resubmit to continue")
            return

    # ---- 1. Normalize: human reference summaries should average reward 0 ----
    tldr_train = TLDRDataset("train", cfg.tldr_dir)
    rng = np.random.default_rng(cfg.seed)
    norm_idx = rng.choice(
        len(tldr_train), size=min(cfg.norm_examples, len(tldr_train)), replace=False
    )
    ref_rewards = score(model, [tldr_train[int(i)] for i in norm_idx], device, use_bf16)
    offset = float(ref_rewards.mean())
    with torch.no_grad():
        model.head.bias -= offset  # every reward shifts down by `offset`
    checkpoint(extra={"reward_offset": offset})
    checkpoint(name="final.pt", extra={"reward_offset": offset})

    # ---- 2. Accuracy on held-out human comparisons ----
    eval_idx = list(range(len(valid)))
    if cfg.final_eval_pairs:
        eval_idx = sorted(
            rng.choice(
                len(valid), size=min(cfg.final_eval_pairs, len(valid)), replace=False
            ).tolist()
        )
    final = pair_accuracy(model, valid, eval_idx, cfg, device, use_bf16)

    # ---- 3. Length bias: does the RM simply prefer longer summaries? ----
    policy = GPT(model_cfg)
    if cfg.init_from:
        policy.load_state_dict(load_checkpoint(cfg.init_from, map_location="cpu")["model"])
    policy.to(device).eval()
    tldr_valid = TLDRDataset("valid", cfg.tldr_dir)
    prompts = [
        tldr_valid[int(i)][0]
        for i in rng.choice(
            len(tldr_valid), size=min(cfg.length_samples, len(tldr_valid)), replace=False
        )
    ]
    samples = []
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for start in range(0, len(prompts), cfg.eval_batch_size):
        chunk = prompts[start : start + cfg.eval_batch_size]
        batch = to_device(collate([(p, []) for p in chunk]), device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = generate(
                policy,
                batch["input_ids"],
                batch["attention_mask"],
                max_new_tokens=RESPONSE_CAP,
                temperature=0.7,
                generator=gen if device.type == "cpu" else None,
            )
        for row, p in enumerate(chunk):
            n = int(out["response_mask"][row].sum())
            samples.append((p, out["tokens"][row, :n].tolist()))
    sample_rewards = score(model, samples, device, use_bf16)
    lengths = np.array([len(r) for _, r in samples])
    ref_after = score(model, [tldr_train[int(i)] for i in norm_idx[:1000]], device, use_bf16)

    final.update(
        {
            "reward_offset": offset,
            "reference_reward_mean_after_norm": float(ref_after.mean()),
            "sft_sample_reward_mean": float(sample_rewards.mean()),
            "sft_sample_reward_std": float(sample_rewards.std()),
            "length_spearman": spearman(sample_rewards, lengths),
            "n_length_samples": len(samples),
        }
    )
    (run_dir / "reward_distributions.json").write_text(
        json.dumps(
            {
                "reference": ref_after.tolist(),
                "sft_samples": sample_rewards.tolist(),
                "sft_lengths": lengths.tolist(),
            }
        )
    )
    logger.log({f"final_{k}": v for k, v in final.items() if isinstance(v, float)}, step)
    summary = {
        "model": cfg.model,
        "init_from": cfg.init_from,
        "lr": cfg.lr,
        "steps": step,
        "pairs_seen": step * cfg.batch_size,
        "train_pairs": len(train),
        **final,
        "gpu_hours": round(train_seconds / 3600, 3),
        "hardware": hardware_info(),
        "config": asdict(cfg),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    if cfg.write_results:
        write_results("reward_model", summary)
    delete_rolling_checkpoints(ckpt_dir)  # finished: keep final.pt (and PPO milestones) only
    logger.finish()


if __name__ == "__main__":
    main(load_config(RMConfig))
