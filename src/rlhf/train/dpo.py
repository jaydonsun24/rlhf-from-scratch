"""DPO baseline (docs/05-dpo.md): train the SFT model directly on the human comparisons.

    sbatch scripts/slurm/dpo.slurm lr=1e-6 max_steps=300 run_name=dpo_lr1e-6 write_results=false
    sbatch scripts/slurm/dpo.slurm lr=<best>

Two models: the policy (trained, starts as SFT) and the reference (a frozen copy of SFT).
For each comparison, the loss raises the chosen summary's log-probability relative to the
reference and lowers the rejected one's (rl/losses.py: dpo_loss). No reward model, no sampling
during training; the training step is the same train_step as every other phase.

Logged every step (the DPO-specific diagnostics):
  pref_accuracy       how often log-ratio(chosen) > log-ratio(rejected) in the batch
  reward_chosen/rej   beta * log-ratio: DPO's implicit rewards
  logp_chosen/rej     the policy's raw log-probs. A known DPO effect is that BOTH fall
                      (the margin grows because rejected falls faster); worth plotting.

At the end, on held-out data:
  - preference accuracy and DPO loss on 2,000 held-out comparisons,
  - greedy summaries for the same 1,000 posts as SFT's eval: length, EOT rate, ROUGE,
    and reward-model score for DPO vs. SFT (RM score is a diagnostic: DPO never saw the RM),
  - KL from SFT on 200 posts (eval/kl.py), samples side by side.
"""

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rlhf.data.preferences import PrefsDataset, collate_pairs
from rlhf.data.tldr import RESPONSE_CAP, TLDRDataset, collate
from rlhf.eval.kl import estimate_kl
from rlhf.model.generate import generate
from rlhf.model.gpt import GPT
from rlhf.model.reward_model import RewardModel
from rlhf.rl.logprobs import sequence_logprobs
from rlhf.rl.losses import dpo_loss
from rlhf.train.lr_schedule import lr_at_step
from rlhf.train.reward import score
from rlhf.train.sft import generation_eval
from rlhf.train.step import train_step
from rlhf.utils.checkpoint import (
    PreemptionFlag,
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
class DPOConfig:
    model: str = "gpt124m"
    init_from: str = "checkpoints/sft/final.pt"  # policy AND reference start here; empty = random
    rm_checkpoint: str = "checkpoints/rm/final.pt"  # only for the diagnostic RM score; empty = skip
    data_dir: str = "data/prefs"
    tldr_dir: str = "data/tldr"
    run_name: str = "dpo"
    checkpoint_root: str = "checkpoints"
    runs_root: str = "runs"
    seed: int = 1337
    beta: float = 0.1
    batch_size: int = 64  # comparisons per optimizer step
    micro_batch_size: int = 8
    epochs: int = 1
    max_steps: int = 0  # > 0 overrides epochs (LR sweep)
    lr: float = 3e-6  # docs/05: sweep 1e-6 .. 5e-6
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    eps: float = 1e-5
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 200  # held-out preference accuracy on 2,000 comparisons
    eval_batch_size: int = 32
    ckpt_every: int = 200
    # final evaluation sizes
    eval_examples: int = 1000  # TL;DR posts for generation (same subset as SFT's eval)
    kl_prompts: int = 200
    bf16: bool = True
    use_wandb: bool = True
    write_results: bool = True


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def pair_logprobs(model, batch):
    """Summed log-probs of each row's summary. Rows [0, n) are chosen, [n, 2n) rejected."""
    logits = model(batch["input_ids"], batch["position_ids"], batch["attention_mask"])
    return sequence_logprobs(logits, batch["input_ids"], batch["response_mask"])


def make_loss_fn(reference, beta, stats):
    """loss_fn for train_step. The reference is frozen: its log-probs are computed without
    gradients each micro-batch. Diagnostics are appended to `stats` (detached)."""

    def loss_fn(model, batch):
        n = batch["n_pairs"]
        lp = pair_logprobs(model, batch)
        with torch.no_grad():
            ref = pair_logprobs(reference, batch)
        loss = dpo_loss(lp[:n], lp[n:], ref[:n], ref[n:], beta)
        with torch.no_grad():
            ratio_c, ratio_r = lp[:n] - ref[:n], lp[n:] - ref[n:]
            stats.append(
                {
                    "pref_accuracy": (ratio_c > ratio_r).float().mean().item(),
                    "reward_chosen": (beta * ratio_c).mean().item(),
                    "reward_rejected": (beta * ratio_r).mean().item(),
                    "reward_margin": (beta * (ratio_c - ratio_r)).mean().item(),
                    "logp_chosen": lp[:n].mean().item(),
                    "logp_rejected": lp[n:].mean().item(),
                }
            )
        return loss

    return loss_fn


@torch.no_grad()
def heldout_preferences(policy, reference, prefs, indices, cfg, device, use_bf16) -> dict:
    """DPO loss and implicit-reward accuracy on held-out comparisons."""
    policy.eval()
    stats = []
    loss_fn = make_loss_fn(reference, cfg.beta, stats)
    losses = []
    for start in range(0, len(indices), cfg.eval_batch_size):
        batch = to_device(
            collate_pairs([prefs[i] for i in indices[start : start + cfg.eval_batch_size]]), device
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            losses.append(loss_fn(policy, batch).item())
    policy.train()
    return {
        "heldout_dpo_loss": float(np.mean(losses)),
        "heldout_pref_accuracy": float(np.mean([s["pref_accuracy"] for s in stats])),
    }


def load_gpt(path, model_cfg):
    model = GPT(model_cfg)
    if path:
        model.load_state_dict(load_checkpoint(path, map_location="cpu")["model"])
    return model


def main(cfg: DPOConfig):
    device = get_device()
    use_bf16 = cfg.bf16 and device.type == "cuda"
    seed_everything(cfg.seed)
    train, valid = PrefsDataset("train", cfg.data_dir), PrefsDataset("valid", cfg.data_dir)
    val_subset = json.loads((Path(cfg.data_dir) / "val_subset.json").read_text())["indices"]
    max_steps = cfg.max_steps or (len(train) // cfg.batch_size) * cfg.epochs
    warmup = max(1, int(cfg.warmup_ratio * max_steps))
    ckpt_dir, run_dir = Path(cfg.checkpoint_root) / cfg.run_name, Path(cfg.runs_root) / cfg.run_name
    model_cfg = load_model_config(cfg.model)

    policy = load_gpt(cfg.init_from, model_cfg).to(device)
    reference = copy.deepcopy(policy).eval()  # frozen SFT copy
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )

    step, epoch, pos, train_seconds = 0, 0, 0, 0.0
    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, map_location=device)
        step = restore(state, policy, optimizer)
        epoch, pos = state["data_state"]["epoch"], state["data_state"]["pos"]
        train_seconds = state["extra"].get("train_seconds", 0.0)

    def order(ep):
        return np.random.default_rng(cfg.seed + ep).permutation(len(train))

    perm = order(epoch)
    print(
        f"DPO {cfg.model}: beta {cfg.beta}, {len(train):,} comparisons, {max_steps} steps x {cfg.batch_size}"
    )
    print(
        f"resuming from {latest} at step {step}"
        if latest
        else f"policy and reference from {cfg.init_from or 'random init'}"
    )
    save_config(cfg, run_dir / "config.yaml")
    logger = MetricsLogger(run_dir, cfg.run_name, "dpo", asdict(cfg), cfg.use_wandb)

    def checkpoint(name=None):
        save_checkpoint(
            ckpt_dir,
            step=step,
            model=policy,
            optimizer=optimizer,
            config=asdict(cfg),
            phase="dpo",
            data_state={"epoch": epoch, "pos": pos},
            extra={"train_seconds": train_seconds},
            name=name,
        )

    preempt = PreemptionFlag()
    policy.train()
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
        stats = []
        lr = lr_at_step(step, cfg.lr, warmup, max_steps, cfg.min_lr_ratio)
        out = train_step(
            policy,
            optimizer,
            micro,
            lr,
            cfg.grad_clip,
            use_bf16,
            loss_fn=make_loss_fn(reference, cfg.beta, stats),
        )
        train_seconds += time.time() - t0
        step += 1

        if step % cfg.log_every == 0 or step == 1:
            diag = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            print(
                f"step {step}/{max_steps} | loss {out['loss']:.4f} | acc {diag['pref_accuracy']:.3f} | "
                f"margin {diag['reward_margin']:.3f} | logp c/r {diag['logp_chosen']:.1f}/{diag['logp_rejected']:.1f}",
                flush=True,
            )
            logger.log({**out, **diag, "lr": lr}, step)
        if cfg.eval_every and step % cfg.eval_every == 0 and step < max_steps:
            held = heldout_preferences(policy, reference, valid, val_subset, cfg, device, use_bf16)
            print(
                f"step {step} | heldout_pref_accuracy {held['heldout_pref_accuracy']:.4f}",
                flush=True,
            )
            logger.log(held, step)
        if (cfg.ckpt_every and step % cfg.ckpt_every == 0) or preempt.set:
            checkpoint()
        if preempt.set:
            print(f"preempted: saved at step {step}; resubmit to continue")
            return

    checkpoint()
    checkpoint(name="final.pt")

    # ---- final evaluation ----
    final = heldout_preferences(policy, reference, valid, val_subset, cfg, device, use_bf16)
    tldr_val = TLDRDataset("valid", cfg.tldr_dir)
    eval_idx = json.loads((Path(cfg.tldr_dir) / "val_subset.json").read_text())["indices"][
        : cfg.eval_examples
    ]

    # generation_eval (from sft.py) only reads cfg.eval_batch_size, which DPOConfig also has.
    dpo_gen = generation_eval(
        policy, tldr_val, eval_idx, cfg, device, use_bf16, run_dir / "samples.md"
    )
    sft_gen = generation_eval(
        reference, tldr_val, eval_idx, cfg, device, use_bf16, run_dir / "samples_sft.md"
    )
    final.update({f"dpo_{k}": v for k, v in dpo_gen.items()})
    final.update({f"sft_{k}": v for k, v in sft_gen.items() if k != "n_eval"})

    if cfg.rm_checkpoint:  # diagnostic: what does the reward model think of each model's summaries?
        rm = RewardModel(model_cfg)
        rm.load_state_dict(load_checkpoint(cfg.rm_checkpoint, map_location="cpu")["model"])
        rm.to(device)
        for name, model in (("dpo", policy), ("sft", reference)):
            final[f"{name}_rm_score"] = float(
                np.mean(
                    score(
                        rm,
                        greedy_pairs(model, tldr_val, eval_idx, cfg, device, use_bf16),
                        device,
                        use_bf16,
                    )
                )
            )

    rng = np.random.default_rng(cfg.seed)
    kl_idx = rng.choice(len(tldr_val), size=min(cfg.kl_prompts, len(tldr_val)), replace=False)
    gen = torch.Generator().manual_seed(cfg.seed) if device.type == "cpu" else None
    final["kl_from_sft"], _ = estimate_kl(
        policy,
        reference,
        [tldr_val[int(i)][0] for i in kl_idx],
        device,
        use_bf16=use_bf16,
        generator=gen,
    )
    final["kl_prompts"] = len(kl_idx)

    logger.log({f"final_{k}": v for k, v in final.items() if isinstance(v, float)}, step)
    summary = {
        "model": cfg.model,
        "beta": cfg.beta,
        "lr": cfg.lr,
        "steps": step,
        "pairs_seen": step * cfg.batch_size,
        **final,
        "gpu_hours": round(train_seconds / 3600, 3),
        "hardware": hardware_info(),
        "config": asdict(cfg),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    if cfg.write_results:
        write_results("dpo", summary)
    logger.finish()


@torch.no_grad()
def greedy_pairs(model, tldr_val, indices, cfg, device, use_bf16):
    """(prompt, greedy summary) pairs for the RM to score."""
    model.eval()
    pairs = []
    for start in range(0, len(indices), cfg.eval_batch_size):
        prompts = [tldr_val[i][0] for i in indices[start : start + cfg.eval_batch_size]]
        batch = to_device(collate([(p, []) for p in prompts]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = generate(
                model, batch["input_ids"], batch["attention_mask"], max_new_tokens=RESPONSE_CAP
            )
        for row, p in enumerate(prompts):
            pairs.append((p, out["tokens"][row, : int(out["response_mask"][row].sum())].tolist()))
    return pairs


if __name__ == "__main__":
    main(load_config(DPOConfig))
