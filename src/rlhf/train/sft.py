"""Supervised fine-tuning on Reddit TL;DR.

    # laptop smoke test (the real model; the tiny one's 256-token context is too short for TL;DR)
    python -m rlhf.train.sft --config configs/sft.yaml run_name=sft_smoke max_steps=3 \
        batch_size=8 micro_batch_size=4 eval_every=0 eval_examples=16 eval_batch_size=8 \
        use_wandb=false write_results=false
    # Adroit: LR sweep (3 short runs), then the full epoch
    sbatch scripts/slurm/sft.slurm lr=1e-5 max_steps=200 run_name=sft_lr1e-5 write_results=false save_checkpoints=false
    sbatch scripts/slurm/sft.slurm lr=<best>

Starts from the pretrained model and keeps training it on (post, summary) pairs, grading
only the summary tokens (rl/losses.py: sft_loss). The training step is the same one
pretraining uses (train/step.py), given the SFT loss via loss_fn.

At the end it evaluates on a fixed 1,000-post validation subset: teacher-forced val loss,
greedy summaries -> ROUGE vs. the human summaries, how often the model stops by itself (EOT),
average summary length, and 10 side-by-side samples.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rlhf.data.tldr import RESPONSE_CAP, TLDRDataset, collate
from rlhf.data.tokenizer import EOT_ID, get_tokenizer
from rlhf.eval.rouge import rouge_scores
from rlhf.model.generate import generate
from rlhf.model.gpt import GPT
from rlhf.rl.losses import sft_loss
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
class SFTConfig:
    model: str = "gpt124m"
    init_from: str = "checkpoints/pretrain/final.pt"  # empty = random init (smoke tests only)
    data_dir: str = "data/tldr"
    run_name: str = "sft"
    checkpoint_root: str = "checkpoints"
    runs_root: str = "runs"
    seed: int = 1337
    # 128 sequences per optimizer step, built from micro-batches
    batch_size: int = 128
    micro_batch_size: int = 16
    epochs: int = 1
    max_steps: int = 0  # > 0 overrides epochs (used for the short LR-sweep runs)
    lr: float = 3e-5  # middle of the docs' sweep {1e-5, 3e-5, 1e-4}
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    eps: float = 1e-5
    grad_clip: float = 1.0
    # evaluation
    log_every: int = 10
    eval_every: int = 100  # teacher-forced val loss
    eval_examples: int = 1000  # from data/tldr/val_subset.json
    eval_batch_size: int = 32
    ckpt_every: int = 200
    bf16: bool = True
    use_wandb: bool = True
    write_results: bool = True  # sweep runs turn this off; only the final run writes results/
    save_checkpoints: bool = (
        True  # sweep runs: false (their numbers are in runs/<name>/summary.json)
    )


def sft_batch_loss(model, batch):
    """loss_fn for train_step: one micro-batch in the shared layout -> mean loss on summary tokens."""
    logits = model(batch["input_ids"], batch["position_ids"], batch["attention_mask"])
    return sft_loss(logits, batch["input_ids"], batch["response_mask"])


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_initial_model(cfg: SFTConfig, device):
    model = GPT(load_model_config(cfg.model))
    if cfg.init_from:
        state = load_checkpoint(cfg.init_from, map_location="cpu")
        model.load_state_dict(state["model"])
    return model.to(device)


def eval_indices(cfg: SFTConfig) -> list[int]:
    subset = json.loads((Path(cfg.data_dir) / "val_subset.json").read_text())["indices"]
    return subset[: cfg.eval_examples]


@torch.no_grad()
def val_loss(model, val, indices, cfg, device, use_bf16) -> float:
    """Mean summary-token loss on the human references (teacher forcing)."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(indices), cfg.eval_batch_size):
        chunk = indices[start : start + cfg.eval_batch_size]
        batch = to_device(collate([val[i] for i in chunk]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = sft_batch_loss(model, batch)
        n_tokens = batch["response_mask"][:, 1:].sum().item()
        total, count = total + loss.item() * n_tokens, count + n_tokens
    model.train()
    return total / count


@torch.no_grad()
def generation_eval(model, val, indices, cfg, device, use_bf16, sample_path=None) -> dict:
    """Greedy summaries for the eval posts: ROUGE vs. references, EOT rate, mean length."""
    model.eval()
    enc = get_tokenizer()
    predictions, references, lengths, stopped = [], [], [], []
    for start in range(0, len(indices), cfg.eval_batch_size):
        chunk = indices[start : start + cfg.eval_batch_size]
        prompts = collate([(val[i][0], []) for i in chunk])
        prompts = to_device(prompts, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = generate(
                model, prompts["input_ids"], prompts["attention_mask"], max_new_tokens=RESPONSE_CAP
            )
        for row, i in enumerate(chunk):
            n = int(out["response_mask"][row].sum())
            tokens = out["tokens"][row, :n].tolist()
            stopped.append(EOT_ID in tokens)
            lengths.append(n)
            predictions.append(enc.decode([t for t in tokens if t != EOT_ID]).strip())
            references.append(enc.decode(val[i][1][:-1]).strip())  # drop the reference's EOT
    model.train()
    if sample_path is not None:
        lines = ["# SFT samples (greedy) vs. human summaries\n"]
        for i, pred, ref in zip(indices[:10], predictions, references, strict=False):
            post = enc.decode(val[i][0])
            lines.append(
                f"## Example {i}\n\n```\n{post}\n```\n\n**Model:** {pred}\n\n**Human:** {ref}\n"
            )
        Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
        Path(sample_path).write_text("\n".join(lines))
    return {
        **rouge_scores(predictions, references),
        "eot_rate": float(np.mean(stopped)),  # should be > 0.95
        "mean_length": float(np.mean(lengths)),
        "n_eval": len(indices),
    }


def main(cfg: SFTConfig):
    device = get_device()
    use_bf16 = cfg.bf16 and device.type == "cuda"
    seed_everything(cfg.seed)
    train, val = TLDRDataset("train", cfg.data_dir), TLDRDataset("valid", cfg.data_dir)
    eval_idx = eval_indices(cfg)
    assert cfg.batch_size % cfg.micro_batch_size == 0
    steps_per_epoch = len(train) // cfg.batch_size
    max_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_ratio * max_steps))
    ckpt_dir, run_dir = Path(cfg.checkpoint_root) / cfg.run_name, Path(cfg.runs_root) / cfg.run_name

    # The longest prompt + summary in the data must fit in the model's context (RoPE table).
    longest = max(ds.longest() for ds in (train, val))
    context_len = load_model_config(cfg.model).context_len
    if longest > context_len:
        raise ValueError(
            f"model '{cfg.model}' has context {context_len} but the data has {longest}-token "
            "sequences; use gpt124m (1024)"
        )
    model = load_initial_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )

    # Data order: a fresh seeded shuffle of the training set each epoch. The position is
    # (epoch, pos): saved in checkpoints so a resumed run continues with the next example.
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
    print(
        f"SFT {cfg.model}: {len(train):,} examples, {max_steps} steps x {cfg.batch_size} sequences"
    )
    print(
        f"resuming from {latest} at step {step}"
        if latest
        else f"starting from {cfg.init_from or 'random init'}"
    )
    save_config(cfg, run_dir / "config.yaml")
    logger = MetricsLogger(run_dir, cfg.run_name, "sft", asdict(cfg), cfg.use_wandb)

    def checkpoint(name=None):
        if not cfg.save_checkpoints:
            return
        save_checkpoint(
            ckpt_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            config=asdict(cfg),
            phase="sft",
            data_state={"epoch": epoch, "pos": pos},
            extra={"train_seconds": train_seconds},
            name=name,
        )

    preempt = PreemptionFlag()
    model.train()
    while step < max_steps:
        t0 = time.time()
        if pos + cfg.batch_size > len(perm):  # epoch finished: reshuffle
            epoch, pos = epoch + 1, 0
            perm = order(epoch)
        idx = perm[pos : pos + cfg.batch_size]
        pos += cfg.batch_size
        micro = [
            to_device(collate([train[int(i)] for i in idx[m : m + cfg.micro_batch_size]]), device)
            for m in range(0, cfg.batch_size, cfg.micro_batch_size)
        ]
        lr = lr_at_step(step, cfg.lr, warmup, max_steps, cfg.min_lr_ratio)
        out = train_step(
            model, optimizer, micro, lr, cfg.grad_clip, use_bf16, loss_fn=sft_batch_loss
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
            vl = val_loss(model, val, eval_idx, cfg, device, use_bf16)
            print(f"step {step} | val_loss {vl:.4f}", flush=True)
            logger.log({"val_loss": vl}, step)
        if (cfg.ckpt_every and step % cfg.ckpt_every == 0) or preempt.set:
            checkpoint()
        if preempt.set:
            print(f"preempted: saved at step {step}; resubmit to continue")
            return

    checkpoint()
    checkpoint(name="final.pt")
    final = {"val_loss": val_loss(model, val, eval_idx, cfg, device, use_bf16)}
    final.update(
        generation_eval(model, val, eval_idx, cfg, device, use_bf16, run_dir / "samples.md")
    )
    logger.log(final, step)
    summary = {
        "model": cfg.model,
        "init_from": cfg.init_from,
        "lr": cfg.lr,
        "steps": step,
        "examples_seen": step * cfg.batch_size,
        **final,
        "gpu_hours": round(train_seconds / 3600, 3),
        "hardware": hardware_info(),
        "config": asdict(cfg),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))  # every run, incl. sweeps
    print(summary)
    if cfg.write_results:
        write_results("sft", summary)
    delete_rolling_checkpoints(ckpt_dir)  # finished: keep final.pt (and PPO milestones) only
    logger.finish()


if __name__ == "__main__":
    main(load_config(SFTConfig))
