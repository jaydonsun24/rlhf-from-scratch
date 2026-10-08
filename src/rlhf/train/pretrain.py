"""Pretrain the GPT from random weights on FineWeb-Edu (docs/02-pretraining.md).

    # laptop smoke test
    python -m rlhf.train.pretrain --config configs/pretrain.yaml model=tiny max_steps=20 \
        micro_batch_size=4 seq_len=64 total_batch_tokens=1024 compile=false use_wandb=false
    # Adroit
    sbatch scripts/slurm/pretrain.slurm

The core of each step lives in train/step.py and train/lr_schedule.py. This file is the
loop around them: setup, resume, logging, evals, samples, checkpoints.

Resumable: every checkpoint holds the model, optimizer, step (which also fixes the LR,
since the schedule is a pure function of the step), data position and RNG state. Starting
the script again picks up the latest checkpoint automatically.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from rlhf.data.shards import ShardLoader
from rlhf.data.tokenizer import get_tokenizer
from rlhf.model.generate import generate
from rlhf.model.gpt import GPT
from rlhf.train.lr_schedule import lr_at_step
from rlhf.train.step import train_step
from rlhf.utils.checkpoint import (
    PreemptionFlag,
    latest_checkpoint,
    load_checkpoint,
    restore,
    save_checkpoint,
)
from rlhf.utils.config import load_config, load_model_config, save_config
from rlhf.utils.distributed import cleanup_distributed, setup_distributed
from rlhf.utils.logging import MetricsLogger, hardware_info, write_results
from rlhf.utils.seed import seed_everything

SAMPLE_PROMPTS = [
    "The most important thing to understand about photosynthesis is",
    "In 1905, Albert Einstein",
    "A good way to learn a new programming language is",
    "The main causes of the French Revolution were",
    "Water boils at a lower temperature at high altitude because",
]


@dataclass
class PretrainConfig:
    model: str = "gpt124m"
    data_dir: str = "data/fineweb"
    run_name: str = "pretrain"
    checkpoint_root: str = "checkpoints"
    runs_root: str = "runs"
    seed: int = 1337
    # batch: total_batch_tokens per optimizer step, built from micro-batches
    micro_batch_size: int = 32
    seq_len: int = 1024
    total_batch_tokens: int = 524_288  # 2^19
    tokens: int = 5_000_000_000  # training budget; sets max_steps unless max_steps > 0
    max_steps: int = 0
    # optimizer and schedule
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 700
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    # periodic work (in optimizer steps); 0 disables
    log_every: int = 10
    val_every: int = 250
    val_batches: int = 20
    sample_every: int = 1000
    ckpt_every: int = 500
    # speed / infrastructure
    compile: bool = True
    bf16: bool = True
    use_wandb: bool = True
    write_results: bool = True  # tests turn this off so they don't overwrite results/


def make_optimizer(model, cfg: PretrainConfig, device_type: str):
    # Weight decay on matrices (linear weights and the tied embedding), not on 1-D norm gains:
    # decay pulls weights toward 0, which makes sense for big matrices but would fight
    # RMSNorm's per-feature scales, which are supposed to sit around 1.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=(device_type == "cuda"),  # one GPU kernel for the whole update
    )


@torch.no_grad()
def evaluate(model, cfg, dist_info, device, use_bf16):
    """Mean loss on the same fixed validation batches every time (a fresh loader from the start)."""
    model.eval()
    val = ShardLoader(
        cfg.data_dir, "val", cfg.micro_batch_size, cfg.seq_len, dist_info.rank, dist_info.world_size
    )
    total = torch.zeros((), device=device)
    for _ in range(cfg.val_batches):
        x, y = (t.to(device) for t in val.next_batch())
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
        total += torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
    if dist_info.is_distributed:
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.AVG)
    model.train()
    return (total / cfg.val_batches).item()


@torch.no_grad()
def write_samples(model, path: Path, device):
    enc = get_tokenizer()
    lines = []
    for prompt in SAMPLE_PROMPTS:
        ids = torch.tensor([enc.encode_ordinary(prompt)], device=device)
        out = generate(model, ids, max_new_tokens=48, temperature=0.8, top_k=50)
        lines.append(f"### {prompt}\n{prompt}{enc.decode(out['tokens'][0].tolist())}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(cfg: PretrainConfig):
    dist_info = setup_distributed()
    device = dist_info.device
    use_bf16 = cfg.bf16 and device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmuls for any fp32 work on A100
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(cfg.seed)

    model_cfg = load_model_config(cfg.model)
    assert cfg.seq_len <= model_cfg.context_len
    tokens_per_micro = cfg.micro_batch_size * cfg.seq_len * dist_info.world_size
    assert cfg.total_batch_tokens % tokens_per_micro == 0, (
        "total batch must split into micro-batches"
    )
    accum_steps = cfg.total_batch_tokens // tokens_per_micro
    max_steps = cfg.max_steps or cfg.tokens // cfg.total_batch_tokens

    ckpt_dir = Path(cfg.checkpoint_root) / cfg.run_name
    run_dir = Path(cfg.runs_root) / cfg.run_name

    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = make_optimizer(model, cfg, device.type)
    train_loader = ShardLoader(
        cfg.data_dir,
        "train",
        cfg.micro_batch_size,
        cfg.seq_len,
        dist_info.rank,
        dist_info.world_size,
        cfg.seed,
    )

    # Resume from the latest checkpoint if there is one.
    step, train_seconds = 0, 0.0
    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, map_location=device)
        step = restore(state, model, optimizer)
        train_loader.load_state_dict(state["data_state"])
        train_seconds = state["extra"].get("train_seconds", 0.0)

    if dist_info.is_main:
        print(f"model {cfg.model}: {n_params:,} parameters")
        print(
            f"{max_steps} steps x {cfg.total_batch_tokens:,} tokens, {accum_steps} micro-batches/step"
        )
        print(f"resuming from {latest} at step {step}" if latest else "starting from scratch")
        save_config(cfg, run_dir / "config.yaml")
        if cfg.write_results and cfg.model != "tiny":
            write_results(
                "model", {"model": cfg.model, "parameters": n_params, **asdict(model_cfg)}
            )
        logger = MetricsLogger(run_dir, cfg.run_name, "pretrain", asdict(cfg), cfg.use_wandb)

    raw_model = model  # un-compiled, un-wrapped: used for generation and saving
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)
    if dist_info.is_distributed:
        model = DDP(model, device_ids=[dist_info.local_rank])

    def checkpoint(next_step, name=None):
        if dist_info.is_main:
            save_checkpoint(
                ckpt_dir,
                step=next_step,
                model=model,
                optimizer=optimizer,
                config=asdict(cfg),
                phase="pretrain",
                data_state=train_loader.state_dict(),
                extra={"train_seconds": train_seconds},
                name=name,
            )

    preempt = PreemptionFlag()
    model.train()
    val_loss = None
    while step < max_steps:
        t0 = time.time()
        lr = lr_at_step(step, cfg.max_lr, cfg.warmup_steps, max_steps, cfg.min_lr_ratio)
        micro = [tuple(t.to(device) for t in train_loader.next_batch()) for _ in range(accum_steps)]
        out = train_step(model, optimizer, micro, lr, cfg.grad_clip, use_bf16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_seconds += dt
        step += 1  # number of completed steps

        if dist_info.is_main and (step % cfg.log_every == 0 or step == 1):
            tok_per_sec = cfg.total_batch_tokens / dt
            print(
                f"step {step}/{max_steps} | loss {out['loss']:.4f} | lr {lr:.2e} | "
                f"grad_norm {out['grad_norm']:.3f} | {tok_per_sec:,.0f} tok/s",
                flush=True,
            )
            logger.log({**out, "lr": lr, "tokens_per_sec": tok_per_sec, "step_time": dt}, step)

        if cfg.val_every and (step % cfg.val_every == 0 or step == max_steps):
            val_loss = evaluate(model, cfg, dist_info, device, use_bf16)
            if dist_info.is_main:
                print(f"step {step} | val_loss {val_loss:.4f}", flush=True)
                logger.log({"val_loss": val_loss}, step)

        if dist_info.is_main and cfg.sample_every and step % cfg.sample_every == 0:
            write_samples(raw_model, run_dir / "samples" / f"step_{step:06d}.txt", device)

        if (cfg.ckpt_every and step % cfg.ckpt_every == 0) or preempt.set:
            checkpoint(step)
        if preempt.set:
            if dist_info.is_main:
                print(f"preempted: saved at step {step}, exiting so the job can be resubmitted")
            cleanup_distributed()
            return

    # Finished.
    checkpoint(step)
    checkpoint(step, name="final.pt")
    if dist_info.is_main:
        write_samples(raw_model, run_dir / "samples" / f"step_{step:06d}.txt", device)
        gpu_hours = train_seconds * dist_info.world_size / 3600
        summary = {
            "model": cfg.model,
            "parameters": n_params,
            "tokens_trained": step * cfg.total_batch_tokens,
            "steps": step,
            "final_val_loss": val_loss,
            "tokens_per_sec": round(step * cfg.total_batch_tokens / train_seconds, 1),
            "gpu_hours": round(gpu_hours, 2),
            "hardware": hardware_info(),
            "config": asdict(cfg),
        }
        print(summary)
        if cfg.write_results:
            write_results("pretrain", summary)
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main(load_config(PretrainConfig))
