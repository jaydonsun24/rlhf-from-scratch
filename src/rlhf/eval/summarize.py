"""Step 1 of the final evaluation (GPU): write each policy's summaries for the 500 test posts.

    sbatch scripts/slurm/summarize.slurm                      # sft, dpo, ppo, ppo_beta0
    python -m rlhf.eval.summarize --policies ppo_iter0100=checkpoints/ppo_default/iter_0100.pt

The judge (step 2, eval/judge.py) needs the internet, which Adroit's compute nodes don't
have, so the GPU work happens here and the judge runs on a laptop from these files:

    runs/eval/summaries/<policy>.jsonl       one row per test post: the post, the human
                                             reference, the policy's greedy summary, lengths,
                                             and the reward model's score for both summaries
    runs/eval/summaries/<policy>.meta.json   means, ROUGE, and KL from SFT

Decoding is greedy (deterministic). Same post order in every file, so rows line up.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rlhf.data.tldr import TLDRDataset
from rlhf.data.tokenizer import EOT_ID, get_tokenizer
from rlhf.eval.kl import estimate_kl
from rlhf.eval.rouge import rouge_scores
from rlhf.model.gpt import GPT
from rlhf.model.reward_model import RewardModel
from rlhf.train.dpo import greedy_pairs
from rlhf.train.reward import score as rm_score
from rlhf.utils.checkpoint import load_checkpoint
from rlhf.utils.config import load_model_config
from rlhf.utils.distributed import get_device

DEFAULT_POLICIES = [
    "sft=checkpoints/sft/final.pt",
    "dpo=checkpoints/dpo/final.pt",
    "ppo=checkpoints/ppo_default/final.pt",
    "ppo_beta0=checkpoints/ppo_beta0/final.pt",
]


def load_weights(model, path):
    model.load_state_dict(load_checkpoint(path, map_location="cpu")["model"])
    return model


def post_text(prompt_tokens, enc) -> str:
    """The prompt the model saw, minus the trailing 'TL;DR:' cue (the judge doesn't need it)."""
    return enc.decode(prompt_tokens).removesuffix("TL;DR:").strip()


def summarize_policy(policy, reference, reward_model, test, indices, args, device, use_bf16):
    enc = get_tokenizer()
    pairs = greedy_pairs(policy, test, indices, args, device, use_bf16)  # [(prompt, tokens)]
    refs = [test[i] for i in indices]  # [(prompt, reference tokens ending in EOT)]
    rm_policy = rm_score(reward_model, pairs, device, use_bf16)
    rm_ref = rm_score(reward_model, refs, device, use_bf16)

    rows = []
    for k, i in enumerate(indices):
        prompt, tokens = pairs[k]
        ref_tokens = refs[k][1]
        rows.append(
            {
                "index": i,
                "id": str(test.ids[i]),
                "post": post_text(prompt, enc),
                "reference": enc.decode(ref_tokens[:-1]).strip(),
                "summary": enc.decode([t for t in tokens if t != EOT_ID]).strip(),
                "length": len(tokens),  # tokens, including EOT if the model stopped
                "reference_length": len(ref_tokens),
                "eot": EOT_ID in tokens,
                "rm_summary": float(rm_policy[k]),
                "rm_reference": float(rm_ref[k]),
            }
        )

    # KL from SFT: same Monte Carlo definition as training (T = 0.7), on a fixed prompt subset.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    kl, _ = estimate_kl(
        policy,
        reference,
        [test[i][0] for i in indices[: args.kl_prompts]],
        device,
        temperature=0.7,
        batch_size=args.eval_batch_size,
        use_bf16=use_bf16,
        generator=gen,
    )
    meta = {
        "n": len(rows),
        "decoding": "greedy",
        "mean_length": float(np.mean([r["length"] for r in rows])),
        "eot_rate": float(np.mean([r["eot"] for r in rows])),
        "rm_score": float(np.mean(rm_policy)),
        "rm_score_reference": float(np.mean(rm_ref)),
        "kl_from_sft": float(kl),
        "kl_prompts": min(args.kl_prompts, len(indices)),
        **rouge_scores([r["summary"] for r in rows], [r["reference"] for r in rows]),
    }
    return rows, meta


def main(args):
    device = get_device()
    use_bf16 = device.type == "cuda"
    model_cfg = load_model_config(args.model)
    torch.manual_seed(args.seed)

    test = TLDRDataset("test", args.tldr_dir)
    subset = json.loads((Path(args.tldr_dir) / "test_subset.json").read_text())
    indices = subset["indices"][: args.n]

    reference = load_weights(GPT(model_cfg), args.sft_checkpoint).to(device).eval()
    reward_model = load_weights(RewardModel(model_cfg), args.rm_checkpoint).to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.policies:
        name, path = spec.split("=", 1)
        out = out_dir / f"{name}.jsonl"
        if not Path(path).exists():
            print(f"skip {name}: {path} not found")
            continue
        if out.exists() and not args.overwrite:
            print(f"skip {name}: {out} already exists (--overwrite to redo)")
            continue
        policy = load_weights(GPT(model_cfg), path).to(device).eval()
        rows, meta = summarize_policy(
            policy, reference, reward_model, test, indices, args, device, use_bf16
        )
        meta = {"policy": name, "checkpoint": path, **meta}
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))
        del policy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Greedy test-set summaries for the judge")
    p.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES, help="name=checkpoint")
    p.add_argument("--model", default="gpt124m")
    p.add_argument("--sft-checkpoint", default="checkpoints/sft/final.pt")
    p.add_argument("--rm-checkpoint", default="checkpoints/rm/final.pt")
    p.add_argument("--tldr-dir", default="data/tldr")
    p.add_argument("--out-dir", default="runs/eval/summaries")
    p.add_argument("--n", type=int, default=500, help="test posts (from test_subset.json)")
    p.add_argument("--kl-prompts", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=16)  # 32 hits OOM on a 20 GB MIG slice
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
