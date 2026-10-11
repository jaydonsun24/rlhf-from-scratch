"""Play with a trained checkpoint: type the start of some text, the model continues it.

    python scripts/sample.py --checkpoint checkpoints/pretrain/final.pt
    python scripts/sample.py --checkpoint checkpoints/pretrain/final.pt --prompt "In 1905, Albert Einstein"

The pretrained model is a BASE model: it continues text, it doesn't follow instructions.
Give it the beginning of a sentence ("The heart pumps blood by"), not a question.
"""

import argparse

import torch

from rlhf.data.tokenizer import get_tokenizer
from rlhf.model.generate import generate
from rlhf.model.gpt import GPT
from rlhf.utils.checkpoint import load_checkpoint
from rlhf.utils.config import load_model_config
from rlhf.utils.distributed import get_device


def load_model(path, device):
    state = load_checkpoint(path, map_location="cpu")
    model = GPT(load_model_config(state["config"]["model"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def continue_text(model, prompt, device, args):
    enc = get_tokenizer()
    ids = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    out = generate(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    tokens = out["tokens"][0][out["response_mask"][0].bool()].tolist()  # drop padding after EOT
    return prompt + enc.decode(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample text from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None, help="omit for an interactive loop")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = always the top token")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    model = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} on {device}")
    if args.prompt is not None:
        print(continue_text(model, args.prompt, device, args))
    else:
        print("Type the start of some text (empty line to quit).")
        while prompt := input("\n> ").strip():
            print("\n" + continue_text(model, prompt, device, args))
