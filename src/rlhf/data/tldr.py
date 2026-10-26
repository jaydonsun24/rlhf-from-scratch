"""Reddit TL;DR data for SFT, and the batch layout every later phase reuses.

    python -m rlhf.data.tldr        # once: tokenizes data/raw/tldr/*.jsonl -> data/tldr/

Each example is a prompt and a response:

    prompt   = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    response = " " + summary, then <|endoftext|>

Batch layout (SFT, reward model, DPO, PPO all use this):

    [pad pad pad][prompt tokens][response tokens][EOT][pad pad]
     <- left pad ->                                  <- right pad ->

Prompts are LEFT-padded so every row's response starts at the same column, P. Pads use
the dedicated pad id (50257), never EOT: an EOT must always mean "the model chose to stop".
"""

import json
from pathlib import Path

import numpy as np
import torch

from rlhf.data.tokenizer import EOT_ID, PAD_ID, get_tokenizer

RAW_DIR = "data/raw/tldr"
OUT_DIR = "data/tldr"
PROMPT_TEMPLATE = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
MAX_PROMPT_TOKENS = 512
# Max response length in tokens, including EOT. Huang et al. use 53; every human summary
# in this dataset fits in 48 (it was pre-filtered), so SFT drops nothing. PPO generates
# at most this many tokens.
RESPONSE_CAP = 53
SPLITS = ("train", "valid", "test")


def build_prompt(subreddit: str, title: str, post: str, max_tokens: int = MAX_PROMPT_TOKENS):
    """Prompt token ids, truncating the POST at paragraph boundaries if it's too long.

    Drops trailing paragraphs (split on newlines) until the prompt fits, so the cut never
    lands mid-sentence and the "\\n\\nTL;DR:" ending is always kept. If a single giant
    paragraph is still too long, falls back to dropping trailing words.
    """
    enc = get_tokenizer()

    def encode(text):
        return enc.encode_ordinary(
            PROMPT_TEMPLATE.format(subreddit=subreddit, title=title, post=text.strip())
        )

    tokens = encode(post)
    paragraphs = post.split("\n")
    while len(tokens) > max_tokens and len(paragraphs) > 1:
        paragraphs = paragraphs[:-1]
        tokens = encode("\n".join(paragraphs))
    words = "\n".join(paragraphs).split(" ")
    while len(tokens) > max_tokens and len(words) > 1:
        words = words[: int(len(words) * 0.9)]  # shrink 10% at a time
        tokens = encode(" ".join(words))
    return tokens


def build_response(summary: str) -> list[int]:
    return get_tokenizer().encode_ordinary(" " + summary.strip()) + [EOT_ID]


def collate(examples, pad_id: int = PAD_ID) -> dict:
    """examples: list of (prompt_ids, response_ids). response_ids may be [] (prompts only,
    for generation). Returns the shared layout as tensors:

      input_ids      (B, P + R)  P = longest prompt, R = longest response in this batch
      attention_mask (B, P + R)  1 on real tokens (prompt, response, EOT), 0 on pads
      position_ids   (B, P + R)  0, 1, 2, ... counting real tokens only
      response_mask  (B, P + R)  1 on response tokens + EOT only
      prompt_len     int         P: every response starts at this column
    """
    P = max(len(p) for p, _ in examples)
    R = max(len(r) for _, r in examples)
    B = len(examples)
    input_ids = torch.full((B, P + R), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, P + R, dtype=torch.long)
    response_mask = torch.zeros(B, P + R, dtype=torch.long)
    for i, (prompt, response) in enumerate(examples):
        start = P - len(prompt)  # left padding
        input_ids[i, start:P] = torch.tensor(prompt)
        attention_mask[i, start:P] = 1
        if response:
            input_ids[i, P : P + len(response)] = torch.tensor(response)
            attention_mask[i, P : P + len(response)] = 1
            response_mask[i, P : P + len(response)] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "prompt_len": P,
    }


class TLDRDataset:
    """Tokenized split stored compactly: all prompts in one flat uint16 array plus offsets
    (same for responses), instead of 117K separate Python lists."""

    def __init__(self, split: str, data_dir: str = OUT_DIR):
        data = np.load(Path(data_dir) / f"{split}.npz")
        self.prompts, self.prompt_offsets = data["prompts"], data["prompt_offsets"]
        self.responses, self.response_offsets = data["responses"], data["response_offsets"]
        self.ids = list(data["ids"])

    def __len__(self):
        return len(self.ids)

    def longest(self) -> int:
        """Length of the longest prompt + response pair, in tokens."""
        return int((np.diff(self.prompt_offsets) + np.diff(self.response_offsets)).max())

    def __getitem__(self, i):
        p = self.prompts[self.prompt_offsets[i] : self.prompt_offsets[i + 1]].tolist()
        r = self.responses[self.response_offsets[i] : self.response_offsets[i + 1]].tolist()
        return p, r


def prepare(raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "vwxyzjn/summarize_from_feedback_tldr_3_filtered (Hugging Face)",
        "prompt_template": PROMPT_TEMPLATE,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "response_cap": RESPONSE_CAP,
        "splits": {},
    }
    for split in SPLITS:
        with open(Path(raw_dir) / f"{split}.jsonl") as f:
            rows = [json.loads(line) for line in f]
        prompts, responses, ids = [], [], []
        dropped = 0
        for row in rows:
            response = build_response(row["summary"])
            if len(response) > RESPONSE_CAP:
                dropped += 1
                continue
            prompts.append(build_prompt(row["subreddit"], row["title"], row["post"]))
            responses.append(response)
            ids.append(row["id"])
        np.savez(
            out / f"{split}.npz",
            prompts=np.concatenate(prompts).astype(np.uint16),
            prompt_offsets=np.cumsum([0] + [len(p) for p in prompts]),
            responses=np.concatenate(responses).astype(np.uint16),
            response_offsets=np.cumsum([0] + [len(r) for r in responses]),
            ids=np.array(ids),
        )
        meta["splits"][split] = {"raw": len(rows), "kept": len(ids), "dropped_over_cap": dropped}
        print(f"{split}: kept {len(ids):,} of {len(rows):,}", flush=True)

    # Fixed, seeded eval subsets (1,000 val ids; 500 test ids).
    rng = np.random.default_rng(seed)
    for split, n, name in (("valid", 1000, "val_subset.json"), ("test", 500, "test_subset.json")):
        all_ids = list(np.load(out / f"{split}.npz")["ids"])
        chosen = sorted(rng.choice(len(all_ids), size=n, replace=False).tolist())
        (out / name).write_text(json.dumps({"split": split, "indices": chosen, "seed": seed}))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    prepare()
