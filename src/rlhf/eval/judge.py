"""Final eval step 2: an LLM judge (Claude) compares two sets of summaries post by post.

    python -m rlhf.eval.judge --config configs/judge.yaml a=ppo b=reference
    python -m rlhf.eval.judge --config configs/judge.yaml a=ppo b=sft      # head-to-head

Why a judge at all: PPO was trained to maximize the reward model, so the RM's score can't
tell "better summaries" from "found a loophole in the RM". The judge never took part in
training, so it's an independent check (the DPO paper used GPT-4 the same way).

For each post the judge sees the post and two summaries labeled A and B and answers
"Preferred: A" or "Preferred: B" (prompt fixed in configs/judge_prompt.txt).

  Position bias: LLMs tend to favor whichever option comes first, so every pair is judged
  twice with the order swapped. Win = preferred in both orders (1), loss = in neither (0),
  otherwise tie (0.5). The tie rate is the "order disagreement" rate.

  Cost control: every answer is cached (data/judge_cache.jsonl) by a hash of the model, the
  effort and the full prompt, so re-running never pays twice. New calls are capped by
  max_calls and need a y/N confirmation after a rough cost estimate.

Inputs: runs/eval/summaries/<name>.jsonl from eval/summarize.py ("reference" = the human
TL;DRs). Outputs: runs/eval/judge/<a>_vs_<b>.jsonl (every pair, with the judge's sentences)
and results/judge_<a>_vs_<b>.json (win rate, 95% bootstrap CI, counts).
"""

import hashlib
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import numpy as np

from rlhf.eval.bootstrap import bootstrap_ci
from rlhf.utils.config import load_config
from rlhf.utils.logging import write_results

REFERENCE = "reference"
LENGTH_MATCH_TOKENS = 5  # length-controlled win rate: pairs within this many tokens
ASSUMED_OUTPUT_TOKENS = 300  # per call, thinking included; only used for the cost estimate


@dataclass
class JudgeConfig:
    a: str = "ppo"
    b: str = REFERENCE
    model: str = "claude-opus-5-5"
    effort: str = "low"
    max_tokens: int = 8000
    prompt_file: str = "configs/judge_prompt.txt"
    summaries_dir: str = "runs/eval/summaries"
    out_dir: str = "runs/eval/judge"
    cache_path: str = "data/judge_cache.jsonl"
    max_calls: int = 2500
    workers: int = 8
    yes: bool = False
    write_results: bool = True
    price_in_per_mtok: float = 4.0  # USD, Claude Opus 5.5; only used for the cost estimate
    price_out_per_mtok: float = 20.0
    bootstrap_resamples: int = 10_000
    seed: int = 0


# ---------- pure pieces (tested in tests/test_judge.py) ----------


def build_prompt(template: str, post: str, summary_a: str, summary_b: str) -> str:
    return template.format(post=post, summary_a=summary_a, summary_b=summary_b)


def parse_preference(text: str) -> str | None:
    """'A' or 'B' from the judge's last 'Preferred: X' line; None if there isn't one."""
    matches = re.findall(r"preferred:\s*\W*([ab])\b", text, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else None


def pair_score(pref_a_first: str | None, pref_a_second: str | None) -> float | None:
    """Score for summary `a` on one post, from the two orderings.

    pref_a_first: the judge's answer when `a` was shown as A; pref_a_second: when `a` was B.
    Returns 1 (won both), 0 (lost both), 0.5 (orders disagree), or None (unparseable answer).
    """
    if pref_a_first is None or pref_a_second is None:
        return None
    return ((pref_a_first == "A") + (pref_a_second == "B")) / 2


def aggregate(scores, n_resamples=10_000, seed=0) -> dict:
    """Win rate of `a` with ties counted as 0.5, and its 95% bootstrap CI over posts."""
    valid = [s for s in scores if s is not None]
    out = {
        "n_scored": len(valid),
        "unscored": len(scores) - len(valid),
        "wins": sum(s == 1.0 for s in valid),
        "ties": sum(s == 0.5 for s in valid),
        "losses": sum(s == 0.0 for s in valid),
    }
    if not valid:
        return {**out, "win_rate": None, "ci95": None, "order_disagreement_rate": None}
    out["win_rate"] = float(np.mean(valid))
    out["ci95"] = list(bootstrap_ci(valid, n_resamples, seed=seed))
    out["order_disagreement_rate"] = out["ties"] / len(valid)
    return out


# ---------- cache ----------


class JudgeCache:
    """Append-only jsonl: one line per answered prompt. Thread-safe appends."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.entries = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.entries[rec["key"]] = rec

    @staticmethod
    def key(model: str, effort: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{effort}\n{prompt}".encode()).hexdigest()

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, record: dict) -> None:
        record = {"key": key, **record}
        with self.lock:
            self.entries[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")


# ---------- calling Claude ----------


def call_judge(client, cfg: JudgeConfig, prompt: str) -> dict:
    """One judge call. Reddit posts are sometimes about sensitive topics, so server-side
    fallbacks are on: if the model declines, the API reruns the request on a fallback model
    (`served_by` records which model actually answered)."""
    response = client.beta.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        output_config={"effort": cfg.effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return {
        "text": text,
        "stop_reason": response.stop_reason,
        "served_by": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def run_prompts(prompts: list[str], cfg: JudgeConfig, cache: JudgeCache, client=None):
    """Answer every prompt, from the cache when possible. Returns (records, stats).
    A record is None if the API call failed (not cached, so a re-run retries it)."""
    keys = [JudgeCache.key(cfg.model, cfg.effort, p) for p in prompts]
    todo = sorted(
        {k: p for k, p in zip(keys, prompts, strict=True) if cache.get(k) is None}.items()
    )
    stats = Counter(cached=len(prompts) - len(todo), new=0, api_errors=0)

    if todo:
        if len(todo) > cfg.max_calls:
            raise SystemExit(
                f"{len(todo)} new calls needed but max_calls={cfg.max_calls}; raise it to proceed"
            )
        if client is None:
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        if not cfg.yes:
            confirm_cost(client, cfg, [p for _, p in todo])

        def work(item):
            key, prompt = item
            try:
                cache.put(key, call_judge(client, cfg, prompt))
                return "new"
            except anthropic.APIError as e:  # after the SDK's own retries; leave uncached
                print(f"judge call failed ({type(e).__name__}): {e}")
                return "api_errors"

        with ThreadPoolExecutor(cfg.workers) as pool:
            for i, outcome in enumerate(pool.map(work, todo), 1):
                stats[outcome] += 1
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)} calls", flush=True)

    return [cache.get(k) for k in keys], dict(stats)


def confirm_cost(client, cfg: JudgeConfig, prompts: list[str]) -> None:
    sample = client.messages.count_tokens(
        model=cfg.model, messages=[{"role": "user", "content": prompts[0]}]
    ).input_tokens
    tokens_in = sample * len(prompts)
    dollars = (
        tokens_in * cfg.price_in_per_mtok
        + len(prompts) * ASSUMED_OUTPUT_TOKENS * cfg.price_out_per_mtok
    ) / 1e6
    print(
        f"{len(prompts)} new calls to {cfg.model}, ~{tokens_in:,} input tokens, "
        f"rough cost ~${dollars:.2f}"
    )
    if input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted, nothing was spent")


# ---------- one comparison ----------


def load_summaries(summaries_dir, name) -> list[dict]:
    with open(Path(summaries_dir) / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def side(rows, name, k):
    """(text, length in tokens) of side `name` on row k."""
    if name == REFERENCE:
        return rows[k]["reference"], rows[k]["reference_length"]
    return rows[k]["summary"], rows[k]["length"]


def compare(cfg: JudgeConfig, client=None) -> dict:
    if cfg.a == REFERENCE:
        raise SystemExit("put the policy in `a`; `b` may be reference")
    rows_a = load_summaries(cfg.summaries_dir, cfg.a)
    rows_b = rows_a if cfg.b == REFERENCE else load_summaries(cfg.summaries_dir, cfg.b)
    if [r["index"] for r in rows_a] != [r["index"] for r in rows_b]:
        raise SystemExit(f"{cfg.a} and {cfg.b} cover different posts")

    template = Path(cfg.prompt_file).read_text()
    prompts = []  # two per post: a shown as A, then a shown as B
    for k, row in enumerate(rows_a):
        text_a, _ = side(rows_a, cfg.a, k)
        text_b, _ = side(rows_b, cfg.b, k)
        prompts.append(build_prompt(template, row["post"], text_a, text_b))
        prompts.append(build_prompt(template, row["post"], text_b, text_a))

    cache = JudgeCache(cfg.cache_path)
    records, call_stats = run_prompts(prompts, cfg, cache, client)

    pairs, scores, length_matched = [], [], []
    refusals, parse_failures, served_by, usage = 0, 0, Counter(), Counter()
    for k, row in enumerate(rows_a):
        first, second = records[2 * k], records[2 * k + 1]
        prefs = []
        for rec in (first, second):
            if rec is None:  # API error: retried on the next run
                prefs.append(None)
                continue
            served_by[rec["served_by"]] += 1
            usage["input_tokens"] += rec["input_tokens"]
            usage["output_tokens"] += rec["output_tokens"]
            if rec["stop_reason"] == "refusal":
                refusals += 1
                prefs.append(None)
                continue
            pref = parse_preference(rec["text"])
            if pref is None:
                parse_failures += 1
                print(f"unparseable judge answer (post {row['index']}): {rec['text'][-200:]!r}")
            prefs.append(pref)
        s = pair_score(*prefs)
        scores.append(s)
        len_a, len_b = side(rows_a, cfg.a, k)[1], side(rows_b, cfg.b, k)[1]
        if abs(len_a - len_b) <= LENGTH_MATCH_TOKENS:
            length_matched.append(s)
        pairs.append(
            {
                "index": row["index"],
                "id": row["id"],
                "pref_a_first": prefs[0],
                "pref_a_second": prefs[1],
                "score": s,
                "judge_a_first": first and first["text"],
                "judge_a_second": second and second["text"],
            }
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.a}_vs_{cfg.b}"
    (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))

    summary = {
        "a": cfg.a,
        "b": cfg.b,
        "n_posts": len(rows_a),
        **aggregate(scores, cfg.bootstrap_resamples, cfg.seed),
        "length_controlled": {
            "max_length_gap_tokens": LENGTH_MATCH_TOKENS,
            **aggregate(length_matched, cfg.bootstrap_resamples, cfg.seed),
        },
        "mean_length_a": float(np.mean([side(rows_a, cfg.a, k)[1] for k in range(len(rows_a))])),
        "mean_length_b": float(np.mean([side(rows_b, cfg.b, k)[1] for k in range(len(rows_b))])),
        "judge_model": cfg.model,
        "judge_effort": cfg.effort,
        "served_by": dict(served_by),
        "refusals": refusals,
        "parse_failures": parse_failures,
        "api_errors": call_stats.get("api_errors", 0),
        "calls": {"cached": call_stats.get("cached", 0), "new": call_stats.get("new", 0)},
        "judge_usage_tokens": dict(usage),
        "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "config": asdict(cfg),
    }
    if cfg.write_results:
        write_results(f"judge_{name}", summary)
    return summary


def main(cfg: JudgeConfig):
    s = compare(cfg)
    lc = s["length_controlled"]
    print(
        f"{s['a']} vs {s['b']}: win rate {s['win_rate']:.3f} "
        f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]  n={s['n_scored']}  "
        f"W/T/L {s['wins']}/{s['ties']}/{s['losses']}  "
        f"order disagreement {s['order_disagreement_rate']:.2f}  "
        f"unscored {s['unscored']} (refusals {s['refusals']}, parse failures "
        f"{s['parse_failures']}, api errors {s['api_errors']})"
    )
    if lc["win_rate"] is not None:
        print(
            f"length-controlled (|gap| <= {LENGTH_MATCH_TOKENS}): {lc['win_rate']:.3f}  n={lc['n_scored']}"
        )


if __name__ == "__main__":
    main(load_config(JudgeConfig))
