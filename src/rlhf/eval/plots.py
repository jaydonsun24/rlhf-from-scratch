"""README figures, generated only from files in runs/ and results/.

    python -m rlhf.eval.plots pretrain --metrics runs/pretrain/metrics.jsonl
    python -m rlhf.eval.plots overopt --csv results/overopt.csv

Plain matplotlib: thin lines, labeled axes with units, a recessive grid, light surface.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display (works on the cluster)
import matplotlib.pyplot as plt

# Validated categorical palette (light mode) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GPT2_HELLASWAG = 0.296  # OpenAI GPT-2 124M, the reference to beat
TOKENS_PER_STEP = 524_288


def read_metrics(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_pretrain(
    metrics_path, out_path="assets/pretrain_loss.png", tokens_per_step=TOKENS_PER_STEP
):
    records = read_metrics(metrics_path)
    to_billions = tokens_per_step / 1e9

    def series(key):
        rows = [r for r in records if key in r]
        return [r["step"] * to_billions for r in rows], [r[key] for r in rows]

    train_x, train_y = series("loss")
    val_x, val_y = series("val_loss")
    hs_x, hs_y = series("hellaswag_acc")

    fig, (ax_loss, ax_hs) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)

    # Left: loss. Train and val share units, so one axis is honest.
    ax_loss.plot(train_x, train_y, color=SERIES[0], linewidth=1.0, alpha=0.8, label="train")
    ax_loss.plot(val_x, val_y, color=SERIES[1], linewidth=2.0, label="validation")
    ax_loss.set_ylim(
        top=min(6.0, max(train_y))
    )  # the first ~100 steps (11 -> 6) would flatten the rest
    style(ax_loss, "Cross-entropy loss", "tokens trained (billions)", "loss (nats/token)")
    ax_loss.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    if val_y:
        ax_loss.annotate(
            f"val {val_y[-1]:.3f}",
            (val_x[-1], val_y[-1]),
            xytext=(-4, 10),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=TEXT,
        )

    # Right: HellaSwag accuracy, with the two reference lines that give it meaning.
    if hs_y:
        ax_hs.plot(
            hs_x, [100 * y for y in hs_y], color=SERIES[0], linewidth=2.0, marker="o", markersize=4
        )
        ax_hs.annotate(
            f"{100 * hs_y[-1]:.2f}%",
            (hs_x[-1], 100 * hs_y[-1]),
            xytext=(-4, -14),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=TEXT,
        )
    ax_hs.axhline(100 * GPT2_HELLASWAG, color=MUTED, linewidth=1, linestyle="--")
    ax_hs.axhline(25, color=MUTED, linewidth=1, linestyle=":")
    ax_hs.text(
        0.01,
        100 * GPT2_HELLASWAG + 0.15,
        "GPT-2 124M (29.6%)",
        transform=ax_hs.get_yaxis_transform(),
        fontsize=8,
        color=MUTED,
        va="bottom",
    )
    ax_hs.text(
        0.01,
        25.15,
        "random guessing (25%)",
        transform=ax_hs.get_yaxis_transform(),
        fontsize=8,
        color=MUTED,
        va="bottom",
    )
    ax_hs.set_ylim(24, 31)
    style(
        ax_hs, "HellaSwag accuracy (10,042 questions)", "tokens trained (billions)", "accuracy (%)"
    )

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def read_curve(path) -> dict[float, list[dict]]:
    """results/overopt.csv as {beta: rows in training order}. Blank judge columns stay ''."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    curve = {}
    for row in rows:
        curve.setdefault(float(row["beta"]), []).append(row)
    return {beta: sorted(curve[beta], key=lambda r: int(r["iteration"])) for beta in sorted(curve)}


def plot_overopt(csv_path="results/overopt.csv", out_path="assets/overoptimization.png"):
    curve = read_curve(csv_path)
    judged = any(r["judge_win_rate"] for rows in curve.values() for r in rows)

    fig, (ax_rm, ax_quality) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.set_facecolor(SURFACE)
    xlabel = "√KL from SFT (KL in nats per summary)"

    def draw(ax, key, scale=1.0, falling=False):
        for color, (beta, rows) in zip(SERIES, curve.items(), strict=False):
            kept = [r for r in rows if r[key]]
            x = [float(r["sqrt_kl"]) for r in kept]
            y = [scale * float(r[key]) for r in kept]
            label = f"β = {beta:g}"
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=label,
            )
            # The lighter hues are hard to tell apart alone, so every line is also labeled.
            ax.annotate(
                label,
                (x[-1], y[-1]),
                xytext=(8, -3) if falling else (-7, 4),  # on the side away from the line
                textcoords="offset points",
                ha="left" if falling else "right",
                fontsize=8,
                color=TEXT,
            )
        start = next(iter(curve.values()))[0]
        ax.annotate(
            "SFT",
            (float(start["sqrt_kl"]), scale * float(start[key])),
            xytext=(0, 7) if falling else (7, -3),
            textcoords="offset points",
            ha="center" if falling else "left",
            fontsize=8,
            color=MUTED,
        )
        ax.legend(
            frameon=False,
            fontsize=8,
            labelcolor=MUTED,
            loc="lower left" if falling else "upper left",
        )
        if falling:
            ax.margins(x=0.12)  # room for the label after the last point

    draw(ax_rm, "rm_score")
    style(ax_rm, "Reward-model score (what PPO optimizes)", xlabel, "mean score, 500 posts")

    if judged:
        draw(ax_quality, "judge_win_rate", scale=100, falling=True)
        for color, rows in zip(SERIES, curve.values(), strict=False):
            kept = [r for r in rows if r["judge_win_rate"]]
            ax_quality.vlines(
                [float(r["sqrt_kl"]) for r in kept],
                [100 * float(r["judge_ci_low"]) for r in kept],
                [100 * float(r["judge_ci_high"]) for r in kept],
                color=color,
                linewidth=1,
                alpha=0.5,
            )
        ax_quality.axhline(50, color=MUTED, linewidth=1, linestyle="--")
        title, ylabel = "Judge win rate vs. human TL;DR", "win rate (%), 95% bootstrap CI"
    else:  # no judge results yet: word overlap with the human TL;DRs is the independent signal
        draw(ax_quality, "rougeL", falling=True)
        title, ylabel = "Word overlap with human TL;DRs (ROUGE-L)", "mean ROUGE-L, 500 posts"
    style(ax_quality, title, xlabel, ylabel)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make README figures")
    sub = parser.add_subparsers(dest="figure", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--metrics", default="runs/pretrain/metrics.jsonl")
    p.add_argument("--out", default="assets/pretrain_loss.png")
    p = sub.add_parser("overopt")
    p.add_argument("--csv", default="results/overopt.csv")
    p.add_argument("--out", default="assets/overoptimization.png")
    args = parser.parse_args()
    if args.figure == "pretrain":
        print("wrote", plot_pretrain(args.metrics, args.out))
    elif args.figure == "overopt":
        print("wrote", plot_overopt(args.csv, args.out))
