"""README figures, generated only from files in runs/ and results/.

    python -m rlhf.eval.plots pretrain --metrics runs/pretrain/metrics.jsonl

Plain matplotlib: thin lines, labeled axes with units, a recessive grid, light surface.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display (works on the cluster)
import matplotlib.pyplot as plt

# Validated categorical palette (light mode) and neutral inks.
SERIES = ["#2a78d6", "#eb6834"]  # blue, orange
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make README figures")
    sub = parser.add_subparsers(dest="figure", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--metrics", default="runs/pretrain/metrics.jsonl")
    p.add_argument("--out", default="assets/pretrain_loss.png")
    args = parser.parse_args()
    if args.figure == "pretrain":
        print("wrote", plot_pretrain(args.metrics, args.out))
