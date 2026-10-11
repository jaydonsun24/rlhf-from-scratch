"""The plotting script runs on a metrics file shaped like the real one and writes a PNG."""

import json

from rlhf.eval.plots import plot_pretrain


def test_plot_pretrain_writes_png(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    lines = [{"step": s, "loss": 11 - s / 20} for s in range(1, 101, 10)]
    lines += [{"step": s, "val_loss": 10 - s / 25} for s in (50, 100)]
    lines += [{"step": 100, "hellaswag_acc": 0.27, "hellaswag_n": 10042}]
    metrics.write_text("\n".join(json.dumps(r) for r in lines))
    out = plot_pretrain(metrics, tmp_path / "fig.png")
    assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"
