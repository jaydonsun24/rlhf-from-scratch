"""95% confidence intervals by bootstrap: how much would this mean move with a different
sample of posts?

Resample the per-post scores WITH replacement (same size as the original), take the mean,
repeat 10,000 times; the middle 95% of those means is the interval. No normality assumption,
which matters for win rates (scores are only ever 0, 0.5 or 1).
"""

import numpy as np


def bootstrap_ci(scores, n_resamples=10_000, confidence=0.95, seed=0) -> tuple[float, float]:
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))  # (n_resamples, n)
    means = x[idx].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)
