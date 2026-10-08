"""Tests for train/lr_schedule.py: linear warmup, then cosine decay to a floor.

Hand-checkable setup: max_lr = 1.0, warmup_steps = 10, max_steps = 110, min_lr_ratio = 0.1
  warmup:  steps 0..9,   lr = max_lr * (step + 1) / warmup_steps   -> 0.1, 0.2, ..., 1.0
  decay:   steps 10..110, cosine from max_lr (1.0) down to min_lr (0.1)
  after:   step > 110,   lr stays at min_lr
"""

import math

import pytest

from rlhf.train.lr_schedule import lr_at_step

CFG = dict(max_lr=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, 0.1),  # first step: 1/10 of max, never exactly 0 (a 0 LR step would be wasted)
        (4, 0.5),
        (9, 1.0),  # last warmup step reaches the peak
        (10, 1.0),  # decay starts at the peak
        (60, 0.55),  # halfway through decay: exactly between max (1.0) and min (0.1)
        (110, 0.1),  # end of decay: the floor
        (500, 0.1),  # past the end: stays at the floor
    ],
)
def test_hand_computed_values(step, expected):
    assert lr_at_step(step, **CFG) == pytest.approx(expected, abs=1e-9)


def test_cosine_shape_in_decay():
    # A quarter of the way through decay: 0.1 + 0.9 * 0.5 * (1 + cos(pi/4))
    expected = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 4))
    assert lr_at_step(35, **CFG) == pytest.approx(expected, abs=1e-9)


def test_warmup_rises_and_decay_falls():
    lrs = [lr_at_step(s, **CFG) for s in range(0, 120)]
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10]))  # strictly up during warmup
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))  # never up during decay


def test_real_pretraining_settings():
    # docs/02: peak 6e-4, warmup 700, ~9,537 steps for 5B tokens, decay to 10% of peak
    real = dict(max_lr=6e-4, warmup_steps=700, max_steps=9537, min_lr_ratio=0.1)
    assert lr_at_step(699, **real) == pytest.approx(6e-4)
    assert lr_at_step(9537, **real) == pytest.approx(6e-5)
