import pytest

from rlhf.eval.rouge import rouge_scores


def test_identical_is_one_disjoint_is_zero():
    same = rouge_scores(["my cat ate the homework"], ["my cat ate the homework"])
    assert all(v == pytest.approx(1.0) for v in same.values())
    none = rouge_scores(["completely different words"], ["my cat ate homework"])
    assert all(v == 0.0 for v in none.values())


def test_partial_overlap_and_averaging():
    # pred has 2 of the reference's 4 words (and 2 of its own 4): precision = recall = 0.5
    out = rouge_scores(["my dog ate food", "a b"], ["my cat ate homework", "a b"])
    assert out["rouge1"] == pytest.approx((0.5 + 1.0) / 2)
