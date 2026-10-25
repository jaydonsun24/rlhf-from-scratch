"""ROUGE: word overlap between generated summaries and the human references.

ROUGE-1 / ROUGE-2 count shared single words / word pairs; ROUGE-L uses the longest common
subsequence. Each is an F1 between 0 and 1, averaged over examples. It's a weak signal for
summary quality (a good summary in different words scores low), reported because it's
standard and cheap; the LLM judge in Phase 7 is the real quality measure.
"""

from rouge_score import rouge_scorer

METRICS = ("rouge1", "rouge2", "rougeL")


def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)
    totals = dict.fromkeys(METRICS, 0.0)
    for pred, ref in zip(predictions, references, strict=True):
        scores = scorer.score(ref, pred)  # rouge-score's argument order is (target, prediction)
        for m in METRICS:
            totals[m] += scores[m].fmeasure
    return {m: totals[m] / len(predictions) for m in METRICS}
