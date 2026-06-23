from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _midrank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values)
    sorted_values = values[order]
    size = values.shape[0]
    ranks = np.zeros(size, dtype=np.float64)
    cursor = 0
    while cursor < size:
        run = cursor
        while run < size and sorted_values[run] == sorted_values[cursor]:
            run += 1
        ranks[cursor:run] = 0.5 * (cursor + run - 1) + 1.0
        cursor = run
    restored = np.empty(size, dtype=np.float64)
    restored[order] = ranks
    return restored


def _structure(
    scores: Sequence[NDArray[np.float64]], labels: NDArray[np.int_]
) -> Tuple[NDArray[np.float64], int]:
    order = np.argsort(-labels, kind="mergesort")
    positives = int(labels.sum())
    stacked = np.vstack([series[order] for series in scores])
    return stacked, positives


def _fast_delong(
    predictions: NDArray[np.float64], positives: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    negatives = predictions.shape[1] - positives
    positive = predictions[:, :positives]
    negative = predictions[:, positives:]
    count = predictions.shape[0]
    tx = np.empty((count, positives), dtype=np.float64)
    ty = np.empty((count, negatives), dtype=np.float64)
    tz = np.empty((count, positives + negatives), dtype=np.float64)
    for row in range(count):
        tx[row] = _midrank(positive[row])
        ty[row] = _midrank(negative[row])
        tz[row] = _midrank(predictions[row])
    aucs = (
        tz[:, :positives].sum(axis=1) / positives / negatives - (positives + 1.0) / 2.0 / negatives
    )
    v01 = (tz[:, :positives] - tx) / negatives
    v10 = 1.0 - (tz[:, positives:] - ty) / positives
    cov = np.atleast_2d(np.cov(v01)) / positives + np.atleast_2d(np.cov(v10)) / negatives
    return aucs, cov


def roc_auc(scores: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    positives = int(labels.sum())
    negatives = labels.shape[0] - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _midrank(scores.astype(np.float64))
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def auc_variance(scores: NDArray[np.float64], labels: NDArray[np.int_]) -> Tuple[float, float]:
    predictions, positives = _structure([scores.astype(np.float64)], labels)
    aucs, cov = _fast_delong(predictions, positives)
    return float(aucs[0]), float(cov[0, 0])


def delong_test(
    scores_a: NDArray[np.float64], scores_b: NDArray[np.float64], labels: NDArray[np.int_]
) -> Tuple[float, float, float]:
    from scipy.stats import norm

    predictions, positives = _structure(
        [scores_a.astype(np.float64), scores_b.astype(np.float64)], labels
    )
    aucs, cov = _fast_delong(predictions, positives)
    variance = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if variance <= 0.0:
        p_value = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        statistic = (aucs[0] - aucs[1]) / np.sqrt(variance)
        p_value = float(2.0 * (1.0 - norm.cdf(abs(statistic))))
    return float(aucs[0]), float(aucs[1]), p_value


def bootstrap_ci(
    values: NDArray[np.float64], iterations: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    size = values.shape[0]
    means = np.empty(iterations, dtype=np.float64)
    for step in range(iterations):
        draw = rng.integers(0, size, size)
        means[step] = values[draw].mean()
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def cohen_dz(differences: NDArray[np.float64]) -> float:
    spread = float(np.std(differences, ddof=1))
    if spread == 0.0:
        return 0.0
    return float(np.mean(differences) / spread)


def benjamini_hochberg(
    p_values: NDArray[np.float64], alpha: float = 0.05
) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
    size = p_values.shape[0]
    order = np.argsort(p_values)
    ranked = p_values[order]
    scale = size / (np.arange(size) + 1.0)
    q_sorted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    q_values = np.empty(size, dtype=np.float64)
    q_values[order] = np.clip(q_sorted, 0.0, 1.0)
    rejected = q_values <= alpha
    return rejected, q_values


def holm_threshold(p_values: List[float], alpha: float = 0.05) -> float:
    if not p_values:
        return alpha
    return alpha / float(len(p_values))
