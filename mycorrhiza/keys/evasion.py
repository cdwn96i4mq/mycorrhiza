from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mycorrhiza.keys.stats import roc_auc


def response_auc(scores: NDArray[np.float64], response: NDArray[np.int_]) -> float:
    return roc_auc(scores, response)


def colocalisation_f1(evasion_zone: NDArray[np.bool_], exclusion_zone: NDArray[np.bool_]) -> float:
    true_positive = float((evasion_zone & exclusion_zone).sum())
    false_positive = float((evasion_zone & ~exclusion_zone).sum())
    false_negative = float((~evasion_zone & exclusion_zone).sum())
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0.0 else 2.0 * true_positive / denominator


def spatial_pearson(program_score: NDArray[np.float64], marker: NDArray[np.float64]) -> float:
    a = program_score - program_score.mean()
    b = marker - marker.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator == 0.0:
        return 0.0
    return float((a * b).sum() / denominator)
