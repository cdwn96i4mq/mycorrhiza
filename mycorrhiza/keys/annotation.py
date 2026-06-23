from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def concordance(truth: NDArray[np.int_], predicted: NDArray[np.int_]) -> float:
    return float((truth == predicted).mean())


def macro_f1(truth: NDArray[np.int_], predicted: NDArray[np.int_]) -> float:
    labels = np.unique(truth)
    scores = []
    for label in labels:
        true_positive = float(((predicted == label) & (truth == label)).sum())
        false_positive = float(((predicted == label) & (truth != label)).sum())
        false_negative = float(((predicted != label) & (truth == label)).sum())
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0
