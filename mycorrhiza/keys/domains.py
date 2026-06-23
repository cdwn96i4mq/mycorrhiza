from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def adjusted_rand_index(truth: NDArray[np.int_], predicted: NDArray[np.int_]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(truth, predicted))


def normalised_mutual_information(truth: NDArray[np.int_], predicted: NDArray[np.int_]) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(truth, predicted))
