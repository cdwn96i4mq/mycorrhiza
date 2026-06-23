from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from mycorrhiza.keys.annotation import concordance


def zero_shot_concordance(
    reference: Tensor,
    reference_labels: NDArray[np.int_],
    query: Tensor,
    query_labels: NDArray[np.int_],
    k: int = 15,
) -> float:
    from sklearn.neighbors import KNeighborsClassifier

    width = min(k, reference.shape[0])
    classifier = KNeighborsClassifier(n_neighbors=width)
    classifier.fit(reference.detach().cpu().numpy(), reference_labels)
    predicted = classifier.predict(query.detach().cpu().numpy())
    return concordance(query_labels, np.asarray(predicted, dtype=np.int_))
