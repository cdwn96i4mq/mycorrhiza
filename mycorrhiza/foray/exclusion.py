from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray

EPSILON = 1e-3


def exclusion_score(
    cluster_labels: NDArray[np.int_],
    immune: NDArray[np.float64],
    neighbours: NDArray[np.int_],
    eps: float = EPSILON,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for cluster in np.unique(cluster_labels):
        members = cluster_labels == cluster
        ring = np.unique(neighbours[members].reshape(-1))
        ring = ring[~np.isin(ring, np.nonzero(members)[0])]
        density_cluster = float(immune[members].mean()) if members.any() else 0.0
        density_ring = float(immune[ring].mean()) if ring.size > 0 else density_cluster
        scores[int(cluster)] = 1.0 - (density_cluster + eps) / (density_ring + eps)
    return scores


def flag_zones(scores: Dict[int, float], threshold: float = 0.5) -> NDArray[np.int_]:
    flagged = [cluster for cluster, value in scores.items() if value > threshold]
    return np.asarray(sorted(flagged), dtype=np.int_)
