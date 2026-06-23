from __future__ import annotations

import numpy as np

from mycorrhiza.foray.exclusion import exclusion_score, flag_zones


def test_cold_cluster_scores_near_one() -> None:
    labels = np.array([0, 0, 1, 1])
    immune = np.array([0.0, 0.0, 1.0, 1.0])
    neighbours = np.array([[2, 3], [2, 3], [0, 1], [0, 1]])
    scores = exclusion_score(labels, immune, neighbours, eps=1e-3)
    assert scores[0] > 0.99
    assert scores[1] < -0.5


def test_equal_density_scores_zero() -> None:
    labels = np.array([0, 0, 1, 1])
    immune = np.array([0.5, 0.5, 0.5, 0.5])
    neighbours = np.array([[2, 3], [2, 3], [0, 1], [0, 1]])
    scores = exclusion_score(labels, immune, neighbours, eps=1e-9)
    assert abs(scores[0]) < 1e-6


def test_flag_zones_thresholds() -> None:
    scores = {0: 0.8, 1: 0.2, 2: 0.6}
    flagged = flag_zones(scores, threshold=0.5)
    assert flagged.tolist() == [0, 2]
