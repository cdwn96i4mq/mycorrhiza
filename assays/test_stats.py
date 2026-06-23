from __future__ import annotations

import numpy as np

from mycorrhiza.keys.stats import benjamini_hochberg, bootstrap_ci, cohen_dz, delong_test, roc_auc


def test_roc_auc_matches_reference() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.normal(size=200) + labels * 0.8
    assert abs(roc_auc(scores, labels) - float(roc_auc_score(labels, scores))) < 1e-9


def test_delong_identical_scores_have_unit_pvalue() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=120)
    scores = rng.normal(size=120) + labels
    auc_a, auc_b, pvalue = delong_test(scores, scores, labels)
    assert abs(auc_a - auc_b) < 1e-9
    assert pvalue > 0.99


def test_delong_separates_distinct_classifiers() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, size=400)
    strong = rng.normal(size=400) + labels * 2.0
    weak = rng.normal(size=400) + labels * 0.1
    auc_strong, auc_weak, pvalue = delong_test(strong, weak, labels)
    assert auc_strong > auc_weak
    assert pvalue < 0.05


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(loc=0.5, scale=0.1, size=80)
    low, high = bootstrap_ci(values, iterations=2000, seed=0)
    assert low < values.mean() < high


def test_benjamini_hochberg_rejects_small_pvalues() -> None:
    pvalues = np.array([0.001, 0.2, 0.04, 0.8])
    rejected, qvalues = benjamini_hochberg(pvalues, alpha=0.05)
    assert rejected[0]
    assert not rejected[3]
    assert np.all((qvalues >= 0.0) & (qvalues <= 1.0))


def test_cohen_dz_known_value() -> None:
    differences = np.array([1.0, 1.0, 1.0, 1.0])
    assert cohen_dz(differences) == 0.0
    spread = np.array([0.0, 2.0, 0.0, 2.0])
    assert cohen_dz(spread) > 0.0
