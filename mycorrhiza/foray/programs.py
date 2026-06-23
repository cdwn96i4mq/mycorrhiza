from __future__ import annotations

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from mycorrhiza.keys.stats import benjamini_hochberg


def _rank_sum_pvalue(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    from scipy.stats import mannwhitneyu

    if left.size == 0 or right.size == 0:
        return 1.0
    try:
        _, pvalue = mannwhitneyu(left, right, alternative="two-sided")
    except ValueError:
        return 1.0
    return float(pvalue)


def cluster_signature(
    expr: NDArray[np.float64],
    cluster_labels: NDArray[np.int_],
    cluster: int,
    top_k: int = 50,
    fdr: float = 0.05,
) -> List[int]:
    members = cluster_labels == cluster
    rest = ~members
    n_genes = expr.shape[1]
    pvalues = np.ones(n_genes, dtype=np.float64)
    effect = np.zeros(n_genes, dtype=np.float64)
    for gene in range(n_genes):
        pvalues[gene] = _rank_sum_pvalue(expr[members, gene], expr[rest, gene])
        effect[gene] = float(expr[members, gene].mean() - expr[rest, gene].mean())
    rejected, _ = benjamini_hochberg(pvalues, alpha=fdr)
    candidates = np.nonzero(rejected & (effect > 0.0))[0]
    ordered = candidates[np.argsort(-effect[candidates])]
    return ordered[:top_k].tolist()


def discover_programs(
    expr: NDArray[np.float64],
    cluster_labels: NDArray[np.int_],
    flagged: NDArray[np.int_],
    top_k: int = 50,
    fdr: float = 0.05,
) -> Dict[int, List[int]]:
    return {
        int(cluster): cluster_signature(expr, cluster_labels, int(cluster), top_k, fdr)
        for cluster in flagged
    }


def _jaccard(left: List[int], right: List[int]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    union = len(a | b)
    return 0.0 if union == 0 else len(a & b) / union


def merge_programs(programs: Dict[int, List[int]], threshold: float = 0.5) -> List[List[int]]:
    remaining = list(programs.values())
    merged: List[List[int]] = []
    for signature in remaining:
        placed = False
        for index, existing in enumerate(merged):
            if _jaccard(signature, existing) >= threshold:
                merged[index] = sorted(set(existing) | set(signature))
                placed = True
                break
        if not placed:
            merged.append(sorted(set(signature)))
    return merged
