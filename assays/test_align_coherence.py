from __future__ import annotations

import torch

from mycorrhiza.substrate.lattice import mutual_nearest_pairs
from mycorrhiza.tropism.align import cross_modal_alignment
from mycorrhiza.tropism.coherence import spatial_coherence


def test_mutual_nearest_pairs_recovers_shared_anchors() -> None:
    torch.manual_seed(0)
    shared = torch.randn(5, 12)
    expr_sc = torch.cat([shared, torch.randn(3, 12)], dim=0)
    expr_sp = torch.cat([shared, torch.randn(2, 12)], dim=0)
    pairs = mutual_nearest_pairs(expr_sc, expr_sp, r_min=0.9)
    recovered = {(int(i), int(j)) for i, j in pairs.tolist()}
    for anchor in range(5):
        assert (anchor, anchor) in recovered


def test_alignment_is_mean_squared_distance() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [1.0, 1.0]])
    pairs = torch.tensor([[0, 1], [2, 3]])
    loss = cross_modal_alignment(embeddings, pairs)
    assert abs(float(loss) - 12.5) < 1e-6


def test_alignment_empty_pairs_is_zero() -> None:
    embeddings = torch.randn(4, 3)
    pairs = torch.zeros((0, 2), dtype=torch.long)
    assert float(cross_modal_alignment(embeddings, pairs)) == 0.0


def test_spatial_coherence_relu_gate() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    expr = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    edges = torch.tensor([[0, 1]])
    loss = spatial_coherence(embeddings, expr, edges, tau=0.3)
    assert abs(float(loss) - (1.0 - 0.3) * 4.0) < 1e-5

    orthogonal = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gated = spatial_coherence(embeddings, orthogonal, edges, tau=0.3)
    assert float(gated) == 0.0
