from __future__ import annotations

import torch

from mycorrhiza.identification.bounds import permutation_equivariant
from mycorrhiza.network import ImmunoFounder
from mycorrhiza.network.hyphae import CrossModalAttention
from mycorrhiza.substrate.corpus import CrossModalBatch


def _tiny_model(n_genes: int) -> ImmunoFounder:
    return ImmunoFounder(n_genes=n_genes, dim=16, depth=2, heads=2, hidden=16, experts=2, top_k=1)


def test_encoder_is_permutation_equivariant_within_modality(
    small_batch: CrossModalBatch,
) -> None:
    model = _tiny_model(small_batch.expr.shape[1])
    assert permutation_equivariant(model.encoder, small_batch, tolerance=1e-3)


def test_rope_logits_depend_only_on_relative_coordinates() -> None:
    torch.manual_seed(0)
    attention = CrossModalAttention(
        dim=16, heads=2, use_cross=False, use_spatial=False, use_rope=True
    )
    attention.eval()
    tokens = torch.randn(8, 16)
    coords = torch.randn(8, 2)
    modality = torch.ones(8, dtype=torch.long)
    adjacency = torch.zeros(8, 8, dtype=torch.bool)
    with torch.no_grad():
        base = attention(tokens, coords, modality, adjacency)
        shifted = attention(tokens, coords + torch.tensor([3.0, -2.0]), modality, adjacency)
    assert torch.allclose(base, shifted, atol=1e-4)


def test_modality_bias_matches_definition() -> None:
    attention = CrossModalAttention(dim=16, heads=2, alpha_init=0.7)
    modality = torch.tensor([0, 0, 1, 1])
    bias = attention._modality_bias(modality)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.7, 0.7],
            [0.0, 0.0, 0.7, 0.7],
            [0.7, 0.7, 0.0, 0.0],
            [0.7, 0.7, 0.0, 0.0],
        ]
    )
    assert torch.allclose(bias, expected, atol=1e-6)
