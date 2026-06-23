from __future__ import annotations

import math

import torch
from torch import Tensor

from mycorrhiza.substrate.corpus import G_MIN, CrossModalBatch


def mutual_information_bound(
    variance: float, mask_ratio: float, masked_loss: float, constant: float = 0.0
) -> float:
    denominator = variance - (1.0 - mask_ratio) * (variance - masked_loss)
    if denominator <= 0.0:
        return math.inf
    return 0.5 * math.log(variance / denominator) - constant


def is_admissible(n_shared_genes: int, has_cross_modal_bias: bool, g_min: int = G_MIN) -> bool:
    return n_shared_genes >= g_min and has_cross_modal_bias


def permutation_equivariant(
    encoder: torch.nn.Module, batch: CrossModalBatch, tolerance: float = 1e-4
) -> bool:
    encoder.eval()
    with torch.no_grad():
        reference = encoder(batch)
        shuffled = torch.randperm(batch.n_sc)
        permuted = _permute_single_cells(batch, shuffled)
        moved = encoder(permuted)
    expected = reference[shuffled]
    realised = moved[: batch.n_sc]
    return bool(torch.allclose(expected, realised, atol=tolerance))


def _permute_single_cells(batch: CrossModalBatch, order: Tensor) -> CrossModalBatch:
    n_sc = batch.n_sc
    full = torch.cat([order, torch.arange(n_sc, batch.expr.shape[0])])
    adjacency = batch.adjacency[full][:, full]
    return CrossModalBatch(
        expr=batch.expr[full],
        expr_masked=batch.expr_masked[full],
        mask=batch.mask[full],
        coords=batch.coords[full],
        modality=batch.modality[full],
        adjacency=adjacency,
        edges=batch.edges,
        pairs=batch.pairs,
        immune=batch.immune[full],
        label=batch.label[full],
        n_sc=batch.n_sc,
        n_sp=batch.n_sp,
    )
