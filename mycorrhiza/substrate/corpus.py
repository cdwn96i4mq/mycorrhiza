from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gin
import numpy as np
import torch
from torch import Tensor

from mycorrhiza.substrate.lattice import mutual_nearest_pairs, spatial_adjacency, spatial_edges

G_MIN = 500


@dataclass
class CrossModalBatch:
    expr: Tensor
    expr_masked: Tensor
    mask: Tensor
    coords: Tensor
    modality: Tensor
    adjacency: Tensor
    edges: Tensor
    pairs: Tensor
    immune: Tensor
    label: Tensor
    n_sc: int
    n_sp: int

    def to(self, device: torch.device) -> "CrossModalBatch":
        return CrossModalBatch(
            expr=self.expr.to(device),
            expr_masked=self.expr_masked.to(device),
            mask=self.mask.to(device),
            coords=self.coords.to(device),
            modality=self.modality.to(device),
            adjacency=self.adjacency.to(device),
            edges=self.edges.to(device),
            pairs=self.pairs.to(device),
            immune=self.immune.to(device),
            label=self.label.to(device),
            n_sc=self.n_sc,
            n_sp=self.n_sp,
        )


def apply_mask(expr: Tensor, ratio: float, generator: torch.Generator) -> Tuple[Tensor, Tensor]:
    draw = torch.rand(expr.shape, generator=generator)
    mask = draw < ratio
    masked = expr.masked_fill(mask, 0.0)
    return masked, mask


def assemble(
    expr_sc: Tensor,
    expr_sp: Tensor,
    coords_sp: Tensor,
    immune_sc: Tensor,
    immune_sp: Tensor,
    label_sc: Tensor,
    label_sp: Tensor,
    mask_ratio: float,
    neighbours: int,
    r_min: float,
    generator: torch.Generator,
) -> CrossModalBatch:
    n_sc = expr_sc.shape[0]
    n_sp = expr_sp.shape[0]
    expr = torch.cat([expr_sc, expr_sp], dim=0)
    coords = torch.cat([torch.zeros((n_sc, 2)), coords_sp], dim=0)
    modality = torch.cat([torch.zeros(n_sc, dtype=torch.long), torch.ones(n_sp, dtype=torch.long)])
    immune = torch.cat([immune_sc, immune_sp]).float()
    label = torch.cat([label_sc, label_sp]).long()
    masked, mask = apply_mask(expr, mask_ratio, generator)
    adjacency = spatial_adjacency(coords, modality, neighbours)
    edges = spatial_edges(coords, modality, neighbours)
    local_pairs = mutual_nearest_pairs(expr_sc, expr_sp, r_min)
    if local_pairs.shape[0] > 0:
        pairs = torch.stack([local_pairs[:, 0], local_pairs[:, 1] + n_sc], dim=1)
    else:
        pairs = torch.zeros((0, 2), dtype=torch.long)
    return CrossModalBatch(
        expr=expr,
        expr_masked=masked,
        mask=mask,
        coords=coords,
        modality=modality,
        adjacency=adjacency,
        edges=edges,
        pairs=pairs,
        immune=immune,
        label=label,
        n_sc=n_sc,
        n_sp=n_sp,
    )


@gin.configurable
def synthetic_corpus(
    n_sc: int = 128,
    n_sp: int = 144,
    n_genes: int = 64,
    n_types: int = 5,
    mask_ratio: float = 0.30,
    neighbours: int = 6,
    r_min: float = 0.6,
    seed: int = 0,
) -> CrossModalBatch:
    rng = np.random.default_rng(seed)
    signal_genes = np.array([0, 1, 2])
    profiles = rng.normal(size=(n_types, n_genes)).astype(np.float32)

    sc_type = rng.integers(0, n_types, size=n_sc)
    expr_sc = profiles[sc_type] + rng.normal(scale=0.3, size=(n_sc, n_genes)).astype(np.float32)
    immune_sc = (sc_type < 2).astype(np.float32)

    side = int(np.ceil(np.sqrt(n_sp)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)
    grid = grid[:n_sp].astype(np.float32)
    centre = grid.mean(axis=0)
    radius = np.linalg.norm(grid - centre, axis=1)
    boundary = (radius > np.quantile(radius, 0.55)) & (radius < np.quantile(radius, 0.85))

    sp_type = rng.integers(0, n_types, size=n_sp)
    expr_sp = profiles[sp_type] + rng.normal(scale=0.3, size=(n_sp, n_genes)).astype(np.float32)
    expr_sp[
        boundary[:, None].repeat(n_genes, axis=1) & np.isin(np.arange(n_genes), signal_genes)
    ] += 3.0
    immune_sp = (sp_type < 2).astype(np.float32)
    immune_sp[boundary] = 0.0

    generator = torch.Generator()
    generator.manual_seed(seed)
    return assemble(
        torch.from_numpy(expr_sc),
        torch.from_numpy(expr_sp),
        torch.from_numpy(grid),
        torch.from_numpy(immune_sc),
        torch.from_numpy(immune_sp),
        torch.from_numpy(sc_type.astype(np.int64)),
        torch.from_numpy(sp_type.astype(np.int64)),
        mask_ratio,
        neighbours,
        r_min,
        generator,
    )
