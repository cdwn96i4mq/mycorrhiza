from __future__ import annotations

from typing import Tuple

import gin
from torch import Tensor, nn

from mycorrhiza.network.hyphae import CrossModalAttention
from mycorrhiza.network.reconstruct import Reconstructor
from mycorrhiza.network.sporocarp import SparseMixture
from mycorrhiza.network.thallus import Thallus
from mycorrhiza.substrate.corpus import CrossModalBatch


@gin.configurable
class ImmunoFounder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        dim: int = 256,
        depth: int = 12,
        heads: int = 8,
        hidden: int = 1024,
        experts: int = 8,
        top_k: int = 2,
        use_cross: bool = True,
        use_spatial: bool = True,
        use_rope: bool = True,
        block_cross: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Thallus(
            n_genes,
            dim=dim,
            depth=depth,
            heads=heads,
            hidden=hidden,
            experts=experts,
            top_k=top_k,
            use_cross=use_cross,
            use_spatial=use_spatial,
            use_rope=use_rope,
            block_cross=block_cross,
        )
        self.reconstructor = Reconstructor(n_genes, dim=dim)

    def embed(self, batch: CrossModalBatch) -> Tensor:
        return self.encoder(batch)

    def forward(self, batch: CrossModalBatch) -> Tuple[Tensor, Tensor]:
        embeddings = self.encoder(batch)
        reconstruction = self.reconstructor(embeddings)
        return embeddings, reconstruction


__all__ = [
    "ImmunoFounder",
    "Thallus",
    "CrossModalAttention",
    "SparseMixture",
    "Reconstructor",
]
