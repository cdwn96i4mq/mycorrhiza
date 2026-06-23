from __future__ import annotations

import gin
from torch import Tensor, nn

from mycorrhiza.network.hyphae import CrossModalAttention
from mycorrhiza.network.sporocarp import SparseMixture
from mycorrhiza.substrate.corpus import CrossModalBatch
from mycorrhiza.substrate.inoculate import Inoculator


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        hidden: int,
        experts: int,
        top_k: int,
        use_cross: bool,
        use_spatial: bool,
        use_rope: bool,
        block_cross: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = CrossModalAttention(
            dim=dim,
            heads=heads,
            use_cross=use_cross,
            use_spatial=use_spatial,
            use_rope=use_rope,
            block_cross=block_cross,
        )
        self.mix_norm = nn.LayerNorm(dim)
        self.mix = SparseMixture(dim=dim, hidden=hidden, experts=experts, top_k=top_k)

    def forward(
        self, hidden: Tensor, coords: Tensor, modality: Tensor, adjacency: Tensor
    ) -> Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden), coords, modality, adjacency)
        hidden = hidden + self.mix(self.mix_norm(hidden))
        return hidden


@gin.configurable
class Thallus(nn.Module):
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
        self.inoculate = Inoculator(n_genes, dim)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim,
                    heads,
                    hidden,
                    experts,
                    top_k,
                    use_cross,
                    use_spatial,
                    use_rope,
                    block_cross,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: CrossModalBatch) -> Tensor:
        hidden = self.inoculate(batch.expr_masked, batch.modality)
        for block in self.blocks:
            hidden = block(hidden, batch.coords, batch.modality, batch.adjacency)
        return self.norm(hidden)
