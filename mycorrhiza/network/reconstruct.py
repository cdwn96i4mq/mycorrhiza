from __future__ import annotations

import gin
from torch import Tensor, nn


@gin.configurable
class Reconstructor(nn.Module):
    def __init__(self, n_genes: int, dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Linear(dim, n_genes)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.head(embeddings)
