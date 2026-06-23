from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor


def spatial_coherence(embeddings: Tensor, expr: Tensor, edges: Tensor, tau: float = 0.3) -> Tensor:
    if edges.shape[0] == 0:
        return embeddings.new_zeros(())
    source = edges[:, 0]
    target = edges[:, 1]
    similarity = functional.cosine_similarity(expr[source], expr[target], dim=-1)
    weight = functional.relu(similarity - tau)
    distance = (embeddings[source] - embeddings[target]).pow(2).sum(dim=-1)
    return (weight * distance).mean()
