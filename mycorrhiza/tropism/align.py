from __future__ import annotations

from torch import Tensor


def cross_modal_alignment(embeddings: Tensor, pairs: Tensor) -> Tensor:
    if pairs.shape[0] == 0:
        return embeddings.new_zeros(())
    left = embeddings[pairs[:, 0]]
    right = embeddings[pairs[:, 1]]
    return (left - right).pow(2).sum(dim=-1).mean()
