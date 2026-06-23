from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gin
from torch import Tensor

from mycorrhiza.substrate.corpus import CrossModalBatch
from mycorrhiza.tropism.align import cross_modal_alignment
from mycorrhiza.tropism.coherence import spatial_coherence
from mycorrhiza.tropism.masked import masked_reconstruction


@dataclass
class LossTerms:
    total: Tensor
    masked: Tensor
    align: Tensor
    spatial: Tensor

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total.item()),
            "masked": float(self.masked.item()),
            "align": float(self.align.item()),
            "spatial": float(self.spatial.item()),
        }


@gin.configurable
def composite_loss(
    model_output: Tuple[Tensor, Tensor],
    batch: CrossModalBatch,
    lam_align: float = 0.5,
    lam_spatial: float = 0.3,
    use_mask: bool = True,
    tau: float = 0.3,
) -> LossTerms:
    embeddings, reconstruction = model_output
    masked = masked_reconstruction(reconstruction, batch.expr, batch.mask)
    align = cross_modal_alignment(embeddings, batch.pairs)
    spatial = spatial_coherence(embeddings, batch.expr, batch.edges, tau=tau)
    mask_weight = 1.0 if use_mask else 0.0
    total = mask_weight * masked + lam_align * align + lam_spatial * spatial
    return LossTerms(total=total, masked=masked, align=align, spatial=spatial)
