from __future__ import annotations

import torch
from torch import Tensor


def masked_reconstruction(reconstruction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return reconstruction.new_zeros(())
    residual = (reconstruction - target)[mask]
    return residual.pow(2).mean()


def residual_variance(target: Tensor) -> Tensor:
    return torch.var(target, unbiased=False)
