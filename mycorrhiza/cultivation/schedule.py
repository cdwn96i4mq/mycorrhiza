from __future__ import annotations

import math
from typing import Iterable, Tuple

import gin
import torch
from torch import Tensor


@gin.configurable
def build_optimizer(
    parameters: Iterable[Tensor],
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)


@gin.configurable
def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int = 2000, max_steps: int = 100000
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
