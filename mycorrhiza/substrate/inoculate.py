from __future__ import annotations

import gin
import torch
from torch import Tensor, nn


@gin.configurable
class Inoculator(nn.Module):
    def __init__(self, n_genes: int, dim: int = 256, n_modalities: int = 2) -> None:
        super().__init__()
        self.expr = nn.Linear(n_genes, dim)
        self.modality = nn.Embedding(n_modalities, dim)

    def forward(self, expr: Tensor, modality: Tensor) -> Tensor:
        return self.expr(expr) + self.modality(modality)


def _rotate_axis(block: Tensor, angle: Tensor) -> Tensor:
    even = block[..., 0]
    odd = block[..., 1]
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1)


class RotaryCoordinate(nn.Module):
    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float = 10000.0, model_dim: int = 256) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                "head dimension must be divisible by four for two-axis rotary encoding"
            )
        self.head_dim = head_dim
        self.pairs = head_dim // 4
        index = torch.arange(self.pairs, dtype=torch.float32)
        inv_freq = base ** (-4.0 * index / float(model_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tokens: Tensor, coords: Tensor) -> Tensor:
        total, heads, _ = tokens.shape
        half = self.head_dim // 2
        angle_x = coords[:, 0:1] * self.inv_freq
        angle_y = coords[:, 1:2] * self.inv_freq
        axis_x = tokens[..., :half].reshape(total, heads, self.pairs, 2)
        axis_y = tokens[..., half:].reshape(total, heads, self.pairs, 2)
        rotated_x = _rotate_axis(axis_x, angle_x).reshape(total, heads, half)
        rotated_y = _rotate_axis(axis_y, angle_y).reshape(total, heads, half)
        return torch.cat((rotated_x, rotated_y), dim=-1)
