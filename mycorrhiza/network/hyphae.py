from __future__ import annotations

import gin
import torch
from torch import Tensor, nn

from mycorrhiza.substrate.inoculate import RotaryCoordinate

_NEG_INF = -1.0e9


@gin.configurable
class CrossModalAttention(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        alpha_init: float = 0.0,
        beta_init: float = 0.1,
        use_cross: bool = True,
        use_spatial: bool = True,
        use_rope: bool = True,
        block_cross: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.use_cross = use_cross
        self.use_spatial = use_spatial
        self.use_rope = use_rope
        self.block_cross = block_cross
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.rotary = RotaryCoordinate(self.head_dim, model_dim=dim)

    def _modality_bias(self, modality: Tensor) -> Tensor:
        different = modality.unsqueeze(0) != modality.unsqueeze(1)
        if self.block_cross:
            return torch.where(
                different,
                torch.full_like(different, _NEG_INF, dtype=torch.float32),
                torch.zeros_like(different, dtype=torch.float32),
            )
        if not self.use_cross:
            return torch.zeros_like(different, dtype=torch.float32)
        return self.alpha * different.float()

    def _spatial_bias(self, adjacency: Tensor) -> Tensor:
        if not self.use_spatial:
            return torch.zeros_like(adjacency, dtype=torch.float32)
        return self.beta * adjacency.float()

    def forward(
        self, hidden: Tensor, coords: Tensor, modality: Tensor, adjacency: Tensor
    ) -> Tensor:
        total = hidden.shape[0]
        query, key, value = self.to_qkv(hidden).chunk(3, dim=-1)
        query = query.view(total, self.heads, self.head_dim)
        key = key.view(total, self.heads, self.head_dim)
        value = value.view(total, self.heads, self.head_dim)
        if self.use_rope:
            query = self.rotary(query, coords)
            key = self.rotary(key, coords)
        logits = torch.einsum("thd,shd->hts", query, key) * self.scale
        bias = self._modality_bias(modality) + self._spatial_bias(adjacency)
        logits = logits + bias.unsqueeze(0)
        attention = logits.softmax(dim=-1)
        context = torch.einsum("hts,shd->thd", attention, value).reshape(total, -1)
        return self.proj(context)
