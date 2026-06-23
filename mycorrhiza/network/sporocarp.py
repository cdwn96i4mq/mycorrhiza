from __future__ import annotations

import gin
import torch
from torch import Tensor, nn


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden, dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(self.act(self.up(hidden)))


@gin.configurable
class SparseMixture(nn.Module):
    def __init__(
        self, dim: int = 256, hidden: int = 1024, experts: int = 8, top_k: int = 2
    ) -> None:
        super().__init__()
        if top_k > experts:
            raise ValueError("top_k cannot exceed the number of experts")
        self.top_k = top_k
        self.gate = nn.Linear(dim, experts)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(experts)])

    def forward(self, hidden: Tensor) -> Tensor:
        scores = self.gate(hidden)
        weight, choice = scores.topk(self.top_k, dim=-1)
        weight = weight.softmax(dim=-1)
        output = torch.zeros_like(hidden)
        for slot in range(self.top_k):
            selected = choice[:, slot]
            gate = weight[:, slot].unsqueeze(-1)
            for index, expert in enumerate(self.experts):
                rows = selected == index
                if bool(rows.any()):
                    output[rows] = output[rows] + gate[rows] * expert(hidden[rows])
        return output
