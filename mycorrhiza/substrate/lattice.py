from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def neighbour_index(coords: Tensor, k: int) -> Tensor:
    if coords.shape[0] <= 1:
        return torch.zeros((coords.shape[0], 0), dtype=torch.long)
    distance = torch.cdist(coords, coords)
    distance.fill_diagonal_(float("inf"))
    width = min(k, coords.shape[0] - 1)
    return distance.topk(width, largest=False).indices


def spatial_adjacency(coords: Tensor, modality: Tensor, k: int) -> Tensor:
    total = coords.shape[0]
    adjacency = torch.zeros((total, total), dtype=torch.bool)
    spot = (modality == 1).nonzero(as_tuple=False).flatten()
    if spot.numel() <= 1:
        return adjacency
    local = neighbour_index(coords[spot], k)
    rows = spot.repeat_interleave(local.shape[1])
    cols = spot[local.reshape(-1)]
    adjacency[rows, cols] = True
    return adjacency


def spatial_edges(coords: Tensor, modality: Tensor, k: int) -> Tensor:
    adjacency = spatial_adjacency(coords, modality, k)
    edges = adjacency.nonzero(as_tuple=False)
    return edges


def _centred_norm(matrix: Tensor) -> Tensor:
    centred = matrix - matrix.mean(dim=1, keepdim=True)
    norm = centred.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return centred / norm


def mutual_nearest_pairs(expr_sc: Tensor, expr_sp: Tensor, r_min: float) -> Tensor:
    if expr_sc.shape[0] == 0 or expr_sp.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    left = _centred_norm(expr_sc)
    right = _centred_norm(expr_sp)
    correlation = left @ right.t()
    sc_to_sp = correlation.argmax(dim=1)
    sp_to_sc = correlation.argmax(dim=0)
    pairs = []
    for i in range(expr_sc.shape[0]):
        j = int(sc_to_sp[i].item())
        if int(sp_to_sc[j].item()) == i and float(correlation[i, j].item()) >= r_min:
            pairs.append((i, j))
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def knn_connectivity(embeddings: Tensor, k: int) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    from sklearn.neighbors import NearestNeighbors

    array = embeddings.detach().cpu().numpy()
    width = min(k + 1, array.shape[0])
    engine = NearestNeighbors(n_neighbors=width)
    engine.fit(array)
    graph = engine.kneighbors_graph(array, mode="connectivity")
    coo = graph.tocoo()
    return coo.row, coo.col
