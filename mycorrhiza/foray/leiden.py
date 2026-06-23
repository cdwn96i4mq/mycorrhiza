from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from mycorrhiza.substrate.lattice import knn_connectivity


def leiden_clusters(
    embeddings: Tensor, k: int = 15, resolution: float = 1.0, seed: int = 0
) -> NDArray[np.int_]:
    import igraph
    import leidenalg

    total = embeddings.shape[0]
    rows, cols = knn_connectivity(embeddings, k)
    edges = list(zip(rows.tolist(), cols.tolist()))
    graph = igraph.Graph(n=total, edges=edges, directed=False)
    graph.simplify(multiple=True, loops=True)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=np.int_)
