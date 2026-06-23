from __future__ import annotations

from mycorrhiza.cultivation.grower import Grower
from mycorrhiza.network import ImmunoFounder
from mycorrhiza.substrate.corpus import CrossModalBatch


def test_single_batch_overfits(small_batch: CrossModalBatch) -> None:
    model = ImmunoFounder(
        n_genes=small_batch.expr.shape[1], dim=32, depth=2, heads=2, hidden=64, experts=2, top_k=1
    )
    grower = Grower(
        model,
        lr=5e-3,
        warmup_steps=1,
        max_steps=200,
        grad_accum=1,
        device="cpu",
        seed=0,
    )
    history = grower.fit([small_batch], epochs=150)
    assert history[-1] < history[0] * 0.5
