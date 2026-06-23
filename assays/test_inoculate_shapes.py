from __future__ import annotations

import torch

from mycorrhiza.substrate.corpus import CrossModalBatch
from mycorrhiza.substrate.inoculate import Inoculator, RotaryCoordinate


def test_inoculator_emits_one_token_per_cell(small_batch: CrossModalBatch) -> None:
    tokenizer = Inoculator(n_genes=small_batch.expr.shape[1], dim=32)
    tokens = tokenizer(small_batch.expr_masked, small_batch.modality)
    assert tokens.shape == (small_batch.expr.shape[0], 32)
    assert tokens.dtype == torch.float32


def test_rotary_preserves_shape() -> None:
    rotary = RotaryCoordinate(head_dim=16, model_dim=32)
    tokens = torch.randn(5, 2, 16)
    coords = torch.randn(5, 2)
    rotated = rotary(tokens, coords)
    assert rotated.shape == tokens.shape


def test_mask_is_boolean_and_bounded(small_batch: CrossModalBatch) -> None:
    assert small_batch.mask.dtype == torch.bool
    assert small_batch.mask.shape == small_batch.expr.shape
    masked_entries = small_batch.expr_masked[small_batch.mask]
    assert torch.allclose(masked_entries, torch.zeros_like(masked_entries))


def test_rotary_requires_divisible_head_dim() -> None:
    try:
        RotaryCoordinate(head_dim=6, model_dim=12)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible head dimension")
