from __future__ import annotations

from mycorrhiza.identification.bounds import is_admissible, mutual_information_bound


def test_bound_decreases_with_reconstruction_loss() -> None:
    variance = 1.0
    mask_ratio = 0.3
    tighter = mutual_information_bound(variance, mask_ratio, masked_loss=0.05)
    looser = mutual_information_bound(variance, mask_ratio, masked_loss=0.4)
    assert tighter > looser


def test_bound_is_zero_when_loss_equals_variance() -> None:
    value = mutual_information_bound(1.0, 0.3, masked_loss=1.0)
    assert abs(value) < 1e-9


def test_admissibility_requires_shared_genes_and_cross_modal_bias() -> None:
    assert is_admissible(600, has_cross_modal_bias=True)
    assert not is_admissible(400, has_cross_modal_bias=True)
    assert not is_admissible(600, has_cross_modal_bias=False)
