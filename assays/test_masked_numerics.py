from __future__ import annotations

import torch

from mycorrhiza.tropism.masked import masked_reconstruction


def test_masked_loss_gradient_matches_finite_difference() -> None:
    torch.manual_seed(0)
    prediction = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randn(6, 5, dtype=torch.float64)
    mask = torch.rand(6, 5) < 0.4
    loss = masked_reconstruction(prediction, target, mask)
    (analytic,) = torch.autograd.grad(loss, prediction)
    count = int(mask.sum())
    expected = torch.where(
        mask, 2.0 * (prediction.detach() - target) / count, torch.zeros_like(target)
    )
    assert torch.allclose(analytic, expected, atol=1e-9)


def test_masked_loss_central_difference() -> None:
    torch.manual_seed(1)
    prediction = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 4, dtype=torch.float64)
    mask = torch.ones(4, 4, dtype=torch.bool)
    loss = masked_reconstruction(prediction, target, mask)
    (analytic,) = torch.autograd.grad(loss, prediction)
    epsilon = 1e-6
    probe = prediction.detach().clone()
    probe[1, 2] += epsilon
    high = masked_reconstruction(probe, target, mask)
    probe[1, 2] -= 2 * epsilon
    low = masked_reconstruction(probe, target, mask)
    numerical = (high - low) / (2 * epsilon)
    assert abs(float(numerical) - float(analytic[1, 2])) < 1e-6


def test_masked_loss_empty_mask_is_zero() -> None:
    prediction = torch.randn(3, 3)
    target = torch.randn(3, 3)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    assert float(masked_reconstruction(prediction, target, mask)) == 0.0
