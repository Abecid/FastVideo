# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from fastvideo.train.methods.rl.common.local_asfmc import (
    local_anchor_gaussian_parameters,
)


def test_local_anchor_matches_flowmap_grpo_formula() -> None:
    target = torch.full((2, 3), 2.0)
    reverse_velocity = torch.full_like(target, 0.5)
    mean, std = local_anchor_gaussian_parameters(
        target,
        reverse_velocity,
        torch.tensor([750.0, 750.0]),
        num_train_timesteps=1000,
        delta_fraction=0.03,
        noise_scale=0.7,
        terminal_base_sigma=0.05,
    )

    paper_time = 0.25
    lambda_sq = 0.7**2
    expected_mean = 2.0 - 0.03 * lambda_sq * (
        2.0 / paper_time + 0.5
    )
    expected_std = (
        0.7
        * math.sqrt(2.0 * 0.75 / paper_time)
        * math.sqrt(0.03)
    )
    assert torch.allclose(mean, torch.full_like(mean, expected_mean))
    assert std.shape == (2, 1)
    assert torch.allclose(std, torch.full_like(std, expected_std))


def test_local_anchor_is_differentiable_through_policy_mean() -> None:
    target = torch.tensor([[1.5]], requires_grad=True)
    reverse_velocity = torch.tensor([[0.2]], requires_grad=True)
    mean, std = local_anchor_gaussian_parameters(
        target,
        reverse_velocity,
        600.0,
        num_train_timesteps=1000,
        delta_fraction=0.03,
        noise_scale=0.7,
        terminal_base_sigma=0.05,
    )
    (mean.sum() + 0.0 * std.sum()).backward()
    assert target.grad is not None
    assert reverse_velocity.grad is not None
    assert torch.isfinite(target.grad).all()
    assert torch.isfinite(reverse_velocity.grad).all()


def test_terminal_base_sigma_stabilizes_near_noise_target() -> None:
    target = torch.ones((1, 2))
    velocity = torch.zeros_like(target)
    mean, std = local_anchor_gaussian_parameters(
        target,
        velocity,
        999.0,
        num_train_timesteps=1000,
        delta_fraction=0.03,
        noise_scale=0.7,
        terminal_base_sigma=0.05,
    )
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.all(std > 0)


@pytest.mark.parametrize(
    ("noise_scale", "delta_fraction", "terminal_base_sigma"),
    [
        (0.0, 0.03, 0.05),
        (0.7, 0.0, 0.05),
        (0.7, 1.0, 0.05),
        (0.7, 0.03, 0.0),
    ],
)
def test_local_anchor_rejects_degenerate_policy_parameters(
    noise_scale: float,
    delta_fraction: float,
    terminal_base_sigma: float,
) -> None:
    with pytest.raises(ValueError):
        local_anchor_gaussian_parameters(
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            500.0,
            num_train_timesteps=1000,
            delta_fraction=delta_fraction,
            noise_scale=noise_scale,
            terminal_base_sigma=terminal_base_sigma,
        )
