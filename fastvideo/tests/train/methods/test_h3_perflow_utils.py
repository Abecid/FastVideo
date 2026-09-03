# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.knowledge_distillation.h3_perflow_utils import (
    compute_h3_perflow_losses,
    interpolate_sigma_segment,
    sample_segment_timestep,
    weighted_regression_loss,
)


def test_sigma_interpolation_preserves_signed_velocity() -> None:
    current = torch.tensor([[2.0, 4.0]])
    next_state = torch.tensor([[0.0, 2.0]])
    sample = interpolate_sigma_segment(
        current,
        next_state,
        sigma_current=torch.tensor([1.0]),
        sigma_next=torch.tensor([0.25]),
        sigma_query=torch.tensor([0.625]),
    )

    torch.testing.assert_close(sample.interpolation_fraction, torch.tensor([0.5]))
    torch.testing.assert_close(sample.state, torch.tensor([[1.0, 3.0]]))
    torch.testing.assert_close(
        sample.velocity_target,
        torch.full((1, 2), 2.0 / 0.75),
    )
    torch.testing.assert_close(sample.sigma_delta, torch.tensor([-0.75]))


def test_reversing_segment_orientation_keeps_same_velocity_field() -> None:
    forward = interpolate_sigma_segment(
        torch.tensor([[2.0, 4.0]]),
        torch.tensor([[0.0, 2.0]]),
        sigma_current=1.0,
        sigma_next=0.25,
        sigma_query=0.625,
    )
    reverse = interpolate_sigma_segment(
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[2.0, 4.0]]),
        sigma_current=0.25,
        sigma_next=1.0,
        sigma_query=0.625,
    )
    torch.testing.assert_close(forward.state, reverse.state)
    torch.testing.assert_close(forward.velocity_target, reverse.velocity_target)


def test_teacher_segment_is_detached_and_exact_field_has_zero_loss() -> None:
    current = torch.tensor([[1.0, 3.0]], requires_grad=True)
    next_state = torch.tensor([[0.0, 2.0]], requires_grad=True)
    sample = interpolate_sigma_segment(
        current,
        next_state,
        sigma_current=0.8,
        sigma_next=0.3,
        sigma_query=0.55,
    )
    assert not sample.state.requires_grad
    assert not sample.velocity_target.requires_grad

    prediction = sample.velocity_target.clone().requires_grad_(True)
    loss = weighted_regression_loss(prediction, sample.velocity_target)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
    assert current.grad is None
    assert next_state.grad is None


def test_query_outside_segment_and_degenerate_interval_fail() -> None:
    current = torch.zeros(1, 2)
    next_state = torch.ones(1, 2)
    with pytest.raises(ValueError, match="outside"):
        interpolate_sigma_segment(
            current,
            next_state,
            sigma_current=1.0,
            sigma_next=0.5,
            sigma_query=0.4,
        )
    with pytest.raises(ValueError, match="degenerate sigma"):
        interpolate_sigma_segment(
            current,
            next_state,
            sigma_current=0.5,
            sigma_next=0.5,
            sigma_query=0.5,
        )


def test_shared_base_timestep_sampling_stays_in_segment() -> None:
    query, fraction = sample_segment_timestep(
        torch.tensor([1000.0, 500.0]),
        torch.tensor([750.0, 250.0]),
        batch_size=2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    assert bool(((fraction >= 0.0) & (fraction < 1.0)).all())
    expected = torch.tensor([1000.0, 500.0]) + fraction * torch.tensor([-250.0, -250.0])
    torch.testing.assert_close(query, expected)
    assert bool((query <= torch.tensor([1000.0, 500.0])).all())
    assert bool((query >= torch.tensor([750.0, 250.0])).all())


def test_weighted_losses_normalize_equal_topq_mass() -> None:
    prediction = torch.tensor([[1.0, 3.0], [2.0, 6.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    weights = torch.tensor([0.5, 0.5])

    weighted = weighted_regression_loss(
        prediction,
        target,
        sample_weight=weights,
        loss_type="mse",
    )
    unweighted = weighted_regression_loss(prediction, target, loss_type="mse")
    torch.testing.assert_close(weighted, unweighted)

    unequal = weighted_regression_loss(
        prediction,
        target,
        sample_weight=torch.tensor([1.0, 3.0]),
        loss_type="mse",
    )
    expected = (1.0 * 5.0 + 3.0 * 20.0) / 4.0
    torch.testing.assert_close(unequal, torch.tensor(expected))


def test_huber_loss_has_expected_piecewise_value_and_gradient() -> None:
    prediction = torch.tensor([[0.5, 2.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = weighted_regression_loss(
        prediction,
        target,
        loss_type="huber",
        huber_delta=1.0,
    )
    # mean([0.5 * 0.5^2, 1 * (2 - 0.5)])
    torch.testing.assert_close(loss, torch.tensor((0.125 + 1.5) / 2.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[0.25, 0.5]]))


def test_packed_modalities_are_reduced_separately() -> None:
    prediction = torch.cat(
        [
            torch.ones(1, 100),
            torch.full((1, 1), 2.0),
        ],
        dim=1,
    )
    losses = compute_h3_perflow_losses(
        prediction,
        video_target=torch.zeros(1, 100),
        audio_target=torch.zeros(1, 1),
        video_slice=slice(0, 100),
        audio_slice=slice(100, 101),
        sample_weight=torch.tensor([0.5]),
        audio_loss_weight=1.0,
        loss_type="mse",
    )
    torch.testing.assert_close(losses["video_perflow_loss"], torch.tensor(1.0))
    torch.testing.assert_close(losses["audio_perflow_loss"], torch.tensor(4.0))
    torch.testing.assert_close(losses["total_loss"], torch.tensor(5.0))


def test_function_anchor_is_optional_and_detached() -> None:
    prediction = torch.tensor([[1.0, 3.0]], requires_grad=True)
    reference = torch.tensor([[0.0, 1.0]], requires_grad=True)
    losses = compute_h3_perflow_losses(
        prediction,
        video_target=torch.tensor([[1.0]]),
        audio_target=torch.tensor([[3.0]]),
        video_slice=slice(0, 1),
        audio_slice=slice(1, 2),
        sample_weight=1.0,
        anchor_weight=0.25,
        reference_prediction=reference,
    )
    torch.testing.assert_close(losses["function_anchor_loss"], torch.tensor(2.5))
    torch.testing.assert_close(losses["total_loss"], torch.tensor(0.625))
    losses["total_loss"].backward()
    assert reference.grad is None
    assert prediction.grad is not None


@pytest.mark.parametrize(
    "weight",
    [
        torch.tensor([-1.0]),
        torch.tensor([float("nan")]),
        torch.tensor([0.0]),
    ],
)
def test_invalid_sample_weights_fail(weight: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        weighted_regression_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            sample_weight=weight,
        )
