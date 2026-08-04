# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import torch
import yaml

from examples.train.pt_pdd.core import (
    append_data_endpoint,
    centered_velocity_correction,
    clipped_grpo_loss,
    effective_sample_size,
    endpoint_anchor_parameters,
    finite_interval_velocity,
    gaussian_log_prob_mean,
    global_group_advantages,
    posterior_distillation_loss,
    posterior_tilted_regression_loss,
    reward_tilted_weights,
    validate_training_schedule,
    verify_group_partition,
)


def test_endpoint_anchor_is_exact_affine_conditional() -> None:
    clean = torch.full((2, 3, 4), 2.0)
    mean, std = endpoint_anchor_parameters(
        clean,
        torch.tensor(250.0),
        num_train_timesteps=1000,
    )
    torch.testing.assert_close(mean, torch.full_like(clean, 1.5))
    torch.testing.assert_close(std, torch.tensor([[[0.25]]]))


def test_schedule_requires_four_stochastic_plus_final() -> None:
    schedule = append_data_endpoint(
        torch.tensor([1000.0, 900.0, 750.0, 500.0, 250.0])
    )
    validate_training_schedule(schedule, stochastic_steps=4)
    assert schedule.shape == (6,)
    assert float(schedule[-1]) == 0.0


def test_reward_tilt_hits_half_group_ess_and_prefers_best() -> None:
    rewards = torch.tensor([0.0, 0.2, 0.5, 2.0, 3.0, 8.0])
    weights, temperature, ess = reward_tilted_weights(
        rewards,
        target_ess_ratio=0.5,
    )
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert abs(float(ess) - 3.0) < 1.0e-2
    assert int(weights.argmax()) == int(rewards.argmax())
    assert float(temperature) > 0.0


def test_degenerate_reward_tilt_is_uniform() -> None:
    rewards = torch.ones(6)
    weights, temperature, ess = reward_tilted_weights(
        rewards,
        target_ess_ratio=0.5,
    )
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 6.0))
    assert math.isinf(float(temperature))
    torch.testing.assert_close(ess, torch.tensor(6.0))


def test_finite_interval_velocity_recovers_anyflow_transition() -> None:
    source = torch.tensor([[[[4.0, 6.0]]]])
    velocity = torch.tensor([[[[2.0, -1.0]]]])
    source_t = torch.tensor(800.0)
    target_t = torch.tensor(300.0)
    delta = (source_t - target_t) / 1000.0
    target = source - delta * velocity
    recovered = finite_interval_velocity(
        source,
        target,
        source_t,
        target_t,
        num_train_timesteps=1000,
    )
    torch.testing.assert_close(recovered, velocity)


def test_uniform_posterior_gives_exact_zero_velocity_correction() -> None:
    candidates = torch.tensor(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
            [[[7.0, 8.0]]],
        ]
    )
    correction, diagnostics = centered_velocity_correction(
        candidates,
        torch.full((4,), 0.25),
        global_group_size=4,
    )
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    torch.testing.assert_close(
        diagnostics["score_coefficient_mass_local"], torch.tensor(0.0)
    )


def test_rewarded_candidate_moves_velocity_target_toward_that_future() -> None:
    candidates = torch.tensor(
        [
            [[[0.0]]],
            [[[0.0]]],
            [[[0.0]]],
            [[[4.0]]],
        ]
    )
    correction, _ = centered_velocity_correction(
        candidates,
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        global_group_size=4,
    )
    # (1 - 1/4) * 4 = 3; the zero-valued candidates contribute nothing.
    torch.testing.assert_close(correction, torch.tensor([[[[3.0]]]]))


def test_posterior_tilted_regression_changes_target_not_loss_weight() -> None:
    student = torch.tensor([[[[2.0]]]], requires_grad=True)
    reference = torch.tensor([[[[1.0]]]])
    correction = torch.tensor([[[[0.5]]]])
    loss, diagnostics = posterior_tilted_regression_loss(
        student,
        reference,
        correction,
    )
    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(
        diagnostics["reward_correction_rms"], torch.tensor(0.5)
    )


def test_reference_regression_control_is_zero_at_frozen_base() -> None:
    reference = torch.randn(1, 2, 3)
    student = reference.clone().requires_grad_(True)
    loss, diagnostics = posterior_tilted_regression_loss(
        student,
        reference,
        torch.zeros_like(reference),
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    torch.testing.assert_close(
        diagnostics["reward_correction_rms"], torch.tensor(0.0)
    )


def test_distributed_posterior_distillation_recovers_global_control_variate() -> None:
    log_prob_rank0 = torch.tensor([-1.0, -2.0], requires_grad=True)
    log_prob_rank1 = torch.tensor([-3.0, -4.0], requires_grad=True)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    loss0, _ = posterior_distillation_loss(
        log_prob_rank0,
        weights[:2],
        distributed_world_size=2,
    )
    loss1, _ = posterior_distillation_loss(
        log_prob_rank1,
        weights[2:],
        distributed_world_size=2,
    )
    ddp_average = 0.5 * (loss0 + loss1)
    baseline = torch.full_like(weights, 0.25)
    expected = -(
        (weights - baseline) * torch.tensor([-1.0, -2.0, -3.0, -4.0])
    ).sum()
    torch.testing.assert_close(ddp_average, expected)


def test_uniform_posterior_weights_produce_exact_zero_likelihood_update() -> None:
    log_prob = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    loss, _ = posterior_distillation_loss(
        log_prob,
        torch.full((4,), 0.25),
        distributed_world_size=1,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.zeros_like(log_prob))


def test_log_prob_reduction_returns_one_scalar_per_video() -> None:
    action = torch.ones(3, 2, 4, 5)
    mean = torch.zeros_like(action)
    std = torch.full((1, 1, 1, 1), 0.5)
    log_prob = gaussian_log_prob_mean(action, mean, std)
    assert log_prob.shape == (3,)
    assert torch.isfinite(log_prob).all()


def test_group_advantage_is_centered_and_clipped() -> None:
    advantages, mean, std = global_group_advantages(
        torch.tensor([-100.0, 0.0, 1.0, 100.0]),
        epsilon=1.0e-4,
        clip=1.0,
    )
    assert float(advantages.abs().max()) <= 1.0
    assert torch.isfinite(mean)
    assert float(std) > 0.0


def test_optional_grpo_baseline_is_finite_at_behavior_policy() -> None:
    old = torch.tensor([-1.0, -2.0])
    new = old.clone().requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0])
    loss, metrics = clipped_grpo_loss(
        new,
        old,
        advantages,
        clip_range=1.0e-4,
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["ratio_mean"], torch.tensor(1.0))


def test_group_partition_matches_video_flow_grpo_on_four_gpus() -> None:
    assert verify_group_partition(4, 4) == 1


def test_effective_sample_size_uniform_group() -> None:
    torch.testing.assert_close(
        effective_sample_size(torch.full((8,), 1.0 / 8.0)),
        torch.tensor(8.0),
    )


def test_checked_in_scientific_config_is_locked_and_valid() -> None:
    from examples.train.pt_pdd.train_anyflow import validate_config

    config_path = Path(
        "examples/train/configs/pt_pdd/wan_anyflow_videoalign_mq.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
