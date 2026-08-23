# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from fastvideo.train.methods.rl.common.anyflow_schedule import (
    anyflow_inference_schedule,
)
from fastvideo.train.methods.rl.common.finite_transition import (
    append_data_endpoint,
    endpoint_anchor_parameters,
    gaussian_log_prob_mean,
    group_advantages,
    mean_pairwise_rms,
    posterior_projection_loss,
    reward_tilted_weights,
    temporal_l1,
    validate_training_schedule,
)


def test_append_and_validate_training_schedule() -> None:
    schedule = append_data_endpoint(
        torch.tensor([1000.0, 800.0, 600.0, 400.0, 200.0])
    )
    validate_training_schedule(schedule, stochastic_steps=4)
    assert schedule.tolist()[-1] == 0.0


def test_released_anyflow_schedule_parity() -> None:
    eval_schedule = anyflow_inference_schedule(
        num_steps=4,
        shift=5.0,
        num_train_timesteps=1000,
        device="cpu",
    )
    expected_eval = torch.tensor(
        [1000.0, 937.5, 833.3333333, 625.0, 0.0],
        dtype=torch.float32,
    )
    assert torch.allclose(
        eval_schedule,
        expected_eval,
        atol=1.0e-4,
        rtol=0.0,
    )

    train_schedule = anyflow_inference_schedule(
        num_steps=5,
        shift=5.0,
        num_train_timesteps=1000,
        device="cpu",
    )
    expected_train = torch.tensor(
        [
            1000.0,
            952.3809524,
            882.3529412,
            769.2307692,
            555.5555556,
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        train_schedule,
        expected_train,
        atol=1.0e-4,
        rtol=0.0,
    )


def test_endpoint_anchor_matches_affine_path() -> None:
    x0 = torch.full((2, 3), 4.0)
    mean, std = endpoint_anchor_parameters(
        x0,
        250.0,
        num_train_timesteps=1000,
    )
    assert torch.allclose(mean, torch.full_like(x0, 3.0))
    assert torch.allclose(std, torch.tensor([[0.25]]))


def test_reward_tilt_hits_target_ess() -> None:
    weights, temperature, ess = reward_tilted_weights(
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        target_ess_ratio=0.5,
    )
    assert torch.isfinite(temperature)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1.0e-6)
    assert float(ess) == pytest.approx(2.0, abs=2.0e-3)
    assert int(weights.argmax()) == 3


def test_degenerate_reward_gives_exact_zero_projection_update() -> None:
    log_prob = torch.tensor([-1.0, -2.0], requires_grad=True)
    local_weights = torch.full((2,), 0.25)
    loss, diagnostics = posterior_projection_loss(
        log_prob,
        local_weights,
        global_group_size=4,
        distributed_world_size=2,
    )
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0)
    assert torch.equal(log_prob.grad, torch.zeros_like(log_prob))
    assert float(diagnostics["score_coefficient_abs_mean_local"]) == 0.0


def test_projection_increases_high_weight_log_probability() -> None:
    log_prob = torch.tensor([-1.0, -1.0], requires_grad=True)
    loss, _ = posterior_projection_loss(
        log_prob,
        torch.tensor([0.4, 0.1]),
        global_group_size=4,
        distributed_world_size=2,
    )
    loss.backward()
    # Gradient descent increases the first log probability and decreases the second.
    assert float(log_prob.grad[0]) < 0.0
    assert float(log_prob.grad[1]) > 0.0


def test_mean_gaussian_log_prob_is_dimension_normalized() -> None:
    action = torch.ones((1, 2))
    mean = torch.zeros_like(action)
    std = torch.ones((1, 1))
    value = gaussian_log_prob_mean(action, mean, std)
    expected = -0.5 - 0.5 * math.log(2.0 * math.pi)
    assert float(value) == pytest.approx(expected)


def test_mean_gaussian_log_prob_preserves_small_bfloat16_policy_shift() -> None:
    action = torch.zeros((1, 4096), dtype=torch.bfloat16)
    mean_before = torch.zeros_like(action)
    mean_after = torch.full_like(action, 0.01)
    std = torch.ones((1, 1), dtype=torch.bfloat16)

    before = gaussian_log_prob_mean(action, mean_before, std)
    after = gaussian_log_prob_mean(action, mean_after, std)
    delta = after - before

    assert before.dtype == torch.float32
    assert float(delta.abs()) > 1.0e-5
    expected = -0.5 * float(mean_after[0, 0]) ** 2
    assert float(delta) == pytest.approx(expected, rel=2.0e-2)


def test_group_advantages_are_centered_and_clipped() -> None:
    advantages, mean, std = group_advantages(
        torch.tensor([0.0, 1.0, 2.0]),
        epsilon=1.0e-4,
        clip=0.5,
    )
    assert float(mean) == pytest.approx(1.0)
    assert float(std) > 0.0
    assert float(advantages.max()) <= 0.5
    assert float(advantages.min()) >= -0.5


def test_video_motion_and_diversity_helpers() -> None:
    video = torch.zeros((2, 3, 3, 4, 4))
    video[1, :, 1:] = 1.0
    motion = temporal_l1(video)
    assert float(motion[0]) == 0.0
    assert float(motion[1]) > 0.0
    diversity = mean_pairwise_rms(torch.stack((video[0], video[1])))
    assert float(diversity) > 0.0
