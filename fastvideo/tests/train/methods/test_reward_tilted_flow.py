# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.train.methods.knowledge_distillation.reward_tilted_flow_utils import (
    build_deployment_flow_schedule,
    reward_tilt_weights,
)


def test_reward_tilt_weights_are_uniform_for_degenerate_rewards() -> None:
    rewards = torch.ones(3, 4)
    weights, temperatures, raw_ess, final_ess = reward_tilt_weights(
        rewards,
        target_ess_ratio=0.5,
        uniform_mix=0.0,
    )
    torch.testing.assert_close(weights, torch.full_like(weights, 0.25))
    torch.testing.assert_close(raw_ess, torch.ones_like(raw_ess))
    torch.testing.assert_close(final_ess, torch.ones_like(final_ess))
    assert torch.isfinite(temperatures).all()


def test_reward_tilt_weights_control_ess_and_preserve_coverage() -> None:
    rewards = torch.tensor([[0.0, 1.0, 2.0, 5.0]], dtype=torch.float32)
    pure, _, pure_ess, pure_final_ess = reward_tilt_weights(
        rewards,
        target_ess_ratio=0.5,
        uniform_mix=0.0,
    )
    mixed, _, mixed_raw_ess, mixed_final_ess = reward_tilt_weights(
        rewards,
        target_ess_ratio=0.5,
        uniform_mix=0.25,
    )

    torch.testing.assert_close(pure.sum(dim=1), torch.ones(1))
    torch.testing.assert_close(mixed.sum(dim=1), torch.ones(1))
    assert abs(float(pure_ess.item()) - 0.5) < 0.03
    torch.testing.assert_close(pure_final_ess, pure_ess)
    torch.testing.assert_close(mixed_raw_ess, pure_ess)
    assert float(mixed.min()) >= 0.25 / 4.0
    assert float(mixed.max()) < float(pure.max())
    assert float(mixed_final_ess) > float(mixed_raw_ess)


def test_uniform_mix_one_is_matched_compute_no_reward_baseline() -> None:
    rewards = torch.tensor([[0.0, 1.0, 2.0, 9.0]], dtype=torch.float32)
    weights, _, _, final_ess = reward_tilt_weights(
        rewards,
        target_ess_ratio=0.25,
        uniform_mix=1.0,
    )
    torch.testing.assert_close(weights, torch.full_like(weights, 0.25))
    torch.testing.assert_close(final_ess, torch.ones_like(final_ess))


def test_deployment_flow_schedule_matches_four_step_grid() -> None:
    timesteps, sigmas, interval_weights = build_deployment_flow_schedule(
        num_steps=4,
        flow_shift=8.0,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    assert timesteps.shape == (4,)
    assert sigmas.shape == (5,)
    assert interval_weights.shape == (4,)
    assert torch.all(sigmas[:-1] > sigmas[1:])
    assert float(sigmas[-1]) == 0.0
    torch.testing.assert_close(interval_weights.sum(), torch.tensor(1.0))
