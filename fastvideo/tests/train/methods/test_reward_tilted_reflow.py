# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.train.methods.knowledge_distillation.reward_tilted_reflow import (
    aggregate_grouped_rewards,
)
from fastvideo.train.methods.knowledge_distillation.reward_tilted_flow_utils import (
    build_deployment_flow_schedule,
)


def test_component_zscore_reward_is_invariant_to_component_scale() -> None:
    rewards = {
        "videoalign_vq": torch.tensor([0.0, 0.2, 0.5, 1.0]),
        "videoalign_mq": torch.tensor([1.0, 0.4, 0.2, 0.0]),
        "videoalign_ta": torch.tensor([-2.0, -1.7, -1.6, -1.5]),
    }
    scaled = dict(rewards)
    scaled["videoalign_ta"] = 1000.0 * rewards["videoalign_ta"]
    weights = {
        "videoalign_vq": 1.0,
        "videoalign_mq": 1.0,
        "videoalign_ta": 1.0,
    }

    signal = aggregate_grouped_rewards(
        rewards,
        weights,
        prompt_batch=1,
        trajectories_per_prompt=4,
        mode="component_zscore",
    )
    scaled_signal = aggregate_grouped_rewards(
        scaled,
        weights,
        prompt_batch=1,
        trajectories_per_prompt=4,
        mode="component_zscore",
    )
    torch.testing.assert_close(signal, scaled_signal)


def test_raw_reward_sum_remains_available_as_ablation() -> None:
    rewards = {
        "videoalign_vq": torch.tensor([0.0, 0.2, 0.5, 1.0]),
        "videoalign_mq": torch.tensor([1.0, 0.4, 0.2, 0.0]),
        "videoalign_ta": torch.tensor([-2.0, -1.7, -1.6, -1.5]),
    }
    rewards["avg"] = sum(rewards.values())
    signal = aggregate_grouped_rewards(
        rewards,
        {
            "videoalign_vq": 1.0,
            "videoalign_mq": 1.0,
            "videoalign_ta": 1.0,
        },
        prompt_batch=1,
        trajectories_per_prompt=4,
        mode="raw_sum",
    )
    torch.testing.assert_close(signal, rewards["avg"].reshape(1, 4))


def test_balanced_student_shift_avoids_single_dominant_four_step_jump() -> None:
    _, _, balanced = build_deployment_flow_schedule(
        num_steps=4,
        flow_shift=1.0,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    _, _, shifted = build_deployment_flow_schedule(
        num_steps=4,
        flow_shift=8.0,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )

    assert float(balanced.max()) < 0.34
    assert float(shifted.max()) > 0.75
