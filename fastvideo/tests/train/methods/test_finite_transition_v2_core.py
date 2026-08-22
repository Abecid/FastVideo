# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.rl.common.finite_transition_v2 import (
    PromptRewardTracker,
    cosine_similarity_flat,
    global_temperature_weights,
    paired_bootstrap_interval,
    paired_difference_statistics,
    running_baseline_advantages,
    stable_global_reward_std,
    update_target_kl_scale,
)


def test_global_temperature_keeps_flat_group_near_uniform() -> None:
    rewards = torch.tensor([0.5000, 0.5001, 0.4999, 0.5000])
    weights, temperature, ess = global_temperature_weights(
        rewards,
        baseline=0.5,
        global_std=0.25,
        temperature_multiplier=1.0,
        minimum_temperature=1.0e-4,
    )
    assert float(temperature) == pytest.approx(0.25)
    assert torch.allclose(weights, torch.full_like(weights, 0.25), atol=2.0e-4)
    assert float(ess) > 3.99


def test_global_temperature_still_selects_clear_winner() -> None:
    weights, _, ess = global_temperature_weights(
        torch.tensor([-1.0, 0.0, 2.0, 0.5]),
        baseline=0.0,
        global_std=0.5,
        temperature_multiplier=1.0,
        minimum_temperature=1.0e-4,
    )
    assert int(weights.argmax()) == 2
    assert float(weights[2]) > 0.9
    assert float(ess) < 1.3


def test_running_prompt_tracker_uses_group_mean_then_history() -> None:
    tracker = PromptRewardTracker()
    first = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert tracker.baseline("p", first) == pytest.approx(2.5)
    tracker.update("p", first)
    second = torch.tensor([5.0, 6.0, 7.0, 8.0])
    assert tracker.baseline("p", second) == pytest.approx(2.5)
    restored = PromptRewardTracker.from_state_dict(tracker.state_dict())
    assert restored.baseline("p", second) == pytest.approx(2.5)


def test_running_baseline_advantage_uses_global_denominator() -> None:
    advantages = running_baseline_advantages(
        torch.tensor([0.0, 1.0, 2.0]),
        baseline=1.0,
        global_std=2.0,
        clip=5.0,
    )
    assert torch.allclose(advantages, torch.tensor([-0.5, 0.0, 0.5]))


def test_reward_std_ema_is_stable() -> None:
    current, ema = stable_global_reward_std(
        torch.tensor([-1.0, 0.0, 1.0, 2.0]),
        previous=2.0,
        decay=0.9,
        floor=0.01,
    )
    assert current > 0.0
    assert current < ema < 2.0


def test_target_kl_controller_increases_tiny_updates_conservatively() -> None:
    updated = update_target_kl_scale(
        1.0,
        1.0e-7,
        target_kl=1.0e-5,
        controller_rate=0.25,
        minimum_scale=0.1,
        maximum_scale=100.0,
    )
    assert 1.0 < updated < 10.0


def test_paired_statistics_use_difference_variance() -> None:
    baseline = torch.tensor([10.0, 20.0, 30.0, 40.0])
    current = baseline + torch.tensor([0.9, 1.1, 1.0, 1.0])
    stats = paired_difference_statistics(current, baseline)
    assert float(stats["mean"]) == pytest.approx(1.0)
    assert float(stats["sem"]) < 0.05
    low, high = paired_bootstrap_interval(
        stats["delta"],
        confidence=0.95,
        num_bootstrap=1000,
        seed=42,
    )
    assert float(low) < 1.0 < float(high)


def test_cosine_similarity_flat() -> None:
    tensor = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert float(cosine_similarity_flat(tensor, tensor)) == pytest.approx(1.0)
    assert float(cosine_similarity_flat(tensor, -tensor)) == pytest.approx(-1.0)
