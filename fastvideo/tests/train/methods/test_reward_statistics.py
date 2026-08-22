# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.rl.common.reward_statistics import (
    PromptRewardNormalizer,
    TargetKLController,
    paired_summary,
    reward_softmax_weights,
)


def test_global_temperature_preserves_reward_confidence() -> None:
    weak = torch.tensor([0.500, 0.501, 0.499, 0.500])
    strong = torch.tensor([-0.4, 0.8, 0.1, 0.7])
    weak_weights, weak_ess = reward_softmax_weights(
        weak,
        temperature=0.2,
    )
    strong_weights, strong_ess = reward_softmax_weights(
        strong,
        temperature=0.2,
    )

    assert float(weak_ess) > float(strong_ess)
    assert float(weak_weights.max()) < float(strong_weights.max())


def test_running_prompt_baseline_uses_previous_observations() -> None:
    normalizer = PromptRewardNormalizer(
        min_prompt_count=2,
        min_global_count=4,
    )
    first, first_diag = normalizer.normalize(
        torch.tensor([1.0, 3.0]),
        prompt="p",
    )
    second, second_diag = normalizer.normalize(
        torch.tensor([2.0, 4.0]),
        prompt="p",
    )

    assert float(first.mean()) == pytest.approx(0.0, abs=1.0e-5)
    assert first_diag["baseline_source"] == 0.0
    assert second_diag["baseline_source"] == 2.0
    assert second_diag["baseline"] == pytest.approx(2.0)
    assert float(second.mean()) > 0.0


def test_target_kl_controller_moves_scale_in_correct_direction() -> None:
    controller = TargetKLController(
        target_kl=1.0e-5,
        initial_scale=1.0,
        max_adjustment=2.0,
    )
    up = controller.update(1.0e-7)
    down = controller.update(1.0e-3)

    assert up > 1.0
    assert down < up


def test_target_kl_controller_does_not_escalate_zero_signal() -> None:
    controller = TargetKLController(
        target_kl=1.0e-5,
        initial_scale=3.0,
        max_adjustment=2.0,
    )
    assert controller.update(0.0) == pytest.approx(3.0)
    assert controller.update(float("nan")) == pytest.approx(1.5)


def test_paired_summary_detects_consistent_gain() -> None:
    baseline = torch.tensor([0.0, 1.0, 2.0, 3.0])
    current = baseline + 0.5
    summary = paired_summary(
        current,
        baseline,
        bootstrap_samples=500,
        seed=7,
    )

    assert summary["mean_delta"] == pytest.approx(0.5)
    assert summary["sem_delta"] == pytest.approx(0.0)
    assert summary["ci_lower"] == pytest.approx(0.5)
    assert summary["ci_upper"] == pytest.approx(0.5)
    assert summary["positive_fraction"] == pytest.approx(1.0)
