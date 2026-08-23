# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.rl.common.finite_transition_grpo_v3 import (
    group_normalized_advantages,
    shuffled_group_minibatches,
)


def test_group_normalization_uses_only_same_prompt_candidates() -> None:
    result = group_normalized_advantages(
        torch.tensor([10.0, 12.0, 8.0, 10.0]),
        epsilon=1.0e-4,
        clip=5.0,
        minimum_std=1.0e-4,
    )

    assert result.active
    assert float(result.advantages.mean()) == pytest.approx(0.0, abs=1.0e-6)
    assert float(result.advantages.std(unbiased=False)) == pytest.approx(
        1.0,
        rel=2.0e-4,
    )


def test_low_variance_group_produces_exact_zero_update() -> None:
    result = group_normalized_advantages(
        torch.tensor([0.5, 0.5, 0.5, 0.5]),
        epsilon=1.0e-4,
        clip=5.0,
        minimum_std=1.0e-4,
    )

    assert not result.active
    assert torch.equal(result.advantages, torch.zeros_like(result.advantages))


def test_shuffled_minibatches_are_deterministic_and_complete() -> None:
    first = shuffled_group_minibatches(
        7,
        groups_per_minibatch=2,
        seed=123,
    )
    second = shuffled_group_minibatches(
        7,
        groups_per_minibatch=2,
        seed=123,
    )

    assert first == second
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(7))
    assert len(set(flattened)) == 7
