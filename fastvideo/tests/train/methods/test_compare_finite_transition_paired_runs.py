# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from examples.train.compare_finite_transition_paired_runs import compare


def _artifact(values: list[float]):
    return {
        "iteration": 10,
        "mode": "ema",
        "metrics": {
            "sample_key": [0, 1, 1000, 1001],
            "sample_seed": [42, 43, 10042, 10043],
            "prompt_index": [0, 0, 1, 1],
            "reward/videoalign_mq_audited": values,
        },
    }


def test_cross_arm_comparison_bootstraps_prompts() -> None:
    result = compare(
        _artifact([1.0, 2.0, 3.0, 4.0]),
        _artifact([0.5, 1.5, 2.5, 3.5]),
        bootstrap_samples=500,
        confidence=0.95,
        seed=42,
    )
    summary = result["metrics"]["reward/videoalign_mq_audited"]
    assert result["bootstrap_unit"] == "prompt"
    assert summary["sample_count"] == 4.0
    assert summary["prompt_count"] == 2.0
    assert summary["mean_delta"] == pytest.approx(0.5)
    assert summary["ci_lower"] == pytest.approx(0.5)
    assert summary["ci_upper"] == pytest.approx(0.5)


def test_cross_arm_comparison_rejects_mismatched_identity() -> None:
    right = _artifact([0.5, 1.5, 2.5, 3.5])
    right["metrics"]["sample_seed"][0] = 99
    with pytest.raises(RuntimeError):
        compare(
            _artifact([1.0, 2.0, 3.0, 4.0]),
            right,
            bootstrap_samples=100,
            confidence=0.95,
            seed=42,
        )
