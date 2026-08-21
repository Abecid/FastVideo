# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.train.methods.rl.finite_transition_posterior_repro import (
    _FiniteTransitionRunState,
)


def test_run_state_round_trip_preserves_scientific_baseline() -> None:
    source = SimpleNamespace(
        _validation_baseline={
            "reward/videoalign_mq": 0.42,
            "temporal_l1": 0.031,
        },
        _validation_best_primary_delta=0.08,
        _steps_to_primary_target=300,
        _cumulative_train_seconds=1234.5,
    )
    state = _FiniteTransitionRunState(source).state_dict()

    target = SimpleNamespace(
        _validation_baseline={},
        _validation_best_primary_delta=float("-inf"),
        _steps_to_primary_target=-1,
        _cumulative_train_seconds=0.0,
    )
    _FiniteTransitionRunState(target).load_state_dict(state)

    assert target._validation_baseline == pytest.approx(
        source._validation_baseline
    )
    assert target._validation_best_primary_delta == pytest.approx(0.08)
    assert target._steps_to_primary_target == 300
    assert target._cumulative_train_seconds == pytest.approx(1234.5)
