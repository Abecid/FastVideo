# SPDX-License-Identifier: Apache-2.0
"""Reproducible finite-transition posterior training entry point.

This thin subclass keeps the scientific method in
``finite_transition_posterior.py`` and adds checkpoint persistence for the
held-out evaluation baseline and efficiency counters.  Without this state, a
resumed run would incorrectly treat its first post-resume validation as a new
step-zero baseline.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.rl.finite_transition_posterior import (
    FiniteTransitionPosteriorMethod,
)


class _FiniteTransitionRunState:
    """DCP-compatible wrapper for method-owned scientific run state."""

    def __init__(self, method: "ReproducibleFiniteTransitionPosteriorMethod") -> None:
        self._method = method

    def state_dict(self) -> dict[str, Any]:
        return {
            "validation_baseline": {
                key: torch.tensor(float(value), dtype=torch.float64)
                for key, value in self._method._validation_baseline.items()
            },
            "validation_best_primary_delta": torch.tensor(
                float(self._method._validation_best_primary_delta),
                dtype=torch.float64,
            ),
            "steps_to_primary_target": torch.tensor(
                int(self._method._steps_to_primary_target),
                dtype=torch.long,
            ),
            "cumulative_train_seconds": torch.tensor(
                float(self._method._cumulative_train_seconds),
                dtype=torch.float64,
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raw_baseline = state_dict.get("validation_baseline", {})
        if isinstance(raw_baseline, dict):
            baseline: dict[str, float] = {}
            for key, value in raw_baseline.items():
                if torch.is_tensor(value):
                    value = float(value.item())
                baseline[str(key)] = float(value)
            self._method._validation_baseline = baseline

        best = state_dict.get("validation_best_primary_delta", float("-inf"))
        if torch.is_tensor(best):
            best = float(best.item())
        self._method._validation_best_primary_delta = float(best)

        steps = state_dict.get("steps_to_primary_target", -1)
        if torch.is_tensor(steps):
            steps = int(steps.item())
        self._method._steps_to_primary_target = int(steps)

        seconds = state_dict.get("cumulative_train_seconds", 0.0)
        if torch.is_tensor(seconds):
            seconds = float(seconds.item())
        self._method._cumulative_train_seconds = float(seconds)


class ReproducibleFiniteTransitionPosteriorMethod(
    FiniteTransitionPosteriorMethod
):
    """FTPP with resume-safe held-out evaluation and efficiency accounting."""

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["finite_transition_posterior.run_state"] = (
            _FiniteTransitionRunState(self)
        )
        return states


__all__ = [
    "ReproducibleFiniteTransitionPosteriorMethod",
]
