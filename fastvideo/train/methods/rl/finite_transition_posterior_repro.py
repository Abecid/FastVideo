# SPDX-License-Identifier: Apache-2.0
"""Reproducible finite-transition posterior training entry point.

This thin subclass keeps the scientific method in
``finite_transition_posterior.py`` and adds checkpoint persistence for the
held-out evaluation baseline and efficiency counters. Without this state, a
resumed run would incorrectly treat its first post-resume validation as a new
step-zero baseline.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from fastvideo.train.methods.rl.finite_transition_posterior import (
    FiniteTransitionPosteriorMethod,
)

_BASELINE_BUFFER_BYTES = 65536


class _FiniteTransitionRunState:
    """DCP-compatible wrapper for method-owned scientific run state.

    DCP builds a load plan from the tensors returned by ``state_dict()``. A
    dynamically keyed dictionary is therefore unsafe when a freshly constructed
    method has an empty validation baseline. We serialize the baseline into a
    fixed-size uint8 tensor so save and load always expose identical keys and
    shapes.
    """

    def __init__(self, method: "ReproducibleFiniteTransitionPosteriorMethod") -> None:
        self._method = method

    def _baseline_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        payload = json.dumps(
            self._method._validation_baseline,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > _BASELINE_BUFFER_BYTES:
            raise RuntimeError(
                "Serialized FTPP validation baseline exceeds the fixed DCP "
                f"buffer ({len(payload)} > {_BASELINE_BUFFER_BYTES} bytes)"
            )
        buffer = torch.zeros(_BASELINE_BUFFER_BYTES, dtype=torch.uint8)
        if payload:
            buffer[: len(payload)] = torch.tensor(
                list(payload),
                dtype=torch.uint8,
            )
        length = torch.tensor(len(payload), dtype=torch.long)
        return buffer, length

    def state_dict(self) -> dict[str, Any]:
        baseline_buffer, baseline_length = self._baseline_tensors()
        return {
            "validation_baseline_json": baseline_buffer,
            "validation_baseline_length": baseline_length,
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
        raw_buffer = state_dict.get("validation_baseline_json")
        raw_length = state_dict.get("validation_baseline_length", 0)
        if torch.is_tensor(raw_length):
            raw_length = int(raw_length.item())
        length = int(raw_length)
        if length < 0 or length > _BASELINE_BUFFER_BYTES:
            raise RuntimeError(
                f"Invalid FTPP validation baseline payload length: {length}"
            )
        if raw_buffer is not None:
            if not torch.is_tensor(raw_buffer):
                raise TypeError("validation_baseline_json must be a tensor")
            payload = bytes(
                raw_buffer.detach().cpu().to(torch.uint8)[:length].tolist()
            )
            decoded = json.loads(payload.decode("utf-8")) if payload else {}
            if not isinstance(decoded, dict):
                raise RuntimeError("Decoded FTPP validation baseline is not a mapping")
            self._method._validation_baseline = {
                str(key): float(value)
                for key, value in decoded.items()
            }

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
