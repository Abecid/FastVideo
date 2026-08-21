# SPDX-License-Identifier: Apache-2.0
"""Reproducible finite-transition posterior training entry point.

This subclass keeps the posterior-projection experiment in
``finite_transition_posterior.py`` while adding three pieces needed for the
scientific AnyFlow run:

* the released AnyFlow finite-map timestep grid;
* local-anchor ASFMC, the Flow-Map-GRPO construction designed for a two-time
  flow map; and
* checkpoint persistence for held-out baselines and efficiency counters.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from typing import Any

import torch

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common.anyflow_schedule import (
    anyflow_inference_schedule,
)
from fastvideo.train.methods.rl.common.local_asfmc import (
    local_anchor_gaussian_parameters,
)
from fastvideo.train.methods.rl.finite_transition_posterior import (
    FiniteTransitionPosteriorMethod,
)

_BASELINE_BUFFER_BYTES = 65536


def _effective_local_anchor_delta_fraction(
    target_time: torch.Tensor | float,
    *,
    num_train_timesteps: int,
    configured_delta_fraction: float,
) -> float:
    """Truncate the local interval at AnyFlow's data endpoint."""
    target = torch.as_tensor(target_time).detach().float()
    if target.numel() != 1:
        raise ValueError("local ASFMC target time must be scalar")
    target_fraction = float(target.item()) / float(num_train_timesteps)
    if target_fraction <= 0.0:
        raise ValueError(
            "local ASFMC requires a target strictly before the data endpoint"
        )
    return min(float(configured_delta_fraction), target_fraction)


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
    """FTPP with the official AnyFlow grid and resume-safe evaluation."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config
        self._anchor_type = str(
            mcfg.get("anchor_type", "local") or "local"
        ).strip().lower()
        if self._anchor_type not in {"local", "endpoint"}:
            raise ValueError("method.anchor_type must be local or endpoint")
        self._local_anchor_delta = float(
            mcfg.get("local_anchor_delta", 0.03) or 0.03
        )
        self._local_noise_scale = float(
            mcfg.get("local_noise_scale", 0.7)
        )
        self._local_terminal_base_sigma = float(
            mcfg.get("local_terminal_base_sigma", 0.05) or 0.05
        )
        if not 0.0 < self._local_anchor_delta < 1.0:
            raise ValueError("method.local_anchor_delta must lie in (0, 1)")
        if self._local_noise_scale <= 0.0:
            raise ValueError("method.local_noise_scale must be positive")
        if not 0.0 < self._local_terminal_base_sigma < 1.0:
            raise ValueError(
                "method.local_terminal_base_sigma must lie in (0, 1)"
            )
        self._last_effective_local_anchor_delta = self._local_anchor_delta

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["finite_transition_posterior.run_state"] = (
            _FiniteTransitionRunState(self)
        )
        return states

    def _build_schedule(
        self,
        *,
        steps: int,
        override: list[float] | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Use the exact grid from AnyFlow's released FlowMap scheduler."""
        if override is not None:
            return super()._build_schedule(
                steps=steps,
                override=override,
                device=device,
            )
        return anyflow_inference_schedule(
            num_steps=steps,
            shift=self._flow_shift,
            num_train_timesteps=self.student.num_train_timesteps,
            device=device,
        )

    def _branch_policy(
        self,
        shared_state: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._anchor_type == "endpoint":
            return super()._branch_policy(
                shared_state,
                source_time,
                target_time,
                batch,
            )

        deterministic_target, _ = self._flow_map(
            shared_state,
            source_time,
            target_time,
            batch,
        )
        effective_delta = _effective_local_anchor_delta_fraction(
            target_time,
            num_train_timesteps=self.student.num_train_timesteps,
            configured_delta_fraction=self._local_anchor_delta,
        )
        self._last_effective_local_anchor_delta = effective_delta

        # The closed-form local conditional is evaluated at x_r. AnyFlow's
        # instantaneous reverse velocity is obtained by setting both time
        # arguments to r. The conceptual anchor time tau=r+delta in the paper
        # corresponds to q_tau=q_r-delta in AnyFlow's reverse convention.
        batch_size = int(deterministic_target.shape[0])
        target_batch = target_time.reshape(1).to(
            device=deterministic_target.device,
            dtype=torch.float32,
        ).expand(batch_size)
        batch.timesteps = target_batch
        instantaneous_reverse_velocity = self.student.predict_velocity_with_r(
            deterministic_target,
            target_batch,
            target_batch,
            batch,
            conditional=True,
            attn_kind=self._attn_kind,  # type: ignore[arg-type]
        )
        mean, std = local_anchor_gaussian_parameters(
            deterministic_target,
            instantaneous_reverse_velocity,
            target_batch,
            num_train_timesteps=self.student.num_train_timesteps,
            delta_fraction=effective_delta,
            noise_scale=self._local_noise_scale,
            terminal_base_sigma=self._local_terminal_base_sigma,
        )
        return mean, std, deterministic_target

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        losses, outputs, metrics = super().managed_train_step(
            data_stream,
            iteration,
        )
        is_local = float(self._anchor_type == "local")
        metrics["ftp/schedule_is_official_anyflow"] = 1.0
        metrics["ftp/anchor_is_local"] = is_local
        metrics["ftp/local_anchor_delta"] = self._local_anchor_delta
        metrics["ftp/local_noise_scale"] = self._local_noise_scale
        metrics["ftp/local_terminal_base_sigma"] = (
            self._local_terminal_base_sigma
        )
        if self._anchor_type == "local":
            target = float(metrics["ftp/target_timestep"])
            effective_delta = self._last_effective_local_anchor_delta
            metrics["ftp/local_anchor_effective_delta"] = effective_delta
            metrics["ftp/local_anchor_delta_was_clipped"] = float(
                effective_delta < self._local_anchor_delta
            )
            metrics["ftp/local_anchor_timestep"] = max(0.0, target - (
                effective_delta
                * float(self.student.num_train_timesteps)
            ))
        return losses, outputs, metrics


__all__ = [
    "ReproducibleFiniteTransitionPosteriorMethod",
    "_effective_local_anchor_delta_fraction",
]
