# SPDX-License-Identifier: Apache-2.0
"""Reproducible finite-transition posterior training entry point.

This subclass keeps the posterior-projection experiment in
``finite_transition_posterior.py`` while adding two pieces needed for the
scientific AnyFlow run:

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
from fastvideo.train.methods.rl.common.local_asfmc import (
    local_anchor_gaussian_parameters,
)
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
    """FTPP with local ASFMC and resume-safe scientific evaluation."""

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
        if self._local_noise_scale < 0.0:
            raise ValueError("method.local_noise_scale must be non-negative")
        if not 0.0 < self._local_terminal_base_sigma < 1.0:
            raise ValueError(
                "method.local_terminal_base_sigma must lie in (0, 1)"
            )

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["finite_transition_posterior.run_state"] = (
            _FiniteTransitionRunState(self)
        )
        return states

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
        delta_absolute = (
            self._local_anchor_delta
            * float(self.student.num_train_timesteps)
        )
        anchor_time = target_time.to(torch.float32) - delta_absolute
        if torch.any(anchor_time < 0.0):
            raise ValueError(
                "local ASFMC anchor crosses the data endpoint: target="
                f"{float(target_time)}, delta={delta_absolute}"
            )

        # AnyFlow can query arbitrary time pairs, so move a short additional
        # interval toward data. This is the local anchor recommended for
        # two-time maps, rather than re-predicting the full clean endpoint.
        anchor_state, _ = self._flow_map(
            deterministic_target,
            target_time,
            anchor_time,
            batch,
        )

        batch_size = int(anchor_state.shape[0])
        anchor_batch = anchor_time.reshape(1).to(
            device=anchor_state.device,
            dtype=torch.float32,
        ).expand(batch_size)
        batch.timesteps = anchor_batch
        instantaneous_reverse_velocity = self.student.predict_velocity_with_r(
            anchor_state,
            anchor_batch,
            anchor_batch,
            batch,
            conditional=True,
            attn_kind=self._attn_kind,  # type: ignore[arg-type]
        )
        mean, std = local_anchor_gaussian_parameters(
            anchor_state,
            instantaneous_reverse_velocity,
            anchor_batch,
            num_train_timesteps=self.student.num_train_timesteps,
            delta_fraction=self._local_anchor_delta,
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
        metrics["ftp/anchor_is_local"] = is_local
        metrics["ftp/local_anchor_delta"] = self._local_anchor_delta
        metrics["ftp/local_noise_scale"] = self._local_noise_scale
        metrics["ftp/local_terminal_base_sigma"] = (
            self._local_terminal_base_sigma
        )
        if self._anchor_type == "local":
            target = float(metrics["ftp/target_timestep"])
            metrics["ftp/local_anchor_timestep"] = target - (
                self._local_anchor_delta
                * float(self.student.num_train_timesteps)
            )
        return losses, outputs, metrics


__all__ = [
    "ReproducibleFiniteTransitionPosteriorMethod",
]
