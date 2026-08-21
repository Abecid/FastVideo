# SPDX-License-Identifier: Apache-2.0
"""Local-anchor ASFMC math for reverse-time rectified-flow maps.

Flow-Map GRPO derives a short-interval Gaussian policy for two-time flow maps.
AnyFlow uses the reverse coordinate ``q`` (noise at 1, data at 0), whereas the
paper uses ``s = 1-q`` (noise at 0, data at 1). This module performs that
coordinate and velocity-sign conversion explicitly.
"""

from __future__ import annotations

import math

import torch


def local_anchor_gaussian_parameters(
    target_state: torch.Tensor,
    instantaneous_reverse_velocity: torch.Tensor,
    target_timestep: torch.Tensor | float,
    *,
    num_train_timesteps: int,
    delta_fraction: float,
    noise_scale: float,
    terminal_base_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local-anchor ASFMC Gaussian at the policy target time.

    Flow-Map GRPO writes the affine path in a coordinate ``s`` that increases
    from noise to data. For ``a_s=s`` and a local anchor ``tau=s+delta``, its
    two-time policy is, up to the local Euler--Maruyama error,

        mean_s = x_s - delta * lambda^2 * (x_s / s - u_s(x_s))
        std_s  = lambda * sqrt(2 * (1-s) / s) * sqrt(delta).

    FastVideo/AnyFlow uses ``q=1-s`` and predicts
    ``v_q=dX/dq=-u_s``. Substitution therefore gives

        mean_q = x_q - delta * lambda^2 * (x_q / (1-q) + v_q(x_q))
        std_q  = lambda * sqrt(2 * q / (1-q)) * sqrt(delta).

    ``terminal_base_sigma`` lower-bounds ``1-q`` near the initial noise
    endpoint, matching the stabilizing terminal coefficient used by the
    released Flow-Map-GRPO configuration.
    """
    if target_state.shape != instantaneous_reverse_velocity.shape:
        raise ValueError(
            "target_state and instantaneous_reverse_velocity must have equal "
            f"shapes, got {target_state.shape} and "
            f"{instantaneous_reverse_velocity.shape}"
        )
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    if not 0.0 < float(delta_fraction) < 1.0:
        raise ValueError("delta_fraction must lie in (0, 1)")
    if float(noise_scale) <= 0.0:
        raise ValueError("noise_scale must be positive")
    if not 0.0 < float(terminal_base_sigma) < 1.0:
        raise ValueError("terminal_base_sigma must lie in (0, 1)")

    reverse_time = torch.as_tensor(
        target_timestep,
        device=target_state.device,
        dtype=torch.float32,
    ) / float(num_train_timesteps)
    if torch.any((reverse_time < 0.0) | (reverse_time >= 1.0)):
        raise ValueError(
            "local-anchor reverse target time must lie in [0, 1); got "
            f"{target_timestep!r}"
        )

    paper_time = 1.0 - reverse_time
    stabilized_time = paper_time.clamp_min(float(terminal_base_sigma))
    while stabilized_time.ndim < target_state.ndim:
        stabilized_time = stabilized_time.unsqueeze(-1)
    while reverse_time.ndim < target_state.ndim:
        reverse_time = reverse_time.unsqueeze(-1)

    lambda_sq = float(noise_scale) ** 2
    delta = float(delta_fraction)
    mean = target_state - delta * lambda_sq * (
        target_state / stabilized_time.to(target_state.dtype)
        + instantaneous_reverse_velocity
    )
    sigma = float(noise_scale) * torch.sqrt(
        2.0 * reverse_time / stabilized_time
    )
    std = sigma.to(target_state.dtype) * math.sqrt(delta)
    return mean.to(target_state.dtype), std


__all__ = ["local_anchor_gaussian_parameters"]
