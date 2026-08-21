# SPDX-License-Identifier: Apache-2.0
"""Local-anchor ASFMC math for reverse-time rectified-flow maps.

Flow-Map GRPO recommends a local anchor for two-time maps such as MeanFlow:
after a long deterministic transition to the policy target, move a short
additional distance toward the data endpoint, then sample back through the
short reverse-SDE Gaussian. AnyFlow uses the reverse convention ``q=1`` at
Gaussian noise and ``q=0`` at data, while the paper uses ``s=1-q``. The helper
below performs that coordinate conversion explicitly.
"""

from __future__ import annotations

import math

import torch


def local_anchor_gaussian_parameters(
    anchor_state: torch.Tensor,
    instantaneous_reverse_velocity: torch.Tensor,
    anchor_timestep: torch.Tensor | float,
    *,
    num_train_timesteps: int,
    delta_fraction: float,
    noise_scale: float,
    terminal_base_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local-anchor Gaussian mean and standard deviation.

    Args:
        anchor_state: State at the short anchor time, in FastVideo's reverse
            convention (noise time decreases toward zero during generation).
        instantaneous_reverse_velocity: ``dX/dq`` predicted at the anchor. The
            Flow-Map-GRPO paper writes velocity in the opposite coordinate
            ``s=1-q``, so its instantaneous velocity is ``-dX/dq``.
        anchor_timestep: Absolute reverse-time model timestep of the anchor.
        num_train_timesteps: Usually 1000 for Wan.
        delta_fraction: Short anchor interval in normalized time; the paper's
            released MeanFlow setting uses 0.03.
        noise_scale: Positive local reverse-SDE stochasticity coefficient; the
            released Flow-Map-GRPO setting uses 0.7.
        terminal_base_sigma: Lower bound for the paper-coordinate data
            coefficient. It stabilizes the first near-noise transition; the
            released setting is 0.05.

    For the rectified path in the paper coordinate ``s``,

        sigma(s) = lambda * sqrt(2 * (1-s) / s).

    The short reverse conditional has, to first order,

        mean = x_tau - delta * [(1-lambda^2) u_tau
                                + lambda^2 x_tau / tau]

    with ``u_tau = -v_reverse``. The returned standard deviation is
    ``sigma(tau) * sqrt(delta)``. The approximation error is the local-anchor
    discretization error; no long-range Gaussian approximation is made.
    """
    if anchor_state.shape != instantaneous_reverse_velocity.shape:
        raise ValueError(
            "anchor_state and instantaneous_reverse_velocity must have equal "
            f"shapes, got {anchor_state.shape} and "
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
        anchor_timestep,
        device=anchor_state.device,
        dtype=torch.float32,
    ) / float(num_train_timesteps)
    if torch.any((reverse_time < 0.0) | (reverse_time >= 1.0)):
        raise ValueError(
            "local anchor reverse time must lie in [0, 1); got "
            f"{anchor_timestep!r}"
        )

    paper_time = 1.0 - reverse_time
    stabilized_time = paper_time.clamp_min(float(terminal_base_sigma))
    while stabilized_time.ndim < anchor_state.ndim:
        stabilized_time = stabilized_time.unsqueeze(-1)
    while reverse_time.ndim < anchor_state.ndim:
        reverse_time = reverse_time.unsqueeze(-1)

    lambda_sq = float(noise_scale) ** 2
    delta = float(delta_fraction)
    mean = (
        anchor_state
        + delta * (1.0 - lambda_sq) * instantaneous_reverse_velocity
        - delta * lambda_sq * anchor_state / stabilized_time.to(anchor_state.dtype)
    )

    sigma = float(noise_scale) * torch.sqrt(
        2.0
        * reverse_time
        / stabilized_time
    )
    std = sigma.to(anchor_state.dtype) * math.sqrt(delta)
    return mean.to(anchor_state.dtype), std


__all__ = ["local_anchor_gaussian_parameters"]
