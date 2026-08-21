# SPDX-License-Identifier: Apache-2.0
"""Official AnyFlow finite-map timestep construction.

AnyFlow does not use the generic Diffusers FlowMatchEulerDiscreteScheduler grid.
Its released ``FlowMapDiscreteScheduler`` builds ``K`` source nodes from
``linspace(1, 0, K + 1)[:-1]``, applies the flow shift exactly once, and then
appends the clean endpoint. Reusing the generic scheduler here applies a
non-zero training ``sigma_min`` and shifts an already shifted range, which
produces a materially different grid (for example a penultimate 4-step node near
0.024 instead of the released 0.625 at shift 5).
"""

from __future__ import annotations

import torch


def anyflow_inference_schedule(
    *,
    num_steps: int,
    shift: float,
    num_train_timesteps: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the released AnyFlow schedule including the clean endpoint.

    For ``num_steps=4``, ``shift=5`` and ``N=1000`` this returns approximately
    ``[1000, 937.5, 833.3333, 625, 0]``.
    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if float(shift) <= 0.0:
        raise ValueError("shift must be positive")
    if int(num_train_timesteps) <= 0:
        raise ValueError("num_train_timesteps must be positive")

    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps) + 1,
        device=device,
        dtype=torch.float64,
    )[:-1]
    shifted = float(shift) * base / (
        1.0 + (float(shift) - 1.0) * base
    )
    nodes = shifted * float(num_train_timesteps)
    schedule = torch.cat((nodes, nodes.new_zeros((1,))), dim=0)
    if not torch.all(schedule[:-1] > schedule[1:]):
        raise RuntimeError("AnyFlow schedule is not strictly descending")
    return schedule.to(dtype=torch.float32)


__all__ = ["anyflow_inference_schedule"]
