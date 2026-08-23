# SPDX-License-Identifier: Apache-2.0
"""Reusable math for finite-transition RL on deterministic flow maps.

The module contains path-preserving endpoint-anchor policy math, group-relative
reward tilting, direct posterior projection, the matched GRPO loss, and video
motion/diversity diagnostics.  It intentionally has no model-family imports and
is unit-testable on CPU.
"""

from __future__ import annotations

import math

import torch

LOG_TWO_PI = math.log(2.0 * math.pi)


def append_data_endpoint(timesteps: torch.Tensor) -> torch.Tensor:
    """Append the clean endpoint ``t=0`` to a descending reverse schedule."""
    if timesteps.ndim != 1 or timesteps.numel() < 1:
        raise ValueError("timesteps must be a non-empty rank-1 tensor")
    schedule = torch.cat((timesteps, timesteps.new_zeros((1,))), dim=0)
    if not torch.all(schedule[:-1] > schedule[1:]):
        raise ValueError("reverse-time schedule must be strictly decreasing")
    return schedule


def validate_training_schedule(
    schedule: torch.Tensor,
    *,
    stochastic_steps: int,
) -> None:
    """Validate ``K`` stochastic transitions plus one deterministic finish."""
    expected_nodes = int(stochastic_steps) + 2
    if schedule.ndim != 1 or schedule.numel() != expected_nodes:
        raise ValueError(
            "expected K stochastic transitions plus one deterministic final "
            f"transition ({expected_nodes} nodes), got {tuple(schedule.shape)}"
        )
    if float(schedule[-1]) != 0.0:
        raise ValueError("the final schedule node must be t=0")
    if not torch.all(schedule[:-1] > schedule[1:]):
        raise ValueError("training schedule must be strictly decreasing")


def endpoint_anchor_parameters(
    clean_endpoint: torch.Tensor,
    target_timestep: torch.Tensor | float,
    *,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact affine-path endpoint-anchor Gaussian parameters.

    With reverse convention ``x_r=(1-r)x_0+r eps``, conditioning on ``x_0``
    yields ``N((1-r)x_0, r^2 I)``. ``target_timestep`` is expressed in the
    model's absolute training-timestep units.
    """
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    r = torch.as_tensor(
        target_timestep,
        device=clean_endpoint.device,
        dtype=torch.float32,
    ) / float(num_train_timesteps)
    if torch.any((r <= 0.0) | (r > 1.0)):
        raise ValueError("stochastic target time must lie in (0, 1]")
    while r.ndim < clean_endpoint.ndim:
        r = r.unsqueeze(-1)
    mean = (1.0 - r).to(clean_endpoint.dtype) * clean_endpoint
    return mean, r.to(clean_endpoint.dtype)


def sample_diagonal_gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a diagonal Gaussian action and return the standard noise."""
    noise = torch.randn(
        mean.shape,
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    return mean + std * noise, noise


def gaussian_log_prob_mean(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Mean Gaussian log density over all non-batch latent dimensions.

    Averaging, rather than summing, avoids likelihood ratios whose scale grows
    with the millions of coordinates in a video latent.
    """
    if action.shape != mean.shape:
        raise ValueError(
            f"action and mean shapes must match, got {action.shape} and {mean.shape}"
        )
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")
    # The latent policy normally runs in bf16. Reducing millions of bf16 log
    # density terms directly can quantize a real optimizer-step change to
    # exactly zero, which also blinds the target-KL controller. Keep gradients
    # through ``mean`` while doing the likelihood arithmetic and reduction in
    # float32.
    action_float = action.detach().float()
    mean_float = mean.float()
    std_float = std.float()
    standardized = (action_float - mean_float) / std_float
    log_prob = (
        -0.5 * standardized.square()
        - torch.log(std_float)
        - 0.5 * LOG_TWO_PI
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def diagonal_gaussian_kl_mean(
    reference_mean: torch.Tensor,
    reference_std: torch.Tensor,
    updated_mean: torch.Tensor,
    updated_std: torch.Tensor,
) -> torch.Tensor:
    """Coordinate-normalized KL from a reference to an updated policy.

    Computing KL from a sampled, dimension-averaged log-probability delta can
    cancel positive and negative coordinate contributions. The analytic
    diagonal-Gaussian expression stays non-negative and remains comparable
    across latent resolutions by averaging all non-batch dimensions.
    """
    if reference_mean.shape != updated_mean.shape:
        raise ValueError("reference and updated mean shapes must match")
    if torch.any(reference_std <= 0) or torch.any(updated_std <= 0):
        raise ValueError("standard deviations must be strictly positive")

    reference_mean_float = reference_mean.detach().float()
    updated_mean_float = updated_mean.float()
    reference_std_float = reference_std.detach().float()
    updated_std_float = updated_std.float()
    mean_delta = reference_mean_float - updated_mean_float
    log_std_ratio = torch.log(updated_std_float) - torch.log(reference_std_float)
    variance_ratio_minus_one = torch.expm1(-2.0 * log_std_ratio)
    kl = (
        log_std_ratio
        + 0.5
        * (
            variance_ratio_minus_one
            + mean_delta.square() / updated_std_float.square()
        )
    )
    return kl.mean(dim=tuple(range(1, kl.ndim)))


def group_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute clipped group-relative advantages for a single prompt."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be rank-1 with at least two samples")
    if epsilon <= 0.0 or clip <= 0.0:
        raise ValueError("epsilon and clip must be positive")
    rewards = rewards.float()
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = ((rewards - mean) / (std + float(epsilon))).clamp(
        min=-float(clip),
        max=float(clip),
    )
    return advantages, mean, std


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Effective sample size for normalized non-negative weights."""
    if weights.ndim != 1:
        raise ValueError("weights must be rank-1")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    normalized = weights / weights.sum().clamp_min(1.0e-12)
    return normalized.square().sum().reciprocal()


def reward_tilted_weights(
    rewards: torch.Tensor,
    *,
    target_ess_ratio: float,
    bisection_steps: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct Feynman--Kac weights with a scale-free ESS target.

    ``tau`` is solved per group such that
    ``ESS(softmax((R-mean(R))/tau)) = target_ess_ratio * G``. Degenerate
    reward groups return exactly uniform weights and an infinite temperature.
    """
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be rank-1 with at least two samples")
    if not 0.0 < target_ess_ratio <= 1.0:
        raise ValueError("target_ess_ratio must lie in (0, 1]")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")

    centered = rewards.float() - rewards.float().mean()
    scale = centered.std(unbiased=False)
    group_size = centered.numel()
    uniform = torch.full_like(centered, 1.0 / float(group_size))
    if float(scale) < 1.0e-8 or target_ess_ratio >= 1.0:
        return (
            uniform,
            centered.new_tensor(float("inf")),
            centered.new_tensor(float(group_size)),
        )

    standardized = centered / scale
    target_ess = float(target_ess_ratio) * float(group_size)
    log_lo = standardized.new_tensor(-8.0)
    log_hi = standardized.new_tensor(8.0)
    for _ in range(int(bisection_steps)):
        log_mid = 0.5 * (log_lo + log_hi)
        candidate = torch.softmax(standardized / log_mid.exp(), dim=0)
        if float(effective_sample_size(candidate)) < target_ess:
            log_lo = log_mid
        else:
            log_hi = log_mid
    temperature = log_hi.exp() * scale
    weights = torch.softmax(
        centered / temperature.clamp_min(1.0e-12),
        dim=0,
    )
    return weights, temperature, effective_sample_size(weights)


def posterior_projection_loss(
    new_log_prob: torch.Tensor,
    local_weights: torch.Tensor,
    *,
    global_group_size: int,
    distributed_world_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward-KL posterior projection with a behavior-score baseline.

    At the behavior parameters, subtracting ``1/G`` leaves the expected
    first-order projection gradient unchanged because the behavior score has
    zero expectation. It also guarantees an exactly zero update for an
    uninformative reward group.

    DDP/FSDP averages gradients across ranks, so the local weighted sum is
    multiplied by ``distributed_world_size`` to recover the intended global
    objective.
    """
    if new_log_prob.ndim != 1 or local_weights.ndim != 1:
        raise ValueError("new_log_prob and local_weights must be rank-1")
    if new_log_prob.shape != local_weights.shape:
        raise ValueError("new_log_prob and local_weights must have equal shapes")
    if global_group_size <= 1:
        raise ValueError("global_group_size must exceed one")
    if distributed_world_size <= 0:
        raise ValueError("distributed_world_size must be positive")
    if torch.any(local_weights < 0):
        raise ValueError("local_weights must be non-negative")

    coefficients = local_weights.detach() - (1.0 / float(global_group_size))
    loss = (
        -float(distributed_world_size)
        * (coefficients * new_log_prob).sum()
    )
    safe_weights = local_weights.clamp_min(1.0e-12)
    return loss, {
        "posterior_weight_mass_local": local_weights.sum().detach(),
        "posterior_weight_max_local": local_weights.max().detach(),
        "posterior_weight_entropy_local": (
            -(safe_weights * safe_weights.log()).sum().detach()
        ),
        "score_coefficient_mass_local": coefficients.sum().detach(),
        "score_coefficient_abs_mean_local": coefficients.abs().mean().detach(),
        "weighted_log_prob_local": (
            local_weights * new_log_prob.detach()
        ).sum(),
    }


def clipped_grpo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Matched Flow-Map-GRPO likelihood-ratio ablation."""
    if not (
        new_log_prob.ndim == 1
        and new_log_prob.shape == old_log_prob.shape == advantages.shape
    ):
        raise ValueError(
            "new_log_prob, old_log_prob and advantages must be equal rank-1 tensors"
        )
    if not 0.0 < clip_range < 1.0:
        raise ValueError("clip_range must lie in (0, 1)")
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    unclipped = ratio * advantages
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = -torch.minimum(unclipped, clipped_ratio * advantages).mean()
    clip_event = (ratio - 1.0).abs() > clip_range
    return loss, {
        "ratio_mean": ratio.mean().detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "clip_fraction": clip_event.float().mean().detach(),
        "approx_kl": (0.5 * log_ratio.square().mean()).detach(),
        "log_ratio_abs_mean": log_ratio.abs().mean().detach(),
    }


def temporal_l1(video: torch.Tensor) -> torch.Tensor:
    """Mean adjacent-frame absolute difference for ``[B,C,T,H,W]`` video."""
    if video.ndim != 5:
        raise ValueError(f"video must have shape [B,C,T,H,W], got {video.shape}")
    if video.shape[2] <= 1:
        return torch.zeros(
            video.shape[0],
            device=video.device,
            dtype=torch.float32,
        )
    return (
        video[:, :, 1:].float() - video[:, :, :-1].float()
    ).abs().mean(dim=(1, 2, 3, 4))


def mean_pairwise_rms(samples: torch.Tensor) -> torch.Tensor:
    """Mean pairwise RMS distance for samples with shape ``[G, ...]``."""
    if samples.ndim < 2:
        raise ValueError("samples must have shape [G, ...]")
    group_size = int(samples.shape[0])
    if group_size < 2:
        return samples.new_zeros((), dtype=torch.float32)
    flat = samples.float().flatten(1)
    distances = []
    for i in range(group_size):
        for j in range(i + 1, group_size):
            distances.append((flat[i] - flat[j]).square().mean().sqrt())
    return torch.stack(distances).mean()


__all__ = [
    "append_data_endpoint",
    "clipped_grpo_loss",
    "effective_sample_size",
    "endpoint_anchor_parameters",
    "gaussian_log_prob_mean",
    "group_advantages",
    "mean_pairwise_rms",
    "posterior_projection_loss",
    "reward_tilted_weights",
    "sample_diagonal_gaussian",
    "temporal_l1",
    "validate_training_schedule",
]
