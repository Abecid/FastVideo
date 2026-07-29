# SPDX-License-Identifier: Apache-2.0
"""Pure math utilities for Video Posterior Transition Distillation (VPTD).

The implementation is intentionally small and source-traceable:

* AnyFlow supplies the pretrained two-time Wan flow map and shift-5 schedule.
* Flow-Map GRPO supplies endpoint-anchor ASFMC, K=4, and the optional
  clipped likelihood-ratio baseline; released video Flow-GRPO supplies G=4.
* Diamond Maps/Feynman--Kac steering motivates reward tilting a conditional
  posterior rather than weighting unrelated terminal videos.
* Advantage Weighted Matching and reward-weighted regression motivate the
  default forward-KL projection of the tilted posterior back into the model.
* BranchGRPO/Flow-GRPO-Fast motivates branching one shared-prefix transition so
  terminal reward differences have local credit.

This module has no AnyFlow or FastVideo dependency and is unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class PosteriorPolicyConfig:
    """Released algorithm constants used by the experiment."""

    stochastic_steps: int = 4
    group_size: int = 4
    target_ess_ratio: float = 0.5
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0
    advantage_epsilon: float = 1.0e-4

    def validate(self) -> None:
        if self.stochastic_steps <= 0:
            raise ValueError("stochastic_steps must be positive")
        if self.group_size <= 1:
            raise ValueError("group_size must exceed one")
        if not 0.0 < self.target_ess_ratio <= 1.0:
            raise ValueError("target_ess_ratio must lie in (0, 1]")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must lie in (0, 1)")
        if self.advantage_clip <= 0.0:
            raise ValueError("advantage_clip must be positive")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")


def append_data_endpoint(timesteps: torch.Tensor) -> torch.Tensor:
    """Append the clean-data endpoint ``t=0`` to an AnyFlow reverse schedule."""

    if timesteps.ndim != 1 or timesteps.numel() < 1:
        raise ValueError("timesteps must be a non-empty rank-1 tensor")
    schedule = torch.cat([timesteps, timesteps.new_zeros((1,))], dim=0)
    if not torch.all(schedule[:-1] > schedule[1:]):
        raise ValueError("reverse-time schedule must be strictly decreasing")
    return schedule


def validate_training_schedule(schedule: torch.Tensor, *, stochastic_steps: int) -> None:
    """Validate ``K stochastic transitions + one deterministic final step``."""

    expected_transitions = int(stochastic_steps) + 1
    if schedule.numel() != expected_transitions + 1:
        raise ValueError(
            "expected K stochastic transitions plus one deterministic final "
            f"transition ({expected_transitions + 1} nodes), got {schedule.numel()}"
        )
    if float(schedule[-1]) != 0.0:
        raise ValueError("the final schedule node must be t=0")


def endpoint_anchor_parameters(
    clean_endpoint: torch.Tensor,
    target_timestep: torch.Tensor | float,
    *,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact endpoint-anchor ASFMC Gaussian in AnyFlow's reverse convention.

    AnyFlow uses the affine path ``x_r=(1-r)x_0+r eps`` with ``r=1`` at
    Gaussian noise and ``r=0`` at clean video. Conditioning on the clean
    endpoint therefore yields ``N((1-r)x_0, r^2 I)``. No free noise multiplier
    or variance floor is introduced, because either would break path
    preservation. The final segment is deterministic, so stochastic targets
    always have ``r>0``.
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
    """Sample a diagonal Gaussian action and return its standard noise."""

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

    The reduction follows the released Flow-GRPO Wan implementation. Summing
    millions of video-latent dimensions makes likelihood ratios numerically
    unusable; averaging preserves the gradient direction up to a fixed scale.
    """

    if action.shape != mean.shape:
        raise ValueError(
            f"action and mean shapes must match, got {action.shape} and {mean.shape}"
        )
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")
    standardized = (action.detach() - mean) / std
    log_prob = -0.5 * standardized.square() - torch.log(std) - 0.5 * LOG_TWO_PI
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def global_group_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one-prompt group-relative advantages over the global group."""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be rank-1 with at least two samples")
    if epsilon <= 0.0 or clip <= 0.0:
        raise ValueError("epsilon and clip must be positive")
    rewards = rewards.float()
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = ((rewards - mean) / (std + float(epsilon))).clamp(
        min=-float(clip), max=float(clip)
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
    """Feynman--Kac weights with a scale-free effective-sample-size target.

    The temperature is solved per group so that
    ``ESS(softmax((R-mean(R))/tau)) = target_ess_ratio * G``. This avoids
    transferring a reward-temperature number across incompatible reward
    models. Degenerate rewards return the exact uniform distribution.
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
    weights = torch.softmax(centered / temperature.clamp_min(1.0e-12), dim=0)
    return weights, temperature, effective_sample_size(weights)


def posterior_distillation_loss(
    new_log_prob: torch.Tensor,
    weights: torch.Tensor,
    *,
    distributed_world_size: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-step forward-KL posterior projection with an exact score baseline.

    Let ``q_old(a|s)`` be the ASFMC behavior posterior and
    ``q_R(a|s) ∝ q_old(a|s) exp(R(a)/tau)`` its Feynman--Kac tilt. At the
    behavior parameters, the gradient of ``KL(q_R || pi_theta)`` is unchanged
    if we subtract the behavior score expectation, because
    ``E_q_old[grad log pi_old]=0``. The empirical coefficient is therefore
    ``w_j - 1/G`` rather than ``w_j``. This control variate has two important
    properties: it is the same first-order posterior-projection gradient, and
    it gives exactly zero update when all rewards are equal instead of causing
    a finite-sample behavior-cloning random walk.

    ``weights`` are the local slice of globally normalized Feynman--Kac
    weights. DDP averages gradients across ranks, so local weighted sums are
    multiplied by world size to recover the global objective exactly.
    """

    if (
        new_log_prob.ndim != 1
        or weights.ndim != 1
        or new_log_prob.shape != weights.shape
    ):
        raise ValueError("new_log_prob and weights must be equal rank-1 tensors")
    if distributed_world_size <= 0:
        raise ValueError("distributed_world_size must be positive")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    global_group_size = int(weights.numel()) * int(distributed_world_size)
    score_coefficients = weights.detach() - (1.0 / float(global_group_size))
    loss = -float(distributed_world_size) * (score_coefficients * new_log_prob).sum()
    safe_weights = weights.clamp_min(1.0e-12)
    entropy = -(safe_weights * safe_weights.log()).sum()
    return loss, {
        "posterior_weight_mass_local": weights.sum().detach(),
        "posterior_weight_max_local": weights.max().detach(),
        "posterior_weight_entropy_local": entropy.detach(),
        "score_coefficient_mass_local": score_coefficients.sum().detach(),
        "score_coefficient_abs_mean_local": score_coefficients.abs().mean().detach(),
        "weighted_log_prob": (weights * new_log_prob.detach()).sum(),
    }


def clipped_grpo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optional Flow-Map GRPO likelihood-ratio baseline."""

    if not (
        new_log_prob.shape == old_log_prob.shape == advantages.shape
        and new_log_prob.ndim == 1
    ):
        raise ValueError("log probabilities and advantages must be equal rank-1 tensors")
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
    """Mean absolute change between adjacent frames for ``[B,C,T,H,W]``."""

    if video.ndim != 5:
        raise ValueError(f"video must have shape [B,C,T,H,W], got {video.shape}")
    if video.shape[2] <= 1:
        return torch.zeros(video.shape[0], device=video.device, dtype=torch.float32)
    return (video[:, :, 1:].float() - video[:, :, :-1].float()).abs().mean(
        dim=(1, 2, 3, 4)
    )


def component_summary(values: Iterable[float]) -> dict[str, float]:
    """JSON/W&B-friendly scalar summary."""

    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def verify_group_partition(group_size: int, world_size: int) -> int:
    """Return the exact number of posterior branches collected per rank."""

    if group_size <= 1:
        raise ValueError("group_size must exceed one")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if group_size % world_size != 0:
        raise ValueError(
            f"group_size={group_size} must be divisible by world_size={world_size}"
        )
    return group_size // world_size


def provenance_table() -> dict[str, str]:
    """Machine-readable provenance for scientific defaults."""

    return {
        "base_checkpoint_and_video_shape": "NVlabs/AnyFlow released Wan-1.3B config",
        "flow_shift_and_eval_nfe": "NVlabs/AnyFlow released Wan-1.3B config",
        "lora_rank_alpha_targets": "NVlabs/AnyFlow released Wan-1.3B config",
        "optimizer_and_ema": "NVlabs/AnyFlow Wan on-policy config",
        "K_clip_advantage_epsilon": "Flow-Map GRPO Appendix C.1",
        "video_group_size": "released Flow-GRPO Wan configuration",
        "endpoint_posterior": "Flow-Map GRPO Eq. (13), exact affine-path conditional",
        "shared_prefix_transition_credit": "BranchGRPO / Flow-GRPO-Fast",
        "reward_tilt_projection": "Diamond Maps Feynman-Kac steering + AWM/reward-weighted regression",
        "target_ess_ratio": "Pyro SMCFilter default half-particle ESS threshold",
        "log_prob_reduction": "released Flow-GRPO Wan implementation",
    }
