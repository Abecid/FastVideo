# SPDX-License-Identifier: Apache-2.0
"""Statistics and controllers for finite-transition RL v2 experiments.

The first FTPP/GRPO study used one four-sample group per optimizer update,
standardized every group independently, forced ESS=2 for every non-flat group,
and then evaluated only an EMA model.  This module contains the small,
model-independent pieces needed for a statistically meaningful follow-up:

* running prompt/global reward statistics;
* a global-temperature reward posterior that becomes uniform on weak groups;
* a conservative target-KL loss-scale controller;
* paired-difference and paired-bootstrap validation statistics; and
* deterministic transition-shift diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch


@dataclass
class RunningMoments:
    """Numerically stable scalar running mean and variance."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor | list[float] | tuple[float, ...]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64).flatten().cpu()
        for raw in tensor.tolist():
            value = float(raw)
            self.count += 1
            delta = value - self.mean
            self.mean += delta / float(self.count)
            delta2 = value - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / float(self.count)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def state_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "m2": float(self.m2),
        }

    @classmethod
    def from_state_dict(cls, raw: dict[str, Any]) -> "RunningMoments":
        return cls(
            count=int(raw.get("count", 0)),
            mean=float(raw.get("mean", 0.0)),
            m2=float(raw.get("m2", 0.0)),
        )


@dataclass
class PromptRewardTracker:
    """Running prompt baselines with a global fallback.

    For a prompt observed for the first time, ``baseline()`` returns the current
    group mean.  This avoids introducing a common non-zero score coefficient
    from an uncalibrated global baseline.  Once a prompt has history, its running
    mean is used.  The denominator is supplied separately from a much larger
    global rollout pool instead of a four-sample group standard deviation.
    """

    prompts: dict[str, RunningMoments] = field(default_factory=dict)
    global_moments: RunningMoments = field(default_factory=RunningMoments)

    def baseline(self, prompt: str, group_rewards: torch.Tensor) -> float:
        stats = self.prompts.get(str(prompt))
        if stats is None or stats.count == 0:
            return float(group_rewards.float().mean().item())
        return float(stats.mean)

    def update(self, prompt: str, group_rewards: torch.Tensor) -> None:
        key = str(prompt)
        stats = self.prompts.setdefault(key, RunningMoments())
        stats.update(group_rewards)
        self.global_moments.update(group_rewards)

    def state_dict(self) -> dict[str, Any]:
        return {
            "prompts": {
                key: value.state_dict()
                for key, value in self.prompts.items()
            },
            "global": self.global_moments.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, raw: dict[str, Any]) -> "PromptRewardTracker":
        tracker = cls()
        prompt_raw = raw.get("prompts", {})
        if isinstance(prompt_raw, dict):
            tracker.prompts = {
                str(key): RunningMoments.from_state_dict(value)
                for key, value in prompt_raw.items()
                if isinstance(value, dict)
            }
        global_raw = raw.get("global", {})
        if isinstance(global_raw, dict):
            tracker.global_moments = RunningMoments.from_state_dict(global_raw)
        return tracker


def stable_global_reward_std(
    rewards: torch.Tensor,
    *,
    previous: float,
    decay: float,
    floor: float,
) -> tuple[float, float]:
    """Return current and EMA reward standard deviations.

    ``rewards`` should contain every candidate from every rollout group in one
    optimizer update.  Unlike four-sample group normalization, this denominator
    becomes more stable as rollout accumulation increases.
    """
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be a rank-1 tensor with at least two values")
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("decay must lie in [0, 1)")
    if float(floor) <= 0.0:
        raise ValueError("floor must be positive")
    current = max(float(rewards.float().std(unbiased=False).item()), float(floor))
    if not math.isfinite(previous) or previous <= 0.0:
        ema = current
    else:
        ema = float(decay) * float(previous) + (1.0 - float(decay)) * current
    return current, max(ema, float(floor))


def running_baseline_advantages(
    rewards: torch.Tensor,
    *,
    baseline: float,
    global_std: float,
    clip: float,
) -> torch.Tensor:
    """Advantages using a running prompt mean and global rollout std."""
    if rewards.ndim != 1:
        raise ValueError("rewards must be rank-1")
    if float(global_std) <= 0.0:
        raise ValueError("global_std must be positive")
    if float(clip) <= 0.0:
        raise ValueError("clip must be positive")
    return ((rewards.float() - float(baseline)) / float(global_std)).clamp(
        min=-float(clip),
        max=float(clip),
    )


def global_temperature_weights(
    rewards: torch.Tensor,
    *,
    baseline: float,
    global_std: float,
    temperature_multiplier: float,
    minimum_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward-posterior weights with a cross-group temperature.

    Unlike exact per-group ESS targeting, the absolute reward spread is retained.
    A weak/noisy group therefore produces nearly uniform weights and a nearly
    zero centered update.
    """
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be rank-1 with at least two samples")
    if float(global_std) <= 0.0:
        raise ValueError("global_std must be positive")
    if float(temperature_multiplier) <= 0.0:
        raise ValueError("temperature_multiplier must be positive")
    if float(minimum_temperature) <= 0.0:
        raise ValueError("minimum_temperature must be positive")
    temperature = max(
        float(global_std) * float(temperature_multiplier),
        float(minimum_temperature),
    )
    logits = (rewards.float() - float(baseline)) / temperature
    weights = torch.softmax(logits, dim=0)
    ess = weights.square().sum().reciprocal()
    return (
        weights,
        rewards.new_tensor(temperature, dtype=torch.float32),
        ess,
    )


def update_target_kl_scale(
    current_scale: float,
    observed_kl: float,
    *,
    target_kl: float,
    controller_rate: float,
    minimum_scale: float,
    maximum_scale: float,
) -> float:
    """Conservatively adapt loss scale toward a target post-update KL.

    The square-root ratio follows the local ``KL ~ step_size^2`` approximation.
    Raising it to ``controller_rate`` prevents one noisy probe from changing the
    next update by orders of magnitude.
    """
    if current_scale <= 0.0:
        raise ValueError("current_scale must be positive")
    if target_kl <= 0.0:
        return float(current_scale)
    if not 0.0 < controller_rate <= 1.0:
        raise ValueError("controller_rate must lie in (0, 1]")
    if minimum_scale <= 0.0 or maximum_scale < minimum_scale:
        raise ValueError("invalid scale bounds")
    if not math.isfinite(observed_kl) or observed_kl <= 0.0:
        return min(max(float(current_scale), minimum_scale), maximum_scale)
    raw_factor = math.sqrt(float(target_kl) / float(observed_kl))
    factor = raw_factor ** float(controller_rate)
    updated = float(current_scale) * factor
    return min(max(updated, float(minimum_scale)), float(maximum_scale))


def paired_difference_statistics(
    current: torch.Tensor,
    baseline: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mean, std and SEM of paired differences."""
    if current.ndim != 1 or baseline.ndim != 1 or current.shape != baseline.shape:
        raise ValueError("current and baseline must be equal rank-1 tensors")
    if current.numel() == 0:
        raise ValueError("paired tensors must be non-empty")
    delta = current.float() - baseline.float()
    std = delta.std(unbiased=False)
    return {
        "delta": delta,
        "mean": delta.mean(),
        "std": std,
        "sem": std / math.sqrt(float(delta.numel())),
    }


def paired_bootstrap_interval(
    deltas: torch.Tensor,
    *,
    confidence: float = 0.95,
    num_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic percentile-bootstrap interval for paired mean deltas."""
    if deltas.ndim != 1 or deltas.numel() == 0:
        raise ValueError("deltas must be a non-empty rank-1 tensor")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    cpu = deltas.detach().float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        cpu.numel(),
        (int(num_bootstrap), cpu.numel()),
        generator=generator,
    )
    means = cpu[indices].mean(dim=1)
    alpha = 0.5 * (1.0 - float(confidence))
    return (
        torch.quantile(means, alpha),
        torch.quantile(means, 1.0 - alpha),
    )


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def cosine_similarity_flat(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Cosine similarity over all non-batch dimensions, then batch mean."""
    if left.shape != right.shape:
        raise ValueError("left and right must have equal shapes")
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return (numerator / denominator.clamp_min(1.0e-12)).mean()


__all__ = [
    "PromptRewardTracker",
    "RunningMoments",
    "cosine_similarity_flat",
    "global_temperature_weights",
    "paired_bootstrap_interval",
    "paired_difference_statistics",
    "rms",
    "running_baseline_advantages",
    "stable_global_reward_std",
    "update_target_kl_scale",
]
