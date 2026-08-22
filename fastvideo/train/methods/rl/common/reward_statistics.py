# SPDX-License-Identifier: Apache-2.0
"""Stateful reward normalization, target-KL control, and paired statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class RunningMoments:
    """Numerically stable scalar running moments."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor | list[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
        for value in tensor.tolist():
            self.count += 1
            delta = float(value) - self.mean
            self.mean += delta / float(self.count)
            delta2 = float(value) - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / float(self.count)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "m2": float(self.m2),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.count = int(state_dict.get("count", 0))
        self.mean = float(state_dict.get("mean", 0.0))
        self.m2 = float(state_dict.get("m2", 0.0))


class PromptRewardNormalizer:
    """Running per-prompt baseline with a global reward scale.

    The current rollout is normalized using statistics from previous rollouts,
    then incorporated into the tracker. This avoids using a noisy four-sample
    denominator as the sole scale of every update.
    """

    def __init__(
        self,
        *,
        min_prompt_count: int = 4,
        min_global_count: int = 32,
        epsilon: float = 1.0e-4,
        clip: float = 5.0,
    ) -> None:
        if min_prompt_count < 1 or min_global_count < 2:
            raise ValueError("minimum counts must be positive")
        if epsilon <= 0.0 or clip <= 0.0:
            raise ValueError("epsilon and clip must be positive")
        self.min_prompt_count = int(min_prompt_count)
        self.min_global_count = int(min_global_count)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.global_moments = RunningMoments()
        self.prompt_moments: dict[str, RunningMoments] = {}

    def normalize(
        self,
        rewards: torch.Tensor,
        *,
        prompt: str,
        update: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if rewards.ndim != 1 or rewards.numel() < 2:
            raise ValueError("rewards must be rank-1 with at least two values")
        values = rewards.detach().float()
        group_mean = float(values.mean())
        group_std = float(values.std(unbiased=False))
        prompt_key = str(prompt)
        prompt_stats = self.prompt_moments.setdefault(
            prompt_key,
            RunningMoments(),
        )

        if prompt_stats.count >= self.min_prompt_count:
            baseline = prompt_stats.mean
            baseline_source = 2.0
        elif self.global_moments.count >= self.min_global_count:
            baseline = self.global_moments.mean
            baseline_source = 1.0
        else:
            baseline = group_mean
            baseline_source = 0.0

        if (
            self.global_moments.count >= self.min_global_count
            and self.global_moments.std > self.epsilon
        ):
            scale = self.global_moments.std
            scale_source = 1.0
        else:
            scale = max(group_std, self.epsilon)
            scale_source = 0.0

        advantages = ((values - baseline) / (scale + self.epsilon)).clamp(
            -self.clip,
            self.clip,
        )
        prompt_count_before = prompt_stats.count
        global_count_before = self.global_moments.count
        if update:
            cpu_values = values.cpu()
            prompt_stats.update(cpu_values)
            self.global_moments.update(cpu_values)

        return advantages, {
            "baseline": float(baseline),
            "scale": float(scale),
            "group_mean": group_mean,
            "group_std": group_std,
            "baseline_source": baseline_source,
            "scale_source": scale_source,
            "prompt_count_before": float(prompt_count_before),
            "global_count_before": float(global_count_before),
        }

    def temperature(
        self,
        rewards: torch.Tensor,
        *,
        scale_multiplier: float,
        minimum: float = 1.0e-4,
    ) -> torch.Tensor:
        if scale_multiplier <= 0.0 or minimum <= 0.0:
            raise ValueError("temperature parameters must be positive")
        group_std = float(rewards.detach().float().std(unbiased=False))
        if (
            self.global_moments.count >= self.min_global_count
            and self.global_moments.std > self.epsilon
        ):
            scale = self.global_moments.std
        else:
            scale = max(group_std, self.epsilon)
        return rewards.new_tensor(
            max(float(scale_multiplier) * scale, float(minimum)),
            dtype=torch.float32,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "min_prompt_count": self.min_prompt_count,
            "min_global_count": self.min_global_count,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "global": self.global_moments.state_dict(),
            "prompts": {
                key: value.state_dict()
                for key, value in self.prompt_moments.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.global_moments.load_state_dict(
            dict(state_dict.get("global", {}))
        )
        prompts = state_dict.get("prompts", {})
        if not isinstance(prompts, dict):
            raise TypeError("reward normalizer prompts state must be a mapping")
        self.prompt_moments = {}
        for key, value in prompts.items():
            moments = RunningMoments()
            moments.load_state_dict(dict(value))
            self.prompt_moments[str(key)] = moments


class TargetKLController:
    """Multiplicative loss-scale controller driven by incremental update KL."""

    def __init__(
        self,
        *,
        target_kl: float,
        initial_scale: float = 1.0,
        min_scale: float = 0.05,
        max_scale: float = 128.0,
        max_adjustment: float = 2.0,
    ) -> None:
        if target_kl <= 0.0:
            raise ValueError("target_kl must be positive")
        if not 0.0 < min_scale <= initial_scale <= max_scale:
            raise ValueError("loss-scale bounds are inconsistent")
        if max_adjustment <= 1.0:
            raise ValueError("max_adjustment must exceed one")
        self.target_kl = float(target_kl)
        self.scale = float(initial_scale)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_adjustment = float(max_adjustment)
        self.last_kl = 0.0

    def update(self, measured_kl: float | torch.Tensor) -> float:
        value = float(torch.as_tensor(measured_kl).detach().float())
        self.last_kl = value
        if not math.isfinite(value):
            # A non-finite probe is a stability failure, never a reason to make
            # the next update larger.
            adjustment = 1.0 / self.max_adjustment
        elif value <= 1.0e-16:
            # Exactly zero commonly means an equal-reward/no-gradient group or
            # a skipped probe. Preserve the current scale instead of escalating
            # toward the maximum without evidence.
            adjustment = 1.0
        else:
            adjustment = math.sqrt(self.target_kl / value)
            adjustment = min(
                self.max_adjustment,
                max(1.0 / self.max_adjustment, adjustment),
            )
        self.scale = min(
            self.max_scale,
            max(self.min_scale, self.scale * adjustment),
        )
        return self.scale

    def state_dict(self) -> dict[str, float]:
        return {
            "target_kl": self.target_kl,
            "scale": self.scale,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "max_adjustment": self.max_adjustment,
            "last_kl": self.last_kl,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scale = float(state_dict.get("scale", self.scale))
        self.last_kl = float(state_dict.get("last_kl", 0.0))


def reward_softmax_weights(
    rewards: torch.Tensor,
    *,
    temperature: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized Boltzmann weights and their ESS."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be rank-1 with at least two values")
    tau = torch.as_tensor(
        temperature,
        device=rewards.device,
        dtype=torch.float32,
    )
    if tau.numel() != 1 or float(tau) <= 0.0:
        raise ValueError("temperature must be a positive scalar")
    centered = rewards.float() - rewards.float().mean()
    weights = torch.softmax(centered / tau, dim=0)
    ess = weights.square().sum().reciprocal()
    return weights, ess


def paired_summary(
    current: torch.Tensor,
    baseline: torch.Tensor,
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired mean/SEM and deterministic percentile-bootstrap interval."""
    current = torch.as_tensor(current, dtype=torch.float64).flatten()
    baseline = torch.as_tensor(baseline, dtype=torch.float64).flatten()
    if current.shape != baseline.shape or current.numel() < 2:
        raise ValueError("paired vectors must have equal shape and >=2 values")
    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")

    delta = current - baseline
    count = int(delta.numel())
    mean = float(delta.mean())
    std = float(delta.std(unbiased=False))
    sem = std / math.sqrt(float(count))

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chunk = min(512, int(bootstrap_samples))
    samples: list[torch.Tensor] = []
    remaining = int(bootstrap_samples)
    cpu_delta = delta.cpu()
    while remaining > 0:
        size = min(chunk, remaining)
        indices = torch.randint(
            0,
            count,
            (size, count),
            generator=generator,
        )
        samples.append(cpu_delta[indices].mean(dim=1))
        remaining -= size
    bootstrap = torch.cat(samples)
    alpha = (1.0 - float(confidence)) / 2.0
    lower = float(torch.quantile(bootstrap, alpha))
    upper = float(torch.quantile(bootstrap, 1.0 - alpha))
    return {
        "count": float(count),
        "mean_delta": mean,
        "std_delta": std,
        "sem_delta": sem,
        "ci_lower": lower,
        "ci_upper": upper,
        "positive_fraction": float((delta > 0).double().mean()),
    }


__all__ = [
    "PromptRewardNormalizer",
    "RunningMoments",
    "TargetKLController",
    "paired_summary",
    "reward_softmax_weights",
]
