# SPDX-License-Identifier: Apache-2.0
"""Small deterministic helpers for the multi-minibatch GRPO v3 baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupAdvantageBatch:
    """Normalized rewards for one prompt's candidate group."""

    advantages: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    active: bool


def group_normalized_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float,
    clip: float,
    minimum_std: float,
) -> GroupAdvantageBatch:
    """Normalize candidates only against other candidates for the same prompt.

    The v2 experiment subtracted a per-prompt mean but divided by a standard
    deviation pooled across different prompts. Between-prompt difficulty can be
    much larger than the within-prompt preference signal. This helper follows
    the reference Wan GRPO recipe: both centering and scaling are local to the
    prompt group.

    Groups whose reward spread is below ``minimum_std`` produce an exact zero
    update rather than amplifying a numerically meaningless ranking.
    """
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be rank-1 with at least two candidates")
    if epsilon <= 0.0 or clip <= 0.0 or minimum_std < 0.0:
        raise ValueError("epsilon/clip must be positive and minimum_std non-negative")

    values = rewards.float()
    mean = values.mean()
    std = values.std(unbiased=False)
    active = bool(torch.isfinite(std)) and float(std) >= float(minimum_std)
    if not active:
        advantages = torch.zeros_like(values)
    else:
        advantages = ((values - mean) / (std + float(epsilon))).clamp(
            min=-float(clip),
            max=float(clip),
        )
    return GroupAdvantageBatch(
        advantages=advantages,
        mean=mean,
        std=std,
        active=active,
    )


def shuffled_group_minibatches(
    num_groups: int,
    *,
    groups_per_minibatch: int,
    seed: int,
) -> list[list[int]]:
    """Return a deterministic complete partition of rollout-group indices."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if groups_per_minibatch <= 0:
        raise ValueError("groups_per_minibatch must be positive")
    if groups_per_minibatch > num_groups:
        raise ValueError("groups_per_minibatch cannot exceed num_groups")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(num_groups, generator=generator).tolist()
    return [
        order[start : start + groups_per_minibatch]
        for start in range(0, num_groups, groups_per_minibatch)
    ]


__all__ = [
    "GroupAdvantageBatch",
    "group_normalized_advantages",
    "shuffled_group_minibatches",
]
