# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for Reward-Tilted Flow Distillation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


@dataclass(slots=True)
class RTFDConfig:
    student_num_steps: int = 4
    student_guidance_scale: float = 1.0
    trajectories_per_prompt: int = 4
    transition_batch_size: int = 1
    reward_ess_ratio: float = 0.6
    uniform_mix: float = 0.25
    reward_bisection_steps: int = 32
    max_grad_norm: float = 1.0
    validation_every_steps: int = 5


def build_deployment_flow_schedule(
    *,
    num_steps: int,
    flow_shift: float,
    num_train_timesteps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deployment timesteps, sigmas, and normalized interval weights."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(num_train_timesteps),
        shift=float(flow_shift),
    )
    scheduler.set_timesteps(num_inference_steps=int(num_steps), device=device)
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    if timesteps.numel() != num_steps or sigmas.numel() != num_steps + 1:
        raise RuntimeError(
            "Unexpected FlowMatch Euler schedule shape: "
            f"timesteps={tuple(timesteps.shape)} sigmas={tuple(sigmas.shape)}"
        )
    interval_lengths = (sigmas[1:] - sigmas[:-1]).abs()
    interval_weights = interval_lengths / interval_lengths.sum().clamp_min(1e-12)
    return timesteps, sigmas, interval_weights


def reward_tilt_weights(
    rewards: torch.Tensor,
    *,
    target_ess_ratio: float,
    uniform_mix: float,
    bisection_steps: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create per-prompt softmax weights with target ESS and teacher mixing."""
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [P, M], got {tuple(rewards.shape)}")
    if not 0.0 < target_ess_ratio <= 1.0:
        raise ValueError("target_ess_ratio must lie in (0, 1]")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must lie in [0, 1]")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")

    num_prompts, num_trajectories = rewards.shape
    if num_trajectories <= 0:
        raise ValueError("at least one trajectory per prompt is required")

    centered = rewards.float() - rewards.float().mean(dim=1, keepdim=True)
    std = centered.std(dim=1, unbiased=False, keepdim=True)
    standardized = centered / std.clamp_min(1e-6)
    uniform = torch.full_like(standardized, 1.0 / float(num_trajectories))
    degenerate = std.squeeze(1) < 1e-6
    target_ess = float(target_ess_ratio) * float(num_trajectories)

    log_lo = torch.full((num_prompts,), -8.0, device=rewards.device, dtype=torch.float32)
    log_hi = torch.full((num_prompts,), 8.0, device=rewards.device, dtype=torch.float32)
    for _ in range(int(bisection_steps)):
        log_mid = 0.5 * (log_lo + log_hi)
        candidate = torch.softmax(standardized / log_mid.exp().unsqueeze(1), dim=1)
        too_concentrated = candidate.square().sum(dim=1).reciprocal() < target_ess
        log_lo = torch.where(too_concentrated, log_mid, log_lo)
        log_hi = torch.where(too_concentrated, log_hi, log_mid)

    temperatures = log_hi.exp()
    tilted = torch.softmax(standardized / temperatures.unsqueeze(1), dim=1)
    tilted = torch.where(degenerate.unsqueeze(1), uniform, tilted)
    raw_ess_ratio = tilted.square().sum(dim=1).reciprocal() / float(num_trajectories)
    final_weights = (1.0 - float(uniform_mix)) * tilted + float(uniform_mix) * uniform
    final_weights = final_weights / final_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    final_ess_ratio = final_weights.square().sum(dim=1).reciprocal() / float(num_trajectories)
    return final_weights, temperatures, raw_ess_ratio, final_ess_ratio


def infer_batch_size(raw_batch: dict[str, Any]) -> int:
    for key in ("text_embedding", "text_attention_mask", "vae_latent"):
        value = raw_batch.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    info_list = raw_batch.get("info_list")
    if isinstance(info_list, list) and info_list:
        return len(info_list)
    raise ValueError("Could not infer batch size from raw training batch")


def repeat_batch_rows(
    raw_batch: dict[str, Any],
    *,
    repeats: int,
    batch_size: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw_batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            out[key] = value.repeat_interleave(repeats, dim=0)
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = [copy.deepcopy(item) for item in value for _ in range(repeats)]
        else:
            out[key] = value
    return out


def extract_prompts(raw_batch: dict[str, Any]) -> list[str]:
    infos = raw_batch.get("info_list")
    if isinstance(infos, list) and infos:
        return [
            str(info.get("prompt") or info.get("caption") or "")
            if isinstance(info, dict) else ""
            for info in infos
        ]
    captions = raw_batch.get("caption_text")
    if isinstance(captions, list):
        return [str(caption) for caption in captions]
    raise ValueError("Could not find prompts in info_list or caption_text")


def repeat_condition_rows(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = hidden.shape[0]
    row_hidden = hidden.unsqueeze(1).expand(
        num_samples,
        num_steps,
        *hidden.shape[1:],
    ).reshape(num_samples * num_steps, *hidden.shape[1:])
    row_mask = mask.unsqueeze(1).expand(
        num_samples,
        num_steps,
        *mask.shape[1:],
    ).reshape(num_samples * num_steps, *mask.shape[1:])
    return row_hidden, row_mask


def require_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"RTFD requires {name}")
    return value
