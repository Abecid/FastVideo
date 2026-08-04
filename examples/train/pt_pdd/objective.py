# SPDX-License-Identifier: Apache-2.0
"""Centered posterior velocity objective for the PT-PDD AnyFlow proxy."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from examples.train.pt_pdd.core import (
    centered_velocity_correction,
    clipped_grpo_loss,
    finite_interval_velocity,
    gaussian_log_prob_mean,
    global_group_advantages,
    posterior_distillation_loss,
    posterior_tilted_regression_loss,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
    verify_group_partition,
)
from examples.train.vptd import train_anyflow as base


def all_reduce_sum(value: torch.Tensor, info: base.DistInfo) -> torch.Tensor:
    reduced = value.clone()
    if info.world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def predict_velocity(
    model: torch.nn.Module,
    latents: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    enable_grad: bool,
) -> torch.Tensor:
    context = torch.enable_grad() if enable_grad else torch.no_grad()
    with context:
        source = base.expand_time(source_time, latents)
        target = base.expand_time(target_time, latents)
        embeds = prompt_embeds.expand(latents.shape[0], -1, -1)
        return model(
            hidden_states=latents,
            timestep=source,
            r_timestep=target,
            encoder_hidden_states=embeds,
            return_dict=False,
            is_causal=False,
        )[0]


def train_one_step(
    runtime: base.AnyFlowRuntime,
    cfg: dict[str, Any],
    info: base.DistInfo,
    optimizer: torch.optim.Optimizer,
    ema: base.LoRAEMA,
    scorer: Any,
    prompt: str,
    step: int,
) -> dict[str, float]:
    policy_cfg = cfg["posterior_policy"]
    local_group = verify_group_partition(
        int(policy_cfg["group_size"]),
        info.world_size,
    )
    model = runtime.transformer
    model.eval()
    embeds = base.encode_prompt(runtime, prompt, cfg, info)
    time_grid = base.schedule(runtime, int(cfg["model"]["train_map_steps"]), info)
    validate_training_schedule(
        time_grid,
        stochastic_steps=int(policy_cfg["stochastic_steps"]),
    )

    if info.is_main:
        generator = torch.Generator(device=info.device).manual_seed(
            int(cfg["experiment"]["seed"]) + int(step)
        )
        selected = int(
            torch.randint(
                0,
                int(policy_cfg["stochastic_steps"]),
                (1,),
                generator=generator,
                device=info.device,
            ).item()
        )
    else:
        selected = 0
    branch_index = base.broadcast_int(selected, info)

    shared_seed = int(cfg["experiment"]["seed"]) + 10_000_000 + int(step)
    shared_state = base.initial_noise(runtime, cfg, info, seed=shared_seed)
    with torch.no_grad():
        for index in range(branch_index):
            shared_state = base.flow_map(
                model,
                shared_state,
                time_grid[index],
                time_grid[index + 1],
                embeds,
                num_train_timesteps=runtime.num_train_timesteps,
                enable_grad=False,
            )
        old_mean, old_std, deterministic_target = base.branch_policy(
            runtime,
            model,
            shared_state,
            embeds,
            time_grid[branch_index],
            time_grid[branch_index + 1],
            enable_grad=False,
        )

    action_chunks: list[torch.Tensor] = []
    log_prob_chunks: list[torch.Tensor] = []
    reward_chunks: list[torch.Tensor] = []
    temporal_chunks: list[torch.Tensor] = []
    for local_index in range(local_group):
        branch_seed = (
            int(cfg["experiment"]["seed"])
            + int(step) * 100_000
            + info.rank * 1_000
            + local_index
        )
        branch_generator = torch.Generator(device=info.device).manual_seed(
            branch_seed
        )
        with torch.no_grad():
            action, _ = sample_diagonal_gaussian(
                old_mean,
                old_std,
                generator=branch_generator,
            )
            old_log_prob = gaussian_log_prob_mean(action, old_mean, old_std)
            endpoint = base.complete_from_action(
                runtime,
                model,
                action,
                embeds,
                time_grid,
                branch_index,
            )
            media = base.decode_latents(runtime, endpoint)
            scores = base.score_video(scorer, media, prompt)
            reward = scores[str(cfg["reward"]["optimize"])].to(
                info.device
            ).reshape(-1)
            action_chunks.append(action.detach())
            log_prob_chunks.append(old_log_prob.detach())
            reward_chunks.append(reward.detach())
            temporal_chunks.append(temporal_l1(media).to(info.device).detach())
            del endpoint, media, scores

    actions = torch.cat(action_chunks, dim=0)
    old_log_probs = torch.cat(log_prob_chunks, dim=0)
    local_rewards = torch.cat(reward_chunks, dim=0)
    local_temporal = torch.cat(temporal_chunks, dim=0)
    global_rewards = base.all_gather_1d(local_rewards, info)
    global_temporal = base.all_gather_1d(local_temporal, info)
    advantages, reward_mean, reward_std = global_group_advantages(
        global_rewards,
        epsilon=float(policy_cfg["advantage_epsilon"]),
        clip=float(policy_cfg["advantage_clip"]),
    )
    global_weights, temperature, ess = reward_tilted_weights(
        global_rewards,
        target_ess_ratio=float(policy_cfg["target_ess_ratio"]),
    )
    local_start = info.rank * local_group
    local_end = local_start + local_group
    local_advantages = advantages[local_start:local_end].to(info.device)
    local_weights = global_weights[local_start:local_end].to(info.device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    objective = str(policy_cfg["objective"])

    if objective in {"posterior_tilted_regression", "reference_regression"}:
        candidate_velocities = finite_interval_velocity(
            shared_state.expand_as(actions),
            actions,
            time_grid[branch_index],
            time_grid[branch_index + 1],
            num_train_timesteps=runtime.num_train_timesteps,
        )
        local_correction, correction_diagnostics = centered_velocity_correction(
            candidate_velocities,
            local_weights,
            global_group_size=int(policy_cfg["group_size"]),
        )
        global_correction = all_reduce_sum(local_correction, info)
        if objective == "reference_regression":
            global_correction = torch.zeros_like(global_correction)

        unwrapped = base.unwrap(model)
        disable_adapter = getattr(unwrapped, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(
                "PT-PDD requires PEFT disable_adapter() for the frozen target"
            )
        with disable_adapter(), torch.no_grad():
            reference_velocity = predict_velocity(
                unwrapped,
                shared_state.detach(),
                time_grid[branch_index],
                time_grid[branch_index + 1],
                embeds.detach(),
                enable_grad=False,
            ).detach()
        student_velocity = predict_velocity(
            model,
            shared_state.detach(),
            time_grid[branch_index],
            time_grid[branch_index + 1],
            embeds.detach(),
            enable_grad=True,
        )
        loss, diagnostics = posterior_tilted_regression_loss(
            student_velocity,
            reference_velocity,
            global_correction,
        )
        diagnostics.update(correction_diagnostics)
    else:
        new_mean, new_std, _ = base.branch_policy(
            runtime,
            model,
            shared_state.detach(),
            embeds.detach(),
            time_grid[branch_index],
            time_grid[branch_index + 1],
            enable_grad=True,
        )
        expanded_mean = new_mean.expand(actions.shape[0], *new_mean.shape[1:])
        expanded_std = new_std.expand(actions.shape[0], *new_std.shape[1:])
        new_log_probs = gaussian_log_prob_mean(
            actions,
            expanded_mean,
            expanded_std,
        )
        if objective == "posterior_distillation":
            loss, diagnostics = posterior_distillation_loss(
                new_log_probs,
                local_weights,
                distributed_world_size=info.world_size,
            )
        else:
            loss, diagnostics = clipped_grpo_loss(
                new_log_probs,
                old_log_probs,
                local_advantages,
                clip_range=float(policy_cfg["clip_range"]),
            )

    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        float(cfg["optimizer"]["max_grad_norm"]),
    )
    optimizer.step()
    ema.update(base.unwrap(model), step)

    action_deviation = (
        actions.float() - deterministic_target.expand_as(actions).float()
    ).square().mean().sqrt()
    safe_weights = global_weights.clamp_min(1.0e-12)
    metrics: dict[str, float] = {
        "train/loss": float(base.reduce_mean(loss, info)),
        "train/grad_norm": float(
            base.reduce_mean(torch.as_tensor(grad_norm, device=info.device), info)
        ),
        "train/reward_mean": float(reward_mean),
        "train/reward_std": float(reward_std),
        "train/reward_min": float(global_rewards.min()),
        "train/reward_max": float(global_rewards.max()),
        "train/advantage_abs_mean": float(advantages.abs().mean()),
        "train/zero_std_group": float(
            reward_std < float(policy_cfg["advantage_epsilon"])
        ),
        "train/transition_index": float(branch_index),
        "train/source_timestep": float(time_grid[branch_index]),
        "train/target_timestep": float(time_grid[branch_index + 1]),
        "train/posterior_std": float(old_std.float().mean()),
        "train/posterior_action_deviation": float(
            base.reduce_mean(action_deviation, info)
        ),
        "train/temporal_delta_l1": float(global_temporal.mean()),
        "train/group_size": float(global_rewards.numel()),
        "train/reward_temperature": (
            float(temperature) if torch.isfinite(temperature) else 0.0
        ),
        "train/reward_temperature_is_infinite": float(
            not bool(torch.isfinite(temperature))
        ),
        "train/posterior_ess": float(ess),
        "train/posterior_ess_ratio": float(ess / global_rewards.numel()),
        "train/posterior_weight_max": float(global_weights.max()),
        "train/posterior_weight_entropy": float(
            -(safe_weights * safe_weights.log()).sum()
        ),
        "train/objective_is_posterior_tilted_regression": float(
            objective == "posterior_tilted_regression"
        ),
        "train/objective_is_reference_regression": float(
            objective == "reference_regression"
        ),
        "train/objective_is_posterior_distillation": float(
            objective == "posterior_distillation"
        ),
        "train/objective_is_flowmap_grpo": float(objective == "flowmap_grpo"),
    }
    for name, value in diagnostics.items():
        metrics[f"train/{name}"] = float(base.reduce_mean(value, info))

    del (
        actions,
        old_log_probs,
        local_rewards,
        global_rewards,
        shared_state,
        embeds,
    )
    return metrics
