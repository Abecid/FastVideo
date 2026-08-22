# SPDX-License-Identifier: Apache-2.0
"""Statistically reliable finite-transition RL for AnyFlow video models.

This follow-up fixes the shared substrate exposed by the first 200-step run:
multiple rollout groups per optimizer update, all stochastic transitions in the
trajectory, running reward statistics, global-scale posterior temperature,
target-KL update calibration, paired raw/EMA validation, strict shared-behavior
rollouts when requested, and an optional direct finite-velocity objective.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import contextlib
import json
import time
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common.finite_transition import (
    clipped_grpo_loss,
    gaussian_log_prob_mean,
    group_advantages,
    posterior_projection_loss,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
)
from fastvideo.train.methods.rl.common.reward_statistics import (
    PromptRewardNormalizer,
    TargetKLController,
    reward_softmax_weights,
)
from fastvideo.train.methods.rl.finite_transition_paired_validation import (
    PairedFiniteTransitionValidationMixin,
)
from fastvideo.train.methods.rl.finite_transition_posterior_repro import (
    ReproducibleFiniteTransitionPosteriorMethod,
)

_STATE_BUFFER_BYTES = 2 * 1024 * 1024


class _ReliableFiniteTransitionState:
    """Fixed-shape DCP wrapper for normalizer/controller/evaluation state."""

    def __init__(self, method: "ReliableFiniteTransitionMethod") -> None:
        self._method = method

    def state_dict(self) -> dict[str, torch.Tensor]:
        payload = json.dumps(
            {
                "reward_normalizer": self._method._reward_normalizer.state_dict(),
                "kl_controller": self._method._kl_controller.state_dict(),
                "paired_validation_baselines": getattr(
                    self._method,
                    "_paired_validation_baselines",
                    {},
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > _STATE_BUFFER_BYTES:
            raise RuntimeError(
                "Reliable finite-transition state exceeds fixed DCP buffer "
                f"({len(payload)} > {_STATE_BUFFER_BYTES})"
            )
        buffer = torch.zeros(_STATE_BUFFER_BYTES, dtype=torch.uint8)
        if payload:
            buffer[: len(payload)] = torch.tensor(
                list(payload),
                dtype=torch.uint8,
            )
        return {
            "json": buffer,
            "length": torch.tensor(len(payload), dtype=torch.long),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        length = state_dict.get("length", 0)
        if torch.is_tensor(length):
            length = int(length.item())
        length = int(length)
        if length < 0 or length > _STATE_BUFFER_BYTES:
            raise RuntimeError(f"Invalid reliable-state length: {length}")
        buffer = state_dict.get("json")
        if buffer is None:
            return
        if not torch.is_tensor(buffer):
            raise TypeError("reliable-state json must be a tensor")
        payload = bytes(
            buffer.detach().cpu().to(torch.uint8)[:length].tolist()
        )
        decoded = json.loads(payload.decode("utf-8")) if payload else {}
        self._method._reward_normalizer.load_state_dict(
            dict(decoded.get("reward_normalizer", {}))
        )
        self._method._kl_controller.load_state_dict(
            dict(decoded.get("kl_controller", {}))
        )
        paired = decoded.get("paired_validation_baselines", {})
        if isinstance(paired, dict):
            self._method._paired_validation_baselines = {
                str(mode): {
                    str(key): [float(item) for item in values]
                    for key, values in dict(mode_values).items()
                }
                for mode, mode_values in paired.items()
            }


class ReliableFiniteTransitionMethod(
    PairedFiniteTransitionValidationMixin,
    ReproducibleFiniteTransitionPosteriorMethod,
):
    """Full-trajectory GRPO/FTPP and finite-velocity alignment."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        requested_objective = str(
            cfg.method.get("objective", "flowmap_grpo")
        ).strip().lower()
        allowed = {
            "flowmap_grpo",
            "posterior_projection",
            "finite_velocity_regression",
        }
        if requested_objective not in allowed:
            raise ValueError(
                f"method.objective must be one of {sorted(allowed)}"
            )
        # The parent validates only its original two likelihood objectives.
        if requested_objective == "finite_velocity_regression":
            cfg.method["objective"] = "posterior_projection"
        super().__init__(cfg=cfg, role_models=role_models)
        cfg.method["objective"] = requested_objective
        self._objective = requested_objective

        mcfg = self.method_config
        self._rollout_mode = str(
            mcfg.get("rollout_mode", "full_trajectory")
        ).strip().lower()
        if self._rollout_mode not in {
            "full_trajectory",
            "single_transition",
        }:
            raise ValueError(
                "method.rollout_mode must be full_trajectory or "
                "single_transition"
            )
        if (
            self._objective == "finite_velocity_regression"
            and self._rollout_mode != "single_transition"
        ):
            raise ValueError(
                "finite_velocity_regression requires single_transition rollouts"
            )

        self._rollout_groups_per_update = int(
            mcfg.get("rollout_groups_per_update", 2)
        )
        if self._rollout_groups_per_update <= 0:
            raise ValueError(
                "method.rollout_groups_per_update must be positive"
            )
        self._transition_loss_reduction = str(
            mcfg.get("transition_loss_reduction", "mean")
        ).strip().lower()
        if self._transition_loss_reduction not in {"mean", "sum"}:
            raise ValueError(
                "method.transition_loss_reduction must be mean or sum"
            )

        self._reward_normalization = str(
            mcfg.get("reward_normalization", "running_prompt_global")
        ).strip().lower()
        if self._reward_normalization not in {
            "group",
            "running_prompt_global",
        }:
            raise ValueError(
                "method.reward_normalization must be group or "
                "running_prompt_global"
            )
        self._posterior_temperature_mode = str(
            mcfg.get("posterior_temperature_mode", "global_std")
        ).strip().lower()
        if self._posterior_temperature_mode not in {
            "fixed_ess",
            "global_std",
        }:
            raise ValueError(
                "method.posterior_temperature_mode must be fixed_ess or "
                "global_std"
            )
        self._posterior_temperature_scale = float(
            mcfg.get("posterior_temperature_scale", 1.0)
        )
        if self._posterior_temperature_scale <= 0.0:
            raise ValueError(
                "method.posterior_temperature_scale must be positive"
            )

        normalizer_cfg = mcfg.get("running_reward_stats", {}) or {}
        if not isinstance(normalizer_cfg, dict):
            raise ValueError("method.running_reward_stats must be a mapping")
        self._reward_normalizer = PromptRewardNormalizer(
            min_prompt_count=int(
                normalizer_cfg.get("min_prompt_count", 4)
            ),
            min_global_count=int(
                normalizer_cfg.get("min_global_count", 32)
            ),
            epsilon=float(
                normalizer_cfg.get("epsilon", self._advantage_epsilon)
            ),
            clip=float(
                normalizer_cfg.get("clip", self._advantage_clip)
            ),
        )

        controller_cfg = mcfg.get("target_kl_controller", {}) or {}
        if not isinstance(controller_cfg, dict):
            raise ValueError(
                "method.target_kl_controller must be a mapping"
            )
        self._kl_controller_enabled = bool(
            controller_cfg.get("enabled", True)
        )
        self._kl_controller = TargetKLController(
            target_kl=float(controller_cfg.get("target_kl", 1.0e-5)),
            initial_scale=float(
                controller_cfg.get("initial_loss_scale", 1.0)
            ),
            min_scale=float(
                controller_cfg.get("min_loss_scale", 0.05)
            ),
            max_scale=float(
                controller_cfg.get("max_loss_scale", 128.0)
            ),
            max_adjustment=float(
                controller_cfg.get("max_adjustment", 2.0)
            ),
        )

        self._behavior_policy = str(
            mcfg.get("behavior_policy", "current")
        ).strip().lower()
        if self._behavior_policy not in {
            "current",
            "base_adapter_disabled",
        }:
            raise ValueError(
                "method.behavior_policy must be current or "
                "base_adapter_disabled"
            )
        self._finite_velocity_target_rms = float(
            mcfg.get("finite_velocity_target_rms", 0.002)
        )
        self._finite_velocity_max_eta = float(
            mcfg.get("finite_velocity_max_eta", 4.0)
        )
        if (
            self._finite_velocity_target_rms <= 0.0
            or self._finite_velocity_max_eta <= 0.0
        ):
            raise ValueError(
                "finite-velocity target controls must be positive"
            )
        self._transfer_probe_every = max(
            0,
            int(mcfg.get("transfer_probe_every", 10)),
        )

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["finite_transition_reliable.state"] = (
            _ReliableFiniteTransitionState(self)
        )
        return states

    @contextlib.contextmanager
    def _behavior_context(self) -> Iterator[None]:
        if self._behavior_policy == "current":
            yield
            return

        transformer = self.student.transformer
        candidates = [transformer, getattr(transformer, "module", None)]
        candidates.extend(list(transformer.modules()))
        seen: set[int] = set()
        for module in candidates:
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            disable = getattr(module, "disable_adapter", None)
            if callable(disable):
                with disable():
                    yield
                return
        raise RuntimeError(
            "behavior_policy=base_adapter_disabled requires a PEFT module "
            "with disable_adapter()"
        )

    def _reward_coefficients(
        self,
        rewards: torch.Tensor,
        *,
        prompt: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
    ]:
        if self._reward_normalization == "running_prompt_global":
            if self._posterior_temperature_mode == "global_std":
                temperature = self._reward_normalizer.temperature(
                    rewards,
                    scale_multiplier=self._posterior_temperature_scale,
                )
            else:
                temperature = rewards.new_tensor(float("nan"))
            advantages, diagnostics = self._reward_normalizer.normalize(
                rewards,
                prompt=prompt,
                update=True,
            )
        else:
            advantages, mean, std = group_advantages(
                rewards,
                epsilon=self._advantage_epsilon,
                clip=self._advantage_clip,
            )
            diagnostics = {
                "baseline": float(mean),
                "scale": float(std),
                "group_mean": float(mean),
                "group_std": float(std),
                "baseline_source": -1.0,
                "scale_source": -1.0,
                "prompt_count_before": 0.0,
                "global_count_before": 0.0,
            }
            temperature = rewards.new_tensor(float("nan"))

        if self._posterior_temperature_mode == "fixed_ess":
            weights, temperature, ess = reward_tilted_weights(
                rewards,
                target_ess_ratio=self._target_ess_ratio,
                bisection_steps=self._reward_bisection_steps,
            )
        else:
            weights, ess = reward_softmax_weights(
                rewards,
                temperature=temperature,
            )
        return advantages, weights, ess, {
            **diagnostics,
            "temperature": float(temperature),
        }

    def _sample_rollout_group(
        self,
        *,
        rollout_iteration: int,
        local_branches: int,
        rank: int,
    ) -> dict[str, Any]:
        raw_batch = self._sample_prompt_batch(
            iteration=rollout_iteration,
            local_branches=local_branches,
        )
        prompts = self._extract_prompts(raw_batch)
        if len(prompts) != local_branches or len(set(prompts)) != 1:
            raise RuntimeError(
                "every reliable rollout group must contain one repeated prompt"
            )
        batch = self.student.prepare_batch(
            raw_batch,
            generator=self.cuda_generator,
            latents_source="zeros",
        )
        schedule = self._build_schedule(
            steps=self._train_map_steps,
            override=self._train_schedule_override,
            device=self.student.device,
        )
        validate_training_schedule(
            schedule,
            stochastic_steps=self._stochastic_steps,
        )
        initial = self._shared_initial_noise(
            batch,
            iteration=rollout_iteration,
            local_branches=local_branches,
        )

        transitions: list[dict[str, Any]] = []
        current = initial
        selected_index = -1
        if self._rollout_mode == "single_transition":
            selected_index = self._select_branch_index(
                iteration=rollout_iteration,
                device=self.student.device,
            )

        with self._behavior_context(), torch.no_grad():
            for index in range(self._stochastic_steps):
                if (
                    self._rollout_mode == "single_transition"
                    and index != selected_index
                ):
                    current, _ = self._flow_map(
                        current,
                        schedule[index],
                        schedule[index + 1],
                        batch,
                    )
                    continue

                state = current.detach()
                mean, std, deterministic_target = self._branch_policy(
                    state,
                    schedule[index],
                    schedule[index + 1],
                    batch,
                )
                generator = torch.Generator(
                    device=self.student.device
                ).manual_seed(
                    int(self.training_config.data.seed)
                    + rollout_iteration * 1_000_003
                    + rank * 10_009
                    + index * 101
                )
                action, _ = sample_diagonal_gaussian(
                    mean,
                    std,
                    generator=generator,
                )
                old_log_prob = gaussian_log_prob_mean(
                    action,
                    mean,
                    std,
                )
                old_velocity = None
                if self._objective == "finite_velocity_regression":
                    _, old_velocity = self._flow_map(
                        state,
                        schedule[index],
                        schedule[index + 1],
                        batch,
                    )
                transitions.append(
                    {
                        "index": index,
                        "state": state,
                        "action": action.detach(),
                        "old_mean": mean.detach(),
                        "old_std": std.detach(),
                        "old_log_prob": old_log_prob.detach(),
                        "old_velocity": (
                            None
                            if old_velocity is None
                            else old_velocity.detach()
                        ),
                        "deterministic_target": deterministic_target.detach(),
                        "source_time": schedule[index].detach(),
                        "target_time": schedule[index + 1].detach(),
                    }
                )
                current = action

                if self._rollout_mode == "single_transition":
                    for suffix_index in range(
                        index + 1,
                        schedule.numel() - 1,
                    ):
                        current, _ = self._flow_map(
                            current,
                            schedule[suffix_index],
                            schedule[suffix_index + 1],
                            batch,
                        )
                    break

            if self._rollout_mode == "full_trajectory":
                for index in range(
                    self._stochastic_steps,
                    schedule.numel() - 1,
                ):
                    current, _ = self._flow_map(
                        current,
                        schedule[index],
                        schedule[index + 1],
                        batch,
                    )

            media = self.student.decode_latents(current).detach().cpu()
            local_rewards = self._score_media(media, prompts)
            local_temporal = temporal_l1(media).to(self.student.device)

        if not transitions:
            raise RuntimeError("rollout produced no stochastic transitions")
        return {
            "batch": batch,
            "schedule": schedule,
            "prompts": prompts,
            "transitions": transitions,
            "endpoints": current.detach(),
            "media": media,
            "local_rewards": local_rewards,
            "local_temporal": local_temporal,
        }

    def _new_transition_log_prob(
        self,
        transition: dict[str, Any],
        batch: TrainingBatch,
    ) -> torch.Tensor:
        mean, std, _ = self._branch_policy(
            transition["state"].detach(),
            transition["source_time"],
            transition["target_time"],
            batch,
        )
        return gaussian_log_prob_mean(
            transition["action"],
            mean,
            std,
        )

    def _finite_velocity_loss(
        self,
        rollout: dict[str, Any],
        *,
        local_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        transition = rollout["transitions"][0]
        state = transition["state"].detach()
        action = transition["action"].detach()
        source = transition["source_time"]
        target = transition["target_time"]
        delta = (source - target) / float(
            self.student.num_train_timesteps
        )
        if float(delta) <= 0.0:
            raise RuntimeError("finite transition delta must be positive")

        coefficients = local_weights.detach() - (
            1.0 / float(self._group_size)
        )
        view = [coefficients.shape[0]] + [1] * (action.ndim - 1)
        finite_velocity = (state - action) / delta.to(action.dtype)
        local_correction = (
            coefficients.view(*view).to(finite_velocity.dtype)
            * finite_velocity
        ).sum(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_correction, op=dist.ReduceOp.SUM)

        old_velocity = transition["old_velocity"]
        if old_velocity is None:
            raise RuntimeError("finite-velocity rollout is missing its reference")
        old_velocity = old_velocity[:1]
        transition_shift = delta.to(local_correction.dtype) * local_correction
        base_rms = transition_shift.float().square().mean().sqrt()
        eta = min(
            self._finite_velocity_max_eta,
            self._finite_velocity_target_rms
            / max(float(base_rms), 1.0e-12),
        )
        target_velocity = (
            old_velocity + float(eta) * local_correction
        ).detach()

        batch: TrainingBatch = rollout["batch"]
        _, new_velocity = self._flow_map(
            state,
            source,
            target,
            batch,
        )
        loss = (
            new_velocity.float()
            - target_velocity.expand_as(new_velocity).float()
        ).square().mean()
        return loss, {
            "finite_velocity_eta": loss.new_tensor(eta),
            "finite_velocity_correction_rms": local_correction.float()
            .square()
            .mean()
            .sqrt()
            .detach(),
            "finite_transition_target_shift_rms": (
                float(eta) * transition_shift.float()
            )
            .square()
            .mean()
            .sqrt()
            .detach(),
        }

    def _backward_likelihood_transitions(
        self,
        rollout: dict[str, Any],
        *,
        local_weights: torch.Tensor,
        local_advantages: torch.Tensor,
        loss_scale: float,
    ) -> tuple[list[torch.Tensor], dict[str, list[float]]]:
        losses: list[torch.Tensor] = []
        diagnostics: dict[str, list[float]] = defaultdict(list)
        transition_count = len(rollout["transitions"])
        if self._transition_loss_reduction == "mean":
            denominator = self._rollout_groups_per_update * transition_count
        else:
            denominator = self._rollout_groups_per_update

        batch: TrainingBatch = rollout["batch"]
        for transition in rollout["transitions"]:
            new_log_prob = self._new_transition_log_prob(transition, batch)
            if self._objective == "posterior_projection":
                raw_loss, transition_diagnostics = posterior_projection_loss(
                    new_log_prob,
                    local_weights,
                    global_group_size=self._group_size,
                    distributed_world_size=self._world_size(),
                )
            else:
                raw_loss, transition_diagnostics = clipped_grpo_loss(
                    new_log_prob,
                    transition["old_log_prob"],
                    local_advantages,
                    clip_range=self._clip_range,
                )
            self.student.backward(
                raw_loss * float(loss_scale),
                (batch.timesteps, batch.attn_metadata),
                grad_accum_rounds=denominator,
            )
            losses.append(raw_loss.detach())
            for name, value in transition_diagnostics.items():
                diagnostics[name].append(float(value))
        return losses, diagnostics

    @torch.no_grad()
    def _post_update_policy_probe(
        self,
        rollout: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deltas = []
        batch: TrainingBatch = rollout["batch"]
        for transition in rollout["transitions"]:
            post_log_prob = self._new_transition_log_prob(transition, batch)
            deltas.append(post_log_prob - transition["old_log_prob"])
        stacked = torch.stack(deltas, dim=0)
        return (
            0.5 * stacked.square().mean(),
            stacked.abs().mean(),
        )

    @torch.no_grad()
    def _deterministic_transfer_probe(
        self,
        rollout: dict[str, Any],
        *,
        global_weights: torch.Tensor,
        local_branches: int,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rollout_mode != "single_transition":
            zero = torch.zeros((), device=self.student.device)
            return zero, zero
        transition = rollout["transitions"][0]
        new_target, _ = self._flow_map(
            transition["state"],
            transition["source_time"],
            transition["target_time"],
            rollout["batch"],
        )
        deterministic_shift = (
            new_target[:1]
            - transition["deterministic_target"][:1]
        ).float()
        local_start = rank * local_branches
        local_end = local_start + local_branches
        coefficients = (
            global_weights[local_start:local_end].to(self.student.device)
            - 1.0 / float(self._group_size)
        )
        action = transition["action"].float()
        view = [coefficients.shape[0]] + [1] * (action.ndim - 1)
        preferred = (coefficients.view(*view) * action).sum(
            dim=0,
            keepdim=True,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(preferred, op=dist.ReduceOp.SUM)
        cosine = torch.nn.functional.cosine_similarity(
            deterministic_shift.flatten(),
            preferred.flatten(),
            dim=0,
            eps=1.0e-12,
        )
        return cosine, deterministic_shift.square().mean().sqrt()

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, Any],
        dict[str, LogScalar],
    ]:
        del data_stream
        if self._reward_scorer is None or self.cuda_generator is None:
            raise RuntimeError("method was not initialized with on_train_start")

        started = time.perf_counter()
        rank = self._rank()
        world_size = self._world_size()
        if self._group_size % world_size != 0:
            raise ValueError("group_size must be divisible by world size")
        local_branches = self._group_size // world_size
        self.student.transformer.eval()
        self._student_optimizer.zero_grad(set_to_none=True)

        metric_lists: dict[str, list[float]] = defaultdict(list)
        loss_values: list[torch.Tensor] = []
        last_rollout: dict[str, Any] | None = None
        last_global_weights: torch.Tensor | None = None
        loss_scale_used = float(self._kl_controller.scale)

        for group_index in range(self._rollout_groups_per_update):
            rollout_iteration = (
                int(iteration) * self._rollout_groups_per_update
                + group_index
            )
            rollout = self._sample_rollout_group(
                rollout_iteration=rollout_iteration,
                local_branches=local_branches,
                rank=rank,
            )
            global_reward_components = {
                key: self._all_gather_1d(value)
                for key, value in rollout["local_rewards"].items()
            }
            if self._optimize_reward not in global_reward_components:
                raise RuntimeError(
                    f"reward scorer did not return {self._optimize_reward!r}"
                )
            global_rewards = global_reward_components[self._optimize_reward]
            prompt = str(rollout["prompts"][0])
            advantages, weights, ess, reward_diag = (
                self._reward_coefficients(global_rewards, prompt=prompt)
            )
            local_start = rank * local_branches
            local_end = local_start + local_branches
            local_weights = weights[local_start:local_end].to(
                self.student.device
            )
            local_advantages = advantages[local_start:local_end].to(
                self.student.device
            )

            self.student.transformer.train()
            if self._objective == "finite_velocity_regression":
                raw_loss, transition_diagnostics = self._finite_velocity_loss(
                    rollout,
                    local_weights=local_weights,
                )
                self.student.backward(
                    raw_loss * loss_scale_used,
                    (
                        rollout["batch"].timesteps,
                        rollout["batch"].attn_metadata,
                    ),
                    grad_accum_rounds=self._rollout_groups_per_update,
                )
                group_losses = [raw_loss.detach()]
                diagnostics = {
                    name: [float(value)]
                    for name, value in transition_diagnostics.items()
                }
            else:
                group_losses, diagnostics = (
                    self._backward_likelihood_transitions(
                        rollout,
                        local_weights=local_weights,
                        local_advantages=local_advantages,
                        loss_scale=loss_scale_used,
                    )
                )
            loss_values.extend(group_losses)

            safe_weights = weights.clamp_min(1.0e-12)
            metric_lists["reward_mean"].append(float(global_rewards.mean()))
            metric_lists["reward_std"].append(
                float(global_rewards.std(unbiased=False))
            )
            metric_lists["reward_selection_gain"].append(
                float((weights * global_rewards).sum() - global_rewards.mean())
            )
            metric_lists["posterior_ess"].append(float(ess))
            metric_lists["posterior_weight_max"].append(float(weights.max()))
            metric_lists["posterior_entropy"].append(
                float(-(safe_weights * safe_weights.log()).sum())
            )
            metric_lists["temperature"].append(
                float(reward_diag["temperature"])
            )
            metric_lists["normalizer_baseline"].append(
                float(reward_diag["baseline"])
            )
            metric_lists["normalizer_scale"].append(
                float(reward_diag["scale"])
            )
            metric_lists["normalizer_baseline_source"].append(
                float(reward_diag["baseline_source"])
            )
            metric_lists["normalizer_scale_source"].append(
                float(reward_diag["scale_source"])
            )
            metric_lists["temporal_l1"].append(
                float(
                    self._all_gather_1d(
                        rollout["local_temporal"]
                    ).mean()
                )
            )
            metric_lists["transition_count"].append(
                float(len(rollout["transitions"]))
            )
            for name, values in diagnostics.items():
                metric_lists[f"lossdiag/{name}"].extend(values)
            for name, values in global_reward_components.items():
                metric_lists[f"reward/{name}"].append(float(values.mean()))
                metric_lists[f"reward_std/{name}"].append(
                    float(values.std(unbiased=False))
                )

            last_rollout = rollout
            last_global_weights = weights.detach()
            self.student.transformer.eval()

        grad_norm = self._clip_student_grads()
        self._student_optimizer.step()
        self._student_lr_scheduler.step()
        self._update_ema()
        self._student_optimizer.zero_grad(set_to_none=True)

        post_update_kl = torch.zeros((), device=self.student.device)
        post_update_logprob_delta = torch.zeros(
            (),
            device=self.student.device,
        )
        transfer_cosine = torch.zeros((), device=self.student.device)
        transfer_shift_rms = torch.zeros((), device=self.student.device)
        if last_rollout is not None and last_global_weights is not None:
            self.student.transformer.eval()
            with torch.no_grad():
                post_update_kl, post_update_logprob_delta = (
                    self._post_update_policy_probe(last_rollout)
                )
                if (
                    self._transfer_probe_every > 0
                    and iteration % self._transfer_probe_every == 0
                ):
                    transfer_cosine, transfer_shift_rms = (
                        self._deterministic_transfer_probe(
                            last_rollout,
                            global_weights=last_global_weights,
                            local_branches=local_branches,
                            rank=rank,
                        )
                    )

        measured_kl = self._mean_across_ranks(post_update_kl)
        if self._kl_controller_enabled:
            self._kl_controller.update(measured_kl)

        step_seconds = time.perf_counter() - started
        self._cumulative_train_seconds += step_seconds

        def mean_metric(name: str, default: float = 0.0) -> float:
            values = metric_lists.get(name, [])
            return sum(values) / len(values) if values else default

        metrics: dict[str, LogScalar] = {
            "reliable/objective_is_posterior_projection": float(
                self._objective == "posterior_projection"
            ),
            "reliable/objective_is_grpo": float(
                self._objective == "flowmap_grpo"
            ),
            "reliable/objective_is_finite_velocity": float(
                self._objective == "finite_velocity_regression"
            ),
            "reliable/rollout_is_full_trajectory": float(
                self._rollout_mode == "full_trajectory"
            ),
            "reliable/behavior_is_fixed_base": float(
                self._behavior_policy == "base_adapter_disabled"
            ),
            "reliable/rollout_groups_per_update": float(
                self._rollout_groups_per_update
            ),
            "reliable/reward_samples_per_update": float(
                self._rollout_groups_per_update * self._group_size
            ),
            "reliable/stochastic_transitions_per_trajectory": mean_metric(
                "transition_count"
            ),
            "reliable/reward_mean": mean_metric("reward_mean"),
            "reliable/reward_std": mean_metric("reward_std"),
            "reliable/reward_selection_gain": mean_metric(
                "reward_selection_gain"
            ),
            "reliable/posterior_ess": mean_metric("posterior_ess"),
            "reliable/posterior_weight_max": mean_metric(
                "posterior_weight_max"
            ),
            "reliable/posterior_entropy": mean_metric(
                "posterior_entropy"
            ),
            "reliable/posterior_temperature": mean_metric("temperature"),
            "reliable/reward_baseline": mean_metric(
                "normalizer_baseline"
            ),
            "reliable/reward_scale": mean_metric("normalizer_scale"),
            "reliable/reward_baseline_source": mean_metric(
                "normalizer_baseline_source"
            ),
            "reliable/reward_scale_source": mean_metric(
                "normalizer_scale_source"
            ),
            "reliable/temporal_l1": mean_metric("temporal_l1"),
            "reliable/loss_scale_used": loss_scale_used,
            "reliable/loss_scale_next": float(self._kl_controller.scale),
            "reliable/target_kl": float(self._kl_controller.target_kl),
            "reliable/post_update_approx_kl": measured_kl,
            "reliable/post_update_logprob_delta_abs": (
                self._mean_across_ranks(post_update_logprob_delta)
            ),
            "reliable/deterministic_transfer_cosine": (
                self._mean_across_ranks(transfer_cosine)
            ),
            "reliable/deterministic_shift_rms": self._mean_across_ranks(
                transfer_shift_rms
            ),
            "reliable/grad_norm": self._mean_across_ranks(
                torch.as_tensor(grad_norm, device=self.student.device)
            ),
            "reliable/train_step_seconds": step_seconds,
            "reliable/cumulative_gpu_hours": (
                self._cumulative_train_seconds * world_size / 3600.0
            ),
        }
        for name, values in metric_lists.items():
            if name.startswith(("reward/", "reward_std/", "lossdiag/")):
                metrics[f"reliable/{name}"] = sum(values) / len(values)

        mean_loss = (
            torch.stack(loss_values).mean()
            if loss_values
            else torch.zeros((), device=self.student.device)
        )
        return {"total_loss": mean_loss}, {}, metrics


__all__ = ["ReliableFiniteTransitionMethod"]
