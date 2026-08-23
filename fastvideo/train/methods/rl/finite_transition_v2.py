# SPDX-License-Identifier: Apache-2.0
"""Finite-transition RL v2 for AnyFlow-Wan.

This method is the follow-up to the first 200-step FTPP/GRPO experiment.  The
first study used one prompt, four candidates, one selected transition and one
optimizer step at learning rate 2e-6.  Both objectives produced post-update KL
near 1e-7 and no held-out reward gain.  The v2 implementation changes the shared
training substrate before asking whether one update rule is better:

* accumulate several prompt groups before one optimizer step;
* optionally use every stochastic transition in the rollout;
* normalize rewards with a running prompt mean and a global rollout std;
* optionally use a global/EMA reward temperature instead of forced per-group ESS;
* adapt update scale toward an explicit post-update KL target;
* support a frozen-base behavior policy for literal shared-rollout comparisons;
* evaluate raw and EMA weights separately; and
* expose a finite-velocity posterior-regression objective that directly changes
  the deterministic AnyFlow transition used at inference.

The old ``posterior_projection`` and ``flowmap_grpo`` modes remain available as
matched score-function objectives.  The new objective is selected through
``method.v2_objective`` so the parent class can retain its compatibility checks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common.finite_transition import (
    clipped_grpo_loss,
    diagonal_gaussian_kl_mean,
    gaussian_log_prob_mean,
    posterior_projection_loss,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
)
from fastvideo.train.methods.rl.common.finite_transition_v2 import (
    PromptRewardTracker,
    cosine_similarity_flat,
    global_temperature_weights,
    rms,
    running_baseline_advantages,
    stable_global_reward_std,
    update_target_kl_scale,
)
from fastvideo.train.methods.rl.finite_transition_posterior_repro import (
    ReproducibleFiniteTransitionPosteriorMethod,
)
from fastvideo.train.methods.rl.rewards.videoalign_audit import (
    audit_videoalign_checkpoint,
    repeatability_probe,
    write_audit_report,
)

_STATE_BUFFER_BYTES = 2 * 1024 * 1024


@dataclass
class _TransitionRecord:
    state: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    source_time: torch.Tensor
    target_time: torch.Tensor
    deterministic_target: torch.Tensor
    batch: TrainingBatch


@dataclass
class _RolloutGroup:
    prompt: str
    prompts: list[str]
    transitions: list[_TransitionRecord]
    global_rewards: torch.Tensor
    global_reward_components: dict[str, torch.Tensor]
    global_temporal: torch.Tensor
    local_start: int
    local_end: int


class _FiniteTransitionV2State:
    """Fixed-shape DCP state for online statistics and KL control."""

    def __init__(self, method: "FiniteTransitionV2Method") -> None:
        self._method = method

    def _payload(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = {
            "prompt_tracker": self._method._prompt_tracker.state_dict(),
            "reward_std_ema": self._method._reward_std_ema,
            "loss_scale": self._method._loss_scale,
            "raw_validation_state": self._method._raw_validation_state,
            "ema_validation_state": self._method._ema_validation_state,
        }
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > _STATE_BUFFER_BYTES:
            raise RuntimeError(
                "FiniteTransitionV2 state exceeds fixed DCP buffer: "
                f"{len(payload)} > {_STATE_BUFFER_BYTES} bytes"
            )
        buffer = torch.zeros(_STATE_BUFFER_BYTES, dtype=torch.uint8)
        if payload:
            buffer[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
        return buffer, torch.tensor(len(payload), dtype=torch.long)

    def state_dict(self) -> dict[str, torch.Tensor]:
        payload, length = self._payload()
        return {"json": payload, "length": length}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        buffer = state_dict.get("json")
        length = state_dict.get("length", 0)
        if torch.is_tensor(length):
            length = int(length.item())
        if not torch.is_tensor(buffer):
            return
        payload = bytes(buffer.detach().cpu().to(torch.uint8)[: int(length)].tolist())
        raw = json.loads(payload.decode("utf-8")) if payload else {}
        tracker = raw.get("prompt_tracker", {})
        if isinstance(tracker, dict):
            self._method._prompt_tracker = PromptRewardTracker.from_state_dict(tracker)
        self._method._reward_std_ema = float(raw.get("reward_std_ema", 1.0))
        self._method._loss_scale = float(raw.get("loss_scale", 1.0))
        raw_state = raw.get("raw_validation_state", {})
        ema_state = raw.get("ema_validation_state", {})
        if isinstance(raw_state, dict):
            self._method._raw_validation_state = raw_state
        if isinstance(ema_state, dict):
            self._method._ema_validation_state = ema_state


class FiniteTransitionV2Method(ReproducibleFiniteTransitionPosteriorMethod):
    """Scaled Flow-Map GRPO, posterior fitting, and velocity regression."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config
        self._v2_objective = str(
            mcfg.get("v2_objective", self._objective) or self._objective
        ).strip().lower()
        if self._v2_objective not in {
            "flowmap_grpo",
            "posterior_projection",
            "finite_velocity_regression",
        }:
            raise ValueError(
                "method.v2_objective must be flowmap_grpo, "
                "posterior_projection, or finite_velocity_regression"
            )

        self._transition_mode = str(
            mcfg.get("transition_mode", "all") or "all"
        ).strip().lower()
        if self._transition_mode not in {"single", "all"}:
            raise ValueError("method.transition_mode must be single or all")
        if (
            self._v2_objective == "finite_velocity_regression"
            and self._transition_mode != "single"
        ):
            raise ValueError(
                "finite_velocity_regression currently requires shared-state "
                "single-transition branching"
            )

        self._rollout_groups_per_update = self._read_positive_int(
            "rollout_groups_per_update",
            4,
        )
        self._behavior_policy = str(
            mcfg.get("behavior_policy", "on_policy") or "on_policy"
        ).strip().lower()
        if self._behavior_policy not in {"on_policy", "frozen_base"}:
            raise ValueError("method.behavior_policy must be on_policy or frozen_base")

        self._reward_std_decay = float(mcfg.get("reward_std_decay", 0.95))
        self._reward_std_floor = float(mcfg.get("reward_std_floor", 0.05))
        if not 0.0 <= self._reward_std_decay < 1.0:
            raise ValueError("method.reward_std_decay must lie in [0, 1)")
        if self._reward_std_floor <= 0.0:
            raise ValueError("method.reward_std_floor must be positive")
        self._reward_std_ema = float(
            mcfg.get("initial_reward_std", self._reward_std_floor)
        )
        self._prompt_tracker = PromptRewardTracker()

        self._posterior_temperature_mode = str(
            mcfg.get("posterior_temperature_mode", "global_std") or "global_std"
        ).strip().lower()
        if self._posterior_temperature_mode not in {"global_std", "group_ess"}:
            raise ValueError(
                "method.posterior_temperature_mode must be global_std or group_ess"
            )
        self._posterior_temperature_multiplier = float(
            mcfg.get("posterior_temperature_multiplier", 1.0)
        )
        self._posterior_min_temperature = float(
            mcfg.get("posterior_min_temperature", 1.0e-3)
        )
        if self._posterior_temperature_multiplier <= 0.0:
            raise ValueError("posterior_temperature_multiplier must be positive")
        if self._posterior_min_temperature <= 0.0:
            raise ValueError("posterior_min_temperature must be positive")

        self._target_post_update_kl = float(
            mcfg.get("target_post_update_kl", 3.0e-5)
        )
        self._kl_controller_rate = float(mcfg.get("kl_controller_rate", 0.25))
        self._loss_scale = float(mcfg.get("initial_loss_scale", 1.0))
        self._minimum_loss_scale = float(mcfg.get("minimum_loss_scale", 0.1))
        self._maximum_loss_scale = float(mcfg.get("maximum_loss_scale", 100.0))
        if self._loss_scale <= 0.0:
            raise ValueError("initial_loss_scale must be positive")

        self._velocity_correction_scale = float(
            mcfg.get("velocity_correction_scale", 1.0)
        )
        self._velocity_target_state_shift_rms = float(
            mcfg.get("velocity_target_state_shift_rms", 0.01)
        )
        if self._velocity_correction_scale <= 0.0:
            raise ValueError("velocity_correction_scale must be positive")
        if self._velocity_target_state_shift_rms <= 0.0:
            raise ValueError("velocity_target_state_shift_rms must be positive")

        self._diagnostic_reward = str(
            mcfg.get("diagnostic_reward", "none") or "none"
        ).strip().lower()
        if self._diagnostic_reward not in {"none", "temporal_l1"}:
            raise ValueError("method.diagnostic_reward must be none or temporal_l1")

        self._ema_update_interval = max(
            1,
            int(mcfg.get("ema_update_interval", 8) or 8),
        )
        self._validate_raw_model = bool(mcfg.get("validate_raw_model", True))
        self._validate_ema_model = bool(mcfg.get("validate_ema_model", True))
        self._active_validation_variant = "ema"
        self._raw_validation_state = self._new_validation_state()
        self._ema_validation_state = self._new_validation_state()

        audit_cfg = mcfg.get("videoalign_audit", {}) or {}
        if not isinstance(audit_cfg, dict):
            raise ValueError("method.videoalign_audit must be a mapping")
        self._videoalign_audit_enabled = bool(audit_cfg.get("enabled", True))
        self._videoalign_audit_min_coverage = float(
            audit_cfg.get("minimum_checkpoint_numel_coverage", 0.97)
        )
        self._videoalign_audit_min_component = float(
            audit_cfg.get("minimum_component_coverage", 0.95)
        )
        self._videoalign_require_head = bool(
            audit_cfg.get("require_reward_head", False)
        )
        self._videoalign_repeat_tolerance = float(
            audit_cfg.get("repeatability_tolerance", 1.0e-6)
        )
        self._startup_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle and checkpointing

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["finite_transition_v2.state"] = _FiniteTransitionV2State(self)
        return states

    def on_train_start(self) -> None:
        super().on_train_start()
        if self._videoalign_audit_enabled:
            checkpoint_path = None
            try:
                import os

                checkpoint_path = os.environ.get("VIDEOALIGN_CHECKPOINT_PATH")
                report = audit_videoalign_checkpoint(
                    device=self.student.device,
                    checkpoint_path=checkpoint_path,
                    minimum_checkpoint_numel_coverage=(
                        self._videoalign_audit_min_coverage
                    ),
                    minimum_component_coverage=(
                        self._videoalign_audit_min_component
                    ),
                    require_reward_head=self._videoalign_require_head,
                )
                output_dir = Path(self.training_config.checkpoint.output_dir)
                write_audit_report(
                    report,
                    output_dir / "videoalign_checkpoint_audit.json",
                )
                overall = report["overall"]
                self._startup_metrics.update(
                    {
                        "audit/videoalign_checkpoint_numel_coverage": float(
                            overall["numel_ratio"]
                        ),
                        "audit/videoalign_checkpoint_tensor_coverage": float(
                            overall["tensor_ratio"]
                        ),
                        "audit/videoalign_unmatched_keys": float(
                            report["unmatched_key_count"]
                        ),
                    }
                )
                for name, component in report.get("components", {}).items():
                    self._startup_metrics[
                        f"audit/videoalign_{name}_numel_coverage"
                    ] = float(component["numel_ratio"])
            except Exception:
                # This audit is intentionally fail-fast. A finite reward is not
                # evidence that the adapter/head loaded correctly.
                raise

            if self._reward_scorer is None:
                raise RuntimeError("reward scorer was not initialized")
            repeat = repeatability_probe(
                self._reward_scorer,
                device=self.student.device,
                tolerance=self._videoalign_repeat_tolerance,
            )
            self._startup_metrics.update(
                {f"audit/videoalign_{key}": float(value) for key, value in repeat.items()}
            )

    def _update_ema(self) -> None:
        if not self._ema_enabled or self._student_ema is None:
            return
        if (
            self._ema_update_count >= self._ema_update_after_step
            and self._ema_update_count % self._ema_update_interval == 0
        ):
            self._student_ema.update(self.student.transformer)
        self._ema_update_count += 1

    # ------------------------------------------------------------------
    # Rollout collection

    @contextlib.contextmanager
    def _behavior_context(self) -> Iterator[None]:
        if self._behavior_policy == "on_policy":
            yield
            return
        for module in (self.student.transformer, *self.student.transformer.modules()):
            disable = getattr(module, "disable_adapter", None)
            if not callable(disable):
                continue
            context = disable()
            if hasattr(context, "__enter__"):
                with context:
                    yield
                return
        raise RuntimeError(
            "behavior_policy=frozen_base requires a PEFT disable_adapter() "
            "context on the trainable AnyFlow transformer"
        )

    def _collect_rollout_group(
        self,
        *,
        iteration: int,
        group_offset: int,
    ) -> _RolloutGroup:
        if self._reward_scorer is None or self.cuda_generator is None:
            raise RuntimeError("method was not initialized with on_train_start")
        rank = self._rank()
        world_size = self._world_size()
        local_branches = self._group_size // world_size
        sample_id = int(iteration) * self._rollout_groups_per_update + int(group_offset)
        raw_batch = self._sample_prompt_batch(
            iteration=sample_id,
            local_branches=local_branches,
        )
        prompts = self._extract_prompts(raw_batch)
        if len(prompts) != local_branches or len(set(prompts)) != 1:
            raise RuntimeError("every candidate in one rollout group must share a prompt")

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
        current = self._shared_initial_noise(
            batch,
            iteration=sample_id,
            local_branches=local_branches,
        )
        records: list[_TransitionRecord] = []

        self.student.transformer.eval()
        with torch.no_grad(), self._behavior_context():
            if self._transition_mode == "single":
                branch_indices = [
                    self._select_branch_index(
                        iteration=sample_id,
                        device=self.student.device,
                    )
                ]
                for index in range(branch_indices[0]):
                    current, _ = self._flow_map(
                        current,
                        schedule[index],
                        schedule[index + 1],
                        batch,
                    )
            else:
                branch_indices = list(range(self._stochastic_steps))

            for branch_index in branch_indices:
                old_mean, old_std, deterministic_target = self._branch_policy(
                    current,
                    schedule[branch_index],
                    schedule[branch_index + 1],
                    batch,
                )
                generator = torch.Generator(device=self.student.device).manual_seed(
                    int(self.training_config.data.seed)
                    + sample_id * 1_000_003
                    + branch_index * 10_007
                    + rank * 101
                )
                actions, _ = sample_diagonal_gaussian(
                    old_mean,
                    old_std,
                    generator=generator,
                )
                records.append(
                    _TransitionRecord(
                        state=current.detach(),
                        action=actions.detach(),
                        old_log_prob=gaussian_log_prob_mean(
                            actions,
                            old_mean,
                            old_std,
                        ).detach(),
                        source_time=schedule[branch_index].detach(),
                        target_time=schedule[branch_index + 1].detach(),
                        deterministic_target=deterministic_target.detach(),
                        batch=batch,
                    )
                )
                current = actions

            final_branch_index = branch_indices[-1]
            endpoint = self._complete_from_action(
                current,
                batch,
                schedule,
                branch_index=final_branch_index,
            )
            media = self.student.decode_latents(endpoint).detach().cpu()
            local_rewards = self._score_media(media, prompts)
            local_temporal = temporal_l1(media).to(self.student.device)
            if self._diagnostic_reward == "temporal_l1":
                local_rewards["diagnostic_temporal_l1"] = local_temporal

        global_components = {
            key: self._all_gather_1d(value)
            for key, value in local_rewards.items()
        }
        reward_key = (
            "diagnostic_temporal_l1"
            if self._diagnostic_reward == "temporal_l1"
            else self._optimize_reward
        )
        if reward_key not in global_components:
            raise RuntimeError(
                f"rollout reward {reward_key!r} is unavailable; "
                f"found {sorted(global_components)}"
            )
        local_start = rank * local_branches
        return _RolloutGroup(
            prompt=prompts[0],
            prompts=prompts,
            transitions=records,
            global_rewards=global_components[reward_key],
            global_reward_components=global_components,
            global_temporal=self._all_gather_1d(local_temporal),
            local_start=local_start,
            local_end=local_start + local_branches,
        )

    # ------------------------------------------------------------------
    # Objective helpers

    def _posterior_weights(
        self,
        rewards: torch.Tensor,
        *,
        baseline: float,
        global_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._posterior_temperature_mode == "group_ess":
            return reward_tilted_weights(
                rewards,
                target_ess_ratio=self._target_ess_ratio,
                bisection_steps=self._reward_bisection_steps,
            )
        return global_temperature_weights(
            rewards,
            baseline=baseline,
            global_std=global_std,
            temperature_multiplier=self._posterior_temperature_multiplier,
            minimum_temperature=self._posterior_min_temperature,
        )

    def _global_centered_action_shift(
        self,
        record: _TransitionRecord,
        weights: torch.Tensor,
        local_start: int,
        local_end: int,
    ) -> torch.Tensor:
        local = weights[local_start:local_end].to(record.action.device)
        shape = [local.shape[0]] + [1] * (record.action.ndim - 1)
        weighted = (local.view(*shape) * record.action.float()).sum(dim=0, keepdim=True)
        uniform = record.action.float().sum(dim=0, keepdim=True) / float(self._group_size)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            dist.all_reduce(uniform, op=dist.ReduceOp.SUM)
        return weighted - uniform

    def _finite_velocity_loss(
        self,
        record: _TransitionRecord,
        weights: torch.Tensor,
        *,
        local_start: int,
        local_end: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta = (
            record.source_time.float() - record.target_time.float()
        ) / float(self.student.num_train_timesteps)
        if float(delta) <= 0.0:
            raise ValueError("finite velocity regression requires positive interval")
        candidate_velocity = (
            record.state.float() - record.action.float()
        ) / delta
        local_weights = weights[local_start:local_end].to(record.action.device)
        coefficients = local_weights - (1.0 / float(self._group_size))
        shape = [coefficients.shape[0]] + [1] * (candidate_velocity.ndim - 1)
        correction = (
            coefficients.view(*shape) * candidate_velocity
        ).sum(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correction, op=dist.ReduceOp.SUM)

        reference_velocity = (
            record.state[:1].float() - record.deterministic_target[:1].float()
        ) / delta
        raw_state_shift = delta * correction
        raw_shift_rms = rms(raw_state_shift)
        eta = self._velocity_correction_scale
        if float(raw_shift_rms) > self._velocity_target_state_shift_rms:
            eta *= self._velocity_target_state_shift_rms / max(
                float(raw_shift_rms),
                1.0e-12,
            )
        target = reference_velocity + float(eta) * correction
        _, predicted_velocity = self._flow_map(
            record.state.detach(),
            record.source_time,
            record.target_time,
            record.batch,
        )
        target_expanded = target.to(predicted_velocity.dtype).expand_as(predicted_velocity)
        loss = (predicted_velocity.float() - target_expanded.float().detach()).square().mean()
        return loss, {
            "velocity_correction_eta": loss.new_tensor(float(eta)),
            "velocity_correction_rms": rms(correction),
            "velocity_target_state_shift_rms": rms(float(eta) * raw_state_shift),
            "velocity_reference_rms": rms(reference_velocity),
        }

    # ------------------------------------------------------------------
    # Managed optimization

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del data_stream
        started = time.perf_counter()
        groups = [
            self._collect_rollout_group(iteration=iteration, group_offset=index)
            for index in range(self._rollout_groups_per_update)
        ]
        pooled_rewards = torch.cat([group.global_rewards for group in groups])
        current_std, self._reward_std_ema = stable_global_reward_std(
            pooled_rewards,
            previous=self._reward_std_ema,
            decay=self._reward_std_decay,
            floor=self._reward_std_floor,
        )
        global_std = self._reward_std_ema

        group_payloads: list[dict[str, Any]] = []
        for group in groups:
            baseline = self._prompt_tracker.baseline(group.prompt, group.global_rewards)
            advantages = running_baseline_advantages(
                group.global_rewards,
                baseline=baseline,
                global_std=global_std,
                clip=self._advantage_clip,
            )
            weights, temperature, ess = self._posterior_weights(
                group.global_rewards,
                baseline=baseline,
                global_std=global_std,
            )
            group_payloads.append(
                {
                    "group": group,
                    "baseline": baseline,
                    "advantages": advantages,
                    "weights": weights,
                    "temperature": temperature,
                    "ess": ess,
                }
            )
            self._prompt_tracker.update(group.prompt, group.global_rewards)

        total_records = sum(len(group.transitions) for group in groups)
        if total_records <= 0:
            raise RuntimeError("no finite transitions were collected")

        self.student.transformer.train()
        self._student_optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.student.device)
        diagnostic_sums: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((), device=self.student.device)
        )
        probe: dict[str, Any] | None = None

        for payload in group_payloads:
            group: _RolloutGroup = payload["group"]
            weights: torch.Tensor = payload["weights"]
            advantages: torch.Tensor = payload["advantages"]
            local_weights = weights[group.local_start:group.local_end].to(
                self.student.device
            )
            local_advantages = advantages[group.local_start:group.local_end].to(
                self.student.device
            )
            for record in group.transitions:
                if self._v2_objective == "finite_velocity_regression":
                    raw_loss, diagnostics = self._finite_velocity_loss(
                        record,
                        weights,
                        local_start=group.local_start,
                        local_end=group.local_end,
                    )
                    pre_log_prob = None
                    pre_deterministic = None
                else:
                    new_mean, new_std, pre_deterministic = self._branch_policy(
                        record.state.detach(),
                        record.source_time,
                        record.target_time,
                        record.batch,
                    )
                    new_log_prob = gaussian_log_prob_mean(
                        record.action,
                        new_mean,
                        new_std,
                    )
                    pre_log_prob = new_log_prob.detach()
                    if self._v2_objective == "flowmap_grpo":
                        raw_loss, diagnostics = clipped_grpo_loss(
                            new_log_prob,
                            record.old_log_prob,
                            local_advantages,
                            clip_range=self._clip_range,
                        )
                    else:
                        raw_loss, diagnostics = posterior_projection_loss(
                            new_log_prob,
                            local_weights,
                            global_group_size=self._group_size,
                            distributed_world_size=self._world_size(),
                        )

                normalized = (
                    raw_loss
                    * float(self._loss_scale)
                    / float(total_records)
                )
                self.student.backward(
                    normalized,
                    (record.batch.timesteps, record.batch.attn_metadata),
                    grad_accum_rounds=1,
                )
                total_loss = total_loss + normalized.detach()
                for key, value in diagnostics.items():
                    diagnostic_sums[key] += torch.as_tensor(
                        value,
                        device=self.student.device,
                    ).detach().float() / float(total_records)

                if probe is None and pre_log_prob is not None:
                    probe = {
                        "record": record,
                        "pre_mean": new_mean.detach(),
                        "pre_std": new_std.detach(),
                        "pre_log_prob": pre_log_prob,
                        "pre_deterministic": pre_deterministic.detach(),
                        "preferred_shift": self._global_centered_action_shift(
                            record,
                            weights,
                            group.local_start,
                            group.local_end,
                        ).detach(),
                    }

        grad_norm = self._clip_student_grads()
        self._student_optimizer.step()
        self._student_lr_scheduler.step()
        self._update_ema()
        self._student_optimizer.zero_grad(set_to_none=True)

        observed_kl = 0.0
        logprob_delta = 0.0
        deterministic_shift_rms = 0.0
        deterministic_alignment = 0.0
        preferred_shift_rms = 0.0
        if probe is not None:
            record = probe["record"]
            self.student.transformer.eval()
            with torch.no_grad():
                post_mean, post_std, post_deterministic = self._branch_policy(
                    record.state.detach(),
                    record.source_time,
                    record.target_time,
                    record.batch,
                )
                post_log_prob = gaussian_log_prob_mean(
                    record.action,
                    post_mean,
                    post_std,
                )
                delta_log_prob = post_log_prob - probe["pre_log_prob"]
                observed_kl = float(
                    diagonal_gaussian_kl_mean(
                        probe["pre_mean"],
                        probe["pre_std"],
                        post_mean,
                        post_std,
                    ).mean().item()
                )
                logprob_delta = float(delta_log_prob.abs().mean().item())
                map_shift = post_deterministic.float() - probe["pre_deterministic"].float()
                preferred = probe["preferred_shift"].to(map_shift.device)
                deterministic_shift_rms = float(rms(map_shift).item())
                preferred_shift_rms = float(rms(preferred).item())
                deterministic_alignment = float(
                    cosine_similarity_flat(
                        map_shift[:1],
                        preferred.expand_as(map_shift)[:1],
                    ).item()
                )

        previous_scale = self._loss_scale
        self._loss_scale = update_target_kl_scale(
            self._loss_scale,
            observed_kl,
            target_kl=self._target_post_update_kl,
            controller_rate=self._kl_controller_rate,
            minimum_scale=self._minimum_loss_scale,
            maximum_scale=self._maximum_loss_scale,
        )

        step_seconds = time.perf_counter() - started
        self._cumulative_train_seconds += step_seconds
        all_weights = torch.cat([payload["weights"] for payload in group_payloads])
        all_ess = torch.stack([payload["ess"].float() for payload in group_payloads])
        temperatures = torch.stack(
            [payload["temperature"].float() for payload in group_payloads]
        )
        selection_gain = torch.stack(
            [
                (payload["weights"] * payload["group"].global_rewards).sum()
                - payload["group"].global_rewards.mean()
                for payload in group_payloads
            ]
        ).mean()

        metrics: dict[str, LogScalar] = {
            "ftv2/objective_is_grpo": float(self._v2_objective == "flowmap_grpo"),
            "ftv2/objective_is_posterior": float(
                self._v2_objective == "posterior_projection"
            ),
            "ftv2/objective_is_velocity_regression": float(
                self._v2_objective == "finite_velocity_regression"
            ),
            "ftv2/transition_mode_is_all": float(self._transition_mode == "all"),
            "ftv2/behavior_is_frozen_base": float(
                self._behavior_policy == "frozen_base"
            ),
            "ftv2/rollout_groups_per_update": float(
                self._rollout_groups_per_update
            ),
            "ftv2/reward_videos_per_update": float(
                self._rollout_groups_per_update * self._group_size
            ),
            "ftv2/transition_records_per_update": float(total_records),
            "ftv2/reward_mean": pooled_rewards.mean(),
            "ftv2/reward_std_current": float(current_std),
            "ftv2/reward_std_ema": float(self._reward_std_ema),
            "ftv2/reward_selection_gain": selection_gain,
            "ftv2/posterior_ess_mean": all_ess.mean(),
            "ftv2/posterior_ess_min": all_ess.min(),
            "ftv2/posterior_temperature_mean": temperatures.mean(),
            "ftv2/posterior_weight_max": all_weights.max(),
            "ftv2/loss_scale_before": float(previous_scale),
            "ftv2/loss_scale_after": float(self._loss_scale),
            "ftv2/target_post_update_kl": float(self._target_post_update_kl),
            "ftv2/post_update_approx_kl": float(observed_kl),
            "ftv2/post_update_logprob_delta_abs": float(logprob_delta),
            "ftv2/deterministic_map_shift_rms": float(deterministic_shift_rms),
            "ftv2/preferred_action_shift_rms": float(preferred_shift_rms),
            "ftv2/deterministic_preference_alignment": float(
                deterministic_alignment
            ),
            "ftv2/grad_norm": self._mean_across_ranks(
                torch.as_tensor(grad_norm, device=self.student.device)
            ),
            "ftv2/train_step_seconds": float(step_seconds),
            "ftv2/cumulative_gpu_hours": (
                self._cumulative_train_seconds * self._world_size() / 3600.0
            ),
            "ftv2/prompt_tracker_size": float(len(self._prompt_tracker.prompts)),
        }
        for key, value in diagnostic_sums.items():
            metrics[f"ftv2/{key}"] = self._mean_across_ranks(value)
        for key, value in self._startup_metrics.items():
            metrics[key] = float(value)
        for reward_name in sorted(groups[0].global_reward_components):
            pooled = torch.cat(
                [group.global_reward_components[reward_name] for group in groups]
            )
            metrics[f"ftv2/reward/{reward_name}"] = pooled.mean()
            metrics[f"ftv2/reward_std/{reward_name}"] = pooled.std(unbiased=False)
        return {"total_loss": total_loss}, {}, metrics

    # ------------------------------------------------------------------
    # Raw and EMA validation

    @staticmethod
    def _new_validation_state() -> dict[str, Any]:
        return {
            "baseline": {},
            "best": float("-inf"),
            "steps": -1,
        }

    @contextlib.contextmanager
    def _validation_state_context(self, variant: str) -> Iterator[None]:
        state = (
            self._raw_validation_state
            if variant == "raw"
            else self._ema_validation_state
        )
        saved = (
            self._validation_baseline,
            self._validation_best_primary_delta,
            self._steps_to_primary_target,
            self._active_validation_variant,
        )
        self._validation_baseline = dict(state.get("baseline", {}))
        self._validation_best_primary_delta = float(
            state.get("best", float("-inf"))
        )
        self._steps_to_primary_target = int(state.get("steps", -1))
        self._active_validation_variant = variant
        try:
            yield
        finally:
            state["baseline"] = dict(self._validation_baseline)
            state["best"] = float(self._validation_best_primary_delta)
            state["steps"] = int(self._steps_to_primary_target)
            (
                self._validation_baseline,
                self._validation_best_primary_delta,
                self._steps_to_primary_target,
                self._active_validation_variant,
            ) = saved

    @staticmethod
    def _variant_metrics(
        metrics: dict[str, LogScalar],
        variant: str,
    ) -> dict[str, LogScalar]:
        prefixes = (
            "validation/",
            "validation_std/",
            "validation_sem/",
            "validation_delta/",
            "validation_baseline/",
            "validation_success/",
        )
        result: dict[str, LogScalar] = {}
        for key, value in metrics.items():
            replaced = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    stem = prefix[:-1]
                    result[f"{stem}_{variant}/{key[len(prefix):]}"] = value
                    replaced = True
                    break
            if not replaced:
                result[f"validation_{variant}_meta/{key}"] = value
        return result

    def on_validation_begin(
        self,
        iteration: int = 0,
    ) -> dict[str, LogScalar]:
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}
        metrics: dict[str, LogScalar] = {}
        if self._validate_raw_model:
            with self._validation_state_context("raw"):
                raw = super()._run_validation(iteration)
            metrics.update(self._variant_metrics(raw, "raw"))
        if self._validate_ema_model:
            with self._validation_state_context("ema"), self._ema_context():
                ema = super()._run_validation(iteration)
            # Preserve the original EMA keys for existing dashboards and add an
            # explicit variant namespace for raw/EMA comparison.
            metrics.update(ema)
            metrics.update(self._variant_metrics(ema, "ema"))
        return metrics

    def _log_validation_samples(
        self,
        local_logs: list[dict[str, Any]],
        iteration: int,
    ) -> None:
        if self._active_validation_variant != "ema":
            return
        super()._log_validation_samples(local_logs, iteration)


__all__ = ["FiniteTransitionV2Method"]
