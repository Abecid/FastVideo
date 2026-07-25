# SPDX-License-Identifier: Apache-2.0
"""Motion-preserving Reward-Tilted Reflow Distillation.

The original RTFD baseline sampled a teacher endpoint and then paired it with
fresh independent Gaussian noise. That product coupling preserves the Gaussian
source marginal, but it discards the teacher's noise-to-video assignment. For
multimodal video prompts, deterministic MSE flow matching can then average
incompatible motions into a nearly static conditional mean.

This variant keeps the teacher's original initial noise paired with its endpoint
(reflow coupling), balances the student's four-step sigma schedule separately
from the teacher schedule, and optionally normalizes each reward component
within prompt before scalarization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.knowledge_distillation.reward_tilted_flow import (
    RewardTiltedFlowDistillationMethod,
)
from fastvideo.train.methods.knowledge_distillation.reward_tilted_flow_utils import (
    build_deployment_flow_schedule,
    extract_prompts,
    infer_batch_size,
    repeat_batch_rows,
    repeat_condition_rows,
    require_tensor,
    reward_tilt_weights,
)
from fastvideo.train.methods.rl.common import (
    DiffusionSampler,
    SamplingConfig,
    validation_caption,
)


def aggregate_grouped_rewards(
    rewards: dict[str, torch.Tensor],
    reward_weights: dict[str, float],
    *,
    prompt_batch: int,
    trajectories_per_prompt: int,
    mode: str,
) -> torch.Tensor:
    """Return one scalar reward signal per prompt and teacher trajectory.

    ``component_zscore`` standardizes each component within a prompt before
    applying the configured reward weights. This prevents differently-scaled
    VideoAlign heads from silently dominating the trajectory selection.
    """
    expected = int(prompt_batch) * int(trajectories_per_prompt)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"raw", "raw_sum", "avg"}:
        avg = rewards.get("avg")
        if avg is None or int(avg.numel()) != expected:
            raise ValueError("rewards['avg'] must contain one value per trajectory")
        return avg.float().reshape(prompt_batch, trajectories_per_prompt)

    if normalized_mode not in {"component_zscore", "zscore", "per_component_zscore"}:
        raise ValueError(
            "reward_aggregation must be one of "
            "{raw_sum, component_zscore}, got "
            f"{mode!r}"
        )

    combined: torch.Tensor | None = None
    weight_norm = 0.0
    for name, configured_weight in reward_weights.items():
        value = rewards.get(name)
        if value is None:
            raise ValueError(f"Reward scorer did not return configured component {name!r}")
        if int(value.numel()) != expected:
            raise ValueError(
                f"Reward {name!r} must contain {expected} values, "
                f"got {int(value.numel())}"
            )
        grouped = value.float().reshape(prompt_batch, trajectories_per_prompt)
        centered = grouped - grouped.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, unbiased=False, keepdim=True)
        standardized = torch.where(
            scale > 1e-6,
            centered / scale.clamp_min(1e-6),
            torch.zeros_like(centered),
        )
        weight = float(configured_weight)
        term = weight * standardized
        combined = term if combined is None else combined + term
        weight_norm += abs(weight)

    if combined is None or weight_norm <= 0.0:
        raise ValueError("At least one non-zero reward component weight is required")
    return combined / weight_norm


class RewardTiltedReflowDistillationMethod(RewardTiltedFlowDistillationMethod):
    """Reward-weighted four-step reflow using teacher-coupled noise/video pairs."""

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        self._coupling_mode = str(
            self.method_config.get("coupling_mode", "teacher_noise") or "teacher_noise"
        ).strip().lower()
        if self._coupling_mode not in {"teacher_noise", "independent"}:
            raise ValueError(
                "method.coupling_mode must be one of "
                "{teacher_noise, independent}"
            )

        self._reward_aggregation = str(
            self.method_config.get("reward_aggregation", "component_zscore")
            or "component_zscore"
        ).strip().lower()
        if self._reward_aggregation not in {
            "raw",
            "raw_sum",
            "avg",
            "component_zscore",
            "zscore",
            "per_component_zscore",
        }:
            raise ValueError(
                "method.reward_aggregation must be one of "
                "{raw_sum, component_zscore}"
            )

        self._student_flow_shift = float(
            self.method_config.get("student_flow_shift", 1.0) or 1.0
        )
        if self._student_flow_shift <= 0.0:
            raise ValueError("method.student_flow_shift must be positive")

        validation_raw = self.method_config.get("validation", {}) or {}
        self._validation_video_fps = max(
            1, int(validation_raw.get("fps", 8) or 8)
        )
        self._validation_max_log_videos = max(
            1,
            int(
                validation_raw.get(
                    "max_log_videos",
                    self._validation_config.num_prompts,
                )
                or self._validation_config.num_prompts
            ),
        )

        # Teacher quality can keep the original Wan shift, while the four-step
        # student uses a separately controlled schedule. With shift=8, the
        # default four-step grid puts roughly 79% of the sigma path in one jump.
        self._student_sampler = DiffusionSampler(
            SamplingConfig(
                num_steps=self._rtfd.student_num_steps,
                scheduler="flow_match_euler",
                trajectory="ode",
                flow_shift=self._student_flow_shift,
                guidance_scale=self._rtfd.student_guidance_scale,
            )
        )
        self._validation_sampler = DiffusionSampler(
            SamplingConfig(
                num_steps=self._validation_config.num_steps,
                scheduler="flow_match_euler",
                trajectory="ode",
                flow_shift=self._student_flow_shift,
                guidance_scale=self._rtfd.student_guidance_scale,
            )
        )

    def managed_train_step(
        self,
        data_stream: Any,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        if self._reward_scorer is None or self.cuda_generator is None:
            raise RuntimeError("RTFD has not been initialized with on_train_start()")

        raw_batch = next(data_stream)
        prompt_batch = infer_batch_size(raw_batch)
        repeated_raw = repeat_batch_rows(
            raw_batch,
            repeats=self._rtfd.trajectories_per_prompt,
            batch_size=prompt_batch,
        )
        prompts = extract_prompts(repeated_raw)
        num_samples = prompt_batch * self._rtfd.trajectories_per_prompt
        if len(prompts) != num_samples:
            raise RuntimeError(f"Expected {num_samples} prompts, found {len(prompts)}")

        self.student.transformer.train()
        self.teacher.transformer.eval()
        self._student_optimizer.zero_grad(set_to_none=True)
        prepared = self.student.prepare_batch(
            repeated_raw,
            generator=self.cuda_generator,
            latents_source="zeros",
        )
        if prepared.latents is None:
            raise RuntimeError(
                "student.prepare_batch did not produce latent shape metadata"
            )

        with torch.no_grad():
            teacher_endpoints, teacher_source_noise = self._sample_teacher_pairs(
                prepared
            )
            teacher_media = self.student.decode_latents(teacher_endpoints).detach()
            teacher_rewards = self._score_media(
                teacher_media.cpu(),
                prompts,
            )
            grouped_signal = aggregate_grouped_rewards(
                teacher_rewards,
                self._reward_fn_config,
                prompt_batch=prompt_batch,
                trajectories_per_prompt=self._rtfd.trajectories_per_prompt,
                mode=self._reward_aggregation,
            )
            trajectory_weights, temperatures, raw_ess, final_ess = (
                reward_tilt_weights(
                    grouped_signal,
                    target_ess_ratio=self._rtfd.reward_ess_ratio,
                    uniform_mix=self._rtfd.uniform_mix,
                    bisection_steps=self._rtfd.reward_bisection_steps,
                )
            )

            if self._coupling_mode == "teacher_noise":
                source_noise = teacher_source_noise
            else:
                source_noise = torch.randn(
                    teacher_endpoints.shape,
                    device=teacher_endpoints.device,
                    dtype=teacher_endpoints.dtype,
                    generator=self.cuda_generator,
                )

            target_velocity = source_noise - teacher_endpoints
            timesteps, sigmas, interval_weights = build_deployment_flow_schedule(
                num_steps=self._rtfd.student_num_steps,
                flow_shift=self._student_flow_shift,
                num_train_timesteps=int(self.student.num_train_timesteps),
                device=teacher_endpoints.device,
            )
            noisy_states = torch.stack(
                [
                    (1.0 - sigma) * teacher_endpoints + sigma * source_noise
                    for sigma in sigmas[:-1]
                ],
                dim=1,
            )

        rows = self._build_rows(
            noisy_states=noisy_states,
            target_velocity=target_velocity,
            timesteps=timesteps,
            interval_weights=interval_weights,
            trajectory_weights=trajectory_weights,
            prompt_batch=prompt_batch,
        )
        row_states, row_targets, row_timesteps, row_weights, row_steps = rows
        row_hidden, row_mask = repeat_condition_rows(
            require_tensor(
                prepared.encoder_hidden_states,
                "encoder_hidden_states",
            ),
            require_tensor(
                prepared.encoder_attention_mask,
                "encoder_attention_mask",
            ),
            self._rtfd.student_num_steps,
        )

        total_loss = torch.zeros(
            (),
            device=self.student.device,
            dtype=torch.float32,
        )
        step_loss = torch.zeros(
            self._rtfd.student_num_steps,
            device=self.student.device,
        )
        step_weight = torch.zeros_like(step_loss)
        microbatch = min(
            self._rtfd.transition_batch_size,
            int(row_states.shape[0]),
        )
        for start in range(0, int(row_states.shape[0]), microbatch):
            end = min(start + microbatch, int(row_states.shape[0]))
            train_batch = self._make_training_batch(
                states=row_states[start:end],
                timesteps=row_timesteps[start:end],
                hidden=row_hidden[start:end],
                mask=row_mask[start:end],
            )
            prediction = self.student.predict_noise(
                row_states[start:end],
                row_timesteps[start:end],
                train_batch,
                conditional=True,
                attn_kind="dense",
            )
            mse = (
                prediction.float() - row_targets[start:end].float()
            ).square().mean(dim=tuple(range(1, prediction.ndim)))
            loss = (row_weights[start:end] * mse).sum()
            self.student.backward(
                loss,
                (train_batch.timesteps, train_batch.attn_metadata),
                grad_accum_rounds=1,
            )
            total_loss += loss.detach()
            step_loss.scatter_add_(
                0,
                row_steps[start:end],
                row_weights[start:end] * mse.detach(),
            )
            step_weight.scatter_add_(
                0,
                row_steps[start:end],
                row_weights[start:end],
            )

        self._clip_student_grads()
        self._student_optimizer.step()
        self._student_lr_scheduler.step()
        self._student_optimizer.zero_grad(set_to_none=True)

        metrics: dict[str, LogScalar] = {
            "rtfd/loss": self._mean_scalar(total_loss),
            "rtfd/reward_ess_ratio_raw": self._mean_scalar(raw_ess.mean()),
            "rtfd/reward_ess_ratio_final": self._mean_scalar(final_ess.mean()),
            "rtfd/reward_temperature": self._mean_scalar(temperatures.mean()),
            "rtfd/max_trajectory_weight": self._mean_scalar(
                trajectory_weights.max(dim=1).values.mean()
            ),
            "rtfd/uniform_mix": float(self._rtfd.uniform_mix),
            "rtfd/teacher_steps": float(self._teacher_sampling.num_steps),
            "rtfd/student_steps": float(self._rtfd.student_num_steps),
            "rtfd/transition_rows": float(row_states.shape[0]),
            "rtfd/teacher_noise_coupling": float(
                self._coupling_mode == "teacher_noise"
            ),
            "rtfd/component_zscore_reward": float(
                self._reward_aggregation
                in {"component_zscore", "zscore", "per_component_zscore"}
            ),
            "rtfd/teacher_flow_shift": float(self._flow_shift),
            "rtfd/student_flow_shift": float(self._student_flow_shift),
        }
        transition_mse = step_loss / step_weight.clamp_min(1e-12)
        for step in range(self._rtfd.student_num_steps):
            metrics[f"rtfd/transition_{step}_mse"] = self._mean_scalar(
                transition_mse[step]
            )
            metrics[f"rtfd/schedule/sigma_{step}"] = float(sigmas[step])
            metrics[f"rtfd/schedule/interval_weight_{step}"] = float(
                interval_weights[step]
            )
        metrics.update(self._reward_metrics("teacher", teacher_rewards))
        metrics.update(
            self._selection_metrics(
                teacher_rewards,
                trajectory_weights,
                prompt_batch=prompt_batch,
            )
        )
        return {"total_loss": total_loss}, {}, metrics

    @torch.no_grad()
    def _sample_teacher_pairs(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher endpoint and the exact Gaussian noise that generated it."""
        latents = require_tensor(batch.latents, "latents")
        source_noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=self.cuda_generator,
        )
        current = source_noise.clone()
        scheduler = self._teacher_sampler._prepare_scheduler(
            self.teacher,
            current.device,
        )
        timesteps = scheduler.timesteps.to(device=current.device)
        original_timesteps = batch.timesteps
        try:
            for timestep in timesteps:
                model_timestep = self._teacher_sampler._model_timestep(
                    timestep,
                    current,
                )
                batch.timesteps = model_timestep
                prediction = self._teacher_sampler._predict_with_cfg(
                    self.teacher,
                    current,
                    model_timestep,
                    batch,
                )
                current = scheduler.step(
                    prediction.flatten(0, 1),
                    timestep,
                    current.flatten(0, 1),
                    return_dict=False,
                )[0].unflatten(0, prediction.shape[:2])
        finally:
            batch.timesteps = original_timesteps
        return current.detach(), source_noise.detach()

    def _selection_metrics(
        self,
        rewards: dict[str, torch.Tensor],
        trajectory_weights: torch.Tensor,
        *,
        prompt_batch: int,
    ) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}
        trajectories = self._rtfd.trajectories_per_prompt
        for name in self._reward_fn_config:
            value = rewards.get(name)
            if value is None:
                continue
            grouped = value.float().reshape(prompt_batch, trajectories)
            selected = (trajectory_weights * grouped).sum(dim=1)
            uniform = grouped.mean(dim=1)
            metrics[f"rtfd/selection/{name}_gain"] = self._mean_scalar(
                (selected - uniform).mean()
            )
            metrics[f"rtfd/selection/{name}_std"] = self._mean_scalar(
                grouped.std(dim=1, unbiased=False).mean()
            )
        return metrics

    def _score_media(
        self,
        media: torch.Tensor,
        prompts: list[str],
    ) -> dict[str, torch.Tensor]:
        scores = super()._score_media(media, prompts)
        if media.ndim == 5 and int(media.shape[2]) > 1:
            temporal_delta = (
                media[:, :, 1:].float() - media[:, :, :-1].float()
            ).abs().mean(dim=(1, 2, 3, 4))
        else:
            temporal_delta = torch.zeros(
                media.shape[0],
                device=media.device,
                dtype=torch.float32,
            )
        scores["temporal_delta_l1"] = temporal_delta.to(
            self.student.device,
            dtype=torch.float32,
        )
        return scores

    def _log_validation_samples(
        self,
        local_logs: list[dict[str, Any]],
        iteration: int,
    ) -> None:
        gathered_logs: list[list[dict[str, Any]]] = [local_logs]
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            gathered_logs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_logs, local_logs)
        if rank != 0 or self.tracker is None:
            return

        logs = sorted(
            (
                log
                for rank_logs in gathered_logs
                for log in rank_logs
            ),
            key=lambda item: int(item["index"]),
        )[: self._validation_max_log_videos]
        artifacts = []
        for item in logs:
            artifact = self.tracker.video(
                item["video"],
                caption=validation_caption(
                    item["prompt"],
                    item["rewards"],
                ),
                fps=self._validation_video_fps,
            )
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            self.tracker.log_artifacts(
                {"validation/videos": artifacts},
                step=iteration,
            )


RTRFDMethod = RewardTiltedReflowDistillationMethod

__all__ = [
    "RTRFDMethod",
    "RewardTiltedReflowDistillationMethod",
    "aggregate_grouped_rewards",
]
