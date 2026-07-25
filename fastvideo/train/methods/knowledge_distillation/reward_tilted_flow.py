# SPDX-License-Identifier: Apache-2.0
"""Reward-Tilted Flow Distillation (RTFD).

A frozen multi-step teacher supplies terminal samples, a black-box reward tilts
that endpoint distribution, and one conditional flow-matching objective trains
the student on its exact few-step deployment grid:

    q_beta(x0 | c) proportional to q_teacher(x0 | c) exp(R(x0, c) / tau)
    x_sigma = (1 - sigma) x0 + sigma eps
    u_target = eps - x0.

The fresh ``eps`` is independent of teacher sampling, so the source marginal
remains the standard Gaussian used at inference. A target-ESS temperature and a
uniform teacher mixture preserve coverage. ``uniform_mix=1`` is a matched-
compute no-reward baseline. No old policy, fake-score critic, policy ratio, or
cold-start stage is used.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.dataset.parquet_dataset_map_style import (
    get_parquet_files_and_length,
    read_row_from_parquet_file,
)
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.knowledge_distillation.reward_tilted_flow_utils import (
    RTFDConfig,
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
    media_to_video_array,
    RLValidationConfig,
    SamplingConfig,
    validation_caption,
    validation_shard_indices,
)
from fastvideo.train.methods.rl.rewards import (
    build_multi_reward_scorer,
    normalize_reward_weights,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
)


class RewardTiltedFlowDistillationMethod(TrainingMethod):
    """Single-stage few-step distillation into a reward-tilted teacher law."""

    def __init__(self, *, cfg: Any, role_models: dict[str, ModelBase]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "teacher" not in role_models:
            raise ValueError("RewardTiltedFlowDistillationMethod requires role 'teacher'")
        self.teacher = role_models["teacher"]
        if not self.student._trainable or self.teacher._trainable:
            raise ValueError("RTFD requires a trainable student and frozen teacher")
        self.student.init_preprocessors(self.training_config)

        self._teacher_sampling = SamplingConfig.from_mapping(self.method_config.get("sampling"))
        if self._teacher_sampling.trajectory != "ode":
            raise ValueError("RTFD currently supports deterministic teacher ODE sampling only")
        if self._teacher_sampling.timesteps is not None or self._teacher_sampling.sigmas is not None:
            raise ValueError("RTFD currently derives schedules from step counts")

        self._validation_config = RLValidationConfig.from_mapping(self.method_config.get("validation"))
        self._rtfd = RTFDConfig(
            student_num_steps=self._read_int("student_num_steps", 4),
            student_guidance_scale=self._read_float("student_guidance_scale", 1.0),
            trajectories_per_prompt=self._read_int(
                "trajectories_per_prompt",
                self._read_int("num_video_per_prompt", 4),
            ),
            transition_batch_size=self._read_int("transition_batch_size", 1),
            reward_ess_ratio=self._read_float("reward_ess_ratio", 0.6),
            uniform_mix=self._read_float("uniform_mix", 0.25),
            reward_bisection_steps=self._read_int("reward_bisection_steps", 32),
            max_grad_norm=self._read_float("max_grad_norm", 1.0),
            validation_every_steps=self._validation_config.every_steps,
        )
        self._validate_config()

        reward_fn = self.method_config.get("reward_fn")
        self._reward_fn_config, inline_backend = normalize_reward_weights(reward_fn)
        self._reward_backend = str(self.method_config.get("reward_backend", "auto") or "auto").strip().lower()
        if inline_backend is not None:
            self._reward_backend = inline_backend
        self._reward_scorer: Any | None = None
        self._init_optimizer()

        flow_shift = self._teacher_sampling.flow_shift
        if flow_shift is None:
            flow_shift = float(getattr(self.teacher.noise_scheduler, "shift", 1.0))
        self._flow_shift = float(flow_shift)
        self._teacher_sampler = DiffusionSampler(self._teacher_sampling)
        self._student_sampler = DiffusionSampler(
            SamplingConfig(
                num_steps=self._rtfd.student_num_steps,
                scheduler="flow_match_euler",
                trajectory="ode",
                flow_shift=self._flow_shift,
                guidance_scale=self._rtfd.student_guidance_scale,
            )
        )
        self._validation_sampler = DiffusionSampler(
            SamplingConfig(
                num_steps=self._validation_config.num_steps,
                scheduler="flow_match_euler",
                trajectory="ode",
                flow_shift=self._flow_shift,
                guidance_scale=self._rtfd.student_guidance_scale,
            )
        )
        self._validation_items: list[tuple[int, bool, dict[str, Any]]] | None = None

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def manages_optimization(self) -> bool:
        return True

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del batch, iteration
        raise RuntimeError("RewardTiltedFlowDistillationMethod uses managed_train_step()")

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        return self._maybe_run_validation(iteration)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.teacher.transformer.eval()
        self._reward_scorer = build_multi_reward_scorer(
            self._reward_fn_config,
            device=self.student.device,
            backend=self._reward_backend,
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
            raise RuntimeError("student.prepare_batch did not produce latent shape metadata")

        with torch.no_grad():
            teacher_endpoints = self._teacher_sampler.sample(
                self.teacher,
                prepared,
                generator=self.cuda_generator,
            ).latents.detach()
            teacher_rewards = self._score_media(
                self.student.decode_latents(teacher_endpoints).detach().cpu(),
                prompts,
            )
            grouped_rewards = teacher_rewards["avg"].reshape(
                prompt_batch,
                self._rtfd.trajectories_per_prompt,
            )
            trajectory_weights, temperatures, raw_ess, final_ess = reward_tilt_weights(
                grouped_rewards,
                target_ess_ratio=self._rtfd.reward_ess_ratio,
                uniform_mix=self._rtfd.uniform_mix,
                bisection_steps=self._rtfd.reward_bisection_steps,
            )
            source_noise = torch.randn(
                teacher_endpoints.shape,
                device=teacher_endpoints.device,
                dtype=teacher_endpoints.dtype,
                generator=self.cuda_generator,
            )
            target_velocity = source_noise - teacher_endpoints
            timesteps, sigmas, interval_weights = build_deployment_flow_schedule(
                num_steps=self._rtfd.student_num_steps,
                flow_shift=self._flow_shift,
                num_train_timesteps=int(self.student.num_train_timesteps),
                device=teacher_endpoints.device,
            )
            noisy_states = torch.stack(
                [(1.0 - sigma) * teacher_endpoints + sigma * source_noise for sigma in sigmas[:-1]],
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
            require_tensor(prepared.encoder_hidden_states, "encoder_hidden_states"),
            require_tensor(prepared.encoder_attention_mask, "encoder_attention_mask"),
            self._rtfd.student_num_steps,
        )

        total_loss = torch.zeros((), device=self.student.device, dtype=torch.float32)
        step_loss = torch.zeros(self._rtfd.student_num_steps, device=self.student.device)
        step_weight = torch.zeros_like(step_loss)
        microbatch = min(self._rtfd.transition_batch_size, int(row_states.shape[0]))
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
            mse = (prediction.float() - row_targets[start:end].float()).square().mean(
                dim=tuple(range(1, prediction.ndim))
            )
            loss = (row_weights[start:end] * mse).sum()
            self.student.backward(
                loss,
                (train_batch.timesteps, train_batch.attn_metadata),
                grad_accum_rounds=1,
            )
            total_loss += loss.detach()
            step_loss.scatter_add_(0, row_steps[start:end], row_weights[start:end] * mse.detach())
            step_weight.scatter_add_(0, row_steps[start:end], row_weights[start:end])

        self._clip_student_grads()
        self._student_optimizer.step()
        self._student_lr_scheduler.step()
        self._student_optimizer.zero_grad(set_to_none=True)

        metrics: dict[str, LogScalar] = {
            "rtfd/loss": self._mean_scalar(total_loss),
            "rtfd/reward_ess_ratio_raw": self._mean_scalar(raw_ess.mean()),
            "rtfd/reward_ess_ratio_final": self._mean_scalar(final_ess.mean()),
            "rtfd/reward_temperature": self._mean_scalar(temperatures.mean()),
            "rtfd/max_trajectory_weight": self._mean_scalar(trajectory_weights.max(dim=1).values.mean()),
            "rtfd/uniform_mix": float(self._rtfd.uniform_mix),
            "rtfd/teacher_steps": float(self._teacher_sampling.num_steps),
            "rtfd/student_steps": float(self._rtfd.student_num_steps),
            "rtfd/transition_rows": float(row_states.shape[0]),
        }
        transition_mse = step_loss / step_weight.clamp_min(1e-12)
        for step in range(self._rtfd.student_num_steps):
            metrics[f"rtfd/transition_{step}_mse"] = self._mean_scalar(transition_mse[step])
        metrics.update(self._reward_metrics("teacher", teacher_rewards))
        return {"total_loss": total_loss}, {}, metrics

    def _build_rows(
        self,
        *,
        noisy_states: torch.Tensor,
        target_velocity: torch.Tensor,
        timesteps: torch.Tensor,
        interval_weights: torch.Tensor,
        trajectory_weights: torch.Tensor,
        prompt_batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples, num_steps = noisy_states.shape[:2]
        states = noisy_states.flatten(0, 1)
        targets = target_velocity.unsqueeze(1).expand(
            num_samples,
            num_steps,
            *target_velocity.shape[1:],
        ).reshape(num_samples * num_steps, *target_velocity.shape[1:])
        row_timesteps = timesteps.view(1, num_steps).expand(num_samples, -1).reshape(-1)
        row_weights = (
            trajectory_weights.reshape(num_samples, 1)
            * interval_weights.view(1, num_steps)
            / float(prompt_batch)
        ).reshape(-1)
        if not torch.allclose(row_weights.sum(), torch.ones((), device=row_weights.device), atol=1e-5):
            raise RuntimeError(f"RTFD row weights must sum to one, got {float(row_weights.sum())}")
        row_steps = torch.arange(num_steps, device=states.device, dtype=torch.long).view(1, -1)
        row_steps = row_steps.expand(num_samples, -1).reshape(-1)
        return states, targets, row_timesteps, row_weights, row_steps

    def _make_training_batch(
        self,
        *,
        states: torch.Tensor,
        timesteps: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> TrainingBatch:
        batch = TrainingBatch(
            encoder_hidden_states=hidden,
            encoder_attention_mask=mask,
            timesteps=timesteps,
            raw_latent_shape=tuple(states.shape),
            conditional_dict={
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": mask,
            },
        )
        metadata_builder = getattr(self.student, "_build_attention_metadata", None)
        if callable(metadata_builder):
            batch = metadata_builder(batch)
            if batch.attn_metadata is not None and hasattr(batch.attn_metadata, "VSA_sparsity"):
                batch.attn_metadata.VSA_sparsity = 0.0
        return batch

    @torch.no_grad()
    def _maybe_run_validation(self, iteration: int) -> dict[str, LogScalar]:
        if not self._should_validate(iteration):
            return {}
        if self._reward_scorer is None:
            raise RuntimeError("RTFD reward scorer has not been initialized")
        return self._run_validation(iteration)

    @torch.no_grad()
    def _run_validation(self, iteration: int) -> dict[str, LogScalar]:
        was_training = self.student.transformer.training
        self.student.transformer.eval()
        try:
            config = self._validation_config
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            prepare_generator = torch.Generator(device=self.student.device).manual_seed(config.seed + 100_000 + rank)
            validation_generator = torch.Generator(device=self.student.device).manual_seed(config.seed + rank)
            local_rewards: dict[str, list[torch.Tensor]] = defaultdict(list)
            local_masks: list[torch.Tensor] = []
            local_logs: list[dict[str, Any]] = []
            items = self._get_validation_items()
            for start in range(0, len(items), config.batch_size):
                batch_items = items[start:start + config.batch_size]
                raw_batch = self._collate_validation_rows([item[2] for item in batch_items])
                prompts = extract_prompts(raw_batch)
                batch = self.student.prepare_batch(
                    raw_batch,
                    generator=prepare_generator,
                    latents_source="zeros",
                )
                latents = self._validation_sampler.sample(
                    self.student,
                    batch,
                    generator=validation_generator,
                ).latents
                media = self.student.decode_latents(latents).detach().cpu()
                rewards = self._score_media(media, prompts)
                valid_mask = torch.tensor(
                    [item[1] for item in batch_items],
                    device=self.student.device,
                    dtype=torch.float32,
                )
                local_masks.append(valid_mask)
                for key, value in rewards.items():
                    local_rewards[key].append(value.to(device=self.student.device, dtype=torch.float32))
                if config.log_samples:
                    for sample_idx, (global_idx, valid, _) in enumerate(batch_items):
                        if not valid:
                            continue
                        local_logs.append({
                            "index": int(global_idx),
                            "prompt": prompts[sample_idx],
                            "video": media_to_video_array(media[sample_idx]),
                            "rewards": {
                                key: float(value[sample_idx].detach().float().cpu())
                                for key, value in rewards.items()
                            },
                        })

            if not local_rewards or not local_masks:
                return {}
            gathered_mask = self._gather_tensor(torch.cat(local_masks, dim=0)).bool()
            metrics: dict[str, LogScalar] = {}
            for key, chunks in local_rewards.items():
                gathered_values = self._gather_tensor(torch.cat(chunks, dim=0).detach().float())
                valid_values = gathered_values[gathered_mask]
                if valid_values.numel() > 0:
                    metrics[f"validation/reward/{key}"] = valid_values.mean()
            metrics["validation/num_prompts"] = gathered_mask.float().sum()
            if config.log_samples:
                self._log_validation_samples(local_logs, iteration)
            return metrics
        finally:
            self.student.transformer.train(was_training)

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

        artifacts = []
        logs = sorted(
            (log for rank_logs in gathered_logs for log in rank_logs),
            key=lambda item: int(item["index"]),
        )
        for item in logs:
            artifact = self.tracker.video(
                item["video"],
                caption=validation_caption(item["prompt"], item["rewards"]),
                fps=1,
            )
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            self.tracker.log_artifacts({"validation/videos": artifacts}, step=iteration)

    def _get_validation_items(self) -> list[tuple[int, bool, dict[str, Any]]]:
        if self._validation_items is not None:
            return self._validation_items

        dataset = getattr(getattr(self.student, "dataloader", None), "dataset", None)
        if dataset is None:
            raise RuntimeError("RTFD validation requires the student dataloader dataset")
        data_path = self._validation_config.data_path or getattr(dataset, "path", None)
        if not data_path:
            data_path = self.training_config.data.data_path
        if self._validation_config.data_path is None or data_path == getattr(dataset, "path", None):
            parquet_files = list(dataset.parquet_files)
            lengths = list(dataset.lengths)
        else:
            parquet_files, lengths = get_parquet_files_and_length(data_path)
            parquet_files = list(parquet_files)
            lengths = list(lengths)

        total_rows = int(sum(lengths))
        if total_rows <= 0:
            raise RuntimeError(f"Validation data_path {data_path!r} has no rows")
        num_prompts = min(self._validation_config.num_prompts, total_rows)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        items: list[tuple[int, bool, dict[str, Any]]] = []
        for prompt_idx, valid in validation_shard_indices(
            num_prompts,
            rank=rank,
            world_size=world_size,
        ):
            row = read_row_from_parquet_file(parquet_files, prompt_idx, lengths)
            row["_sample_index"] = prompt_idx
            items.append((prompt_idx, valid, row))
        self._validation_items = items
        return items

    def _collate_validation_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = getattr(getattr(self.student, "dataloader", None), "dataset", None)
        if dataset is None or not hasattr(dataset, "parquet_schema"):
            raise RuntimeError("RTFD requires a parquet-backed student dataset")
        return collate_rows_from_parquet_schema(
            rows,
            dataset.parquet_schema,
            int(getattr(dataset, "text_padding_length", 512)),
            cfg_rate=0.0,
            seed=self._validation_config.seed,
        )

    @staticmethod
    def _gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _score_media(self, media: torch.Tensor, prompts: list[str]) -> dict[str, torch.Tensor]:
        if self._reward_scorer is None:
            raise RuntimeError("RTFD reward scorer has not been initialized")
        scores = self._reward_scorer(media, prompts)
        out = {key: value.to(self.student.device, torch.float32) for key, value in scores.items()}
        if "avg" not in out:
            raise RuntimeError("Reward scorer must return an 'avg' score")
        return out

    def _reward_metrics(
        self,
        source: str,
        rewards: dict[str, torch.Tensor],
        *,
        prefix: str = "rtfd",
    ) -> dict[str, LogScalar]:
        return {
            f"{prefix}/{source}_reward/{key}": self._mean_scalar(value.detach().float().mean())
            for key, value in rewards.items()
        }

    def _should_validate(self, iteration: int) -> bool:
        every = int(self._rtfd.validation_every_steps)
        return every > 0 and int(iteration) % every == 0

    def _init_optimizer(self) -> None:
        tc = self.training_config
        params = [parameter for parameter in self.student.transformer.parameters() if parameter.requires_grad]
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=float(tc.optimizer.learning_rate),
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    def _clip_student_grads(self) -> None:
        if self._rtfd.max_grad_norm > 0.0:
            clip_grad_norm_while_handling_failing_dtensor_cases(
                list(self.student.transformer.parameters()),
                self._rtfd.max_grad_norm,
                foreach=None,
            )

    def _validate_config(self) -> None:
        cfg = self._rtfd
        if min(cfg.student_num_steps, cfg.trajectories_per_prompt, cfg.transition_batch_size) <= 0:
            raise ValueError("RTFD step and batch sizes must be positive")
        if cfg.student_guidance_scale < 0.0:
            raise ValueError("method.student_guidance_scale must be non-negative")
        if not 0.0 < cfg.reward_ess_ratio <= 1.0:
            raise ValueError("method.reward_ess_ratio must lie in (0, 1]")
        if not 0.0 <= cfg.uniform_mix <= 1.0:
            raise ValueError("method.uniform_mix must lie in [0, 1]")
        if cfg.reward_bisection_steps <= 0:
            raise ValueError("method.reward_bisection_steps must be positive")

    def _read_int(self, key: str, default: int) -> int:
        value = self.method_config.get(key, default)
        return int(default if value is None else value)

    def _read_float(self, key: str, default: float) -> float:
        value = self.method_config.get(key, default)
        return float(default if value is None else value)

    @staticmethod
    def _mean_scalar(value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        return reduced


RTFDMethod = RewardTiltedFlowDistillationMethod

# Re-export pure helpers for unit tests and external ablations.
__all__ = [
    "RTFDMethod",
    "RewardTiltedFlowDistillationMethod",
    "build_deployment_flow_schedule",
    "reward_tilt_weights",
]
