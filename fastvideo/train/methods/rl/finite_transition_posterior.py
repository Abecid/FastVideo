# SPDX-License-Identifier: Apache-2.0
"""Finite-transition posterior alignment for deterministic AnyFlow video maps.

The method keeps Flow-Map GRPO's path-preserving ASFMC rollout fixed and changes
only the optimization rule.  A group of counterfactual next states branches
from one shared AnyFlow prefix, receives terminal video rewards, and defines the
KL-regularized local policy-improvement posterior

    q_R(a | s) proportional to q_old(a | s) exp(R(a) / tau).

The default objective directly projects that posterior into the current LoRA
policy with a centered forward-KL loss.  ``objective=flowmap_grpo`` is a matched
likelihood-ratio ablation using the identical prompts, branches and rewards.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.dataset.parquet_dataset_map_style import (
    get_parquet_files_and_length,
    read_row_from_parquet_file,
)
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.rl.common import (
    RLValidationConfig,
    distributed_k_repeat_indices,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)
from fastvideo.train.methods.rl.finite_transition_posterior_core import (
    append_data_endpoint,
    clipped_grpo_loss,
    endpoint_anchor_parameters,
    gaussian_log_prob_mean,
    group_advantages,
    mean_pairwise_rms,
    posterior_projection_loss,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
)
from fastvideo.train.methods.rl.rewards import (
    GENRL_REWARD_NAMES,
    build_multi_reward_scorer,
    normalize_reward_weights,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.training.training_utils import (
    EMA_FSDP,
    clip_grad_norm_while_handling_failing_dtensor_cases,
)

logger = init_logger(__name__)


def _prepare_validation_log_entry(
    *,
    index: int,
    prompt: str,
    media: torch.Tensor,
    rewards: dict[str, float],
    max_samples: int | None,
) -> dict[str, Any] | None:
    """Select and compact one globally ordered qualitative sample."""
    if max_samples is not None and int(index) >= max_samples:
        return None
    return {
        "index": int(index),
        "prompt": prompt,
        # At 81x480x832, the decoded float tensor is about 390 MiB. Convert
        # selected videos to CPU uint8 before NCCL-backed object gathering.
        "media": media_to_video_array(media),
        "rewards": rewards,
    }


class _FiniteTransitionEMAState:
    """DCP wrapper for the method-owned student EMA."""

    def __init__(self, method: "FiniteTransitionPosteriorMethod") -> None:
        self._method = method

    def state_dict(self) -> dict[str, Any]:
        ema = self._method._student_ema
        return {
            "shadow": ema.state_dict() if ema is not None else {},
            "update_count": torch.tensor(
                self._method._ema_update_count,
                dtype=torch.long,
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ema = self._method._student_ema
        shadow = state_dict.get("shadow", {})
        if ema is not None and isinstance(shadow, dict):
            ema.load_state_dict(shadow)
        update_count = state_dict.get("update_count", 0)
        if torch.is_tensor(update_count):
            update_count = int(update_count.item())
        self._method._ema_update_count = int(update_count)


class FiniteTransitionPosteriorMethod(TrainingMethod):
    """Shared-prefix posterior projection for a two-time AnyFlow policy."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if not self.student._trainable:
            raise ValueError(
                "FiniteTransitionPosteriorMethod requires a trainable student"
            )
        if int(self.training_config.distributed.sp_size or 1) != 1:
            raise ValueError(
                "finite-transition posterior groups currently require "
                "training.distributed.sp_size=1"
            )

        # AnyFlow's released post-training uses CFG=1, so avoid loading an
        # unconditional text branch that this method never calls.
        set_negative = getattr(
            self.student,
            "set_requires_negative_conditioning",
            None,
        )
        if callable(set_negative):
            set_negative(False)
        self.student.init_preprocessors(self.training_config)

        mcfg = self.method_config
        self._objective = str(
            mcfg.get("objective", "posterior_projection")
            or "posterior_projection"
        ).strip().lower()
        if self._objective not in {
            "posterior_projection",
            "flowmap_grpo",
        }:
            raise ValueError(
                "method.objective must be one of "
                "{posterior_projection, flowmap_grpo}"
            )

        self._train_map_steps = self._read_positive_int(
            "train_map_steps",
            5,
        )
        self._eval_map_steps = self._read_positive_int(
            "eval_map_steps",
            4,
        )
        self._stochastic_steps = self._read_positive_int(
            "stochastic_steps",
            self._train_map_steps - 1,
        )
        if self._train_map_steps != self._stochastic_steps + 1:
            raise ValueError(
                "train_map_steps must equal stochastic_steps + 1: the last "
                "transition is deterministic and exists only to produce reward"
            )
        configured_gpus = int(self.training_config.distributed.num_gpus or 1)
        self._group_size = self._read_positive_int(
            "group_size",
            configured_gpus,
        )
        self._target_ess_ratio = self._read_unit_float(
            "target_ess_ratio",
            0.5,
            lower_open=True,
        )
        self._reward_bisection_steps = self._read_positive_int(
            "reward_bisection_steps",
            40,
        )
        self._clip_range = self._read_unit_float(
            "clip_range",
            1.0e-4,
            lower_open=True,
        )
        self._advantage_clip = self._read_positive_float(
            "advantage_clip",
            5.0,
        )
        self._advantage_epsilon = self._read_positive_float(
            "advantage_epsilon",
            1.0e-4,
        )
        self._flow_shift = self._read_positive_float(
            "flow_shift",
            float(
                getattr(
                    getattr(self.training_config, "pipeline_config", None),
                    "flow_shift",
                    5.0,
                )
                or 5.0
            ),
        )
        self._attn_kind = str(
            mcfg.get("attn_kind", "dense") or "dense"
        ).strip().lower()
        if self._attn_kind not in {"dense", "vsa"}:
            raise ValueError("method.attn_kind must be dense or vsa")

        self._train_schedule_override = self._parse_schedule_override(
            "train_t_list_override"
        )
        self._eval_schedule_override = self._parse_schedule_override(
            "eval_t_list_override"
        )
        self._max_grad_norm = self._read_positive_float(
            "max_grad_norm",
            1.0,
        )
        self._post_update_probe_every = max(
            0,
            int(mcfg.get("post_update_probe_every", 10) or 0),
        )
        self._terminal_progress = bool(
            mcfg.get("terminal_progress", True)
        )

        reward_fn = mcfg.get("reward_fn", None)
        self._reward_fn_config, inline_backend = normalize_reward_weights(
            reward_fn
        )
        self._reward_backend = str(
            mcfg.get(
                "reward_backend",
                inline_backend or "auto",
            )
            or "auto"
        ).strip().lower()
        if self._reward_backend not in {
            "auto",
            "diffusion_nft",
            "genrl",
        }:
            raise ValueError(
                "method.reward_backend must be one of "
                "{auto, diffusion_nft, genrl}"
            )
        if (
            self._reward_backend == "genrl"
            and not any(
                name in GENRL_REWARD_NAMES
                for name in self._reward_fn_config
            )
        ):
            raise ValueError(
                "method.reward_backend=genrl requires a GenRL reward"
            )
        self._optimize_reward = str(
            mcfg.get("optimize_reward", "avg") or "avg"
        )
        if (
            self._optimize_reward != "avg"
            and self._optimize_reward not in self._reward_fn_config
        ):
            raise ValueError(
                "method.optimize_reward must be avg or one configured reward"
            )
        self._reward_scorer: Any | None = None

        self._validation_config = RLValidationConfig.from_mapping(
            mcfg.get("validation")
        )
        eval_cfg = mcfg.get("evaluation", {}) or {}
        if not isinstance(eval_cfg, dict):
            raise ValueError("method.evaluation must be a mapping")
        self._validation_samples_per_prompt = max(
            1,
            int(eval_cfg.get("samples_per_prompt", 2) or 2),
        )
        self._static_temporal_threshold = max(
            0.0,
            float(eval_cfg.get("static_temporal_l1_threshold", 0.01)),
        )
        self._success_primary_min_delta = max(
            0.0,
            float(eval_cfg.get("primary_min_delta", 0.02)),
        )
        self._success_significance_z = max(
            0.0,
            float(eval_cfg.get("significance_z", 1.96)),
        )
        self._success_min_motion_ratio = max(
            0.0,
            float(eval_cfg.get("min_motion_ratio", 0.90)),
        )
        self._success_min_diversity_ratio = max(
            0.0,
            float(eval_cfg.get("min_latent_diversity_ratio", 0.80)),
        )
        heldout = eval_cfg.get("heldout_max_drop", {}) or {}
        if not isinstance(heldout, dict):
            raise ValueError(
                "method.evaluation.heldout_max_drop must be a mapping"
            )
        self._heldout_max_drop = {
            str(key): max(0.0, float(value))
            for key, value in heldout.items()
        }
        self._validation_items: (
            list[tuple[int, bool, dict[str, Any]]] | None
        ) = None
        self._validation_baseline: dict[str, float] = {}
        self._validation_best_primary_delta = float("-inf")
        self._steps_to_primary_target = -1
        self._cumulative_train_seconds = 0.0

        ema_cfg = mcfg.get("ema", {}) or {}
        if isinstance(ema_cfg, bool):
            ema_cfg = {"enabled": bool(ema_cfg)}
        if not isinstance(ema_cfg, dict):
            raise ValueError("method.ema must be a bool or mapping")
        self._ema_enabled = bool(ema_cfg.get("enabled", True))
        self._ema_decay = float(ema_cfg.get("decay", 0.99))
        self._ema_update_after_step = max(
            0,
            int(ema_cfg.get("update_after_step", 200) or 0),
        )
        self._validation_use_ema = bool(
            ema_cfg.get("validation", True)
        )
        if not 0.0 <= self._ema_decay <= 1.0:
            raise ValueError("method.ema.decay must lie in [0, 1]")
        self._student_ema: EMA_FSDP | None = None
        self._ema_update_count = 0

        self._init_optimizer_and_scheduler()

    # ------------------------------------------------------------------
    # TrainingMethod contract

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
        raise RuntimeError(
            "FiniteTransitionPosteriorMethod uses managed_train_step()"
        )

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        if self._ema_enabled:
            states["finite_transition_posterior.ema"] = (
                _FiniteTransitionEMAState(self)
            )
        return states

    def on_train_start(self) -> None:
        super().on_train_start()
        self._assert_anyflow_two_time_model()
        world_size = self._world_size()
        if self._group_size % world_size != 0:
            raise ValueError(
                "method.group_size must be divisible by the distributed "
                f"world size ({self._group_size} vs {world_size})"
            )
        if self._ema_enabled:
            self._student_ema = EMA_FSDP(
                self.student.transformer,
                decay=self._ema_decay,
                mode="local_shard",
            )
        self._reward_scorer = build_multi_reward_scorer(
            self._reward_fn_config,
            backend=self._reward_backend,
            device=self.student.device,
        )
        self._log_progress(
            "[FiniteTransitionPosterior] initialized "
            f"objective={self._objective} group={self._group_size} "
            f"ESS={self._target_ess_ratio:.2f}"
        )

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del data_stream
        started = time.perf_counter()
        if self._reward_scorer is None or self.cuda_generator is None:
            raise RuntimeError("method was not initialized with on_train_start")

        rank = self._rank()
        world_size = self._world_size()
        local_branches = self._group_size // world_size
        raw_batch = self._sample_prompt_batch(
            iteration=iteration,
            local_branches=local_branches,
        )
        prompts = self._extract_prompts(raw_batch)
        if len(prompts) != local_branches:
            raise RuntimeError(
                f"expected {local_branches} local prompts, got {len(prompts)}"
            )
        if len(set(prompts)) != 1:
            raise RuntimeError(
                "all finite-transition branches must use one shared prompt"
            )

        self.student.transformer.eval()
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
        branch_index = self._select_branch_index(
            iteration=iteration,
            device=self.student.device,
        )
        shared_state = self._shared_initial_noise(
            batch,
            iteration=iteration,
            local_branches=local_branches,
        )

        with torch.no_grad():
            for index in range(branch_index):
                shared_state, _ = self._flow_map(
                    shared_state,
                    schedule[index],
                    schedule[index + 1],
                    batch,
                )
            old_mean, old_std, deterministic_target = self._branch_policy(
                shared_state,
                schedule[branch_index],
                schedule[branch_index + 1],
                batch,
            )
            action_generator = torch.Generator(
                device=self.student.device
            ).manual_seed(
                int(self.training_config.data.seed)
                + int(iteration) * 100_003
                + rank * 1_009
            )
            actions, _ = sample_diagonal_gaussian(
                old_mean,
                old_std,
                generator=action_generator,
            )
            old_log_prob = gaussian_log_prob_mean(
                actions,
                old_mean,
                old_std,
            )
            endpoints = self._complete_from_action(
                actions,
                batch,
                schedule,
                branch_index=branch_index,
            )
            media = self.student.decode_latents(endpoints).detach().cpu()
            local_rewards = self._score_media(media, prompts)
            local_temporal = temporal_l1(media).to(self.student.device)

        global_reward_components = {
            key: self._all_gather_1d(value)
            for key, value in local_rewards.items()
        }
        if self._optimize_reward not in global_reward_components:
            raise RuntimeError(
                f"reward scorer did not return {self._optimize_reward!r}; "
                f"available={sorted(global_reward_components)}"
            )
        global_rewards = global_reward_components[self._optimize_reward]
        global_temporal = self._all_gather_1d(local_temporal)
        advantages, reward_mean, reward_std = group_advantages(
            global_rewards,
            epsilon=self._advantage_epsilon,
            clip=self._advantage_clip,
        )
        global_weights, temperature, ess = reward_tilted_weights(
            global_rewards,
            target_ess_ratio=self._target_ess_ratio,
            bisection_steps=self._reward_bisection_steps,
        )
        local_start = rank * local_branches
        local_end = local_start + local_branches
        local_weights = global_weights[local_start:local_end].to(
            self.student.device
        )
        local_advantages = advantages[local_start:local_end].to(
            self.student.device
        )

        self.student.transformer.train()
        self._student_optimizer.zero_grad(set_to_none=True)
        new_mean, new_std, _ = self._branch_policy(
            shared_state.detach(),
            schedule[branch_index],
            schedule[branch_index + 1],
            batch,
        )
        new_log_prob = gaussian_log_prob_mean(
            actions,
            new_mean,
            new_std,
        )
        if self._objective == "posterior_projection":
            loss, loss_diagnostics = posterior_projection_loss(
                new_log_prob,
                local_weights,
                global_group_size=self._group_size,
                distributed_world_size=world_size,
            )
        else:
            loss, loss_diagnostics = clipped_grpo_loss(
                new_log_prob,
                old_log_prob,
                local_advantages,
                clip_range=self._clip_range,
            )

        self.student.backward(
            loss,
            (batch.timesteps, batch.attn_metadata),
            grad_accum_rounds=1,
        )
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
        if (
            self._post_update_probe_every > 0
            and iteration % self._post_update_probe_every == 0
        ):
            self.student.transformer.eval()
            with torch.no_grad():
                post_mean, post_std, _ = self._branch_policy(
                    shared_state.detach(),
                    schedule[branch_index],
                    schedule[branch_index + 1],
                    batch,
                )
                post_log_prob = gaussian_log_prob_mean(
                    actions,
                    post_mean,
                    post_std,
                )
                delta = post_log_prob - old_log_prob
                post_update_kl = 0.5 * delta.square().mean()
                post_update_logprob_delta = delta.abs().mean()

        step_seconds = time.perf_counter() - started
        self._cumulative_train_seconds += step_seconds
        safe_weights = global_weights.clamp_min(1.0e-12)
        action_deviation = (
            actions.float() - deterministic_target.float()
        ).square().mean().sqrt()
        posterior_deviation = (
            actions.float() - old_mean.float()
        ).square().mean().sqrt()

        metrics: dict[str, LogScalar] = {
            "ftp/objective_is_posterior_projection": float(
                self._objective == "posterior_projection"
            ),
            "ftp/branch_index": float(branch_index),
            "ftp/source_timestep": float(schedule[branch_index]),
            "ftp/target_timestep": float(schedule[branch_index + 1]),
            "ftp/group_size": float(self._group_size),
            "ftp/reward_mean": reward_mean,
            "ftp/reward_std": reward_std,
            "ftp/reward_min": global_rewards.min(),
            "ftp/reward_max": global_rewards.max(),
            "ftp/reward_selection_gain": (
                (global_weights * global_rewards).sum()
                - global_rewards.mean()
            ),
            "ftp/advantage_abs_mean": advantages.abs().mean(),
            "ftp/zero_std_group": float(
                reward_std < self._advantage_epsilon
            ),
            "ftp/posterior_temperature": (
                temperature
                if torch.isfinite(temperature)
                else torch.zeros_like(temperature)
            ),
            "ftp/posterior_temperature_is_infinite": float(
                not bool(torch.isfinite(temperature))
            ),
            "ftp/posterior_ess": ess,
            "ftp/posterior_ess_ratio": ess / float(self._group_size),
            "ftp/posterior_weight_max": global_weights.max(),
            "ftp/posterior_weight_entropy": (
                -(safe_weights * safe_weights.log()).sum()
            ),
            "ftp/posterior_std": old_std.float().mean(),
            "ftp/action_deviation_from_deterministic": (
                self._mean_across_ranks(action_deviation)
            ),
            "ftp/action_deviation_from_posterior_mean": (
                self._mean_across_ranks(posterior_deviation)
            ),
            "ftp/temporal_l1": global_temporal.mean(),
            "ftp/grad_norm": self._mean_across_ranks(
                torch.as_tensor(grad_norm, device=self.student.device)
            ),
            "ftp/post_update_approx_kl": self._mean_across_ranks(
                post_update_kl
            ),
            "ftp/post_update_logprob_delta_abs": (
                self._mean_across_ranks(post_update_logprob_delta)
            ),
            "ftp/ema_update_count": float(self._ema_update_count),
            "ftp/train_step_seconds": float(step_seconds),
            "ftp/cumulative_gpu_hours": (
                self._cumulative_train_seconds * world_size / 3600.0
            ),
        }
        for name, values in global_reward_components.items():
            metrics[f"ftp/reward/{name}"] = values.mean()
            metrics[f"ftp/reward_std/{name}"] = values.std(
                unbiased=False
            )
        for name, value in loss_diagnostics.items():
            metrics[f"ftp/{name}"] = self._mean_across_ranks(value)

        return {"total_loss": loss.detach()}, {}, metrics

    def on_validation_begin(
        self,
        iteration: int = 0,
    ) -> dict[str, LogScalar]:
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}
        if self._reward_scorer is None:
            raise RuntimeError("reward scorer has not been initialized")
        with self._ema_context():
            return self._run_validation(iteration)

    # ------------------------------------------------------------------
    # Finite flow-map primitives

    def _flow_map(
        self,
        latents: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(latents.shape[0])
        source = source_time.reshape(1).to(
            device=latents.device,
            dtype=torch.float32,
        ).expand(batch_size)
        target = target_time.reshape(1).to(
            device=latents.device,
            dtype=torch.float32,
        ).expand(batch_size)
        batch.timesteps = source
        velocity = self.student.predict_velocity_with_r(
            latents,
            source,
            target,
            batch,
            conditional=True,
            attn_kind=self._attn_kind,  # type: ignore[arg-type]
        )
        delta = (source - target) / float(
            self.student.num_train_timesteps
        )
        while delta.ndim < velocity.ndim:
            delta = delta.unsqueeze(-1)
        next_state = latents - delta.to(velocity.dtype) * velocity
        return next_state.to(velocity.dtype), velocity

    def _branch_policy(
        self,
        shared_state: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deterministic_target, _ = self._flow_map(
            shared_state,
            source_time,
            target_time,
            batch,
        )
        clean_endpoint, _ = self._flow_map(
            deterministic_target,
            target_time,
            target_time.new_zeros(()),
            batch,
        )
        mean, std = endpoint_anchor_parameters(
            clean_endpoint,
            target_time,
            num_train_timesteps=self.student.num_train_timesteps,
        )
        return mean, std, deterministic_target

    def _complete_from_action(
        self,
        action: torch.Tensor,
        batch: TrainingBatch,
        schedule: torch.Tensor,
        *,
        branch_index: int,
    ) -> torch.Tensor:
        current = action
        for index in range(branch_index + 1, schedule.numel() - 1):
            current, _ = self._flow_map(
                current,
                schedule[index],
                schedule[index + 1],
                batch,
            )
        return current

    def _deterministic_rollout(
        self,
        initial: torch.Tensor,
        batch: TrainingBatch,
        schedule: torch.Tensor,
    ) -> torch.Tensor:
        current = initial
        for index in range(schedule.numel() - 1):
            current, _ = self._flow_map(
                current,
                schedule[index],
                schedule[index + 1],
                batch,
            )
        return current

    def _build_schedule(
        self,
        *,
        steps: int,
        override: list[float] | None,
        device: torch.device,
    ) -> torch.Tensor:
        if override is not None:
            schedule = torch.tensor(
                override,
                device=device,
                dtype=torch.float32,
            )
            if float(schedule[-1]) != 0.0:
                schedule = append_data_endpoint(schedule)
            if schedule.numel() != steps + 1:
                raise ValueError(
                    f"schedule override for {steps} steps must have "
                    f"{steps + 1} nodes, got {schedule.numel()}"
                )
            if not torch.all(schedule[:-1] > schedule[1:]):
                raise ValueError("schedule override must be descending")
            return schedule

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.student.num_train_timesteps,
            shift=self._flow_shift,
        )
        scheduler.set_timesteps(
            num_inference_steps=steps,
            device=device,
        )
        return append_data_endpoint(
            scheduler.timesteps.to(device=device, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Validation and success criteria

    @torch.no_grad()
    def _run_validation(self, iteration: int) -> dict[str, LogScalar]:
        self.student.transformer.eval()
        config = self._validation_config
        schedule = self._build_schedule(
            steps=self._eval_map_steps,
            override=self._eval_schedule_override,
            device=self.student.device,
        )
        items = self._get_validation_items()
        local_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        local_masks: list[torch.Tensor] = []
        local_logs: list[dict[str, Any]] = []

        for start in range(0, len(items), config.batch_size):
            batch_items = items[start:start + config.batch_size]
            repeated_rows: list[dict[str, Any]] = []
            expanded_meta: list[tuple[int, bool]] = []
            for global_index, valid, row in batch_items:
                for _ in range(self._validation_samples_per_prompt):
                    repeated_rows.append(copy.deepcopy(row))
                    expanded_meta.append((global_index, valid))
            raw_batch = self._collate_rows(repeated_rows)
            prepare_generator = torch.Generator(
                device=self.student.device
            ).manual_seed(config.seed + 100_000 + self._rank() + start)
            batch = self.student.prepare_batch(
                raw_batch,
                generator=prepare_generator,
                latents_source="zeros",
                num_latent_t=config.num_latent_t,
            )
            prompts = self._extract_prompts(raw_batch)
            noises = []
            if batch.latents is None:
                raise RuntimeError("validation batch is missing latent shape")
            for sample_index, (global_index, _valid) in enumerate(expanded_meta):
                seed = (
                    int(config.seed)
                    + int(global_index) * 10_000
                    + sample_index % self._validation_samples_per_prompt
                )
                generator = torch.Generator(
                    device=self.student.device
                ).manual_seed(seed)
                noises.append(
                    torch.randn(
                        (1, *batch.latents.shape[1:]),
                        device=self.student.device,
                        dtype=batch.latents.dtype,
                        generator=generator,
                    )
                )
            initial = torch.cat(noises, dim=0)
            endpoint = self._deterministic_rollout(
                initial,
                batch,
                schedule,
            )
            media = self.student.decode_latents(endpoint).detach().cpu()
            rewards = self._score_media(media, prompts)
            motion = temporal_l1(media)

            item_count = len(batch_items)
            samples_per_prompt = self._validation_samples_per_prompt
            valid_mask = torch.tensor(
                [bool(item[1]) for item in batch_items],
                device=self.student.device,
                dtype=torch.bool,
            )
            local_masks.append(valid_mask)
            for name, value in rewards.items():
                grouped = value.reshape(item_count, samples_per_prompt)
                local_metrics[f"reward/{name}"].append(grouped.mean(dim=1))
            local_metrics["temporal_l1"].append(
                motion.to(self.student.device)
                .reshape(item_count, samples_per_prompt)
                .mean(dim=1)
            )
            static = (
                motion.to(self.student.device)
                < self._static_temporal_threshold
            ).float().reshape(item_count, samples_per_prompt).mean(dim=1)
            local_metrics["static_sample_ratio"].append(static)

            latent_diversity = []
            video_diversity = []
            for item_index in range(item_count):
                lo = item_index * samples_per_prompt
                hi = lo + samples_per_prompt
                latent_diversity.append(mean_pairwise_rms(endpoint[lo:hi]))
                video_diversity.append(mean_pairwise_rms(media[lo:hi]))
            local_metrics["latent_diversity_rms"].append(
                torch.stack(latent_diversity).to(self.student.device)
            )
            local_metrics["video_diversity_rms"].append(
                torch.stack(video_diversity).to(self.student.device)
            )

            if config.log_samples:
                for item_index, (global_index, valid, _row) in enumerate(
                    batch_items
                ):
                    if not valid:
                        continue
                    sample_offset = item_index * samples_per_prompt
                    sample_rewards = {
                        name: float(value[sample_offset])
                        for name, value in rewards.items()
                    }
                    sample_rewards["temporal_l1"] = float(
                        motion[sample_offset]
                    )
                    log_entry = _prepare_validation_log_entry(
                        index=int(global_index),
                        prompt=prompts[sample_offset],
                        media=media[sample_offset],
                        rewards=sample_rewards,
                        max_samples=config.max_samples,
                    )
                    if log_entry is not None:
                        local_logs.append(log_entry)

        if not local_masks:
            return {}
        gathered_mask = self._all_gather_1d(
            torch.cat(local_masks).float()
        ).bool()
        summary: dict[str, float] = {}
        metrics: dict[str, LogScalar] = {}
        for name, chunks in local_metrics.items():
            local_values = torch.cat(chunks).to(self.student.device)
            gathered = self._all_gather_1d(local_values.float())
            values = gathered[gathered_mask]
            if values.numel() == 0:
                continue
            mean = values.mean()
            std = values.std(unbiased=False)
            sem = std / math.sqrt(float(values.numel()))
            metrics[f"validation/{name}"] = mean
            metrics[f"validation_std/{name}"] = std
            metrics[f"validation_sem/{name}"] = sem
            summary[name] = float(mean)
            summary[f"sem/{name}"] = float(sem)
        metrics["validation/num_prompts"] = float(gathered_mask.sum())
        metrics["validation/samples_per_prompt"] = float(
            self._validation_samples_per_prompt
        )
        metrics.update(self._validation_delta_metrics(summary, iteration))

        if config.log_samples:
            self._log_validation_samples(local_logs, iteration)
        return metrics

    def _validation_delta_metrics(
        self,
        summary: dict[str, float],
        iteration: int,
    ) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}
        if not summary:
            return metrics
        if not self._validation_baseline:
            self._validation_baseline = dict(summary)
            for name, value in summary.items():
                metrics[f"validation_baseline/{name}"] = float(value)
            self._validation_best_primary_delta = 0.0
            return metrics

        for name, current in summary.items():
            if name.startswith("sem/"):
                continue
            baseline = self._validation_baseline.get(name)
            if baseline is not None:
                metrics[f"validation_delta/{name}"] = current - baseline

        primary_name = f"reward/{self._optimize_reward}"
        primary_current = summary.get(primary_name)
        primary_base = self._validation_baseline.get(primary_name)
        if primary_current is None or primary_base is None:
            return metrics
        primary_delta = primary_current - primary_base
        current_sem = summary.get(f"sem/{primary_name}", 0.0)
        base_sem = self._validation_baseline.get(
            f"sem/{primary_name}",
            0.0,
        )
        significance_margin = self._success_significance_z * math.sqrt(
            current_sem**2 + base_sem**2
        )
        primary_success = (
            primary_delta >= self._success_primary_min_delta
            and primary_delta > significance_margin
        )
        if (
            primary_success
            and self._steps_to_primary_target < 0
        ):
            self._steps_to_primary_target = int(iteration)
        self._validation_best_primary_delta = max(
            self._validation_best_primary_delta,
            primary_delta,
        )

        motion_current = summary.get("temporal_l1", 0.0)
        motion_base = self._validation_baseline.get("temporal_l1", 0.0)
        motion_ratio = motion_current / max(abs(motion_base), 1.0e-12)
        diversity_current = summary.get("latent_diversity_rms", 0.0)
        diversity_base = self._validation_baseline.get(
            "latent_diversity_rms",
            0.0,
        )
        diversity_ratio = diversity_current / max(
            abs(diversity_base),
            1.0e-12,
        )
        motion_success = motion_ratio >= self._success_min_motion_ratio
        diversity_success = (
            diversity_ratio >= self._success_min_diversity_ratio
        )

        heldout_success = True
        for reward_name, max_drop in self._heldout_max_drop.items():
            key = f"reward/{reward_name}"
            current = summary.get(key)
            baseline = self._validation_baseline.get(key)
            if current is None or baseline is None:
                heldout_success = False
                metrics[
                    f"validation_success/heldout_{reward_name}"
                ] = 0.0
                continue
            retained = current >= baseline - max_drop
            heldout_success = heldout_success and retained
            metrics[
                f"validation_success/heldout_{reward_name}"
            ] = float(retained)
            metrics[
                f"validation_delta/reward/{reward_name}"
            ] = current - baseline

        gpu_hours = (
            self._cumulative_train_seconds * self._world_size() / 3600.0
        )
        metrics.update(
            {
                "validation/primary_delta": primary_delta,
                "validation/primary_significance_margin": significance_margin,
                "validation/primary_best_delta": (
                    self._validation_best_primary_delta
                ),
                "validation/steps_to_primary_target": float(
                    self._steps_to_primary_target
                ),
                "validation/motion_ratio_to_base": motion_ratio,
                "validation/latent_diversity_ratio_to_base": (
                    diversity_ratio
                ),
                "validation/cumulative_gpu_hours": gpu_hours,
                "validation/primary_gain_per_gpu_hour": (
                    primary_delta / max(gpu_hours, 1.0e-12)
                ),
                "validation/primary_gain_per_100_steps": (
                    primary_delta * 100.0 / max(float(iteration), 1.0)
                ),
                "validation_success/primary_reward": float(primary_success),
                "validation_success/motion_retained": float(motion_success),
                "validation_success/diversity_retained": float(
                    diversity_success
                ),
                "validation_success/heldout_retained": float(
                    heldout_success
                ),
                "validation_success/all": float(
                    primary_success
                    and motion_success
                    and diversity_success
                    and heldout_success
                ),
            }
        )
        return metrics

    def _log_validation_samples(
        self,
        local_logs: list[dict[str, Any]],
        iteration: int,
    ) -> None:
        logs = self._gather_objects(local_logs)
        if self._rank() != 0 or not logs or self.tracker is None:
            return
        logs = sorted(logs, key=lambda item: int(item["index"]))
        max_samples = self._validation_config.max_samples
        if max_samples is not None:
            logs = logs[:max_samples]
        artifacts = []
        for item in logs:
            media = item["media"]
            if isinstance(media, torch.Tensor):
                # Backwards compatibility for callers and old checkpoints
                # that still provide decoded tensors here.
                media = media_to_video_array(media)
            artifact = self.tracker.video(
                media,
                caption=validation_caption(
                    str(item["prompt"]),
                    item["rewards"],
                ),
                fps=int(self._validation_config.fps),
            )
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            self.tracker.log_artifacts(
                {"validation/videos": artifacts},
                step=iteration,
            )

    # ------------------------------------------------------------------
    # Dataset, reward and distributed helpers

    def _sample_prompt_batch(
        self,
        *,
        iteration: int,
        local_branches: int,
    ) -> dict[str, Any]:
        dataset = getattr(
            getattr(self.student, "dataloader", None),
            "dataset",
            None,
        )
        if dataset is None or not all(
            hasattr(dataset, attr)
            for attr in ("parquet_files", "lengths", "parquet_schema")
        ):
            raise RuntimeError(
                "finite-transition posterior training requires a parquet-backed "
                "text dataset"
            )
        total_rows = int(sum(dataset.lengths))
        sample = distributed_k_repeat_indices(
            dataset_length=total_rows,
            batch_size=local_branches,
            repeats_per_prompt=self._group_size,
            world_size=self._world_size(),
            rank=self._rank(),
            seed=int(self.training_config.data.seed) + int(iteration),
        )
        rows = []
        for prompt_index in sample.local_indices:
            row = read_row_from_parquet_file(
                dataset.parquet_files,
                prompt_index,
                dataset.lengths,
            )
            row["_sample_index"] = prompt_index
            rows.append(row)
        return self._collate_rows(rows)

    def _get_validation_items(
        self,
    ) -> list[tuple[int, bool, dict[str, Any]]]:
        if self._validation_items is not None:
            return self._validation_items
        dataset = getattr(
            getattr(self.student, "dataloader", None),
            "dataset",
            None,
        )
        if dataset is None:
            raise RuntimeError("validation requires a student dataset")
        data_path = (
            self._validation_config.data_path
            or getattr(dataset, "path", None)
            or self.training_config.data.data_path
        )
        if (
            self._validation_config.data_path is None
            or data_path == getattr(dataset, "path", None)
        ):
            parquet_files = list(dataset.parquet_files)
            lengths = list(dataset.lengths)
        else:
            parquet_files, lengths = get_parquet_files_and_length(data_path)
            parquet_files = list(parquet_files)
            lengths = list(lengths)
        total_rows = int(sum(lengths))
        if total_rows <= 0:
            raise RuntimeError(f"validation path {data_path!r} has no rows")
        num_prompts = min(self._validation_config.num_prompts, total_rows)
        items: list[tuple[int, bool, dict[str, Any]]] = []
        for prompt_index, valid in validation_shard_indices(
            num_prompts,
            rank=self._rank(),
            world_size=self._world_size(),
        ):
            row = read_row_from_parquet_file(
                parquet_files,
                prompt_index,
                lengths,
            )
            row["_sample_index"] = prompt_index
            items.append((prompt_index, valid, row))
        self._validation_items = items
        return items

    def _collate_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = getattr(
            getattr(self.student, "dataloader", None),
            "dataset",
            None,
        )
        if dataset is None or not hasattr(dataset, "parquet_schema"):
            raise RuntimeError("student dataset has no parquet schema")
        return collate_rows_from_parquet_schema(
            rows,
            dataset.parquet_schema,
            int(getattr(dataset, "text_padding_length", 512)),
            cfg_rate=0.0,
            seed=int(self.training_config.data.seed),
        )

    @staticmethod
    def _extract_prompts(raw_batch: dict[str, Any]) -> list[str]:
        infos = raw_batch.get("info_list")
        if isinstance(infos, list) and infos:
            return [
                str(info.get("prompt") or info.get("caption") or "")
                if isinstance(info, dict)
                else ""
                for info in infos
            ]
        captions = raw_batch.get("caption_text")
        if isinstance(captions, list):
            return [str(caption) for caption in captions]
        raise ValueError("could not find prompts in batch")

    def _score_media(
        self,
        media: torch.Tensor,
        prompts: list[str],
    ) -> dict[str, torch.Tensor]:
        if self._reward_scorer is None:
            raise RuntimeError("reward scorer has not been initialized")
        rewards = self._reward_scorer(media, prompts)
        return {
            key: value.to(
                device=self.student.device,
                dtype=torch.float32,
            ).reshape(-1)
            for key, value in rewards.items()
        }

    def _shared_initial_noise(
        self,
        batch: TrainingBatch,
        *,
        iteration: int,
        local_branches: int,
    ) -> torch.Tensor:
        if batch.latents is None:
            raise RuntimeError("TrainingBatch.latents is required")
        generator = torch.Generator(
            device=self.student.device
        ).manual_seed(
            int(self.training_config.data.seed)
            + 10_000_000
            + int(iteration)
        )
        one = torch.randn(
            (1, *batch.latents.shape[1:]),
            device=batch.latents.device,
            dtype=batch.latents.dtype,
            generator=generator,
        )
        return one.expand(local_branches, *one.shape[1:]).clone()

    def _select_branch_index(
        self,
        *,
        iteration: int,
        device: torch.device,
    ) -> int:
        if self._rank() == 0:
            generator = torch.Generator(device=device).manual_seed(
                int(self.training_config.data.seed) + int(iteration)
            )
            selected = torch.randint(
                0,
                self._stochastic_steps,
                (1,),
                generator=generator,
                device=device,
                dtype=torch.long,
            )
        else:
            selected = torch.zeros(1, device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(selected, src=0)
        return int(selected.item())

    def _all_gather_1d(self, local: torch.Tensor) -> torch.Tensor:
        local = local.contiguous()
        if not dist.is_available() or not dist.is_initialized():
            return local.detach()
        gathered = [torch.empty_like(local) for _ in range(self._world_size())]
        dist.all_gather(gathered, local)
        return torch.cat(gathered, dim=0).detach()

    def _mean_across_ranks(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= float(self._world_size())
        return reduced

    def _gather_objects(self, items: list[Any]) -> list[Any]:
        if not dist.is_available() or not dist.is_initialized():
            return list(items)
        gathered: list[list[Any] | None] = [
            None for _ in range(self._world_size())
        ]
        dist.all_gather_object(gathered, list(items))
        flat: list[Any] = []
        for rank_items in gathered:
            if rank_items is not None:
                flat.extend(rank_items)
        return flat

    # ------------------------------------------------------------------
    # Optimizer, EMA and config helpers

    def _init_optimizer_and_scheduler(self) -> None:
        params = [
            parameter
            for parameter in self.student.transformer.parameters()
            if parameter.requires_grad
        ]
        if not params:
            raise ValueError("student has no trainable parameters")
        self._student_optimizer, self._student_lr_scheduler = (
            build_optimizer_and_scheduler(
                params=params,
                optimizer_config=self.training_config.optimizer,
                loop_config=self.training_config.loop,
                learning_rate=float(
                    self.training_config.optimizer.learning_rate
                ),
                betas=self.training_config.optimizer.betas,
                scheduler_name=str(
                    self.training_config.optimizer.lr_scheduler
                ),
            )
        )

    def _clip_student_grads(self) -> torch.Tensor:
        parameters = [
            parameter
            for parameter in self.student.transformer.parameters()
            if parameter.requires_grad
        ]
        if self._max_grad_norm <= 0.0:
            norms = [
                parameter.grad.detach().float().norm()
                for parameter in parameters
                if parameter.grad is not None
            ]
            return (
                torch.stack(norms).norm()
                if norms
                else torch.zeros((), device=self.student.device)
            )
        clipped = clip_grad_norm_while_handling_failing_dtensor_cases(
            parameters,
            self._max_grad_norm,
            foreach=None,
        )
        if clipped is not None:
            return clipped
        norms = [
            parameter.grad.detach().float().norm()
            for parameter in parameters
            if parameter.grad is not None
        ]
        return (
            torch.stack(norms).norm()
            if norms
            else torch.zeros((), device=self.student.device)
        )

    def _update_ema(self) -> None:
        if not self._ema_enabled or self._student_ema is None:
            return
        if self._ema_update_count >= self._ema_update_after_step:
            self._student_ema.update(self.student.transformer)
        self._ema_update_count += 1

    @contextlib.contextmanager
    def _ema_context(self) -> Iterator[None]:
        if self._validation_use_ema and self._student_ema is not None:
            with self._student_ema.apply_to_model(self.student.transformer):
                yield
            return
        yield

    def _assert_anyflow_two_time_model(self) -> None:
        flags = [
            bool(module._r_embedder_enabled)
            for module in self.student.transformer.modules()
            if hasattr(module, "_r_embedder_enabled")
        ]
        if not flags or not any(flags):
            raise RuntimeError(
                "FiniteTransitionPosteriorMethod requires an AnyFlow two-time "
                "Wan model. Set pipeline.dit_config.r_embedder=true and load "
                "an AnyFlow checkpoint."
            )

    def _parse_schedule_override(self, key: str) -> list[float] | None:
        raw = self.method_config.get(key, None)
        if raw is None:
            return None
        if not isinstance(raw, list) or len(raw) < 2:
            raise ValueError(f"method.{key} must be a list with >=2 entries")
        values = [float(value) for value in raw]
        if any(values[index] <= values[index + 1] for index in range(len(values) - 1)):
            raise ValueError(f"method.{key} must be strictly descending")
        return values

    def _read_positive_int(self, key: str, default: int) -> int:
        value = int(self.method_config.get(key, default) or default)
        if value <= 0:
            raise ValueError(f"method.{key} must be positive")
        return value

    def _read_positive_float(self, key: str, default: float) -> float:
        value = float(self.method_config.get(key, default) or default)
        if value <= 0.0:
            raise ValueError(f"method.{key} must be positive")
        return value

    def _read_unit_float(
        self,
        key: str,
        default: float,
        *,
        lower_open: bool,
    ) -> float:
        value = float(self.method_config.get(key, default))
        lower_ok = value > 0.0 if lower_open else value >= 0.0
        if not lower_ok or value > 1.0:
            bracket = "(0, 1]" if lower_open else "[0, 1]"
            raise ValueError(f"method.{key} must lie in {bracket}")
        return value

    def _log_progress(self, message: str) -> None:
        if self._terminal_progress and self._rank() == 0:
            logger.info(message)

    @staticmethod
    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0

    @staticmethod
    def _world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
        return 1


__all__ = ["FiniteTransitionPosteriorMethod"]
