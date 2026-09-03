# SPDX-License-Identifier: Apache-2.0
"""Reward-filtered Piecewise Rectified Flow from full H3 to FastH3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.knowledge_distillation.h3_perflow_utils import (
    LossType,
    compute_h3_perflow_losses,
    interpolate_sigma_segment,
    sample_segment_timestep,
)
from fastvideo.train.models.minimax_h3.minimax_h3_perflow import (
    MiniMaxH3PeRFlowModel,
)
from fastvideo.train.utils.lora_context import temporarily_disable_lora
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

logger = init_logger(__name__)


class H3PeRFlowMethod(TrainingMethod):
    """Supervised continuous-window velocity matching on selected H3 paths.

    Offline rewards choose the deterministic top-q teacher trajectories. They
    never become signed coefficients in this loss. Every selected path receives
    equal positive mass and RVM remains a separate on-policy second stage.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if set(role_models) != {"student"}:
            raise ValueError(
                "H3PeRFlowMethod requires exactly one role: models.student"
            )
        if not isinstance(self.student, MiniMaxH3PeRFlowModel):
            raise TypeError(
                "H3PeRFlowMethod requires "
                "models.student._target_=MiniMaxH3PeRFlowModel"
            )
        if not self.student._trainable:
            raise ValueError("H3PeRFlowMethod requires a trainable student")

        self.student.init_preprocessors(self.training_config)
        self._attention_kind = self._read_attention_kind()
        self._loss_type = self._read_loss_type()
        self._huber_delta = self._read_positive_float("huber_delta", 1.0)
        self._audio_loss_weight = self._read_nonnegative_float(
            "audio_loss_weight", 1.0
        )
        self._function_anchor_weight = self._read_nonnegative_float(
            "function_anchor_weight", 0.0
        )
        self._sigma_eps = self._read_positive_float("sigma_eps", 1e-8)
        self._interpolation_tolerance = self._read_nonnegative_float(
            "interpolation_tolerance", 1e-5
        )
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._lr_scheduler}

    def on_train_start(self) -> None:
        super().on_train_start()
        summary = self.student.perflow_selection_summary
        logger.info(
            "H3 PeRFlow initialized: cache=%s selection=%s ranking=%s "
            "K=%d q=%d prompts=%d trajectories=%d segments=%d "
            "loss=%s audio_weight=%s anchor_weight=%s grid=%s",
            self.student.rest_cache_fingerprint,
            summary.fingerprint,
            summary.ranking_key,
            summary.samples_per_prompt,
            summary.selected_per_prompt,
            summary.num_prompts,
            summary.num_trajectories,
            summary.num_segments,
            self._loss_type,
            self._audio_loss_weight,
            self._function_anchor_weight,
            list(self.student.rest_student_timesteps),
        )

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        self._validate_selection_fingerprint(batch)
        generator = self._require_generator()
        prepared = self.student.prepare_batch(
            batch,
            generator=generator,
            latents_source="zeros",
        )
        if prepared.latents is None:
            raise RuntimeError(
                "H3 PeRFlow batch preparation returned no packed geometry"
            )

        model_dtype = prepared.latents.dtype
        current = self._batch_tensor(
            batch,
            "trajectory_current",
            dtype=model_dtype,
        )
        next_state = self._batch_tensor(
            batch,
            "trajectory_next",
            dtype=model_dtype,
        )
        timestep = self._batch_tensor(
            batch,
            "trajectory_timestep",
            dtype=torch.float32,
        ).reshape(-1)
        next_timestep = self._batch_tensor(
            batch,
            "trajectory_next_timestep",
            dtype=torch.float32,
        ).reshape(-1)
        selection_weight = self._batch_tensor(
            batch,
            "perflow_selection_weight",
            dtype=torch.float32,
        ).reshape(-1)
        selection_rank = self._batch_tensor(
            batch,
            "perflow_selection_rank",
            dtype=torch.long,
        ).reshape(-1)
        selection_score = self._batch_tensor(
            batch,
            "perflow_selection_score",
            dtype=torch.float32,
        ).reshape(-1)
        segment_index_tensor = self._batch_tensor(
            batch,
            "rest_segment_index",
            dtype=torch.long,
        ).reshape(-1)
        if segment_index_tensor.numel() != 1:
            raise ValueError("H3 PeRFlow requires exactly one segment per rank")
        segment_index = int(segment_index_tensor.item())
        self._validate_segment_contract(
            current,
            next_state,
            timestep,
            next_timestep,
            segment_index,
        )

        query_timestep, shared_fraction = sample_segment_timestep(
            timestep,
            next_timestep,
            batch_size=current.shape[0],
            device=self.student.device,
            generator=generator,
        )
        video_slice, audio_slice = self._modality_slices()
        sigma_video, sigma_audio = self.student.noise_amounts(timestep)
        next_sigma_video, next_sigma_audio = self.student.noise_amounts(
            next_timestep
        )
        query_sigma_video, query_sigma_audio = self.student.noise_amounts(
            query_timestep
        )
        video_segment = interpolate_sigma_segment(
            current[:, video_slice],
            next_state[:, video_slice],
            sigma_current=sigma_video,
            sigma_next=next_sigma_video,
            sigma_query=query_sigma_video,
            eps=self._sigma_eps,
            tolerance=self._interpolation_tolerance,
        )
        audio_segment = interpolate_sigma_segment(
            current[:, audio_slice],
            next_state[:, audio_slice],
            sigma_current=sigma_audio,
            sigma_next=next_sigma_audio,
            sigma_query=query_sigma_audio,
            eps=self._sigma_eps,
            tolerance=self._interpolation_tolerance,
        )

        query_state = current.detach().float().clone()
        query_state[:, video_slice] = video_segment.state
        query_state[:, audio_slice] = audio_segment.state
        query_state = query_state.to(dtype=model_dtype)

        self.student.refresh_vsa_metadata(
            prepared,
            current_timestep=segment_index,
        )
        prediction = self.student.predict_noise(
            query_state,
            query_timestep,
            prepared,
            conditional=True,
            attn_kind=self._attention_kind,
        )
        reference_prediction: torch.Tensor | None = None
        if self._function_anchor_weight != 0.0:
            with temporarily_disable_lora(
                self.student.transformer
            ), torch.no_grad():
                reference_prediction = self.student.predict_noise(
                    query_state,
                    query_timestep,
                    prepared,
                    conditional=True,
                    attn_kind=self._attention_kind,
                ).detach()

        backward_metadata = (
            prepared.attn_metadata_vsa
            if self._attention_kind == "vsa"
            else prepared.attn_metadata
        )
        backward_context = (prepared.timesteps, backward_metadata)
        losses = compute_h3_perflow_losses(
            prediction,
            video_target=video_segment.velocity_target,
            audio_target=audio_segment.velocity_target,
            video_slice=video_slice,
            audio_slice=audio_slice,
            sample_weight=selection_weight,
            audio_loss_weight=self._audio_loss_weight,
            loss_type=self._loss_type,
            huber_delta=self._huber_delta,
            reference_prediction=reference_prediction,
            anchor_weight=self._function_anchor_weight,
        )

        mixed_advantage = self._batch_tensor(
            batch,
            "rest_mixed_advantage",
            dtype=torch.float32,
        ).reshape(-1)
        metrics: dict[str, LogScalar] = {
            "perflow/segment_index": float(segment_index),
            "perflow/query_timestep": query_timestep.mean(),
            "perflow/shared_fraction": shared_fraction.mean(),
            "perflow/video_sigma": query_sigma_video.float().mean(),
            "perflow/audio_sigma": query_sigma_audio.float().mean(),
            "perflow/video_fraction": (
                video_segment.interpolation_fraction.mean()
            ),
            "perflow/audio_fraction": (
                audio_segment.interpolation_fraction.mean()
            ),
            "perflow/video_target_rms": (
                video_segment.velocity_target.square().mean().sqrt()
            ),
            "perflow/audio_target_rms": (
                audio_segment.velocity_target.square().mean().sqrt()
            ),
            "perflow/selection_rank": selection_rank.float().mean(),
            "perflow/selection_score": selection_score.mean(),
            "perflow/selection_weight": selection_weight.mean(),
            "perflow/mixed_advantage_diagnostic": mixed_advantage.mean(),
            "perflow/cache_examples": float(
                self.student.perflow_selection_summary.num_examples
            ),
            "perflow/selected_trajectories": float(
                self.student.perflow_selection_summary.num_trajectories
            ),
        }
        reward_scores = batch.get("rest_reward_scores")
        if isinstance(reward_scores, Mapping):
            for name, value in reward_scores.items():
                metrics[f"perflow/reward/{name}"] = float(value)
        return losses, {"student_ctx": backward_context}, metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        student_ctx = outputs.get("student_ctx")
        if student_ctx is None:
            raise RuntimeError(
                "H3 PeRFlow backward is missing the student forward context"
            )
        self.student.backward(
            loss_map["total_loss"],
            student_ctx,
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )

    def get_optimizers(
        self,
        iteration: int,
    ) -> Sequence[torch.optim.Optimizer]:
        del iteration
        return (self._optimizer,)

    def get_lr_schedulers(self, iteration: int) -> Sequence[Any]:
        del iteration
        return (self._lr_scheduler,)

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        del iteration
        return {"student": self.student.transformer}

    def apply_configured_lrs(self) -> None:
        learning_rate = float(self.training_config.optimizer.learning_rate)
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if "initial_lr" in group:
                group["initial_lr"] = learning_rate
        if hasattr(self._lr_scheduler, "base_lrs"):
            self._lr_scheduler.base_lrs = [
                learning_rate for _ in self._lr_scheduler.base_lrs
            ]

    def _init_optimizer_and_scheduler(self) -> None:
        parameters = [
            parameter
            for parameter in self.student.transformer.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("H3 PeRFlow student has no trainable parameters")
        self._optimizer, self._lr_scheduler = build_optimizer_and_scheduler(
            params=parameters,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=float(
                self.training_config.optimizer.learning_rate
            ),
            betas=tuple(self.training_config.optimizer.betas),
            scheduler_name=str(
                self.training_config.optimizer.lr_scheduler
            ),
        )

    def _validate_selection_fingerprint(
        self,
        batch: Mapping[str, Any],
    ) -> None:
        observed = batch.get("perflow_selection_fingerprint")
        expected = self.student.perflow_selection_fingerprint
        if observed != expected:
            raise ValueError(
                "H3 PeRFlow selection fingerprint mismatch: "
                f"batch={observed!r}, dataset={expected!r}"
            )

    def _validate_segment_contract(
        self,
        current: torch.Tensor,
        next_state: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        segment_index: int,
    ) -> None:
        if current.shape != next_state.shape or current.ndim != 2:
            raise ValueError(
                "Cached PeRFlow states must share packed shape [B, N], got "
                f"{tuple(current.shape)} and {tuple(next_state.shape)}"
            )
        if current.shape[0] != 1:
            raise ValueError(
                "H3 PeRFlow currently requires batch size one per SP group"
            )
        grid = self.student.rest_student_timesteps
        if not 0 <= segment_index < len(grid) - 1:
            raise ValueError(
                "PeRFlow segment_index out of range: "
                f"{segment_index} for grid {grid}"
            )
        if timestep.numel() != 1 or next_timestep.numel() != 1:
            raise ValueError(
                "H3 PeRFlow requires one shared timestep per SP group"
            )
        observed = (
            float(timestep.item()),
            float(next_timestep.item()),
        )
        expected = (
            float(grid[segment_index]),
            float(grid[segment_index + 1]),
        )
        if observed != expected:
            raise ValueError(
                "Cached PeRFlow segment timestep mismatch: "
                f"segment={segment_index}, observed={observed}, "
                f"expected={expected}"
            )
        self.student.unpack_latents(current)
        self.student.unpack_latents(next_state)

    def _modality_slices(self) -> tuple[slice, slice]:
        slices = dict(self.student.modality_slices())
        if "video" not in slices or "audio" not in slices:
            raise RuntimeError(
                "H3 PeRFlow requires video/audio slices, got "
                f"{sorted(slices)}"
            )
        return slices["video"], slices["audio"]

    def _batch_tensor(
        self,
        batch: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise ValueError(
                f"H3 PeRFlow cache batch is missing tensor {key!r}"
            )
        tensor = value.to(
            device=self.student.device,
            dtype=dtype,
            non_blocking=True,
        )
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(
                f"H3 PeRFlow cache tensor {key!r} contains NaN or Inf"
            )
        return tensor

    def _require_generator(self) -> torch.Generator:
        if self.cuda_generator is None:
            raise RuntimeError(
                "H3 PeRFlow CUDA generator is not initialized"
            )
        return self.cuda_generator

    def _read_attention_kind(self) -> Literal["dense", "vsa"]:
        value = str(
            self.method_config.get("attn_kind", "vsa")
        ).strip().lower()
        if value not in {"dense", "vsa"}:
            raise ValueError(
                "method.attn_kind must be one of {dense, vsa}"
            )
        return value  # type: ignore[return-value]

    def _read_loss_type(self) -> LossType:
        value = str(
            self.method_config.get("loss_type", "mse")
        ).strip().lower()
        if value not in {"mse", "huber"}:
            raise ValueError(
                "method.loss_type must be one of {mse, huber}"
            )
        return value  # type: ignore[return-value]

    def _read_float(self, key: str, default: float) -> float:
        raw = self.method_config.get(key, default)
        value = float(default if raw is None else raw)
        if not math.isfinite(value):
            raise ValueError(f"method.{key} must be finite")
        return value

    def _read_nonnegative_float(
        self,
        key: str,
        default: float,
    ) -> float:
        value = self._read_float(key, default)
        if value < 0.0:
            raise ValueError(f"method.{key} must be nonnegative")
        return value

    def _read_positive_float(
        self,
        key: str,
        default: float,
    ) -> float:
        value = self._read_float(key, default)
        if value <= 0.0:
            raise ValueError(f"method.{key} must be positive")
        return value


__all__ = ["H3PeRFlowMethod"]
