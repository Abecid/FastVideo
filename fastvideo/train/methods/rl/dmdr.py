# SPDX-License-Identifier: Apache-2.0
"""DMDR: joint distribution matching distillation and reward optimization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.rl.diffusion_nft import (
    _DiffusionNFTEMAState,
    _FullModelState,
    DiffusionNFTMethod,
)
from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import get_optional_float, parse_betas
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.training.training_utils import (
    EMA_FSDP,
    clip_grad_norm_while_handling_failing_dtensor_cases,
)


@dataclass(slots=True)
class _DMDRTimestepSamplingConfig:
    """Reference-DMDR timestep sampler parameters.

    The public SiT code samples continuous DMD/guidance timesteps from a beta
    distribution whose alpha/beta anneal to 1.0 with cosine decay over
    ``dynamic_step``. ``kind='uniform'`` matches the cold-start generator
    update schedule; ``kind='logit_normal'`` keeps the reference name even
    though the released code samples from ``torch.distributions.Beta``.
    """

    kind: str = "logit_normal"
    alpha: float = 4.0
    beta: float = 1.5
    discrete: bool = False
    min_t: float = 0.001
    max_t: float = 1.0


class DMDRMethod(DiffusionNFTMethod):
    """DMDR joint DMD + reward optimization for few-step distillation.

    The method follows the public DMDR SiT loop at the algorithmic level while
    using FastVideo-native Wan model wrappers and reusable RL reward scorers.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "teacher" not in role_models:
            raise ValueError("DMDRMethod requires role 'teacher'")
        if "critic" not in role_models:
            raise ValueError("DMDRMethod requires role 'critic'")

        self.teacher = role_models["teacher"]
        self.critic = role_models["critic"]
        if self.teacher._trainable:
            raise ValueError("DMDRMethod requires teacher to be non-trainable")
        if not self.critic._trainable:
            raise ValueError("DMDRMethod requires critic to be trainable")

        self._rl_loss_weight = self._read_float("rl_loss_weight", 1.0)
        self._dmd_loss_weight = self._read_float("dmd_loss_weight", 1.0)
        self._fake_score_loss_weight = self._read_float("fake_score_loss_weight", 1.0)
        self._real_score_guidance_scale = self._read_float("real_score_guidance_scale", 1.0)
        self._fake_score_max_grad_norm = self._read_float("fake_score_max_grad_norm", self._max_grad_norm)
        self._cold_start_steps = self._read_int("cold_start_steps", 0)
        self._dynamic_step = self._read_int("dynamic_step", 0)
        self._guidance_update_ratio = self._read_int("guidance_update_ratio", 1)
        if self._guidance_update_ratio <= 0:
            raise ValueError("method.guidance_update_ratio must be positive")
        self._gen_timestep_sampling = self._parse_timestep_sampling(
            "gen_timestep_sampling",
            default=_DMDRTimestepSamplingConfig(kind="uniform", alpha=1.0, beta=1.0, discrete=True),
        )
        self._dmd_timestep_sampling = self._parse_timestep_sampling(
            "dmd_timestep_sampling",
            default=_DMDRTimestepSamplingConfig(kind="logit_normal", alpha=4.0, beta=1.5),
        )
        self._fake_score_timestep_sampling = self._parse_timestep_sampling(
            "fake_score_timestep_sampling",
            default=self._dmd_timestep_sampling,
        )
        self._dmd_cfg_uncond = self._parse_dmd_cfg_uncond()
        self._init_critic_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        optimizers = {"student": self._student_optimizer}
        critic_optimizer = getattr(self, "_critic_optimizer", None)
        if critic_optimizer is not None:
            optimizers["critic"] = critic_optimizer
        return optimizers

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        schedulers = {"student": self._student_lr_scheduler}
        critic_scheduler = getattr(self, "_critic_lr_scheduler", None)
        if critic_scheduler is not None:
            schedulers["critic"] = critic_scheduler
        return schedulers

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer, self._critic_optimizer]

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler, self._critic_lr_scheduler]

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        self._log_progress(f"[DMDR] outer step {iteration}: start sampling "
                           f"{self._num_batches_per_epoch} batches")
        sample_items = self._sample_epoch(data_stream, iteration)
        reward_active = self._reward_is_active(iteration)
        if reward_active:
            self._log_progress(f"[DMDR] outer step {iteration}: scoring rewards")
            rewards = self._score_samples(sample_items)
            self._log_progress(f"[DMDR] outer step {iteration}: computing advantages")
            advantages = self._compute_advantages(sample_items, rewards)
        else:
            self._log_progress(f"[DMDR] outer step {iteration}: cold start DMD-only training")
            rewards = self._zero_reward_dict(sample_items)
            advantages = torch.zeros(
                sum(item["latents_clean"].shape[0] for item in sample_items),
                self._num_train_timesteps(),
                device=self.student.device,
                dtype=torch.float32,
            )
        self._log_progress(f"[DMDR] outer step {iteration}: start inner training")
        loss_map, metrics = self._inner_train(sample_items, advantages, iteration, reward_active=reward_active)
        self._update_old_model(iteration)
        metrics.update(self._reward_metrics(rewards))
        if reward_active:
            metrics.update(self._reward_diagnostic_metrics(sample_items, rewards))
        metrics["dmdr/num_sampled"] = float(sum(item["latents_clean"].shape[0] for item in sample_items))
        metrics["dmdr/reward_active"] = float(reward_active)
        return loss_map, {}, metrics

    def checkpoint_state(self) -> dict[str, Any]:
        states = TrainingMethod.checkpoint_state(self)
        states["roles.old.transformer"] = _FullModelState(self.old.transformer)
        if self._ema_enabled:
            states["dmdr.ema"] = _DiffusionNFTEMAState(self)
        return states

    def on_train_start(self) -> None:
        TrainingMethod.on_train_start(self)
        self._sync_old_from_student()
        if self._ema_enabled:
            self._student_ema = EMA_FSDP(
                self.student.transformer,
                decay=self._ema_decay,
                mode="local_shard",
            )
            self._log_progress(f"[DMDR] EMA enabled (decay={self._ema_decay})")
        self._reward_scorer = build_multi_reward_scorer(
            self._reward_fn_config,
            device=self.student.device,
            backend=self._reward_backend,
        )

    def _init_critic_optimizer_and_scheduler(self) -> None:
        critic_lr_raw = get_optional_float(
            self.method_config,
            "fake_score_learning_rate",
            where="method.fake_score_learning_rate",
        )
        if critic_lr_raw is None or critic_lr_raw <= 0.0:
            raise ValueError("method.fake_score_learning_rate must be set to a positive value")

        critic_betas_raw = self.method_config.get("fake_score_betas", None)
        if critic_betas_raw is None:
            raise ValueError("method.fake_score_betas must be set, for example [0.0, 0.999]")
        critic_betas = parse_betas(critic_betas_raw, where="method.fake_score_betas")

        critic_sched_raw = self.method_config.get("fake_score_lr_scheduler", None)
        if critic_sched_raw is None:
            raise ValueError("method.fake_score_lr_scheduler must be set, for example 'constant'")

        critic_params = [p for p in self.critic.transformer.parameters() if p.requires_grad]
        self._critic_optimizer, self._critic_lr_scheduler = build_optimizer_and_scheduler(
            params=critic_params,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=float(critic_lr_raw),
            betas=critic_betas,
            scheduler_name=str(critic_sched_raw),
        )

    def _parse_dmd_cfg_uncond(self) -> dict[str, Any]:
        raw = self.method_config.get("cfg_uncond", None)
        if raw is None:
            return {"text": "zero", "on_missing": "ignore"}
        if not isinstance(raw, dict):
            raise ValueError(f"method.cfg_uncond must be a mapping, got {type(raw).__name__}")
        cfg = dict(raw)
        text_policy = str(cfg.get("text", "zero") or "zero").strip().lower()
        if text_policy not in {"negative_prompt", "keep", "zero"}:
            raise ValueError("method.cfg_uncond.text must be one of "
                             "{negative_prompt, keep, zero} for DMDR")
        on_missing = str(cfg.get("on_missing", "ignore") or "ignore").strip().lower()
        if on_missing not in {"error", "ignore"}:
            raise ValueError("method.cfg_uncond.on_missing must be one of {error, ignore}")
        cfg["text"] = text_policy
        cfg["on_missing"] = on_missing
        return cfg

    def _parse_timestep_sampling(
        self,
        key: str,
        *,
        default: _DMDRTimestepSamplingConfig,
    ) -> _DMDRTimestepSamplingConfig:
        raw = self.method_config.get(key, None)
        if raw is None:
            return default
        if not isinstance(raw, dict):
            raise ValueError(f"method.{key} must be a mapping, got {type(raw).__name__}")
        cfg = _DMDRTimestepSamplingConfig(
            kind=str(raw.get("type", raw.get("kind", default.kind)) or default.kind).strip().lower(),
            alpha=float(raw.get("alpha", default.alpha)),
            beta=float(raw.get("beta", default.beta)),
            discrete=bool(raw.get("discrete", default.discrete)),
            min_t=float(raw.get("min_t", default.min_t)),
            max_t=float(raw.get("max_t", default.max_t)),
        )
        if cfg.kind not in {"uniform", "logit_normal"}:
            raise ValueError(f"method.{key}.type must be one of {{uniform, logit_normal}}")
        if cfg.alpha <= 0.0 or cfg.beta <= 0.0:
            raise ValueError(f"method.{key}.alpha and method.{key}.beta must be positive")
        if not 0.0 <= cfg.min_t < cfg.max_t <= 1.0:
            raise ValueError(f"method.{key}.min_t/max_t must satisfy 0 <= min_t < max_t <= 1")
        return cfg

    def _reward_is_active(self, iteration: int) -> bool:
        return int(iteration) > int(self._cold_start_steps)

    def _zero_reward_dict(self, sample_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        total = sum(item["latents_clean"].shape[0] for item in sample_items)
        return {"avg": torch.zeros(total, device=self.student.device, dtype=torch.float32)}

    def _inner_train(
        self,
        sample_items: list[dict[str, Any]],
        advantages: torch.Tensor,
        iteration: int,
        *,
        reward_active: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
        self.student.transformer.train()
        self.critic.transformer.train()
        self.old.transformer.eval()
        self.reference.transformer.eval()
        self.teacher.transformer.eval()
        self._student_optimizer.zero_grad(set_to_none=True)
        self._critic_optimizer.zero_grad(set_to_none=True)

        samples = self._collate_samples(sample_items)
        total_samples = int(samples["latents_clean"].shape[0])
        num_train_timesteps = self._num_train_timesteps()
        if total_samples != int(advantages.shape[0]):
            raise RuntimeError("advantages and samples have mismatched batch sizes")

        grad_accum = max(1, int(self.training_config.loop.gradient_accumulation_steps or 1))
        effective_grad_accum = grad_accum * max(1, num_train_timesteps)
        critic_accum = 0
        student_accum = 0
        critic_optimizer_steps = 0
        student_optimizer_steps = 0
        micro_step = 0
        loss_terms: dict[str, list[torch.Tensor]] = defaultdict(list)
        num_batches = max(1, total_samples // max(1, self._train_batch_size))
        training_batch_size = max(1, total_samples // num_batches)
        total_train_batches = self._num_inner_epochs * num_batches

        with tqdm(
                total=total_train_batches,
                desc=f"DMDR step {iteration}: training",
                position=1,
                leave=False,
                disable=not self._show_terminal_progress(),
        ) as progress:
            for _ in range(self._num_inner_epochs):
                perm = torch.randperm(
                    total_samples,
                    device=self.student.device,
                    generator=self.cuda_generator,
                )
                shuffled = {key: value[perm] for key, value in samples.items()}
                shuffled_adv = advantages[perm]
                perms_time = torch.stack([
                    torch.randperm(num_train_timesteps, device=self.student.device, generator=self.cuda_generator)
                    for _ in range(total_samples)
                ])
                row_idx = torch.arange(total_samples, device=self.student.device)[:, None]
                shuffled["timesteps"] = shuffled["timesteps"][row_idx, perms_time]
                shuffled_adv = shuffled_adv[row_idx, perms_time]

                for batch_idx in range(num_batches):
                    start = batch_idx * training_batch_size
                    end = total_samples if batch_idx == num_batches - 1 else (batch_idx + 1) * training_batch_size
                    train_sample = {key: value[start:end] for key, value in shuffled.items()}
                    train_adv = shuffled_adv[start:end]
                    for timestep_idx in range(num_train_timesteps):
                        losses, student_ctx, critic_ctx = self._training_timestep_losses(
                            train_sample,
                            train_adv[:, timestep_idx],
                            timestep_idx,
                            iteration=iteration,
                            reward_active=reward_active,
                        )
                        self.critic.backward(
                            losses["fake_score_loss"] * self._fake_score_loss_weight,
                            critic_ctx,
                            grad_accum_rounds=effective_grad_accum,
                        )
                        critic_accum += 1
                        micro_step += 1
                        should_update_student = micro_step % self._guidance_update_ratio == 0
                        if should_update_student:
                            self.student.backward(
                                losses["student_total_loss"],
                                student_ctx,
                                grad_accum_rounds=effective_grad_accum,
                            )
                            student_accum += 1
                        for key, value in losses.items():
                            loss_terms[key].append(value.detach())

                        if critic_accum % effective_grad_accum == 0:
                            self._critic_optimizer_step()
                            critic_optimizer_steps += 1
                        if should_update_student and student_accum % effective_grad_accum == 0:
                            self._student_optimizer_step()
                            student_optimizer_steps += 1
                    progress.update(1)

        if critic_accum % effective_grad_accum != 0:
            self._critic_optimizer_step()
            critic_optimizer_steps += 1
        if student_accum > 0 and student_accum % effective_grad_accum != 0:
            self._student_optimizer_step()
            student_optimizer_steps += 1

        self._log_progress(f"[DMDR] outer step {iteration}: finished inner training "
                           f"({micro_step} micro-steps, {student_optimizer_steps} student optimizer steps, "
                           f"{critic_optimizer_steps} critic optimizer steps)")

        reduced_local = {
            key: torch.stack(values).mean() if values else torch.zeros((), device=self.student.device)
            for key, values in loss_terms.items()
        }
        reduced = {key: self._mean_scalar_across_ranks(value) for key, value in reduced_local.items()}
        reduced.setdefault("total_loss", torch.zeros((), device=self.student.device))
        metrics: dict[str, LogScalar] = {
            "dmdr/iteration": float(iteration),
            "dmdr/num_inner_epochs": float(self._num_inner_epochs),
            "dmdr/inner_micro_steps": float(micro_step),
            "dmdr/student_optimizer_steps": float(student_optimizer_steps),
            "dmdr/critic_optimizer_steps": float(critic_optimizer_steps),
            "dmdr/guidance_update_ratio": float(self._guidance_update_ratio),
            "ema/update_count": float(self._ema_update_count),
        }
        return reduced, metrics

    def _training_timestep_losses(
        self,
        sample: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        timestep_idx: int,
        *,
        iteration: int,
        reward_active: bool,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, Any], tuple[torch.Tensor, Any]]:
        x0 = sample["latents_clean"]
        timestep = self._generator_training_timestep(
            sample["timesteps"],
            timestep_idx,
            iteration=iteration,
        ).to(device=x0.device)
        t = timestep.float() / float(self.student.num_train_timesteps)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = torch.randn(
            x0.shape,
            device=x0.device,
            dtype=x0.dtype,
            generator=self.cuda_generator,
        )
        xt = ((1 - t_expanded) * x0 + t_expanded * noise).to(dtype=x0.dtype)
        batch = self._make_training_batch(sample, timestep)

        with torch.no_grad():
            old_prediction = self.old.predict_noise(
                xt,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            ).detach()
            ref_forward_prediction = self.reference.predict_noise(
                xt,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            ).detach()

        forward_prediction = self.student.predict_noise(
            xt,
            timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )

        policy_loss, kl_div_loss, policy_metrics, x0_prediction = self._reward_policy_losses(
            x0=x0,
            xt=xt,
            t_expanded=t_expanded,
            advantages=advantages,
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            reference_prediction=ref_forward_prediction,
        )
        if not reward_active:
            policy_loss = policy_loss.new_zeros(())
            kl_div_loss = kl_div_loss.new_zeros(())
        dmd_loss = self._distribution_matching_loss(x0_prediction, batch, iteration=iteration)
        fake_score_loss, critic_ctx = self._fake_score_flow_matching_loss(
            x0_prediction.detach(),
            batch,
            iteration=iteration,
        )
        student_total_loss = self._rl_loss_weight * (policy_loss + self._kl_beta * kl_div_loss)
        student_total_loss = student_total_loss + self._dmd_loss_weight * dmd_loss
        total_loss = student_total_loss + self._fake_score_loss_weight * fake_score_loss.detach()
        losses = {
            "total_loss": total_loss,
            "student_total_loss": student_total_loss,
            "policy_loss": policy_loss,
            "kl_div_loss": kl_div_loss,
            "dmd_loss": dmd_loss,
            "fake_score_loss": fake_score_loss,
            **policy_metrics,
        }
        return losses, (batch.timesteps, batch.attn_metadata), critic_ctx

    def _generator_training_timestep(
        self,
        timesteps: torch.Tensor,
        timestep_idx: int,
        *,
        iteration: int,
    ) -> torch.Tensor:
        if self._gen_timestep_sampling.discrete:
            return timesteps[:, timestep_idx]
        return self._sample_train_timestep(
            batch_size=timesteps.shape[0],
            device=timesteps.device,
            sampling=self._gen_timestep_sampling,
            iteration=iteration,
        )

    def _reward_policy_losses(
        self,
        *,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t_expanded: torch.Tensor,
        advantages: torch.Tensor,
        forward_prediction: torch.Tensor,
        old_prediction: torch.Tensor,
        reference_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        advantages_clip = torch.clamp(advantages, -self._adv_clip_max, self._adv_clip_max)
        if self._adv_mode == "positive_only":
            advantages_clip = torch.clamp(advantages_clip, 0, self._adv_clip_max)
        elif self._adv_mode == "negative_only":
            advantages_clip = torch.clamp(advantages_clip, -self._adv_clip_max, 0)
        elif self._adv_mode == "one_only":
            advantages_clip = torch.where(
                advantages_clip > 0,
                torch.ones_like(advantages_clip),
                torch.zeros_like(advantages_clip),
            )
        elif self._adv_mode == "binary":
            advantages_clip = torch.sign(advantages_clip)

        normalized_advantages_clip = (advantages_clip / self._adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(normalized_advantages_clip, 0, 1)
        positive_prediction = self._nft_beta * forward_prediction + (1 - self._nft_beta) * old_prediction.detach()
        implicit_negative_prediction = ((1.0 + self._nft_beta) * old_prediction.detach() -
                                        self._nft_beta * forward_prediction)

        x0_prediction = xt - t_expanded * positive_prediction
        positive_loss = self._weighted_x0_mse(x0_prediction, x0)
        negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
        negative_loss = self._weighted_x0_mse(negative_x0_prediction, x0)

        ori_policy_loss = r * positive_loss / self._nft_beta + (1.0 - r) * negative_loss / self._nft_beta
        policy_loss = (ori_policy_loss * self._adv_clip_max).mean()
        kl_div_loss = ((forward_prediction - reference_prediction)**2).mean(dim=tuple(range(1, x0.ndim))).mean()
        metrics = {
            "unweighted_policy_loss": ori_policy_loss.mean(),
            "old_deviate": ((forward_prediction - old_prediction)**2).mean(),
            "old_kl_div": ((old_prediction - reference_prediction)**2).mean(),
            "x0_norm": torch.mean(x0**2),
        }
        return policy_loss, kl_div_loss, metrics, x0_prediction

    @staticmethod
    def _weighted_x0_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            weight_factor = torch.abs(prediction.double() - target.double())
            weight_factor = weight_factor.mean(dim=tuple(range(1, target.ndim)), keepdim=True).clip(min=0.00001)
        return ((prediction - target)**2 / weight_factor).mean(dim=tuple(range(1, target.ndim)))

    def _distribution_matching_loss(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
        *,
        iteration: int,
    ) -> torch.Tensor:
        original_timesteps = batch.timesteps
        with torch.no_grad():
            timestep = self._sample_train_timestep(
                batch_size=generator_pred_x0.shape[0],
                device=generator_pred_x0.device,
                sampling=self._dmd_timestep_sampling,
                iteration=iteration,
            )
            timestep = self.student.shift_and_clamp_timestep(timestep)
            noise = torch.randn(
                generator_pred_x0.shape,
                device=generator_pred_x0.device,
                dtype=generator_pred_x0.dtype,
                generator=self.cuda_generator,
            )
            noisy_latents = self.student.add_noise(generator_pred_x0, noise, timestep)
            batch.timesteps = timestep
            try:
                faker_x0 = self.critic.predict_x0(
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                real_cond_x0 = self.teacher.predict_x0(
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                real_uncond_x0 = self.teacher.predict_x0(
                    noisy_latents,
                    timestep,
                    batch,
                    conditional=False,
                    cfg_uncond=self._dmd_cfg_uncond,
                    attn_kind="dense",
                )
            finally:
                batch.timesteps = original_timesteps
            real_guidance_scale = self._real_guidance_scale(iteration)
            real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * real_guidance_scale
            reduce_dims = tuple(range(1, generator_pred_x0.ndim))
            denom = torch.abs(generator_pred_x0 - real_cfg_x0).mean(dim=reduce_dims, keepdim=True).clamp(min=1e-6)
            grad = torch.nan_to_num((faker_x0 - real_cfg_x0) / denom)

        return 0.5 * F.mse_loss(
            generator_pred_x0.float(),
            (generator_pred_x0.float() - grad.float()).detach(),
        )

    def _fake_score_flow_matching_loss(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
        *,
        iteration: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, Any]]:
        original_timesteps = batch.timesteps
        timestep = self._sample_train_timestep(
            batch_size=generator_pred_x0.shape[0],
            device=generator_pred_x0.device,
            sampling=self._fake_score_timestep_sampling,
            iteration=iteration,
        )
        timestep = self.student.shift_and_clamp_timestep(timestep)
        noise = torch.randn(
            generator_pred_x0.shape,
            device=generator_pred_x0.device,
            dtype=generator_pred_x0.dtype,
            generator=self.cuda_generator,
        )
        noisy_x0 = self.student.add_noise(generator_pred_x0, noise, timestep)
        batch.timesteps = timestep
        try:
            pred_noise = self.critic.predict_noise(
                noisy_x0,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
        finally:
            batch.timesteps = original_timesteps
        target = noise - generator_pred_x0
        flow_matching_loss = torch.mean((pred_noise - target)**2)
        return flow_matching_loss, (timestep, batch.attn_metadata)

    def _student_optimizer_step(self) -> None:
        self._clip_student_grads()
        self._student_optimizer.step()
        self._student_lr_scheduler.step()
        self._update_ema()
        self._student_optimizer.zero_grad(set_to_none=True)

    def _critic_optimizer_step(self) -> None:
        self._clip_critic_grads()
        self._critic_optimizer.step()
        self._critic_lr_scheduler.step()
        self._critic_optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self) -> None:
        self._student_optimizer_step()
        self._critic_optimizer_step()

    def _sample_train_timestep(
        self,
        *,
        batch_size: int,
        device: torch.device,
        sampling: _DMDRTimestepSamplingConfig,
        iteration: int,
    ) -> torch.Tensor:
        fractions = self._sample_timestep_fraction(
            batch_size=batch_size,
            device=device,
            sampling=sampling,
            iteration=iteration,
        )
        timestep = fractions * float(self.student.num_train_timesteps)
        return timestep.to(device=device, dtype=torch.float32)

    def _sample_timestep_fraction(
        self,
        *,
        batch_size: int,
        device: torch.device,
        sampling: _DMDRTimestepSamplingConfig,
        iteration: int,
    ) -> torch.Tensor:
        if sampling.kind == "uniform":
            values = torch.rand(
                batch_size,
                device=device,
                dtype=torch.float32,
                generator=self.cuda_generator,
            )
        else:
            alpha, beta = self._annealed_beta_params(sampling, iteration)
            dist = torch.distributions.Beta(
                torch.tensor(alpha, device=device, dtype=torch.float32),
                torch.tensor(beta, device=device, dtype=torch.float32),
            )
            values = dist.sample((batch_size, ))
        return sampling.min_t + values.clamp(0, 1) * (sampling.max_t - sampling.min_t)

    def _annealed_beta_params(
        self,
        sampling: _DMDRTimestepSamplingConfig,
        iteration: int,
    ) -> tuple[float, float]:
        if self._dynamic_step <= 0:
            return sampling.alpha, sampling.beta
        progress = min(max(float(iteration) / float(self._dynamic_step), 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        alpha = 1.0 + (sampling.alpha - 1.0) * cosine_decay
        beta = 1.0 + (sampling.beta - 1.0) * cosine_decay
        return alpha, beta

    def _real_guidance_scale(self, iteration: int) -> float:
        if self._dynamic_step <= 0:
            return self._real_score_guidance_scale
        if iteration >= self._dynamic_step:
            return 0.0
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * float(iteration) / float(self._dynamic_step)))
        return self._real_score_guidance_scale * cosine_factor

    def _clip_critic_grads(self) -> None:
        if self._fake_score_max_grad_norm <= 0.0:
            return
        clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in self.critic.transformer.parameters()],
            self._fake_score_max_grad_norm,
            foreach=None,
        )
