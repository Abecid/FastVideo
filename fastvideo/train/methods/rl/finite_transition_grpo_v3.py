# SPDX-License-Identifier: Apache-2.0
"""Multi-minibatch GRPO v3 for AnyFlow finite-transition policies.

The v2 "GRPO" arm collected an on-policy rollout and took one optimizer step
after computing every loss. At every gradient evaluation the learner therefore
still equaled the behavior policy, the likelihood ratio was exactly one, and
PPO clipping was inert. This implementation freezes a rollout buffer and then
performs several optimizer minibatches/epochs against the stored old policy.

The online reward remains audited VideoAlign MQ only. VQ and TA stay held-out
validation metrics and do not contribute gradients.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import contextlib
import time
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common.finite_transition import (
    clipped_grpo_loss,
    diagonal_gaussian_kl_mean,
    gaussian_log_prob_mean,
)
from fastvideo.train.methods.rl.common.finite_transition_grpo_v3 import (
    GroupAdvantageBatch,
    group_normalized_advantages,
    shuffled_group_minibatches,
)
from fastvideo.train.methods.rl.common.finite_transition_v2 import (
    cosine_similarity_flat,
    rms,
)
from fastvideo.train.methods.rl.finite_transition_v2 import (
    _RolloutGroup,
    _TransitionRecord,
)
from fastvideo.train.methods.rl.finite_transition_v2_final import (
    FiniteTransitionV2FinalMethod,
)


class _GRPOV3State:
    """DCP-compatible scalar state for resume-safe optimizer accounting."""

    def __init__(self, method: "FiniteTransitionGRPOV3Method") -> None:
        self._method = method

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "optimizer_steps": torch.tensor(
                int(self._method._grpo_optimizer_steps),
                dtype=torch.long,
            )
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        value = state_dict.get("optimizer_steps", 0)
        if torch.is_tensor(value):
            value = int(value.item())
        self._method._grpo_optimizer_steps = int(value)


class FiniteTransitionGRPOV3Method(FiniteTransitionV2FinalMethod):
    """Actual clipped GRPO optimization over a frozen finite-transition buffer."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if self._v2_objective != "flowmap_grpo":
            raise ValueError("GRPO v3 requires method.v2_objective=flowmap_grpo")
        if self._transition_mode != "all":
            raise ValueError("GRPO v3 requires method.transition_mode=all")
        if self._behavior_policy != "on_policy":
            raise ValueError(
                "GRPO v3 is an on-policy baseline; behavior_policy must be on_policy"
            )

        mcfg = self.method_config
        self._policy_epochs = int(mcfg.get("policy_epochs", 2))
        self._groups_per_minibatch = int(
            mcfg.get("groups_per_minibatch", 1)
        )
        self._minimum_group_reward_std = float(
            mcfg.get("minimum_group_reward_std", 1.0e-4)
        )
        self._policy_kl_target = float(
            mcfg.get("policy_kl_target", 3.0e-5)
        )
        self._policy_kl_early_stop_multiplier = float(
            mcfg.get("policy_kl_early_stop_multiplier", 4.0)
        )
        self._reference_kl_beta = float(
            mcfg.get("reference_kl_beta", 0.0)
        )
        self._deployment_probe_every = max(
            0,
            int(mcfg.get("deployment_probe_every", 5)),
        )
        self._old_logprob_tolerance = float(
            mcfg.get("old_logprob_tolerance", 2.0e-3)
        )
        self._grpo_optimizer_steps = 0

        if self._policy_epochs <= 0:
            raise ValueError("method.policy_epochs must be positive")
        if not 0 < self._groups_per_minibatch <= self._rollout_groups_per_update:
            raise ValueError(
                "method.groups_per_minibatch must lie in "
                "[1, rollout_groups_per_update]"
            )
        if self._minimum_group_reward_std < 0.0:
            raise ValueError(
                "method.minimum_group_reward_std must be non-negative"
            )
        if self._policy_kl_target <= 0.0:
            raise ValueError("method.policy_kl_target must be positive")
        if self._policy_kl_early_stop_multiplier <= 1.0:
            raise ValueError(
                "method.policy_kl_early_stop_multiplier must exceed one"
            )
        if self._reference_kl_beta < 0.0:
            raise ValueError("method.reference_kl_beta must be non-negative")
        if self._old_logprob_tolerance <= 0.0:
            raise ValueError("method.old_logprob_tolerance must be positive")

    def checkpoint_state(self) -> dict[str, Any]:
        states = super().checkpoint_state()
        states["finite_transition_grpo_v3.state"] = _GRPOV3State(self)
        return states

    @staticmethod
    def _distributed_min(value: torch.Tensor) -> torch.Tensor:
        result = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.MIN)
        return result

    @staticmethod
    def _distributed_max(value: torch.Tensor) -> torch.Tensor:
        result = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.MAX)
        return result

    @contextlib.contextmanager
    def _frozen_base_context(self) -> Iterator[None]:
        """Disable the learner adapter for an optional reference-KL cache."""
        transformer = self.student.transformer
        seen: set[int] = set()
        for module in (transformer, *transformer.modules()):
            if id(module) in seen:
                continue
            seen.add(id(module))
            disable = getattr(module, "disable_adapter", None)
            if not callable(disable):
                continue
            context = disable()
            if hasattr(context, "__enter__"):
                with context:
                    yield
                return
        raise RuntimeError(
            "reference_kl_beta > 0 requires a PEFT disable_adapter() context"
        )

    @torch.no_grad()
    def _cache_policy_parameters(
        self,
        groups: list[_RolloutGroup],
        *,
        frozen_base: bool,
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        """Cache old/base Gaussian parameters before any policy update."""
        cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        mismatch = torch.zeros((), device=self.student.device)
        context = self._frozen_base_context() if frozen_base else contextlib.nullcontext()

        self.student.transformer.eval()
        with context:
            for group in groups:
                for record in group.transitions:
                    mean, std, _ = self._branch_policy(
                        record.state.detach(),
                        record.source_time,
                        record.target_time,
                        record.batch,
                    )
                    cache[id(record)] = (mean.detach(), std.detach())
                    if not frozen_base:
                        recomputed = gaussian_log_prob_mean(
                            record.action,
                            mean,
                            std,
                        )
                        mismatch = torch.maximum(
                            mismatch,
                            (
                                recomputed.detach()
                                - record.old_log_prob.detach()
                            )
                            .abs()
                            .max(),
                        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if not frozen_base and float(mismatch) > self._old_logprob_tolerance:
            raise RuntimeError(
                "rollout old-log-probability mismatch exceeds tolerance: "
                f"{float(mismatch):.6g} > {self._old_logprob_tolerance:.6g}"
            )
        return cache, mismatch

    def _build_group_payloads(
        self,
        groups: list[_RolloutGroup],
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for group in groups:
            normalized = group_normalized_advantages(
                group.global_rewards,
                epsilon=self._advantage_epsilon,
                clip=self._advantage_clip,
                minimum_std=self._minimum_group_reward_std,
            )
            payloads.append(
                {
                    "group": group,
                    "normalized": normalized,
                }
            )
        return payloads

    def _preferred_action_shift(
        self,
        record: _TransitionRecord,
        normalized: GroupAdvantageBatch,
        *,
        local_start: int,
        local_end: int,
    ) -> torch.Tensor:
        local_advantages = normalized.advantages[
            local_start:local_end
        ].to(record.action.device)
        view = [local_advantages.shape[0]] + [1] * (
            record.action.ndim - 1
        )
        displacement = (
            record.action.float() - record.deterministic_target.float()
        )
        numerator = (
            local_advantages.view(*view) * displacement
        ).sum(dim=0, keepdim=True)
        denominator = local_advantages.abs().sum().to(
            device=record.action.device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        if float(denominator) <= 1.0e-12:
            return torch.zeros_like(numerator)
        return numerator / denominator

    @torch.no_grad()
    def _deterministic_alignment_probe(
        self,
        payloads: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        payload = next(
            (
                item
                for item in payloads
                if item["normalized"].active
            ),
            None,
        )
        if payload is None:
            zero = torch.zeros((), device=self.student.device)
            return zero, zero, zero

        group: _RolloutGroup = payload["group"]
        normalized: GroupAdvantageBatch = payload["normalized"]
        record = group.transitions[0]
        _, _, new_deterministic = self._branch_policy(
            record.state.detach(),
            record.source_time,
            record.target_time,
            record.batch,
        )
        map_shift = (
            new_deterministic.float()
            - record.deterministic_target.float()
        )
        preferred = self._preferred_action_shift(
            record,
            normalized,
            local_start=group.local_start,
            local_end=group.local_end,
        )
        cosine = cosine_similarity_flat(
            map_shift[:1],
            preferred.expand_as(map_shift)[:1],
        )
        return cosine, rms(map_shift), rms(preferred)

    @torch.no_grad()
    def _deployment_reward_for_group(
        self,
        group: _RolloutGroup,
    ) -> torch.Tensor:
        """Score the deployed deterministic four-step map on a rollout state."""
        schedule = self._build_schedule(
            steps=self._eval_map_steps,
            override=self._eval_schedule_override,
            device=self.student.device,
        )
        initial = group.transitions[0].state.detach()
        endpoint = self._deterministic_rollout(
            initial,
            group.transitions[0].batch,
            schedule,
        )
        media = self.student.decode_latents(endpoint).detach().cpu()
        local_rewards = self._score_media(media, group.prompts)
        if self._optimize_reward not in local_rewards:
            raise RuntimeError(
                f"deployment probe reward {self._optimize_reward!r} is missing"
            )
        return self._all_gather_1d(
            local_rewards[self._optimize_reward]
        ).mean()

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
        groups = [
            self._collect_rollout_group(
                iteration=iteration,
                group_offset=group_offset,
            )
            for group_offset in range(self._rollout_groups_per_update)
        ]
        payloads = self._build_group_payloads(groups)
        active_payloads = [
            payload
            for payload in payloads
            if payload["normalized"].active
        ]

        old_cache, old_logprob_mismatch = self._cache_policy_parameters(
            groups,
            frozen_base=False,
        )
        reference_cache: dict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        if self._reference_kl_beta > 0.0:
            reference_cache, _ = self._cache_policy_parameters(
                groups,
                frozen_base=True,
            )

        should_probe = (
            self._deployment_probe_every > 0
            and iteration % self._deployment_probe_every == 0
        )
        deployment_before: torch.Tensor | None = None
        if should_probe:
            deployment_before = self._deployment_reward_for_group(groups[0])

        policy_losses: list[float] = []
        reference_kls: list[float] = []
        old_policy_kls: list[float] = []
        ratio_means: list[float] = []
        ratio_minima: list[float] = []
        ratio_maxima: list[float] = []
        clip_fractions: list[float] = []
        sampled_approx_kls: list[float] = []
        log_ratio_abs: list[float] = []
        grad_norms: list[float] = []
        optimizer_steps_this_rollout = 0
        attempted_minibatches = 0
        completed_epochs = 0
        early_stopped = False
        early_stop_kl = 0.0

        for policy_epoch in range(self._policy_epochs):
            minibatches = shuffled_group_minibatches(
                len(payloads),
                groups_per_minibatch=self._groups_per_minibatch,
                seed=(
                    int(self.training_config.data.seed)
                    + int(iteration) * 100_003
                    + int(policy_epoch) * 1_009
                ),
            )
            epoch_had_step = False
            for group_indices in minibatches:
                selected = [
                    payloads[index]
                    for index in group_indices
                    if payloads[index]["normalized"].active
                ]
                if not selected:
                    continue

                attempted_minibatches += 1
                record_count = sum(
                    len(payload["group"].transitions)
                    for payload in selected
                )
                if record_count <= 0:
                    continue

                self.student.transformer.train()
                self._student_optimizer.zero_grad(set_to_none=True)
                minibatch_policy_losses: list[torch.Tensor] = []
                minibatch_reference_kls: list[torch.Tensor] = []
                minibatch_old_kls: list[torch.Tensor] = []
                minibatch_diagnostics: dict[
                    str,
                    list[torch.Tensor],
                ] = defaultdict(list)

                for payload in selected:
                    group: _RolloutGroup = payload["group"]
                    normalized: GroupAdvantageBatch = payload["normalized"]
                    local_advantages = normalized.advantages[
                        group.local_start:group.local_end
                    ].to(self.student.device)

                    for record in group.transitions:
                        new_mean, new_std, _ = self._branch_policy(
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
                        policy_loss, diagnostics = clipped_grpo_loss(
                            new_log_prob,
                            record.old_log_prob,
                            local_advantages,
                            clip_range=self._clip_range,
                        )
                        old_mean, old_std = old_cache[id(record)]
                        old_kl = diagonal_gaussian_kl_mean(
                            old_mean,
                            old_std,
                            new_mean,
                            new_std,
                        ).mean()
                        if self._reference_kl_beta > 0.0:
                            base_mean, base_std = reference_cache[id(record)]
                            reference_kl = diagonal_gaussian_kl_mean(
                                base_mean,
                                base_std,
                                new_mean,
                                new_std,
                            ).mean()
                        else:
                            reference_kl = torch.zeros_like(old_kl)

                        loss = (
                            policy_loss
                            + self._reference_kl_beta * reference_kl
                        ) / float(record_count)
                        self.student.backward(
                            loss,
                            (
                                record.batch.timesteps,
                                record.batch.attn_metadata,
                            ),
                            grad_accum_rounds=1,
                        )

                        minibatch_policy_losses.append(
                            policy_loss.detach()
                        )
                        minibatch_reference_kls.append(
                            reference_kl.detach()
                        )
                        minibatch_old_kls.append(old_kl.detach())
                        for name, value in diagnostics.items():
                            minibatch_diagnostics[name].append(
                                torch.as_tensor(
                                    value,
                                    device=self.student.device,
                                )
                                .detach()
                                .float()
                            )

                local_old_kl = torch.stack(
                    minibatch_old_kls
                ).mean()
                global_old_kl = self._mean_across_ranks(local_old_kl)
                stop_threshold = (
                    self._policy_kl_target
                    * self._policy_kl_early_stop_multiplier
                )
                if (
                    self._grpo_optimizer_steps > 0
                    and float(global_old_kl) > stop_threshold
                ):
                    self._student_optimizer.zero_grad(set_to_none=True)
                    early_stopped = True
                    early_stop_kl = float(global_old_kl)
                    break

                grad_norm = self._clip_student_grads()
                self._student_optimizer.step()
                self._student_lr_scheduler.step()
                self._update_ema()
                self._student_optimizer.zero_grad(set_to_none=True)
                self._grpo_optimizer_steps += 1
                optimizer_steps_this_rollout += 1
                epoch_had_step = True

                policy_losses.append(
                    float(torch.stack(minibatch_policy_losses).mean())
                )
                reference_kls.append(
                    float(torch.stack(minibatch_reference_kls).mean())
                )
                old_policy_kls.append(float(global_old_kl))
                grad_norms.append(
                    float(
                        self._mean_across_ranks(
                            torch.as_tensor(
                                grad_norm,
                                device=self.student.device,
                            )
                        )
                    )
                )
                ratio_means.append(
                    float(
                        self._mean_across_ranks(
                            torch.stack(
                                minibatch_diagnostics["ratio_mean"]
                            ).mean()
                        )
                    )
                )
                ratio_minima.append(
                    float(
                        self._distributed_min(
                            torch.stack(
                                minibatch_diagnostics["ratio_min"]
                            ).min()
                        )
                    )
                )
                ratio_maxima.append(
                    float(
                        self._distributed_max(
                            torch.stack(
                                minibatch_diagnostics["ratio_max"]
                            ).max()
                        )
                    )
                )
                clip_fractions.append(
                    float(
                        self._mean_across_ranks(
                            torch.stack(
                                minibatch_diagnostics["clip_fraction"]
                            ).mean()
                        )
                    )
                )
                sampled_approx_kls.append(
                    float(
                        self._mean_across_ranks(
                            torch.stack(
                                minibatch_diagnostics["approx_kl"]
                            ).mean()
                        )
                    )
                )
                log_ratio_abs.append(
                    float(
                        self._mean_across_ranks(
                            torch.stack(
                                minibatch_diagnostics[
                                    "log_ratio_abs_mean"
                                ]
                            ).mean()
                        )
                    )
                )

            if epoch_had_step:
                completed_epochs = policy_epoch + 1
            if early_stopped:
                break

        self.student.transformer.eval()
        alignment, map_shift_rms, preferred_shift_rms = (
            self._deterministic_alignment_probe(payloads)
        )

        deployment_after: torch.Tensor | None = None
        if should_probe:
            deployment_after = self._deployment_reward_for_group(groups[0])

        step_seconds = time.perf_counter() - started
        self._cumulative_train_seconds += step_seconds
        world_size = self._world_size()

        def mean_or_zero(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        group_stds = [
            float(payload["normalized"].std)
            for payload in payloads
        ]
        advantages_abs = [
            float(payload["normalized"].advantages.abs().mean())
            for payload in payloads
        ]
        pooled_primary = torch.cat(
            [group.global_rewards for group in groups]
        )
        metrics: dict[str, LogScalar] = {
            "grpo_v3/online_reward_is_mq_only": float(
                self._optimize_reward == "videoalign_mq_audited"
                and set(self._reward_fn_config)
                == {"videoalign_mq_audited"}
            ),
            "grpo_v3/policy_epochs_configured": float(
                self._policy_epochs
            ),
            "grpo_v3/policy_epochs_completed": float(
                completed_epochs
            ),
            "grpo_v3/groups_per_minibatch": float(
                self._groups_per_minibatch
            ),
            "grpo_v3/rollout_groups": float(
                self._rollout_groups_per_update
            ),
            "grpo_v3/group_size": float(self._group_size),
            "grpo_v3/reward_videos_per_rollout": float(
                self._rollout_groups_per_update * self._group_size
            ),
            "grpo_v3/optimizer_steps_this_rollout": float(
                optimizer_steps_this_rollout
            ),
            "grpo_v3/optimizer_steps_total": float(
                self._grpo_optimizer_steps
            ),
            "grpo_v3/minibatches_attempted": float(
                attempted_minibatches
            ),
            "grpo_v3/early_stopped": float(early_stopped),
            "grpo_v3/early_stop_old_policy_kl": float(
                early_stop_kl
            ),
            "grpo_v3/policy_kl_target": float(
                self._policy_kl_target
            ),
            "grpo_v3/old_policy_kl_mean": mean_or_zero(
                old_policy_kls
            ),
            "grpo_v3/old_policy_kl_max": max(
                old_policy_kls,
                default=0.0,
            ),
            "grpo_v3/reference_kl_beta": float(
                self._reference_kl_beta
            ),
            "grpo_v3/reference_kl_mean": mean_or_zero(
                reference_kls
            ),
            "grpo_v3/policy_loss_mean": mean_or_zero(
                policy_losses
            ),
            "grpo_v3/ratio_mean": mean_or_zero(ratio_means),
            "grpo_v3/ratio_min": min(
                ratio_minima,
                default=1.0,
            ),
            "grpo_v3/ratio_max": max(
                ratio_maxima,
                default=1.0,
            ),
            "grpo_v3/ratio_abs_deviation_max": max(
                [
                    abs(value - 1.0)
                    for value in ratio_minima + ratio_maxima
                ],
                default=0.0,
            ),
            "grpo_v3/clip_fraction_mean": mean_or_zero(
                clip_fractions
            ),
            "grpo_v3/sampled_approx_kl_mean": mean_or_zero(
                sampled_approx_kls
            ),
            "grpo_v3/log_ratio_abs_mean": mean_or_zero(
                log_ratio_abs
            ),
            "grpo_v3/grad_norm_mean": mean_or_zero(grad_norms),
            "grpo_v3/old_logprob_recompute_max_error": (
                old_logprob_mismatch
            ),
            "grpo_v3/active_group_fraction": float(
                len(active_payloads) / max(len(payloads), 1)
            ),
            "grpo_v3/group_reward_std_mean": (
                sum(group_stds) / len(group_stds)
            ),
            "grpo_v3/group_reward_std_min": min(group_stds),
            "grpo_v3/group_reward_std_max": max(group_stds),
            "grpo_v3/advantage_abs_mean": (
                sum(advantages_abs) / len(advantages_abs)
            ),
            "grpo_v3/online_mq_mean": pooled_primary.mean(),
            "grpo_v3/online_mq_std": pooled_primary.std(
                unbiased=False
            ),
            "grpo_v3/deterministic_preference_alignment": (
                self._mean_across_ranks(alignment)
            ),
            "grpo_v3/deterministic_map_shift_rms": (
                self._mean_across_ranks(map_shift_rms)
            ),
            "grpo_v3/preferred_action_shift_rms": (
                self._mean_across_ranks(preferred_shift_rms)
            ),
            "grpo_v3/train_step_seconds": float(step_seconds),
            "grpo_v3/cumulative_gpu_hours": (
                self._cumulative_train_seconds
                * world_size
                / 3600.0
            ),
        }

        for reward_name in sorted(groups[0].global_reward_components):
            pooled = torch.cat(
                [
                    group.global_reward_components[reward_name]
                    for group in groups
                ]
            )
            metrics[f"grpo_v3/reward/{reward_name}"] = pooled.mean()
            metrics[f"grpo_v3/reward_std/{reward_name}"] = (
                pooled.std(unbiased=False)
            )

        if deployment_before is not None and deployment_after is not None:
            candidate_rewards = groups[0].global_rewards
            metrics.update(
                {
                    "grpo_v3/deployment_mq_before": deployment_before,
                    "grpo_v3/deployment_mq_after": deployment_after,
                    "grpo_v3/deployment_mq_update_delta": (
                        deployment_after - deployment_before
                    ),
                    "grpo_v3/candidate_mean_minus_deployment": (
                        candidate_rewards.mean() - deployment_before
                    ),
                    "grpo_v3/candidate_max_minus_deployment": (
                        candidate_rewards.max() - deployment_before
                    ),
                }
            )

        for key, value in self._startup_metrics.items():
            metrics[key] = float(value)

        mean_loss = torch.tensor(
            mean_or_zero(policy_losses),
            device=self.student.device,
            dtype=torch.float32,
        )
        return {"total_loss": mean_loss}, {}, metrics


__all__ = ["FiniteTransitionGRPOV3Method"]
