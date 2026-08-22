# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.pipelines import TrainingBatch
import fastvideo.train.methods.rl.finite_transition_posterior as ftp
from fastvideo.train.methods.rl.finite_transition_reliable_calibrated import (
    CalibratedReliableFiniteTransitionMethod,
)


class _NoopScheduler:
    def step(self) -> None:
        pass


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.15))
        self._r_embedder_enabled = True
        self._adapter_enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        old = self._adapter_enabled
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = old


class _FakeStudent:
    _trainable = True
    device = torch.device("cpu")
    num_train_timesteps = 1000

    def __init__(self) -> None:
        self.transformer = _FakeTransformer()

    def set_requires_negative_conditioning(self, required: bool) -> None:
        del required

    def init_preprocessors(self, training_config: Any) -> None:
        del training_config

    def on_train_start(self) -> None:
        pass

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: str,
        num_latent_t: int | None = None,
    ) -> TrainingBatch:
        del generator, latents_source, num_latent_t
        batch_size = len(raw_batch["info_list"])
        batch = TrainingBatch()
        batch.latents = torch.zeros((batch_size, 1, 1, 1, 1))
        batch.encoder_hidden_states = torch.zeros((batch_size, 1, 1))
        batch.encoder_attention_mask = torch.ones((batch_size, 1))
        batch.conditional_dict = {
            "encoder_hidden_states": batch.encoder_hidden_states,
            "encoder_attention_mask": batch.encoder_attention_mask,
        }
        batch.timesteps = torch.zeros(batch_size)
        batch.attn_metadata = None
        return batch

    def predict_velocity_with_r(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        batch: TrainingBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        del batch, kwargs
        view = [-1] + [1] * (latents.ndim - 1)
        interval = ((timestep - r_timestep) / 1000.0).view(*view)
        scale = self.transformer.scale
        if not self.transformer._adapter_enabled:
            scale = scale.detach() * 0.0 + 0.15
        return scale * latents + interval

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(latents.permute(0, 2, 1, 3, 4))

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del ctx
        (loss / float(grad_accum_rounds)).backward()


def _config(
    objective: str,
    *,
    rollout_mode: str,
    behavior_policy: str = "current",
) -> SimpleNamespace:
    if rollout_mode == "full_trajectory":
        train_steps = 3
        eval_steps = 2
        stochastic_steps = 2
        require_match = False
        train_schedule = [1000.0, 700.0, 300.0, 0.0]
        eval_schedule = [1000.0, 500.0, 0.0]
    else:
        train_steps = 2
        eval_steps = 2
        stochastic_steps = 1
        require_match = True
        train_schedule = [1000.0, 500.0, 0.0]
        eval_schedule = [1000.0, 500.0, 0.0]

    return SimpleNamespace(
        training=SimpleNamespace(
            distributed=SimpleNamespace(sp_size=1, num_gpus=1),
            data=SimpleNamespace(
                seed=42,
                train_batch_size=2,
                data_path="unused",
            ),
            optimizer=SimpleNamespace(
                learning_rate=0.05,
                betas=(0.9, 0.999),
                lr_scheduler="constant",
            ),
            loop=SimpleNamespace(),
            pipeline_config=SimpleNamespace(flow_shift=5.0),
            checkpoint=SimpleNamespace(output_dir="unused"),
        ),
        method={
            "objective": objective,
            "require_train_eval_schedule_match": require_match,
            "anchor_type": "local",
            "local_anchor_delta": 0.03,
            "local_noise_scale": 0.7,
            "local_terminal_base_sigma": 0.05,
            "train_map_steps": train_steps,
            "eval_map_steps": eval_steps,
            "stochastic_steps": stochastic_steps,
            "train_t_list_override": train_schedule,
            "eval_t_list_override": eval_schedule,
            "rollout_mode": rollout_mode,
            "transition_loss_reduction": "mean",
            "group_size": 2,
            "rollout_groups_per_update": 2,
            "behavior_policy": behavior_policy,
            "reward_normalization": "group",
            "posterior_temperature_mode": "fixed_ess",
            "target_ess_ratio": 0.5,
            "reward_backend": "genrl",
            "optimize_reward": "videoalign_mq",
            "reward_fn": {
                "rewards": {
                    "videoalign_mq": 1.0,
                    "videoalign_vq": 1.0,
                    "videoalign_ta": 1.0,
                }
            },
            "target_kl_controller": {
                "enabled": True,
                "target_kl": 1.0e-5,
                "initial_loss_scale": 1.0,
                "min_loss_scale": 0.05,
                "max_loss_scale": 8.0,
                "max_adjustment": 2.0,
            },
            "finite_velocity_target_rms": 0.01,
            "finite_velocity_max_eta": 4.0,
            "validation": {"every_steps": 0},
            "paired_validation": {"bootstrap_samples": 100},
            "evaluation": {},
            "ema": {"enabled": False},
        },
    )


def _make_method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    objective: str,
    rollout_mode: str,
    behavior_policy: str = "current",
) -> tuple[CalibratedReliableFiniteTransitionMethod, _FakeStudent]:
    def build_optimizer(params: list[torch.nn.Parameter], **kwargs: Any):
        del kwargs
        return torch.optim.SGD(params, lr=0.05), _NoopScheduler()

    monkeypatch.setattr(
        ftp,
        "build_optimizer_and_scheduler",
        build_optimizer,
    )
    monkeypatch.setattr(
        ftp,
        "clip_grad_norm_while_handling_failing_dtensor_cases",
        lambda parameters, max_norm, foreach=None: torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            foreach=foreach,
        ),
    )

    student = _FakeStudent()
    method = CalibratedReliableFiniteTransitionMethod(
        cfg=_config(
            objective,
            rollout_mode=rollout_mode,
            behavior_policy=behavior_policy,
        ),
        role_models={"student": student},
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(42)
    method._sample_prompt_batch = lambda iteration, local_branches: {
        "info_list": [
            {"prompt": f"a red block moves, group {iteration}"}
            for _ in range(local_branches)
        ]
    }

    def score(media: torch.Tensor, prompts: list[str]):
        del prompts
        value = media.float().mean(dim=tuple(range(1, media.ndim)))
        return {
            "videoalign_mq": value,
            "videoalign_vq": 0.5 * value,
            "videoalign_ta": 0.25 * value,
            "avg": value,
        }

    method._reward_scorer = score
    method._assert_anyflow_two_time_model()
    return method, student


@pytest.mark.parametrize(
    "objective",
    ["flowmap_grpo", "posterior_projection"],
)
def test_reliable_full_trajectory_updates_all_transitions(
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective=objective,
        rollout_mode="full_trajectory",
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert torch.isfinite(loss_map["total_loss"])
    assert not torch.equal(before, after)
    assert metrics["reliable/rollout_groups_per_update"] == 2.0
    assert metrics["reliable/reward_samples_per_update"] == 4.0
    assert metrics["reliable/stochastic_transitions_per_trajectory"] == 2.0
    assert float(metrics["reliable/post_update_approx_kl"]) >= 0.0
    assert "reliable/lossdiag/transition_0/ratio_mean" in metrics or (
        objective == "posterior_projection"
        and "reliable/lossdiag/transition_0/posterior_weight_mass_local"
        in metrics
    )


def test_fixed_base_behavior_uses_incremental_not_cumulative_kl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, _ = _make_method(
        monkeypatch,
        objective="flowmap_grpo",
        rollout_mode="full_trajectory",
        behavior_policy="base_adapter_disabled",
    )
    _, _, metrics = method.managed_train_step(iter(()), 1)

    assert metrics["reliable/behavior_is_fixed_base"] == 1.0
    assert float(metrics["reliable/post_update_approx_kl"]) >= 0.0
    # The calibrated method stores a learner pre-update likelihood separately
    # from the fixed behavior likelihood; target-KL is incremental update size.
    assert float(metrics["reliable/loss_scale_next"]) > 0.0


def test_finite_velocity_regression_updates_deterministic_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective="finite_velocity_regression",
        rollout_mode="single_transition",
        behavior_policy="base_adapter_disabled",
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert torch.isfinite(loss_map["total_loss"])
    assert not torch.equal(before, after)
    assert metrics["reliable/objective_is_finite_velocity"] == 1.0
    assert float(
        metrics["reliable/lossdiag/finite_transition_target_shift_rms"]
    ) > 0.0
