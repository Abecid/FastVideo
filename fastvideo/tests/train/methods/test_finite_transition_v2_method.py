# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.pipelines import TrainingBatch
import fastvideo.train.methods.rl.finite_transition_posterior as ftp
from fastvideo.train.methods.rl.finite_transition_v2 import FiniteTransitionV2Method


class _NoopScheduler:
    def step(self) -> None:
        pass


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.15))
        self._r_embedder_enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        yield


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
        return self.transformer.scale * latents + interval

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


def _config(v2_objective: str, *, transition_mode: str = "all") -> SimpleNamespace:
    base_objective = (
        "flowmap_grpo" if v2_objective == "flowmap_grpo" else "posterior_projection"
    )
    return SimpleNamespace(
        training=SimpleNamespace(
            distributed=SimpleNamespace(sp_size=1, num_gpus=1),
            data=SimpleNamespace(seed=42, train_batch_size=2, data_path="unused"),
            optimizer=SimpleNamespace(
                learning_rate=0.05,
                betas=(0.9, 0.999),
                lr_scheduler="constant",
            ),
            loop=SimpleNamespace(),
            checkpoint=SimpleNamespace(output_dir="/tmp/ftv2-test"),
            pipeline_config=SimpleNamespace(flow_shift=5.0),
        ),
        method={
            "objective": base_objective,
            "v2_objective": v2_objective,
            "transition_mode": transition_mode,
            "behavior_policy": "on_policy",
            "rollout_groups_per_update": 2,
            "posterior_temperature_mode": "global_std",
            "posterior_temperature_multiplier": 1.0,
            "posterior_min_temperature": 1.0e-3,
            "reward_std_decay": 0.9,
            "reward_std_floor": 0.05,
            "target_post_update_kl": 1.0e-5,
            "initial_loss_scale": 1.0,
            "minimum_loss_scale": 0.1,
            "maximum_loss_scale": 100.0,
            "velocity_target_state_shift_rms": 0.05,
            "videoalign_audit": {"enabled": False},
            "validate_raw_model": False,
            "validate_ema_model": False,
            "require_train_eval_schedule_match": True,
            "anchor_type": "local",
            "local_anchor_delta": 0.03,
            "local_noise_scale": 0.7,
            "local_terminal_base_sigma": 0.05,
            "train_map_steps": 2,
            "eval_map_steps": 2,
            "stochastic_steps": 1,
            "group_size": 2,
            "train_t_list_override": [1000.0, 500.0, 0.0],
            "eval_t_list_override": [1000.0, 500.0, 0.0],
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
            "validation": {"every_steps": 0},
            "evaluation": {},
            "ema": {"enabled": False},
        },
    )


def _make_method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    objective: str,
    transition_mode: str = "all",
    equal_rewards: bool = False,
) -> tuple[FiniteTransitionV2Method, _FakeStudent]:
    def build_optimizer(params: list[torch.nn.Parameter], **kwargs: Any):
        del kwargs
        return torch.optim.SGD(params, lr=0.05), _NoopScheduler()

    monkeypatch.setattr(ftp, "build_optimizer_and_scheduler", build_optimizer)
    monkeypatch.setattr(
        ftp,
        "clip_grad_norm_while_handling_failing_dtensor_cases",
        lambda parameters, max_norm, foreach=None: torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, foreach=foreach
        ),
    )
    student = _FakeStudent()
    method = FiniteTransitionV2Method(
        cfg=_config(objective, transition_mode=transition_mode),
        role_models={"student": student},
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(42)
    method._sample_prompt_batch = lambda iteration, local_branches: {
        "info_list": [
            {"prompt": f"a block moves smoothly {iteration}"}
            for _ in range(local_branches)
        ]
    }

    def score(media: torch.Tensor, prompts: list[str]):
        del prompts
        reward = media.float().mean(dim=tuple(range(1, media.ndim)))
        if equal_rewards:
            reward = torch.zeros_like(reward)
        return {
            "videoalign_mq": reward,
            "videoalign_vq": 0.5 * reward,
            "videoalign_ta": 0.25 * reward,
            "avg": reward,
        }

    method._reward_scorer = score
    method._assert_anyflow_two_time_model()
    return method, student


@pytest.mark.parametrize("objective", ["flowmap_grpo", "posterior_projection"])
def test_v2_score_objectives_accumulate_groups_and_update(
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
) -> None:
    method, student = _make_method(monkeypatch, objective=objective)
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert torch.isfinite(loss_map["total_loss"])
    assert not torch.equal(before, after)
    assert float(metrics["ftv2/rollout_groups_per_update"]) == 2.0
    assert float(metrics["ftv2/reward_videos_per_update"]) == 4.0
    assert float(metrics["ftv2/reward_std_ema"]) > 0.0


def test_global_temperature_posterior_skips_equal_reward_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective="posterior_projection",
        equal_rewards=True,
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, _ = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert float(loss_map["total_loss"]) == pytest.approx(0.0)
    assert torch.equal(before, after)


def test_finite_velocity_regression_updates_deterministic_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective="finite_velocity_regression",
        transition_mode="single",
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert torch.isfinite(loss_map["total_loss"])
    assert not torch.equal(before, after)
    assert float(metrics["ftv2/objective_is_velocity_regression"]) == 1.0
    assert "ftv2/velocity_target_state_shift_rms" in metrics
