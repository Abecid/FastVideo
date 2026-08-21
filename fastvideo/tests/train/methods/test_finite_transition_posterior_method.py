# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fastvideo.pipelines import TrainingBatch
import fastvideo.train.methods.rl.finite_transition_posterior as ftp
from fastvideo.train.methods.rl.finite_transition_posterior import (
    _prepare_validation_log_entry,
)
from fastvideo.train.methods.rl.finite_transition_posterior_repro import (
    ReproducibleFiniteTransitionPosteriorMethod,
)


class _NoopScheduler:

    def step(self) -> None:
        pass


class _FakeAnyFlowTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.15))
        self._r_embedder_enabled = True


class _FakeAnyFlowStudent:
    _trainable = True
    device = torch.device("cpu")
    num_train_timesteps = 1000

    def __init__(self) -> None:
        self.transformer = _FakeAnyFlowTransformer()

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


def _config(objective: str) -> SimpleNamespace:
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
                betas=(0.0, 0.999),
                lr_scheduler="constant",
            ),
            loop=SimpleNamespace(),
            pipeline_config=SimpleNamespace(flow_shift=5.0),
        ),
        method={
            "objective": objective,
            "anchor_type": "local",
            "local_anchor_delta": 0.03,
            "local_noise_scale": 0.7,
            "local_terminal_base_sigma": 0.05,
            "train_map_steps": 2,
            "eval_map_steps": 1,
            "stochastic_steps": 1,
            "group_size": 2,
            "train_t_list_override": [1000.0, 500.0, 0.0],
            "eval_t_list_override": [1000.0, 0.0],
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
            "post_update_probe_every": 1,
        },
    )


def _make_method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    objective: str,
    equal_rewards: bool,
    local_anchor: bool = False,
) -> tuple[ftp.FiniteTransitionPosteriorMethod, _FakeAnyFlowStudent]:
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

    student = _FakeAnyFlowStudent()
    method_cls = (
        ReproducibleFiniteTransitionPosteriorMethod
        if local_anchor
        else ftp.FiniteTransitionPosteriorMethod
    )
    method = method_cls(
        cfg=_config(objective),
        role_models={"student": student},
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(42)
    method._sample_prompt_batch = lambda iteration, local_branches: {
        "info_list": [
            {"prompt": "a red block moves from left to right"}
            for _ in range(local_branches)
        ]
    }

    def score(media: torch.Tensor, prompts: list[str]):
        del prompts
        base = media.float().mean(dim=tuple(range(1, media.ndim)))
        if equal_rewards:
            base = torch.zeros_like(base)
        return {
            "videoalign_mq": base,
            "videoalign_vq": 0.5 * base,
            "videoalign_ta": 0.25 * base,
            "avg": base,
        }

    method._reward_scorer = score
    method._assert_anyflow_two_time_model()
    return method, student


@pytest.mark.parametrize(
    "objective",
    ["posterior_projection", "flowmap_grpo"],
)
def test_finite_transition_method_updates_with_informative_reward(
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective=objective,
        equal_rewards=False,
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert torch.isfinite(loss_map["total_loss"])
    assert not torch.equal(before, after)
    assert float(metrics["ftp/reward_std"]) > 0.0
    assert float(metrics["ftp/posterior_ess"]) >= 1.0


def test_local_anchor_method_updates_and_logs_anchor_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective="posterior_projection",
        equal_rewards=False,
        local_anchor=True,
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert torch.isfinite(loss_map["total_loss"])
    assert not torch.equal(before, after)
    assert float(metrics["ftp/anchor_is_local"]) == 1.0
    assert float(metrics["ftp/local_anchor_timestep"]) == pytest.approx(470.0)
    assert float(metrics["ftp/local_noise_scale"]) == pytest.approx(0.7)


def test_posterior_projection_has_zero_update_for_equal_rewards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, student = _make_method(
        monkeypatch,
        objective="posterior_projection",
        equal_rewards=True,
        local_anchor=True,
    )
    before = student.transformer.scale.detach().clone()
    loss_map, _, metrics = method.managed_train_step(iter(()), 1)
    after = student.transformer.scale.detach().clone()

    assert float(loss_map["total_loss"].detach()) == pytest.approx(0.0)
    assert torch.equal(before, after)
    assert float(metrics["ftp/zero_std_group"]) == 1.0


def test_validation_log_entry_caps_before_compacting_media() -> None:
    media = torch.rand(3, 5, 8, 12)

    skipped = _prepare_validation_log_entry(
        index=8,
        prompt="skipped",
        media=media,
        rewards={"videoalign_mq": 0.0},
        max_samples=8,
    )
    selected = _prepare_validation_log_entry(
        index=7,
        prompt="selected",
        media=media,
        rewards={"videoalign_mq": 1.0},
        max_samples=8,
    )

    assert skipped is None
    assert selected is not None
    assert isinstance(selected["media"], np.ndarray)
    assert selected["media"].dtype == np.uint8
    assert selected["media"].shape == (5, 3, 8, 12)
