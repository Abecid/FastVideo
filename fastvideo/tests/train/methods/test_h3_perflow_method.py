# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.dataset.h3_perflow_cache import H3PeRFlowSelectionSummary
from fastvideo.train.methods.knowledge_distillation import h3_perflow
from fastvideo.train.methods.knowledge_distillation.h3_perflow import (
    H3PeRFlowMethod,
)
from fastvideo.train.models.minimax_h3.minimax_h3_perflow import (
    MiniMaxH3PeRFlowModel,
)


class _TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))


def _fake_optimizer_builder(*, params, **_kwargs):
    optimizer = torch.optim.SGD(list(params), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    return optimizer, scheduler


def _make_fake_model(
    monkeypatch: pytest.MonkeyPatch,
) -> MiniMaxH3PeRFlowModel:
    monkeypatch.setattr(
        MiniMaxH3PeRFlowModel,
        "device",
        property(lambda _self: torch.device("cpu")),
    )
    model = MiniMaxH3PeRFlowModel.__new__(MiniMaxH3PeRFlowModel)
    model._trainable = True
    model.transformer = _TinyTransformer()
    model.attention_backend = None
    model._rest_student_timesteps = (1000.0, 750.0, 500.0, 250.0, 0.0)
    model._rest_verify_cache_hashes = False
    selection = H3PeRFlowSelectionSummary(
        fingerprint="selection-fingerprint",
        ranking_key="mixed_advantage",
        samples_per_prompt=8,
        selected_per_prompt=2,
        num_prompts=2,
        num_trajectories=4,
        num_segments=4,
        num_examples=16,
        selected_trajectory_ids=("a", "b", "c", "d"),
    )
    model.perflow_cache_dataset = SimpleNamespace(selection_summary=selection)
    model.rest_cache_dataset = SimpleNamespace(
        summary=SimpleNamespace(
            fingerprint="cache-fingerprint",
            num_examples=32,
        )
    )
    model.init_preprocessors = lambda _training_config: None
    model.on_train_start = lambda: None

    def prepare_batch(_raw, *, generator, latents_source):
        del generator
        assert latents_source == "zeros"
        return SimpleNamespace(
            latents=torch.zeros(1, 3, dtype=torch.bfloat16),
            timesteps=torch.tensor([0.0]),
            attn_metadata=None,
            attn_metadata_vsa="vsa-metadata",
        )

    model.prepare_batch = prepare_batch
    model.modality_slices = lambda: (
        ("video", slice(0, 2)),
        ("audio", slice(2, 3)),
    )
    model.unpack_latents = lambda packed: (
        packed[:, :2],
        packed[:, 2:],
    )
    model.noise_amounts = lambda timestep: (
        timestep.float().reshape(-1) / 1000.0,
        (timestep.float().reshape(-1) / 1000.0).square(),
    )

    def refresh_vsa_metadata(batch, *, current_timestep):
        batch.attn_metadata_vsa = f"segment-{current_timestep}"

    model.refresh_vsa_metadata = refresh_vsa_metadata

    def predict_noise(
        packed,
        timestep,
        batch,
        *,
        conditional,
        attn_kind,
    ):
        assert conditional
        assert attn_kind == "vsa"
        batch.timesteps = timestep
        model.last_query = packed.detach().clone()
        model.last_timestep = timestep.detach().clone()
        return packed.float() * model.transformer.scale

    model.predict_noise = predict_noise
    model.backward = lambda loss, _ctx, *, grad_accum_rounds: (
        loss / grad_accum_rounds
    ).backward()
    return model


def _make_method(
    monkeypatch: pytest.MonkeyPatch,
) -> H3PeRFlowMethod:
    monkeypatch.setattr(
        h3_perflow,
        "build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    model = _make_fake_model(monkeypatch)
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            optimizer=SimpleNamespace(
                learning_rate=0.1,
                betas=(0.9, 0.999),
                lr_scheduler="constant",
            ),
            loop=SimpleNamespace(max_train_steps=2),
            data=SimpleNamespace(seed=3),
            distributed=SimpleNamespace(sp_size=1),
        ),
        method={
            "_target_": (
                "fastvideo.train.methods.knowledge_distillation."
                "h3_perflow.H3PeRFlowMethod"
            ),
            "attn_kind": "vsa",
            "loss_type": "mse",
            "audio_loss_weight": 1.0,
            "function_anchor_weight": 0.0,
        },
        validation={},
    )
    method = H3PeRFlowMethod(
        cfg=cfg,
        role_models={"student": model},
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(11)
    return method


def _batch(*, advantage: float = 4.0) -> dict[str, object]:
    return {
        "trajectory_current": torch.tensor(
            [[1.0, 2.0, 3.0]],
            dtype=torch.bfloat16,
        ),
        "trajectory_next": torch.tensor(
            [[0.0, 1.0, 2.0]],
            dtype=torch.bfloat16,
        ),
        "trajectory_timestep": torch.tensor([1000.0]),
        "trajectory_next_timestep": torch.tensor([750.0]),
        "rest_segment_index": torch.tensor([0]),
        "rest_mixed_advantage": torch.tensor([advantage]),
        "rest_reward_scores": {"quality": advantage},
        "perflow_selection_weight": torch.tensor([0.5]),
        "perflow_selection_rank": torch.tensor([0]),
        "perflow_selection_score": torch.tensor([advantage]),
        "perflow_selection_fingerprint": "selection-fingerprint",
    }


def test_method_runs_continuous_segment_step_and_backpropagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = _make_method(monkeypatch)
    losses, outputs, metrics = method.single_train_step(
        _batch(),
        iteration=1,
    )

    assert torch.isfinite(losses["total_loss"])
    assert 750.0 <= float(metrics["perflow/query_timestep"]) <= 1000.0
    assert 0.0 <= float(metrics["perflow/shared_fraction"]) < 1.0
    assert metrics["perflow/video_fraction"] != metrics["perflow/audio_fraction"]
    assert method.student.last_query.shape == (1, 3)
    assert outputs["student_ctx"][1] == "segment-0"

    method.backward(losses, outputs)
    assert method.student.transformer.scale.grad is not None
    assert torch.isfinite(method.student.transformer.scale.grad)


def test_offline_reward_is_diagnostic_not_signed_loss_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = _make_method(monkeypatch)
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(19)
    positive, _outputs, _metrics = method.single_train_step(
        _batch(advantage=8.0),
        iteration=1,
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(19)
    negative, _outputs, _metrics = method.single_train_step(
        _batch(advantage=-8.0),
        iteration=1,
    )
    torch.testing.assert_close(
        positive["total_loss"],
        negative["total_loss"],
    )
    assert positive["total_loss"].item() >= 0.0


def test_selection_fingerprint_and_segment_grid_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = _make_method(monkeypatch)
    wrong_fingerprint = _batch()
    wrong_fingerprint["perflow_selection_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        method.single_train_step(wrong_fingerprint, iteration=1)

    wrong_grid = _batch()
    wrong_grid["trajectory_next_timestep"] = torch.tensor([700.0])
    with pytest.raises(ValueError, match="timestep mismatch"):
        method.single_train_step(wrong_grid, iteration=1)
