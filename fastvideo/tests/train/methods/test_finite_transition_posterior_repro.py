# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.pipelines.wan import WanT2V480PConfig
from fastvideo.registry import get_pipeline_config_cls_from_name
from fastvideo.train.methods.rl.finite_transition_posterior_repro import (
    _BASELINE_BUFFER_BYTES,
    _FiniteTransitionRunState,
    _effective_local_anchor_delta_fraction,
)
from fastvideo.train.utils.config import load_run_config


def test_anyflow_checkpoint_resolves_wan_t2v_pipeline_config() -> None:
    resolved = get_pipeline_config_cls_from_name(
        "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
    )
    assert resolved is WanT2V480PConfig


def test_anyflow_cached_snapshot_resolves_wan_t2v_pipeline_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "AnyFlowPipeline",
                "_diffusers_version": "0.35.1",
            }
        ),
        encoding="utf-8",
    )
    resolved = get_pipeline_config_cls_from_name(str(tmp_path))
    assert resolved is WanT2V480PConfig


def test_finite_transition_config_applies_anyflow_arch_overrides() -> None:
    cfg = load_run_config(
        "examples/train/configs/rl/wan/"
        "finite_transition_posterior_anyflow_videoalign.yaml"
    )
    pipeline = cfg.training.pipeline_config
    assert pipeline is not None
    arch = pipeline.dit_config.arch_config
    assert arch.r_embedder is True
    assert arch.r_embedder_fusion == "gated"
    assert arch.r_embedder_gate_value == pytest.approx(0.25)
    assert arch.r_embedder_deltatime_type == "r"

    # Scientific FTPP must train the first three stochastic decisions on the
    # exact four-step deployment grid, then deterministically finish 625 -> 0.
    assert cfg.method["require_train_eval_schedule_match"] is True
    assert cfg.method["train_map_steps"] == 4
    assert cfg.method["eval_map_steps"] == 4
    assert cfg.method["stochastic_steps"] == 3


@pytest.mark.parametrize(
    ("target_time", "expected"),
    [(500.0, 0.03), (24.414066314697266, 0.024414066314697267)],
)
def test_local_anchor_delta_truncates_at_data_endpoint(
    target_time: float,
    expected: float,
) -> None:
    effective = _effective_local_anchor_delta_fraction(
        target_time,
        num_train_timesteps=1000,
        configured_delta_fraction=0.03,
    )

    assert effective == pytest.approx(expected)
    assert target_time - effective * 1000.0 >= -1.0e-5


def test_run_state_round_trip_preserves_scientific_baseline() -> None:
    source = SimpleNamespace(
        _validation_baseline={
            "reward/videoalign_mq": 0.42,
            "temporal_l1": 0.031,
        },
        _validation_best_primary_delta=0.08,
        _steps_to_primary_target=300,
        _cumulative_train_seconds=1234.5,
    )
    state = _FiniteTransitionRunState(source).state_dict()

    assert state["validation_baseline_json"].dtype == torch.uint8
    assert state["validation_baseline_json"].shape == (
        _BASELINE_BUFFER_BYTES,
    )
    assert int(state["validation_baseline_length"]) > 0

    target = SimpleNamespace(
        _validation_baseline={},
        _validation_best_primary_delta=float("-inf"),
        _steps_to_primary_target=-1,
        _cumulative_train_seconds=0.0,
    )
    _FiniteTransitionRunState(target).load_state_dict(state)

    assert target._validation_baseline == pytest.approx(
        source._validation_baseline
    )
    assert target._validation_best_primary_delta == pytest.approx(0.08)
    assert target._steps_to_primary_target == 300
    assert target._cumulative_train_seconds == pytest.approx(1234.5)
