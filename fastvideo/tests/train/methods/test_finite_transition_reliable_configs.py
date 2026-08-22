# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from fastvideo.train.utils.config import load_run_config


@pytest.mark.parametrize(
    ("filename", "objective", "train_steps", "eval_steps", "stochastic"),
    [
        (
            "finite_transition_reliable_anyflow_videoalign.yaml",
            "flowmap_grpo",
            5,
            4,
            4,
        ),
        (
            "finite_transition_velocity_anyflow_videoalign.yaml",
            "finite_velocity_regression",
            4,
            4,
            3,
        ),
        (
            "finite_transition_reliable_sanity_luminance.yaml",
            "flowmap_grpo",
            5,
            4,
            4,
        ),
    ],
)
def test_reliable_recipe_contracts(
    filename: str,
    objective: str,
    train_steps: int,
    eval_steps: int,
    stochastic: int,
) -> None:
    cfg = load_run_config(
        f"examples/train/configs/rl/wan/{filename}"
    )
    method = cfg.method
    assert method["_target_"] == (
        "fastvideo.train.methods.rl.finite_transition_reliable_audited."
        "AuditedReliableFiniteTransitionMethod"
    )
    assert method["objective"] == objective
    assert method["train_map_steps"] == train_steps
    assert method["eval_map_steps"] == eval_steps
    assert method["stochastic_steps"] == stochastic
    assert cfg.training.pipeline_config is not None
    arch = cfg.training.pipeline_config.dit_config.arch_config
    assert arch.r_embedder is True
    assert arch.r_embedder_gate_value == pytest.approx(0.25)
    assert arch.r_embedder_deltatime_type == "r"


def test_reliable_videoalign_uses_one_training_head_and_three_eval_heads() -> None:
    cfg = load_run_config(
        "examples/train/configs/rl/wan/"
        "finite_transition_reliable_anyflow_videoalign.yaml"
    )
    method = cfg.method
    assert set(method["reward_fn"]["rewards"]) == {
        "videoalign_mq_audited"
    }
    assert set(method["validation_reward_fn"]["rewards"]) == {
        "videoalign_mq_audited",
        "videoalign_vq_audited",
        "videoalign_ta_audited",
    }
