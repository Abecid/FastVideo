# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from fastvideo.train.utils.config import load_run_config

FINAL_TARGET = (
    "fastvideo.train.methods.rl.finite_transition_v2_final."
    "FiniteTransitionV2FinalMethod"
)


@pytest.mark.parametrize(
    ("filename", "v2_objective", "train_steps", "eval_steps", "transition_mode"),
    [
        (
            "finite_transition_grpo_v2_anyflow_videoalign.yaml",
            "flowmap_grpo",
            5,
            4,
            "all",
        ),
        (
            "finite_transition_posterior_v2_anyflow_videoalign.yaml",
            "posterior_projection",
            5,
            4,
            "all",
        ),
        (
            "finite_transition_velocity_v2_anyflow_videoalign.yaml",
            "finite_velocity_regression",
            4,
            4,
            "single",
        ),
        (
            "finite_transition_grpo_v2_diagnostic_luminance.yaml",
            "flowmap_grpo",
            5,
            4,
            "all",
        ),
    ],
)
def test_v2_recipe_contracts(
    filename: str,
    v2_objective: str,
    train_steps: int,
    eval_steps: int,
    transition_mode: str,
) -> None:
    cfg = load_run_config(f"examples/train/configs/rl/wan/{filename}")
    method = cfg.method
    assert method["_target_"] == FINAL_TARGET
    assert method["v2_objective"] == v2_objective
    assert method["train_map_steps"] == train_steps
    assert method["eval_map_steps"] == eval_steps
    assert method["transition_mode"] == transition_mode
    assert method["group_size"] == 8
    assert method["rollout_groups_per_update"] == 4
    assert cfg.training.data.train_batch_size == 2

    pipeline = cfg.training.pipeline_config
    assert pipeline is not None
    arch = pipeline.dit_config.arch_config
    assert arch.r_embedder is True
    assert arch.r_embedder_gate_value == pytest.approx(0.25)
    assert arch.r_embedder_deltatime_type == "r"


def test_videoalign_v2_uses_mq_online_and_vq_ta_only_at_validation() -> None:
    for filename in (
        "finite_transition_grpo_v2_anyflow_videoalign.yaml",
        "finite_transition_posterior_v2_anyflow_videoalign.yaml",
        "finite_transition_velocity_v2_anyflow_videoalign.yaml",
    ):
        cfg = load_run_config(f"examples/train/configs/rl/wan/{filename}")
        method = cfg.method
        assert method["optimize_reward"] == "videoalign_mq_audited"
        assert set(method["reward_fn"]["rewards"]) == {
            "videoalign_mq_audited"
        }
        assert set(method["validation_reward_fn"]["rewards"]) == {
            "videoalign_mq_audited",
            "videoalign_vq_audited",
            "videoalign_ta_audited",
        }
        assert method["videoalign_audit"]["require_reward_head"] is True
        assert method["evaluation"]["samples_per_prompt"] == 2


def test_luminance_gate_has_no_videoalign_dependency() -> None:
    cfg = load_run_config(
        "examples/train/configs/rl/wan/"
        "finite_transition_grpo_v2_diagnostic_luminance.yaml"
    )
    method = cfg.method
    assert method["optimize_reward"] == "mean_luminance"
    assert set(method["reward_fn"]["rewards"]) == {"mean_luminance"}
    assert method["videoalign_audit"]["enabled"] is False
    assert cfg.training.data.num_frames == 17
