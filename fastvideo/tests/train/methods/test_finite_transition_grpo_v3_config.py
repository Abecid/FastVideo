# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from fastvideo.train.utils.config import load_run_config


def test_grpo_v3_config_is_mq_only_online_and_multiminibatch() -> None:
    cfg = load_run_config(
        "examples/train/configs/rl/wan/"
        "finite_transition_grpo_v3_anyflow_videoalign.yaml"
    )
    method = cfg.method

    assert method["_target_"] == (
        "fastvideo.train.methods.rl.finite_transition_grpo_v3."
        "FiniteTransitionGRPOV3Method"
    )
    assert method["objective"] == "flowmap_grpo"
    assert method["v2_objective"] == "flowmap_grpo"
    assert method["transition_mode"] == "all"
    assert method["behavior_policy"] == "on_policy"
    assert method["policy_epochs"] == 2
    assert method["groups_per_minibatch"] == 1
    assert method["rollout_groups_per_update"] == 4
    assert method["group_size"] == 8

    assert set(method["reward_fn"]["rewards"]) == {
        "videoalign_mq_audited"
    }
    assert method["optimize_reward"] == "videoalign_mq_audited"
    assert set(method["validation_reward_fn"]["rewards"]) == {
        "videoalign_mq_audited",
        "videoalign_vq_audited",
        "videoalign_ta_audited",
    }

    assert cfg.training.optimizer.learning_rate == pytest.approx(1.0e-5)
    assert cfg.training.loop.max_train_steps == 20
    assert cfg.training.pipeline_config is not None
    arch = cfg.training.pipeline_config.dit_config.arch_config
    assert arch.r_embedder is True
    assert arch.r_embedder_gate_value == pytest.approx(0.25)
    assert arch.r_embedder_deltatime_type == "r"
