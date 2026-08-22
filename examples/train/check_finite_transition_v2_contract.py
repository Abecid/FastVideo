# SPDX-License-Identifier: Apache-2.0
"""Fail fast when a finite-transition v2 config violates scientific invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastvideo.train.methods.rl.common.anyflow_schedule import (
    anyflow_inference_schedule,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import resolve_target

FINAL_TARGET = (
    "fastvideo.train.methods.rl.finite_transition_v2_final."
    "FiniteTransitionV2FinalMethod"
)


def _reward_names(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    raw = value.get("rewards", value)
    if not isinstance(raw, dict):
        return set()
    return {str(name) for name in raw}


def check_config(path: Path) -> dict[str, Any]:
    cfg = load_run_config(str(path))
    method = cfg.method
    target = str(method.get("_target_", ""))
    target_cls = resolve_target(target)
    if target != FINAL_TARGET:
        raise RuntimeError(
            f"V2 config must use {FINAL_TARGET}, got {target!r}"
        )

    world_size = int(cfg.training.distributed.num_gpus or 1)
    group_size = int(method["group_size"])
    local_branches = group_size // world_size
    if group_size <= 1 or group_size % world_size != 0:
        raise RuntimeError(
            "method.group_size must exceed one and be divisible by num_gpus"
        )
    if int(cfg.training.data.train_batch_size) != local_branches:
        raise RuntimeError(
            "training.data.train_batch_size must equal group_size / num_gpus: "
            f"{cfg.training.data.train_batch_size} != {local_branches}"
        )

    train_steps = int(method["train_map_steps"])
    eval_steps = int(method["eval_map_steps"])
    stochastic_steps = int(method["stochastic_steps"])
    if train_steps != stochastic_steps + 1:
        raise RuntimeError(
            "train_map_steps must equal stochastic_steps + one deterministic "
            "completion"
        )
    transition_mode = str(method.get("transition_mode", "all"))
    objective = str(method.get("v2_objective", method.get("objective", "")))
    require_match = bool(method.get("require_train_eval_schedule_match", False))
    if objective == "finite_velocity_regression":
        if transition_mode != "single":
            raise RuntimeError(
                "finite_velocity_regression requires transition_mode=single"
            )
        if not require_match or train_steps != eval_steps:
            raise RuntimeError(
                "finite-velocity regression must use the deployed eval grid"
            )
    elif transition_mode == "all" and require_match:
        raise RuntimeError(
            "full-trajectory likelihood training should not require the four-"
            "step eval grid; it uses four stochastic transitions plus completion"
        )

    pipeline = cfg.training.pipeline_config
    if pipeline is None:
        raise RuntimeError("V2 config is missing pipeline configuration")
    arch = pipeline.dit_config.arch_config
    if not bool(getattr(arch, "r_embedder", False)):
        raise RuntimeError("AnyFlow r_embedder must be enabled")
    gate = float(getattr(arch, "r_embedder_gate_value", float("nan")))
    if abs(gate - 0.25) > 1.0e-6:
        raise RuntimeError(f"Expected AnyFlow r-embedder gate 0.25, got {gate}")
    if str(getattr(arch, "r_embedder_deltatime_type", "")) != "r":
        raise RuntimeError("AnyFlow r_embedder_deltatime_type must be 'r'")
    if abs(float(pipeline.flow_shift) - 5.0) > 1.0e-6:
        raise RuntimeError("Released AnyFlow scientific configs require shift 5")

    eval_schedule = anyflow_inference_schedule(
        steps=eval_steps,
        flow_shift=float(pipeline.flow_shift),
    )
    if eval_steps == 4:
        expected = [1000.0, 937.5, 833.3333333333, 625.0, 0.0]
        actual = [float(value) for value in eval_schedule]
        for observed, wanted in zip(actual, expected, strict=True):
            if abs(observed - wanted) > 0.01:
                raise RuntimeError(
                    f"Official AnyFlow four-step schedule changed: {actual}"
                )

    behavior = str(method.get("behavior_policy", "on_policy"))
    if behavior not in {"on_policy", "frozen_base"}:
        raise RuntimeError("behavior_policy must be on_policy or frozen_base")
    if int(method.get("rollout_groups_per_update", 0)) <= 0:
        raise RuntimeError("rollout_groups_per_update must be positive")

    train_rewards = _reward_names(method.get("reward_fn"))
    validation_rewards = _reward_names(method.get("validation_reward_fn"))
    optimize_reward = str(method.get("optimize_reward", "avg"))
    uses_videoalign = any(
        name.startswith("videoalign_")
        for name in train_rewards | validation_rewards
    )
    if uses_videoalign:
        if optimize_reward != "videoalign_mq_audited":
            raise RuntimeError(
                "Scientific VideoAlign configs must optimize audited MQ"
            )
        if train_rewards != {"videoalign_mq_audited"}:
            raise RuntimeError(
                "Online VideoAlign rollouts must score MQ only"
            )
        required_validation = {
            "videoalign_mq_audited",
            "videoalign_vq_audited",
            "videoalign_ta_audited",
        }
        if validation_rewards != required_validation:
            raise RuntimeError(
                "Held-out validation must score audited MQ/VQ/TA"
            )
        audit = method.get("videoalign_audit", {}) or {}
        if not bool(audit.get("enabled", False)):
            raise RuntimeError("VideoAlign audit must be enabled")
        if not bool(audit.get("require_reward_head", False)):
            raise RuntimeError("VideoAlign audit must require a reward head")

    validation = method.get("validation", {}) or {}
    evaluation = method.get("evaluation", {}) or {}
    if int(validation.get("num_prompts", 0)) <= 0:
        raise RuntimeError("validation.num_prompts must be positive")
    if int(evaluation.get("samples_per_prompt", 0)) <= 0:
        raise RuntimeError("evaluation.samples_per_prompt must be positive")
    if not bool(method.get("validate_raw_model", False)):
        raise RuntimeError("V2 must validate raw weights")
    if not bool(method.get("validate_ema_model", False)):
        raise RuntimeError("V2 must validate EMA weights")

    return {
        "config": str(path),
        "method_class": target_cls.__name__,
        "objective": objective,
        "transition_mode": transition_mode,
        "behavior_policy": behavior,
        "group_size": group_size,
        "local_branches": local_branches,
        "rollout_groups_per_update": int(
            method["rollout_groups_per_update"]
        ),
        "reward_videos_per_update": group_size
        * int(method["rollout_groups_per_update"]),
        "train_map_steps": train_steps,
        "eval_map_steps": eval_steps,
        "stochastic_steps": stochastic_steps,
        "eval_schedule": [float(value) for value in eval_schedule],
        "online_rewards": sorted(train_rewards),
        "validation_rewards": sorted(validation_rewards),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = check_config(args.config.resolve())
    print("Finite-transition v2 contract passed:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
