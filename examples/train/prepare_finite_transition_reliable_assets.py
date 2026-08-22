# SPDX-License-Identifier: Apache-2.0
"""Prepare data, rewards, and resolved configs for reliable AnyFlow RL.

Recipes:

- ``reliable``: full-trajectory GRPO or posterior likelihood update;
- ``velocity``: single-transition finite-velocity posterior regression;
- ``sanity``: cheap mean-luminance learnability gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

from examples.train.prepare_diffusion_nft_assets import (
    check_reward_runtime,
    derive_wan_num_latent_t,
    ensure_videoalign_checkpoint,
)
from examples.train.prepare_finite_transition_posterior_assets import (
    prepare_split,
)

MODEL_ID = "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
CONFIGS = {
    "reliable": Path(
        "examples/train/configs/rl/wan/"
        "finite_transition_reliable_anyflow_videoalign.yaml"
    ),
    "velocity": Path(
        "examples/train/configs/rl/wan/"
        "finite_transition_velocity_anyflow_videoalign.yaml"
    ),
    "sanity": Path(
        "examples/train/configs/rl/wan/"
        "finite_transition_reliable_sanity_luminance.yaml"
    ),
}


def reward_setup(recipe: str) -> tuple[dict[str, float], str, str]:
    if recipe == "sanity":
        return {"mean_luminance": 1.0}, "auto", "mean_luminance"
    rewards = {
        "videoalign_mq_audited": 1.0,
        "videoalign_vq_audited": 1.0,
        "videoalign_ta_audited": 1.0,
    }
    return rewards, "genrl", "videoalign_mq_audited"


def recipe_defaults(recipe: str) -> dict[str, Any]:
    if recipe == "sanity":
        return {
            "num_frames": 17,
            "num_height": 256,
            "num_width": 448,
            "group_size": 4,
            "rollout_groups_per_update": 2,
            "lora_rank": 32,
            "lora_alpha": 64,
            "learning_rate": 2.0e-5,
            "max_train_steps": 50,
            "validation_prompts": 16,
            "validation_every": 10,
            "validation_samples_per_prompt": 1,
            "checkpoint_every": 10,
        }
    if recipe == "velocity":
        return {
            "num_frames": 81,
            "num_height": 480,
            "num_width": 832,
            "group_size": 8,
            "rollout_groups_per_update": 4,
            "lora_rank": 64,
            "lora_alpha": 128,
            "learning_rate": 1.0e-5,
            "max_train_steps": 100,
            "validation_prompts": 64,
            "validation_every": 25,
            "validation_samples_per_prompt": 2,
            "checkpoint_every": 25,
        }
    return {
        "num_frames": 81,
        "num_height": 480,
        "num_width": 832,
        "group_size": 8,
        "rollout_groups_per_update": 4,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 2.0e-5,
        "max_train_steps": 100,
        "validation_prompts": 64,
        "validation_every": 50,
        "validation_samples_per_prompt": 2,
        "checkpoint_every": 25,
    }


def apply_recipe_defaults(args: argparse.Namespace) -> None:
    defaults = recipe_defaults(args.recipe)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def write_run_config(
    args: argparse.Namespace,
    *,
    train_parquet: Path,
    validation_parquet: Path,
    train_count: int,
    validation_count: int,
    num_latent_t: int,
    reward_map: dict[str, float],
    reward_backend: str,
    optimize_reward: str,
) -> Path:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = config.setdefault("models", {})
    student = models.setdefault("student", {})
    student["init_from"] = args.model_id
    lora = student.setdefault("lora", {})
    lora["enable"] = True
    lora["rank"] = int(args.lora_rank)
    lora["alpha"] = int(args.lora_alpha)

    method = config.setdefault("method", {})
    if args.recipe == "reliable":
        method["objective"] = args.objective
    elif args.recipe == "velocity":
        method["objective"] = "finite_velocity_regression"
    else:
        method["objective"] = "flowmap_grpo"
    method["group_size"] = int(args.group_size)
    method["rollout_groups_per_update"] = int(
        args.rollout_groups_per_update
    )
    method["behavior_policy"] = args.behavior_policy
    method["reward_backend"] = reward_backend
    method["reward_fn"] = {"rewards": reward_map}
    method["optimize_reward"] = optimize_reward
    method["posterior_temperature_mode"] = (
        args.posterior_temperature_mode
    )
    method["posterior_temperature_scale"] = float(
        args.posterior_temperature_scale
    )
    controller = method.setdefault("target_kl_controller", {})
    controller["enabled"] = bool(args.target_kl_enabled)
    controller["target_kl"] = float(args.target_kl)
    controller["initial_loss_scale"] = float(args.initial_loss_scale)
    controller["min_loss_scale"] = float(args.min_loss_scale)
    controller["max_loss_scale"] = float(args.max_loss_scale)

    validation = method.setdefault("validation", {})
    validation["data_path"] = str(validation_parquet)
    validation["num_prompts"] = min(
        int(args.validation_prompts),
        int(validation_count),
    )
    validation["every_steps"] = int(args.validation_every)
    validation["batch_size"] = int(args.validation_batch_size)
    validation["max_samples"] = int(args.validation_log_videos)
    validation["log_samples"] = bool(args.validation_log_videos > 0)
    evaluation = method.setdefault("evaluation", {})
    evaluation["samples_per_prompt"] = int(
        args.validation_samples_per_prompt
    )
    evaluation["primary_min_delta"] = float(args.primary_min_delta)
    paired = method.setdefault("paired_validation", {})
    paired["bootstrap_samples"] = int(args.bootstrap_samples)
    paired["confidence"] = float(args.bootstrap_confidence)

    training = config.setdefault("training", {})
    distributed = training.setdefault("distributed", {})
    distributed["num_gpus"] = int(args.num_gpus)
    distributed["sp_size"] = 1
    distributed["tp_size"] = 1
    distributed["hsdp_replicate_dim"] = int(args.hsdp_replicate_dim)
    distributed["hsdp_shard_dim"] = int(args.hsdp_shard_dim)

    data = training.setdefault("data", {})
    data["data_path"] = str(train_parquet)
    data["preprocessed_data_type"] = "text_only"
    data["dataloader_num_workers"] = int(args.dataloader_num_workers)
    data["train_batch_size"] = max(
        1,
        int(args.group_size) // int(args.num_gpus),
    )
    data["seed"] = int(args.seed)
    data["num_frames"] = int(args.num_frames)
    data["num_latent_t"] = int(num_latent_t)
    data["num_height"] = int(args.num_height)
    data["num_width"] = int(args.num_width)

    optimizer = training.setdefault("optimizer", {})
    optimizer["learning_rate"] = float(args.learning_rate)
    optimizer["betas"] = [float(args.beta1), float(args.beta2)]
    optimizer["weight_decay"] = float(args.weight_decay)

    loop = training.setdefault("loop", {})
    loop["max_train_steps"] = int(args.max_train_steps)
    loop["gradient_accumulation_steps"] = 1

    checkpoint = training.setdefault("checkpoint", {})
    checkpoint["output_dir"] = str(args.output_dir)
    checkpoint["training_state_checkpointing_steps"] = int(
        args.checkpoint_every
    )
    checkpoint["checkpoints_total_limit"] = int(
        args.checkpoints_total_limit
    )

    tracker = training.setdefault("tracker", {})
    tracker["project_name"] = args.project_name
    tracker["run_name"] = args.run_name or args.output_dir.name

    args.run_config_dir.mkdir(parents=True, exist_ok=True)
    run_config = args.run_config_dir / (
        f"finite_transition_{args.recipe}_{method['objective']}_"
        f"train{train_count}_val{validation_count}.yaml"
    )
    run_config.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=tuple(CONFIGS),
        default="reliable",
    )
    parser.add_argument(
        "--objective",
        choices=("flowmap_grpo", "posterior_projection"),
        default="flowmap_grpo",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/ftr"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/ftr"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ftr"))
    parser.add_argument(
        "--run-config-dir",
        type=Path,
        default=Path("outputs/ftr_configs"),
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", default="world-r1-enhanced-dynamic")
    parser.add_argument("--max-train-prompts", default="256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-latent-t", type=int, default=0)
    parser.add_argument("--num-height", type=int)
    parser.add_argument("--num-width", type=int)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--hsdp-replicate-dim", type=int, default=1)
    parser.add_argument("--hsdp-shard-dim", type=int, default=4)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--rollout-groups-per-update", type=int)
    parser.add_argument(
        "--behavior-policy",
        choices=("current", "base_adapter_disabled"),
        default="current",
    )
    parser.add_argument(
        "--posterior-temperature-mode",
        choices=("global_std", "fixed_ess"),
        default="global_std",
    )
    parser.add_argument("--posterior-temperature-scale", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=1.0e-5)
    parser.add_argument("--initial-loss-scale", type=float, default=1.0)
    parser.add_argument("--min-loss-scale", type=float, default=0.05)
    parser.add_argument("--max-loss-scale", type=float, default=128.0)
    parser.add_argument(
        "--target-kl-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--checkpoints-total-limit", type=int, default=4)
    parser.add_argument("--validation-prompts", type=int)
    parser.add_argument("--validation-every", type=int)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--validation-samples-per-prompt", type=int)
    parser.add_argument("--validation-log-videos", type=int, default=8)
    parser.add_argument("--primary-min-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--preprocess-batch-size", type=int, default=128)
    parser.add_argument("--preprocess-num-gpus", type=int, default=1)
    parser.add_argument("--preprocess-master-port", type=int, default=29571)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--samples-per-file", type=int, default=1024)
    parser.add_argument("--flush-frequency", type=int, default=1024)
    parser.add_argument("--project-name", default="finite-transition-reliable-wan")
    parser.add_argument("--run-name")
    parser.add_argument("--diffusion-nft-root", type=Path)
    parser.add_argument("--videoalign-checkpoint-path", type=Path)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--check-rewards", action="store_true")
    parser.add_argument("--reward-device", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    if args.config is None:
        args.config = CONFIGS[args.recipe]
    for name in (
        "config",
        "data_root",
        "cache_root",
        "output_dir",
        "run_config_dir",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            value = (args.repo_root / value).resolve()
        setattr(args, name, value)
    args.diffusion_nft_root = args.diffusion_nft_root or (
        args.cache_root / "DiffusionNFT"
    )
    args.videoalign_checkpoint_path = (
        args.videoalign_checkpoint_path
        or args.cache_root / "VideoReward"
    )
    apply_recipe_defaults(args)
    return args


def validate_args(args: argparse.Namespace) -> int:
    if args.preprocess_num_gpus != 1:
        raise ValueError("text preprocessing currently requires one GPU")
    if args.num_gpus <= 0 or args.group_size <= 1:
        raise ValueError("num-gpus must be positive and group-size > 1")
    if args.group_size % args.num_gpus != 0:
        raise ValueError("group-size must be divisible by num-gpus")
    if args.rollout_groups_per_update <= 0:
        raise ValueError("rollout-groups-per-update must be positive")
    if args.hsdp_replicate_dim * args.hsdp_shard_dim != args.num_gpus:
        raise ValueError(
            "hsdp-replicate-dim * hsdp-shard-dim must equal num-gpus"
        )
    if args.recipe == "velocity":
        args.target_kl_enabled = False
    derived = derive_wan_num_latent_t(args.num_frames)
    if args.num_latent_t not in (0, derived):
        raise ValueError(
            f"num_frames={args.num_frames} implies num_latent_t={derived}"
        )
    return derived


def main() -> None:
    args = parse_args()
    num_latent_t = validate_args(args)
    reward_map, reward_backend, optimize_reward = reward_setup(args.recipe)

    if any(name.startswith("videoalign_") for name in reward_map):
        ensure_videoalign_checkpoint(args.videoalign_checkpoint_path)
        os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = str(
            args.videoalign_checkpoint_path
        )
    if args.diffusion_nft_root:
        os.environ["DIFFUSION_NFT_ROOT"] = str(args.diffusion_nft_root)
    if args.check_rewards:
        check_reward_runtime(
            reward_map,
            reward_backend=reward_backend,
            device=args.reward_device,
        )

    (
        train_parquet,
        validation_parquet,
        train_count,
        validation_count,
        prompt_source,
    ) = prepare_split(args, num_latent_t=num_latent_t)
    run_config = write_run_config(
        args,
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        train_count=train_count,
        validation_count=validation_count,
        num_latent_t=num_latent_t,
        reward_map=reward_map,
        reward_backend=reward_backend,
        optimize_reward=optimize_reward,
    )

    summary = {
        "recipe": args.recipe,
        "objective": (
            args.objective if args.recipe == "reliable" else args.recipe
        ),
        "prompt_source": prompt_source,
        "train_prompt_count": train_count,
        "validation_prompt_count": validation_count,
        "train_parquet": str(train_parquet),
        "validation_parquet": str(validation_parquet),
        "run_config": str(run_config),
        "output_dir": str(args.output_dir),
        "num_frames": int(args.num_frames),
        "num_latent_t": int(num_latent_t),
        "group_size": int(args.group_size),
        "rollout_groups_per_update": int(args.rollout_groups_per_update),
        "reward_samples_per_update": int(
            args.group_size * args.rollout_groups_per_update
        ),
        "behavior_policy": args.behavior_policy,
        "target_kl": float(args.target_kl),
        "reward_map": reward_map,
    }
    print("Prepared reliable finite-transition assets:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
