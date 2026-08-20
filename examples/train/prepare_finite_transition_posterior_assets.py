# SPDX-License-Identifier: Apache-2.0
"""Prepare held-out prompts, rewards and a run config for FTPP training.

Unlike the generic DiffusionNFT asset helper, this script creates a deterministic
train/validation prompt split.  The fixed validation parquet is never sampled by
training, so W&B reward deltas measure held-out improvement rather than prompt
memorization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import yaml

from examples.train.prepare_diffusion_nft_assets import (
    check_reward_runtime,
    derive_wan_num_latent_t,
    ensure_diffusion_nft_repo,
    ensure_videoalign_checkpoint,
    has_parquet,
    load_prompts,
    resolve_max_prompts,
    verify_text_only_dataset,
)

MODEL_ID = "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_CONFIG = (
    "examples/train/configs/rl/wan/"
    "finite_transition_posterior_anyflow_videoalign.yaml"
)
GENRL_REWARDS = frozenset(
    {
        "hpsv3_general",
        "hpsv3_percentile",
        "videoalign_mq",
        "videoalign_vq",
        "videoalign_ta",
    }
)


def resolve_reward_setup(reward: str) -> tuple[dict[str, float], str, str]:
    """Return scorer map, backend and the single optimized reward key."""
    normalized = str(reward).strip().lower()
    if normalized in {"mq", "videoalign_mq", "motion"}:
        reward_map = {
            "videoalign_mq": 1.0,
            "videoalign_vq": 1.0,
            "videoalign_ta": 1.0,
        }
        optimize = "videoalign_mq"
    elif normalized in {"ta", "videoalign_ta", "text"}:
        reward_map = {
            "videoalign_mq": 1.0,
            "videoalign_vq": 1.0,
            "videoalign_ta": 1.0,
        }
        optimize = "videoalign_ta"
    elif normalized in {"vq", "videoalign_vq", "quality"}:
        reward_map = {
            "videoalign_mq": 1.0,
            "videoalign_vq": 1.0,
            "videoalign_ta": 1.0,
        }
        optimize = "videoalign_vq"
    else:
        reward_map = {normalized: 1.0}
        optimize = normalized
    backend = (
        "genrl"
        if any(name in GENRL_REWARDS for name in reward_map)
        else "diffusion_nft"
    )
    return reward_map, backend, optimize


def write_prompt_file(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def run_text_only_preprocess(
    args: argparse.Namespace,
    *,
    prompt_file: Path,
    dataset_root: Path,
    num_latent_t: int,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node",
        str(args.preprocess_num_gpus),
        "--master_port",
        str(args.preprocess_master_port),
        "fastvideo/pipelines/preprocess/v1_preprocess.py",
        "--model_path",
        args.model_id,
        "--data_merge_path",
        str(prompt_file),
        "--preprocess_video_batch_size",
        str(args.preprocess_batch_size),
        "--seed",
        str(args.seed),
        "--max_height",
        str(args.num_height),
        "--max_width",
        str(args.num_width),
        "--num_frames",
        str(args.num_frames),
        "--num_latent_t",
        str(num_latent_t),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--output_dir",
        str(dataset_root),
        "--samples_per_file",
        str(args.samples_per_file),
        "--flush_frequency",
        str(args.flush_frequency),
        "--preprocess_task",
        "text_only",
    ]
    subprocess.run(
        cmd,
        cwd=args.repo_root,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def prepare_split(
    args: argparse.Namespace,
    *,
    num_latent_t: int,
) -> tuple[Path, Path, int, int, str]:
    prompts, source = load_prompts(
        args.dataset,
        diffusion_nft_root=args.diffusion_nft_root,
    )
    prompts = list(
        dict.fromkeys(prompt.strip() for prompt in prompts if prompt.strip())
    )
    rng = random.Random(int(args.seed))
    rng.shuffle(prompts)
    if len(prompts) <= args.validation_prompts + args.num_gpus:
        raise RuntimeError(
            "dataset is too small for a disjoint held-out split: "
            f"{len(prompts)} prompts, validation={args.validation_prompts}"
        )
    validation = prompts[: args.validation_prompts]
    train = prompts[args.validation_prompts :]
    train_limit = resolve_max_prompts(
        args.max_train_prompts,
        total_prompts=len(train),
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=1,
    )
    if train_limit > 0:
        train = train[:train_limit]
    if len(train) < args.num_gpus:
        raise RuntimeError(
            f"need at least {args.num_gpus} training prompts, got {len(train)}"
        )

    suffix = (
        f"{args.dataset}_f{args.num_frames}_s{args.seed}_"
        f"train{len(train)}_val{len(validation)}"
    )
    prompt_root = args.data_root / "prompts" / suffix
    train_prompt_file = prompt_root / "train.txt"
    validation_prompt_file = prompt_root / "validation.txt"
    write_prompt_file(train_prompt_file, train)
    write_prompt_file(validation_prompt_file, validation)

    dataset_root = args.data_root / suffix
    train_root = dataset_root / "train"
    validation_root = dataset_root / "validation"
    train_parquet = train_root / "combined_parquet_dataset"
    validation_parquet = validation_root / "combined_parquet_dataset"

    for prompt_file, root, parquet, expected in (
        (train_prompt_file, train_root, train_parquet, len(train)),
        (
            validation_prompt_file,
            validation_root,
            validation_parquet,
            len(validation),
        ),
    ):
        if has_parquet(parquet):
            verify_text_only_dataset(parquet, expected)
            continue
        if args.skip_preprocess:
            raise RuntimeError(
                f"no parquet under {parquet} and --skip-preprocess was set"
            )
        run_text_only_preprocess(
            args,
            prompt_file=prompt_file,
            dataset_root=root,
            num_latent_t=num_latent_t,
        )
        verify_text_only_dataset(parquet, expected)

    return (
        train_parquet,
        validation_parquet,
        len(train),
        len(validation),
        source,
    )


def write_run_config(
    args: argparse.Namespace,
    *,
    train_parquet: Path,
    validation_parquet: Path,
    num_latent_t: int,
    reward_map: dict[str, float],
    reward_backend: str,
    optimize_reward: str,
    train_count: int,
    validation_count: int,
) -> Path:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = config.setdefault("models", {})
    student = models.setdefault("student", {})
    student["init_from"] = args.model_id
    lora = student.setdefault("lora", {})
    lora["enable"] = True
    lora["rank"] = int(args.lora_rank)
    lora["alpha"] = int(args.lora_alpha or args.lora_rank)

    method = config.setdefault("method", {})
    method["objective"] = args.objective
    method["group_size"] = int(args.group_size)
    method["target_ess_ratio"] = float(args.target_ess_ratio)
    method["optimize_reward"] = optimize_reward
    method["reward_backend"] = reward_backend
    method["reward_fn"] = {"rewards": reward_map}
    method["post_update_probe_every"] = int(args.post_update_probe_every)
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
    data["num_frames"] = int(args.num_frames)
    data["num_latent_t"] = int(num_latent_t)
    data["num_height"] = int(args.num_height)
    data["num_width"] = int(args.num_width)
    data["dataloader_num_workers"] = int(args.dataloader_num_workers)
    data["train_batch_size"] = max(
        1,
        int(args.group_size) // int(args.num_gpus),
    )
    loop = training.setdefault("loop", {})
    loop["max_train_steps"] = int(args.max_train_steps)
    loop["gradient_accumulation_steps"] = 1
    optimizer = training.setdefault("optimizer", {})
    optimizer["learning_rate"] = float(args.learning_rate)
    checkpoint = training.setdefault("checkpoint", {})
    checkpoint["output_dir"] = str(args.output_dir)
    checkpoint["training_state_checkpointing_steps"] = int(
        args.checkpoint_every
    )
    tracker = training.setdefault("tracker", {})
    tracker["project_name"] = args.project_name
    tracker["run_name"] = args.run_name or args.output_dir.name

    args.run_config_dir.mkdir(parents=True, exist_ok=True)
    run_config = args.run_config_dir / (
        f"ftp_{args.objective}_{optimize_reward}_"
        f"train{train_count}_val{validation_count}.yaml"
    )
    run_config.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--data-root", type=Path, default=Path("data/ftp"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/ftp"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ftp"))
    parser.add_argument(
        "--run-config-dir",
        type=Path,
        default=Path("outputs/ftp_configs"),
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", default="world-r1-enhanced-dynamic")
    parser.add_argument("--reward", default="videoalign_mq")
    parser.add_argument(
        "--objective",
        choices=("posterior_projection", "flowmap_grpo"),
        default="posterior_projection",
    )
    parser.add_argument("--max-train-prompts", default="512")
    parser.add_argument("--validation-prompts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-latent-t", type=int, default=0)
    parser.add_argument("--num-height", type=int, default=480)
    parser.add_argument("--num-width", type=int, default=832)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--hsdp-replicate-dim", type=int, default=1)
    parser.add_argument("--hsdp-shard-dim", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--target-ess-ratio", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--max-train-steps", type=int, default=1200)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--post-update-probe-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument(
        "--validation-samples-per-prompt",
        type=int,
        default=2,
    )
    parser.add_argument("--validation-log-videos", type=int, default=8)
    parser.add_argument("--primary-min-delta", type=float, default=0.02)
    parser.add_argument("--preprocess-batch-size", type=int, default=128)
    parser.add_argument("--preprocess-num-gpus", type=int, default=1)
    parser.add_argument("--preprocess-master-port", type=int, default=29561)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--samples-per-file", type=int, default=1024)
    parser.add_argument("--flush-frequency", type=int, default=1024)
    parser.add_argument(
        "--project-name",
        default="finite-transition-posterior-wan",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--diffusion-nft-root", type=Path)
    parser.add_argument("--videoalign-checkpoint-path", type=Path)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--check-rewards", action="store_true")
    parser.add_argument("--reward-device", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
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
    return args


def validate_args(args: argparse.Namespace) -> int:
    if args.preprocess_num_gpus != 1:
        raise ValueError("text preprocessing currently requires one GPU")
    if args.num_gpus <= 0 or args.group_size <= 1:
        raise ValueError(
            "num_gpus must be positive and group_size must exceed one"
        )
    if args.group_size % args.num_gpus != 0:
        raise ValueError("group_size must be divisible by num_gpus")
    if args.hsdp_replicate_dim * args.hsdp_shard_dim != args.num_gpus:
        raise ValueError("HSDP dimensions must multiply to num_gpus")
    if not 0.0 < args.target_ess_ratio <= 1.0:
        raise ValueError("target_ess_ratio must lie in (0, 1]")
    if args.validation_prompts <= 0:
        raise ValueError("validation_prompts must be positive")
    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if args.lora_alpha < 0:
        raise ValueError("lora_alpha must be non-negative")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    derived = derive_wan_num_latent_t(args.num_frames)
    if args.num_latent_t <= 0:
        return derived
    if args.num_latent_t != derived:
        raise ValueError(
            f"num_frames={args.num_frames} implies num_latent_t={derived}, "
            f"got {args.num_latent_t}"
        )
    return int(args.num_latent_t)


def main() -> None:
    args = parse_args()
    num_latent_t = validate_args(args)
    reward_map, reward_backend, optimize_reward = resolve_reward_setup(
        args.reward
    )
    if reward_backend == "diffusion_nft":
        ensure_diffusion_nft_repo(args.diffusion_nft_root)
        os.environ["DIFFUSION_NFT_ROOT"] = str(args.diffusion_nft_root)
    if any(name.startswith("videoalign_") for name in reward_map):
        ensure_videoalign_checkpoint(args.videoalign_checkpoint_path)
        os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = str(
            args.videoalign_checkpoint_path
        )
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
        source,
    ) = prepare_split(args, num_latent_t=num_latent_t)
    run_config = write_run_config(
        args,
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        num_latent_t=num_latent_t,
        reward_map=reward_map,
        reward_backend=reward_backend,
        optimize_reward=optimize_reward,
        train_count=train_count,
        validation_count=validation_count,
    )
    summary: dict[str, Any] = {
        "prompt_source": source,
        "train_parquet": str(train_parquet),
        "validation_parquet": str(validation_parquet),
        "train_prompt_count": train_count,
        "validation_prompt_count": validation_count,
        "run_config": str(run_config),
        "output_dir": str(args.output_dir),
        "num_frames": int(args.num_frames),
        "num_latent_t": int(num_latent_t),
        "objective": args.objective,
        "reward_backend": reward_backend,
        "reward_map": reward_map,
        "optimize_reward": optimize_reward,
    }
    print("Prepared finite-transition posterior assets:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
