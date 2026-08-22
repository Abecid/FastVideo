"""Launch reliable AnyFlow finite-transition experiments on Modal.

Recommended sequence:

1. ``--smoke`` for real model/reward/distributed plumbing.
2. ``--recipe sanity`` to prove deterministic held-out learnability.
3. ``--calibrate-kl`` to choose a non-trivial, stable update scale.
4. ``--recipe reliable --objective flowmap_grpo`` for the baseline.
5. ``--paired`` for strict shared-behavior GRPO versus posterior weighting.
6. ``--recipe velocity`` for direct finite-velocity regression.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import modal

from modal_train_finite_transition_posterior import (
    DIFFUSION_NFT_ROOT,
    GPU_TYPE,
    HF_SECRET,
    MODAL_CACHE_ROOT,
    MODAL_DATA_ROOT,
    NUM_GPUS,
    PROJECT_ROOT,
    VIDEOALIGN_ROOT,
    WANDB_SECRET,
    cache_volume,
    data_volume,
    image as base_image,
    runs_volume,
)

app = modal.App("fastvideo-finite-transition-reliable")
OUTPUT_ROOT = f"{PROJECT_ROOT}/outputs/finite_transition_reliable"

image = (
    base_image.run_commands(
        "cd /root/FastVideo && python -m compileall -q "
        "fastvideo/train/methods/rl/common/reward_statistics.py "
        "fastvideo/train/methods/rl/finite_transition_paired_validation.py "
        "fastvideo/train/methods/rl/finite_transition_reliable.py "
        "fastvideo/train/methods/rl/finite_transition_reliable_calibrated.py "
        "fastvideo/train/methods/rl/finite_transition_reliable_audited.py "
        "fastvideo/train/methods/rl/rewards/videoalign_audit.py "
        "examples/train/prepare_finite_transition_reliable_assets.py "
        "examples/train/prepare_finite_transition_reliable_run.py "
        "examples/train/audit_videoalign_checkpoint.py "
        "modal_train_finite_transition_reliable.py"
    )
    .env(
        {
            "VIDEOALIGN_MIN_OVERALL_COVERAGE": "0.90",
            "VIDEOALIGN_MIN_HEAD_COVERAGE": "0.99",
        }
    )
)


def _defaults(recipe: str) -> dict[str, int | float]:
    if recipe == "sanity":
        return {
            "max_train_steps": 50,
            "group_size": 4,
            "rollout_groups": 2,
            "lora_rank": 32,
            "learning_rate": 2.0e-5,
            "validation_prompts": 16,
            "validation_every": 10,
            "validation_samples": 1,
            "frames": 17,
            "height": 256,
            "width": 448,
        }
    if recipe == "velocity":
        return {
            "max_train_steps": 100,
            "group_size": 8,
            "rollout_groups": 4,
            "lora_rank": 64,
            "learning_rate": 1.0e-5,
            "validation_prompts": 64,
            "validation_every": 25,
            "validation_samples": 2,
            "frames": 81,
            "height": 480,
            "width": 832,
        }
    return {
        "max_train_steps": 100,
        "group_size": 8,
        "rollout_groups": 4,
        "lora_rank": 64,
        "learning_rate": 2.0e-5,
        "validation_prompts": 64,
        "validation_every": 50,
        "validation_samples": 2,
        "frames": 81,
        "height": 480,
        "width": 832,
    }


def _effective_objective(recipe: str, objective: str) -> str:
    if recipe == "velocity":
        return "finite_velocity_regression"
    if recipe == "sanity":
        return "flowmap_grpo"
    return objective


def _complete_qwen_snapshot() -> str:
    snapshots_root = (
        Path(os.environ["HF_HUB_CACHE"])
        / "models--Qwen--Qwen2-VL-2B-Instruct"
        / "snapshots"
    )
    missing_by_snapshot: dict[str, list[str]] = {}
    for snapshot in sorted(snapshots_root.glob("*"), reverse=True):
        index_path = snapshot / "model.safetensors.index.json"
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = sorted(
            {
                shard
                for shard in index["weight_map"].values()
                if not (snapshot / shard).is_file()
            }
        )
        if not missing:
            return str(snapshot.resolve())
        missing_by_snapshot[str(snapshot)] = missing
    raise FileNotFoundError(
        "No complete prepared Qwen2-VL snapshot under "
        f"{snapshots_root}; missing={missing_by_snapshot}"
    )


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    cpu=32,
    memory=131072,
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    volumes={
        MODAL_DATA_ROOT: data_volume,
        f"{PROJECT_ROOT}/outputs": runs_volume,
        MODAL_CACHE_ROOT: cache_volume,
    },
    secrets=[
        modal.Secret.from_name(WANDB_SECRET),
        modal.Secret.from_name(HF_SECRET),
    ],
)
def train(
    recipe: str = "reliable",
    objective: str = "flowmap_grpo",
    max_train_steps: int = 0,
    group_size: int = 0,
    rollout_groups: int = 0,
    behavior_policy: str = "current",
    target_kl: float = 1.0e-5,
    initial_loss_scale: float = 1.0,
    target_kl_enabled: bool = True,
    posterior_temperature_mode: str = "global_std",
    posterior_temperature_scale: float = 1.0,
    lora_rank: int = 0,
    learning_rate: float = 0.0,
    dataset: str = "world-r1-enhanced-dynamic",
    max_train_prompts: str = "256",
    validation_prompts: int = 0,
    validation_every: int = 0,
    validation_samples_per_prompt: int = 0,
    validation_log_videos: int = 8,
    num_frames: int = 0,
    height: int = 0,
    width: int = 0,
    seed: int = 42,
    comparison_id: str = "",
    run_name_override: str = "",
    resume_from_checkpoint: str = "",
    volume_commit_interval_seconds: int = 600,
    smoke: bool = False,
    prepare_only: bool = False,
    skip_preprocess: bool = False,
) -> dict[str, str | int | float | bool]:
    if recipe not in {"reliable", "velocity", "sanity"}:
        raise ValueError("recipe must be reliable, velocity, or sanity")
    if objective not in {"flowmap_grpo", "posterior_projection"}:
        raise ValueError(
            "objective must be flowmap_grpo or posterior_projection"
        )
    if behavior_policy not in {"current", "base_adapter_disabled"}:
        raise ValueError("invalid behavior_policy")

    defaults = _defaults(recipe)
    max_train_steps = max_train_steps or int(defaults["max_train_steps"])
    group_size = group_size or int(defaults["group_size"])
    rollout_groups = rollout_groups or int(defaults["rollout_groups"])
    lora_rank = lora_rank or int(defaults["lora_rank"])
    learning_rate = learning_rate or float(defaults["learning_rate"])
    validation_prompts = validation_prompts or int(
        defaults["validation_prompts"]
    )
    validation_every = validation_every or int(defaults["validation_every"])
    validation_samples_per_prompt = validation_samples_per_prompt or int(
        defaults["validation_samples"]
    )
    num_frames = num_frames or int(defaults["frames"])
    height = height or int(defaults["height"])
    width = width or int(defaults["width"])

    if group_size % NUM_GPUS != 0:
        raise ValueError("group_size must be divisible by four Modal GPUs")
    if rollout_groups <= 0 or volume_commit_interval_seconds <= 0:
        raise ValueError("rollout_groups and commit interval must be positive")
    if recipe == "velocity":
        target_kl_enabled = False
    if smoke:
        max_train_steps = 2
        group_size = 4
        rollout_groups = 1
        validation_prompts = min(validation_prompts, 8)
        validation_every = 1
        validation_samples_per_prompt = 1
        validation_log_videos = min(validation_log_videos, 4)
        lora_rank = min(lora_rank, 32)
        num_frames = 17
        height = 256
        width = 448
        max_train_prompts = "32"
        volume_commit_interval_seconds = min(
            volume_commit_interval_seconds,
            60,
        )

    effective_objective = _effective_objective(recipe, objective)
    repo = Path(PROJECT_ROOT)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or (
        f"ftr_{recipe}_{effective_objective}_s{seed}_{timestamp}"
    )
    run_name = run_name_override.strip() or (
        f"anyflow_{recipe}_{effective_objective}_g{group_size}_"
        f"a{rollout_groups}_s{seed}_{timestamp}"
    )
    output_dir = f"{OUTPUT_ROOT}/{run_name}"
    run_config_dir = f"{PROJECT_ROOT}/outputs/ftr_run_configs/{run_name}"

    os.environ["WANDB_RUN_GROUP"] = comparison_id
    os.environ["WANDB_JOB_TYPE"] = f"{recipe}:{effective_objective}"
    os.environ["WANDB_TAGS"] = (
        "finite-transition-reliable,anyflow,"
        f"{recipe},{effective_objective},seed-{seed}"
    )
    if resume_from_checkpoint:
        os.environ["WANDB_RESUME"] = "allow"

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from flash_attn import flash_attn_func; "
            "q=torch.randn((1,128,8,64),device='cuda',dtype=torch.bfloat16,requires_grad=True); "
            "flash_attn_func(q,q,q).sum().backward(); print('flash attention ok')",
        ],
        cwd=repo,
        check=True,
    )

    focused_tests = [
        "fastvideo/tests/train/methods/test_anyflow_schedule.py",
        "fastvideo/tests/train/methods/test_local_asfmc.py",
        "fastvideo/tests/train/methods/test_finite_transition_posterior_core.py",
        "fastvideo/tests/train/methods/test_finite_transition_posterior_method.py",
        "fastvideo/tests/train/methods/test_finite_transition_posterior_repro.py",
        "fastvideo/tests/train/methods/test_reward_statistics.py",
        "fastvideo/tests/train/methods/test_finite_transition_reliable.py",
        "fastvideo/tests/train/methods/test_finite_transition_reliable_configs.py",
        "fastvideo/tests/train/methods/test_videoalign_audit.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *focused_tests],
        cwd=repo,
        check=True,
    )

    if skip_preprocess and recipe != "sanity":
        os.environ["VIDEOALIGN_BASE_MODEL_PATH"] = _complete_qwen_snapshot()

    config_path = {
        "reliable": (
            "examples/train/configs/rl/wan/"
            "finite_transition_reliable_anyflow_videoalign.yaml"
        ),
        "velocity": (
            "examples/train/configs/rl/wan/"
            "finite_transition_velocity_anyflow_videoalign.yaml"
        ),
        "sanity": (
            "examples/train/configs/rl/wan/"
            "finite_transition_reliable_sanity_luminance.yaml"
        ),
    }[recipe]
    preflight_cmd = [
        sys.executable,
        "-m",
        "examples.train.check_finite_transition_posterior_environment",
        "--repo-root",
        str(repo),
        "--config",
        config_path,
        "--model-id",
        "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
        "--dataset",
        dataset,
        "--diffusion-nft-root",
        DIFFUSION_NFT_ROOT,
        "--validation-prompts",
        str(validation_prompts),
        "--num-gpus",
        str(NUM_GPUS),
        "--require-wandb",
        "--json",
    ]
    subprocess.run(preflight_cmd, cwd=repo, check=True)
    cache_volume.commit()

    prep_cmd = [
        sys.executable,
        "-m",
        "examples.train.prepare_finite_transition_reliable_run",
        "--recipe",
        recipe,
        "--objective",
        objective,
        "--repo-root",
        str(repo),
        "--data-root",
        MODAL_DATA_ROOT,
        "--cache-root",
        MODAL_CACHE_ROOT,
        "--output-dir",
        output_dir,
        "--run-config-dir",
        run_config_dir,
        "--diffusion-nft-root",
        DIFFUSION_NFT_ROOT,
        "--videoalign-checkpoint-path",
        VIDEOALIGN_ROOT,
        "--dataset",
        dataset,
        "--max-train-prompts",
        str(max_train_prompts),
        "--max-train-steps",
        str(max_train_steps),
        "--group-size",
        str(group_size),
        "--rollout-groups-per-update",
        str(rollout_groups),
        "--behavior-policy",
        behavior_policy,
        "--target-kl",
        str(target_kl),
        "--initial-loss-scale",
        str(initial_loss_scale),
        (
            "--target-kl-enabled"
            if target_kl_enabled
            else "--no-target-kl-enabled"
        ),
        "--posterior-temperature-mode",
        posterior_temperature_mode,
        "--posterior-temperature-scale",
        str(posterior_temperature_scale),
        "--lora-rank",
        str(lora_rank),
        "--lora-alpha",
        str(2 * lora_rank),
        "--learning-rate",
        str(learning_rate),
        "--validation-prompts",
        str(validation_prompts),
        "--validation-every",
        str(validation_every),
        "--validation-samples-per-prompt",
        str(validation_samples_per_prompt),
        "--validation-log-videos",
        str(validation_log_videos),
        "--num-frames",
        str(num_frames),
        "--num-height",
        str(height),
        "--num-width",
        str(width),
        "--seed",
        str(seed),
        "--num-gpus",
        str(NUM_GPUS),
        "--hsdp-replicate-dim",
        "1",
        "--hsdp-shard-dim",
        str(NUM_GPUS),
        "--project-name",
        "finite-transition-reliable-wan",
        "--run-name",
        run_name,
        "--check-rewards",
        "--json",
    ]
    if skip_preprocess:
        prep_cmd.append("--skip-preprocess")
    completed = subprocess.run(
        prep_cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        check=True,
    )
    print(completed.stdout, end="", flush=True)
    summary = json.loads(completed.stdout.strip().splitlines()[-1])

    if recipe != "sanity":
        # The preparation check has now completed the immutable Qwen snapshot.
        os.environ["VIDEOALIGN_BASE_MODEL_PATH"] = _complete_qwen_snapshot()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.train.audit_videoalign_checkpoint",
                "--checkpoint-path",
                VIDEOALIGN_ROOT,
                "--device",
                "cuda:0",
                "--json",
            ],
            cwd=repo,
            check=True,
        )

    data_volume.commit()
    cache_volume.commit()
    runs_volume.commit()

    result: dict[str, str | int | float | bool] = {
        "recipe": recipe,
        "objective": effective_objective,
        "comparison_id": comparison_id,
        "run_name": run_name,
        "output_dir": output_dir,
        "run_config": str(summary["run_config"]),
        "group_size": group_size,
        "rollout_groups": rollout_groups,
        "reward_samples_per_update": int(
            summary["reward_samples_per_update"]
        ),
        "behavior_policy": behavior_policy,
        "target_kl": target_kl,
        "initial_loss_scale": initial_loss_scale,
        "smoke": smoke,
        "prepare_only": prepare_only,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "modal_launch_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runs_volume.commit()
    if prepare_only:
        return result

    train_cmd = [
        "bash",
        "examples/train/run.sh",
        str(summary["run_config"]),
    ]
    if resume_from_checkpoint:
        train_cmd.extend(
            [
                "--training.checkpoint.resume_from_checkpoint",
                resume_from_checkpoint,
            ]
        )

    process = subprocess.Popen(
        train_cmd,
        cwd=repo,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        while True:
            try:
                return_code = process.wait(
                    timeout=int(volume_commit_interval_seconds)
                )
                break
            except subprocess.TimeoutExpired:
                runs_volume.commit()
                cache_volume.commit()
                print("Committed Modal volumes during training.", flush=True)
    finally:
        runs_volume.commit()
        cache_volume.commit()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, train_cmd)
    return result


@app.local_entrypoint()
def main(
    recipe: str = "reliable",
    objective: str = "flowmap_grpo",
    max_train_steps: int = 0,
    group_size: int = 0,
    rollout_groups: int = 0,
    behavior_policy: str = "current",
    target_kl: float = 1.0e-5,
    initial_loss_scale: float = 1.0,
    target_kl_enabled: bool = True,
    posterior_temperature_mode: str = "global_std",
    posterior_temperature_scale: float = 1.0,
    lora_rank: int = 0,
    learning_rate: float = 0.0,
    dataset: str = "world-r1-enhanced-dynamic",
    max_train_prompts: str = "256",
    validation_prompts: int = 0,
    validation_every: int = 0,
    validation_samples_per_prompt: int = 0,
    validation_log_videos: int = 8,
    num_frames: int = 0,
    height: int = 0,
    width: int = 0,
    seed: int = 42,
    comparison_id: str = "",
    run_name_override: str = "",
    resume_from_checkpoint: str = "",
    volume_commit_interval_seconds: int = 600,
    smoke: bool = False,
    prepare_only: bool = False,
    skip_preprocess: bool = False,
    paired: bool = False,
    calibrate_kl: bool = False,
) -> None:
    if paired and calibrate_kl:
        raise ValueError("paired and calibrate-kl are mutually exclusive")
    if (paired or calibrate_kl) and recipe != "reliable":
        raise ValueError("paired and calibrate-kl require --recipe reliable")
    if (paired or calibrate_kl) and resume_from_checkpoint:
        raise ValueError("multi-run modes do not accept one shared checkpoint")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or (
        f"ftr_{recipe}_s{seed}_{timestamp}"
    )
    common = dict(
        recipe=recipe,
        max_train_steps=max_train_steps,
        group_size=group_size,
        rollout_groups=rollout_groups,
        target_kl=target_kl,
        initial_loss_scale=initial_loss_scale,
        target_kl_enabled=target_kl_enabled,
        posterior_temperature_mode=posterior_temperature_mode,
        posterior_temperature_scale=posterior_temperature_scale,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        dataset=dataset,
        max_train_prompts=max_train_prompts,
        validation_prompts=validation_prompts,
        validation_every=validation_every,
        validation_samples_per_prompt=validation_samples_per_prompt,
        validation_log_videos=validation_log_videos,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        comparison_id=comparison_id,
        run_name_override=run_name_override,
        resume_from_checkpoint=resume_from_checkpoint,
        volume_commit_interval_seconds=volume_commit_interval_seconds,
        smoke=smoke,
        prepare_only=prepare_only,
        skip_preprocess=skip_preprocess,
    )

    if paired:
        prep = dict(common)
        prep.update(
            objective="flowmap_grpo",
            behavior_policy="base_adapter_disabled",
            prepare_only=True,
            skip_preprocess=False,
            run_name_override="",
        )
        train.remote(**prep)
        if prepare_only:
            return
        jobs = []
        for arm in ("flowmap_grpo", "posterior_projection"):
            kwargs = dict(common)
            kwargs.update(
                objective=arm,
                behavior_policy="base_adapter_disabled",
                prepare_only=False,
                skip_preprocess=True,
                run_name_override=(
                    f"anyflow_reliable_{arm}_shared_behavior_"
                    f"s{seed}_{timestamp}"
                ),
            )
            jobs.append(train.spawn(**kwargs))
        errors = []
        for job in jobs:
            try:
                job.get()
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
        if errors:
            raise RuntimeError(f"paired reliable runs failed: {errors}")
        return

    if calibrate_kl:
        prep = dict(common)
        prep.update(
            objective="flowmap_grpo",
            behavior_policy="current",
            max_train_steps=20,
            prepare_only=True,
            skip_preprocess=False,
            run_name_override="",
        )
        train.remote(**prep)
        if prepare_only:
            return
        jobs = []
        for candidate_target in (1.0e-6, 1.0e-5, 1.0e-4):
            target_label = f"{candidate_target:.0e}".replace("-", "m")
            kwargs = dict(common)
            kwargs.update(
                objective="flowmap_grpo",
                behavior_policy="current",
                target_kl=candidate_target,
                max_train_steps=20,
                validation_every=20,
                prepare_only=False,
                skip_preprocess=True,
                run_name_override=(
                    f"anyflow_reliable_grpo_targetkl_{target_label}_"
                    f"s{seed}_{timestamp}"
                ),
            )
            jobs.append(train.spawn(**kwargs))
        errors = []
        for job in jobs:
            try:
                job.get()
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
        if errors:
            raise RuntimeError(f"target-KL calibration failed: {errors}")
        return

    train.remote(
        objective=objective,
        behavior_policy=behavior_policy,
        **common,
    )
