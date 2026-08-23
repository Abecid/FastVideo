"""Launch the actual multi-minibatch AnyFlow GRPO v3 baseline on Modal.

This launcher intentionally does not train a posterior objective. Its only job is
to establish whether a real frozen-buffer, clipped GRPO loop can improve
deterministic held-out VideoAlign MQ before another novel-method comparison.
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_train_finite_transition_v2_complete import (
    DIFFUSION_NFT_ROOT,
    HF_SECRET,
    MODAL_CACHE_ROOT,
    MODAL_DATA_ROOT,
    NUM_GPUS,
    PROJECT_ROOT,
    VIDEOALIGN_ROOT,
    WANDB_SECRET,
    _complete_qwen_snapshot,
    cache_volume,
    data_volume,
    image as base_image,
    runs_volume,
)

app = modal.App("fastvideo-finite-transition-grpo-v3")

CONFIG_PATH = (
    "examples/train/configs/rl/wan/"
    "finite_transition_grpo_v3_anyflow_videoalign.yaml"
)
OUTPUT_ROOT = f"{PROJECT_ROOT}/outputs/finite_transition_grpo_v3"

image = base_image.run_commands(
    "cd /root/FastVideo && python -m compileall -q "
    "fastvideo/train/methods/rl/common/finite_transition_grpo_v3.py "
    "fastvideo/train/methods/rl/finite_transition_grpo_v3.py "
    "modal_train_finite_transition_grpo_v3.py",
    "cd /root/FastVideo && pytest -q "
    "fastvideo/tests/train/methods/test_anyflow_schedule.py "
    "fastvideo/tests/train/methods/test_local_asfmc.py "
    "fastvideo/tests/train/methods/test_finite_transition_posterior_core.py "
    "fastvideo/tests/train/methods/test_finite_transition_v2_core.py "
    "fastvideo/tests/train/methods/test_finite_transition_v2_configs.py "
    "fastvideo/tests/train/methods/test_finite_transition_v2_method.py "
    "fastvideo/tests/train/methods/test_finite_transition_grpo_v3_core.py "
    "fastvideo/tests/train/methods/test_finite_transition_grpo_v3_method.py "
    "fastvideo/tests/train/methods/test_finite_transition_grpo_v3_config.py "
    "fastvideo/tests/train/methods/test_videoalign_audit.py",
)


@app.function(
    image=image,
    gpu=f"H100:{NUM_GPUS}",
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
    max_train_steps: int = 20,
    validation_prompts: int = 128,
    validation_every: int = 5,
    validation_samples_per_prompt: int = 2,
    seed: int = 42,
    comparison_id: str = "",
    run_name: str = "",
    learning_rate: float = 1.0e-5,
    policy_epochs: int = 2,
    groups_per_minibatch: int = 1,
    rollout_groups_per_update: int = 4,
    group_size: int = 8,
    clip_range: float = 0.02,
    policy_kl_target: float = 3.0e-5,
    policy_kl_early_stop_multiplier: float = 4.0,
    reference_kl_beta: float = 0.0,
    minimum_group_reward_std: float = 1.0e-4,
    deployment_probe_every: int = 5,
    max_train_prompts: str = "512",
    resume_from_checkpoint: str = "",
    prepare_only: bool = False,
    skip_preprocess: bool = False,
    smoke: bool = False,
) -> dict[str, str | int | float | bool]:
    from datetime import datetime, timezone
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    import yaml

    if max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    if group_size % NUM_GPUS != 0:
        raise ValueError("group_size must be divisible by the four Modal GPUs")
    if rollout_groups_per_update <= 0:
        raise ValueError("rollout_groups_per_update must be positive")
    if not 0 < groups_per_minibatch <= rollout_groups_per_update:
        raise ValueError(
            "groups_per_minibatch must lie in "
            "[1, rollout_groups_per_update]"
        )
    if policy_epochs <= 0:
        raise ValueError("policy_epochs must be positive")
    if not 0.0 < clip_range < 1.0:
        raise ValueError("clip_range must lie in (0, 1)")
    if policy_kl_target <= 0.0:
        raise ValueError("policy_kl_target must be positive")
    if policy_kl_early_stop_multiplier <= 1.0:
        raise ValueError(
            "policy_kl_early_stop_multiplier must exceed one"
        )
    if reference_kl_beta < 0.0:
        raise ValueError("reference_kl_beta must be non-negative")

    repo = Path(PROJECT_ROOT)
    preset_cfg = yaml.safe_load(
        (repo / CONFIG_PATH).read_text(encoding="utf-8")
    )
    num_frames = int(preset_cfg["training"]["data"]["num_frames"])
    num_height = int(preset_cfg["training"]["data"]["num_height"])
    num_width = int(preset_cfg["training"]["data"]["num_width"])
    lora_rank = int(
        preset_cfg["models"]["student"]["lora"]["rank"]
    )

    if smoke:
        max_train_steps = 2
        validation_prompts = min(validation_prompts, 8)
        validation_every = 1
        validation_samples_per_prompt = 1
        policy_epochs = 1
        rollout_groups_per_update = 2
        groups_per_minibatch = 1
        group_size = 4
        num_frames = 17
        num_height = 256
        num_width = 448
        lora_rank = 32
        max_train_prompts = "32"
        deployment_probe_every = 1

    from examples.train.prepare_diffusion_nft_assets import (
        derive_wan_num_latent_t,
    )

    num_latent_t = derive_wan_num_latent_t(num_frames)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or (
        f"grpo_v3_s{seed}_{timestamp}"
    )
    run_name = run_name.strip() or (
        f"anyflow_grpo_v3_lr_{learning_rate:.0e}_"
        f"pe{policy_epochs}_s{seed}_{timestamp}"
    )
    output_dir = f"{OUTPUT_ROOT}/{run_name}"
    config_dir = Path(
        f"{PROJECT_ROOT}/outputs/finite_transition_grpo_v3_configs/"
        f"{run_name}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_RUN_GROUP"] = comparison_id
    os.environ["WANDB_JOB_TYPE"] = "actual_grpo_v3"
    os.environ["WANDB_TAGS"] = (
        "finite-transition,grpo-v3,multi-minibatch,mq-only,"
        f"seed-{seed}"
    )
    if resume_from_checkpoint:
        os.environ["WANDB_RESUME"] = "allow"

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from flash_attn import flash_attn_func; "
            "q=torch.randn((1,128,8,64),device='cuda',"
            "dtype=torch.bfloat16,requires_grad=True); "
            "flash_attn_func(q,q,q).sum().backward(); "
            "print('flash attention ok')",
        ],
        cwd=repo,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "fastvideo/tests/train/methods/test_anyflow_schedule.py",
            "fastvideo/tests/train/methods/test_local_asfmc.py",
            "fastvideo/tests/train/methods/"
            "test_finite_transition_posterior_core.py",
            "fastvideo/tests/train/methods/"
            "test_finite_transition_v2_core.py",
            "fastvideo/tests/train/methods/"
            "test_finite_transition_grpo_v3_core.py",
            "fastvideo/tests/train/methods/"
            "test_finite_transition_grpo_v3_method.py",
            "fastvideo/tests/train/methods/"
            "test_finite_transition_grpo_v3_config.py",
            "fastvideo/tests/train/methods/test_videoalign_audit.py",
        ],
        cwd=repo,
        check=True,
    )

    if skip_preprocess:
        os.environ["VIDEOALIGN_BASE_MODEL_PATH"] = (
            _complete_qwen_snapshot(os.environ["HF_HUB_CACHE"])
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.train.check_finite_transition_posterior_environment",
            "--repo-root",
            str(repo),
            "--config",
            CONFIG_PATH,
            "--model-id",
            "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
            "--dataset",
            "world-r1-enhanced-dynamic",
            "--diffusion-nft-root",
            DIFFUSION_NFT_ROOT,
            "--validation-prompts",
            str(validation_prompts),
            "--num-gpus",
            str(NUM_GPUS),
            "--require-wandb",
            "--json",
        ],
        cwd=repo,
        check=True,
    )
    cache_volume.commit()

    prep_cmd = [
        sys.executable,
        "-m",
        "examples.train.prepare_finite_transition_v2_assets",
        "--repo-root",
        str(repo),
        "--config",
        CONFIG_PATH,
        "--data-root",
        MODAL_DATA_ROOT,
        "--cache-root",
        MODAL_CACHE_ROOT,
        "--output-dir",
        output_dir,
        "--run-config-dir",
        str(config_dir),
        "--diffusion-nft-root",
        DIFFUSION_NFT_ROOT,
        "--videoalign-checkpoint-path",
        VIDEOALIGN_ROOT,
        "--dataset",
        "world-r1-enhanced-dynamic",
        "--reward",
        "videoalign_mq_audited",
        "--objective",
        "flowmap_grpo",
        "--max-train-prompts",
        str(max_train_prompts),
        "--validation-prompts",
        str(validation_prompts),
        "--validation-every",
        str(validation_every),
        "--validation-samples-per-prompt",
        str(validation_samples_per_prompt),
        "--validation-log-videos",
        "8",
        "--group-size",
        str(group_size),
        "--target-ess-ratio",
        "0.5",
        "--lora-rank",
        str(lora_rank),
        "--learning-rate",
        str(learning_rate),
        "--max-train-steps",
        str(max_train_steps),
        "--num-frames",
        str(num_frames),
        "--num-height",
        str(num_height),
        "--num-width",
        str(num_width),
        "--seed",
        str(seed),
        "--num-gpus",
        str(NUM_GPUS),
        "--hsdp-replicate-dim",
        "1",
        "--hsdp-shard-dim",
        str(NUM_GPUS),
        "--project-name",
        "finite-transition-v3-wan",
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

    generated = yaml.safe_load(
        Path(str(summary["run_config"])).read_text(encoding="utf-8")
    )
    from examples.train.prepare_finite_transition_v2_assets import (
        merge_generated_with_preset,
    )

    data_path = generated["training"]["data"]["data_path"]
    validation_path = generated["method"]["validation"]["data_path"]
    merged = merge_generated_with_preset(generated, preset_cfg)
    merged["method"]["_target_"] = (
        "fastvideo.train.methods.rl.finite_transition_grpo_v3."
        "FiniteTransitionGRPOV3Method"
    )
    merged["method"]["group_size"] = int(group_size)
    merged["method"]["rollout_groups_per_update"] = int(
        rollout_groups_per_update
    )
    merged["method"]["policy_epochs"] = int(policy_epochs)
    merged["method"]["groups_per_minibatch"] = int(
        groups_per_minibatch
    )
    merged["method"]["clip_range"] = float(clip_range)
    merged["method"]["policy_kl_target"] = float(
        policy_kl_target
    )
    merged["method"]["policy_kl_early_stop_multiplier"] = float(
        policy_kl_early_stop_multiplier
    )
    merged["method"]["reference_kl_beta"] = float(
        reference_kl_beta
    )
    merged["method"]["minimum_group_reward_std"] = float(
        minimum_group_reward_std
    )
    merged["method"]["deployment_probe_every"] = int(
        deployment_probe_every
    )

    # Preserve the scientific reward contract atomically.
    merged["method"]["reward_backend"] = "genrl"
    merged["method"]["optimize_reward"] = (
        "videoalign_mq_audited"
    )
    merged["method"]["reward_fn"] = {
        "rewards": {"videoalign_mq_audited": 1.0}
    }
    merged["method"]["validation_reward_backend"] = "genrl"
    merged["method"]["validation_reward_fn"] = {
        "rewards": {
            "videoalign_mq_audited": 1.0,
            "videoalign_vq_audited": 1.0,
            "videoalign_ta_audited": 1.0,
        }
    }

    merged["training"]["data"]["data_path"] = data_path
    merged["training"]["data"]["train_batch_size"] = (
        group_size // NUM_GPUS
    )
    merged["training"]["data"]["num_frames"] = int(num_frames)
    merged["training"]["data"]["num_latent_t"] = int(
        num_latent_t
    )
    merged["training"]["data"]["num_height"] = int(num_height)
    merged["training"]["data"]["num_width"] = int(num_width)
    merged["method"]["validation"]["data_path"] = validation_path
    merged["method"]["validation"]["num_prompts"] = int(
        validation_prompts
    )
    merged["method"]["validation"]["every_steps"] = int(
        validation_every
    )
    merged["method"]["evaluation"]["samples_per_prompt"] = int(
        validation_samples_per_prompt
    )
    merged["training"]["optimizer"]["learning_rate"] = float(
        learning_rate
    )
    merged["training"]["loop"]["max_train_steps"] = int(
        max_train_steps
    )
    merged["training"]["checkpoint"]["output_dir"] = output_dir
    merged["training"]["tracker"]["project_name"] = (
        "finite-transition-v3-wan"
    )
    merged["training"]["tracker"]["run_name"] = run_name
    merged["models"]["student"]["lora"]["rank"] = int(
        lora_rank
    )
    merged["models"]["student"]["lora"]["alpha"] = int(
        2 * lora_rank
    )

    final_config = config_dir / "resolved_grpo_v3.yaml"
    final_config.write_text(
        yaml.safe_dump(merged, sort_keys=False),
        encoding="utf-8",
    )

    os.environ["VIDEOALIGN_BASE_MODEL_PATH"] = (
        _complete_qwen_snapshot(os.environ["HF_HUB_CACHE"])
    )
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

    result: dict[str, str | int | float | bool] = {
        "comparison_id": comparison_id,
        "run_name": run_name,
        "run_config": str(final_config),
        "output_dir": output_dir,
        "outer_rollout_steps": int(max_train_steps),
        "policy_epochs": int(policy_epochs),
        "groups_per_minibatch": int(groups_per_minibatch),
        "rollout_groups_per_update": int(
            rollout_groups_per_update
        ),
        "group_size": int(group_size),
        "reward_videos_per_rollout": int(
            group_size * rollout_groups_per_update
        ),
        "maximum_optimizer_steps_per_rollout": int(
            policy_epochs
            * (
                (
                    rollout_groups_per_update
                    + groups_per_minibatch
                    - 1
                )
                // groups_per_minibatch
            )
        ),
        "learning_rate": float(learning_rate),
        "clip_range": float(clip_range),
        "policy_kl_target": float(policy_kl_target),
        "reference_kl_beta": float(reference_kl_beta),
        "online_reward": "videoalign_mq_audited",
        "validation_rewards": (
            "videoalign_mq_audited,"
            "videoalign_vq_audited,"
            "videoalign_ta_audited"
        ),
        "smoke": bool(smoke),
        "prepare_only": bool(prepare_only),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "modal_launch_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_volume.commit()
    cache_volume.commit()
    runs_volume.commit()

    if prepare_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    command = [
        "bash",
        "examples/train/run.sh",
        str(final_config),
    ]
    if resume_from_checkpoint:
        command.extend(
            [
                "--training.checkpoint.resume_from_checkpoint",
                resume_from_checkpoint,
            ]
        )

    process = subprocess.Popen(
        command,
        cwd=repo,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        while True:
            try:
                return_code = process.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                runs_volume.commit()
                cache_volume.commit()
                print(
                    "Committed GRPO v3 run/cache volumes.",
                    flush=True,
                )
    finally:
        runs_volume.commit()
        cache_volume.commit()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return result


@app.local_entrypoint()
def main(
    max_train_steps: int = 20,
    validation_prompts: int = 128,
    validation_every: int = 5,
    validation_samples_per_prompt: int = 2,
    seed: int = 42,
    comparison_id: str = "",
    run_name: str = "",
    learning_rate: float = 1.0e-5,
    policy_epochs: int = 2,
    groups_per_minibatch: int = 1,
    rollout_groups_per_update: int = 4,
    group_size: int = 8,
    clip_range: float = 0.02,
    policy_kl_target: float = 3.0e-5,
    policy_kl_early_stop_multiplier: float = 4.0,
    reference_kl_beta: float = 0.0,
    minimum_group_reward_std: float = 1.0e-4,
    deployment_probe_every: int = 5,
    max_train_prompts: str = "512",
    resume_from_checkpoint: str = "",
    prepare_only: bool = False,
    skip_preprocess: bool = False,
    smoke: bool = False,
) -> None:
    train.remote(
        max_train_steps=max_train_steps,
        validation_prompts=validation_prompts,
        validation_every=validation_every,
        validation_samples_per_prompt=validation_samples_per_prompt,
        seed=seed,
        comparison_id=comparison_id,
        run_name=run_name,
        learning_rate=learning_rate,
        policy_epochs=policy_epochs,
        groups_per_minibatch=groups_per_minibatch,
        rollout_groups_per_update=rollout_groups_per_update,
        group_size=group_size,
        clip_range=clip_range,
        policy_kl_target=policy_kl_target,
        policy_kl_early_stop_multiplier=(
            policy_kl_early_stop_multiplier
        ),
        reference_kl_beta=reference_kl_beta,
        minimum_group_reward_std=minimum_group_reward_std,
        deployment_probe_every=deployment_probe_every,
        max_train_prompts=max_train_prompts,
        resume_from_checkpoint=resume_from_checkpoint,
        prepare_only=prepare_only,
        skip_preprocess=skip_preprocess,
        smoke=smoke,
    )
