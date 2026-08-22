"""Authoritative Modal launcher for finite-transition v2.

Execution order:

1. ``diagnostic_luminance``: prove the common RL substrate can learn.
2. ``--lr-sweep``: calibrate update size without the adaptive controller.
3. ``grpo``: establish a positive deterministic held-out baseline.
4. ``--paired``: compare GRPO/posterior on frozen identical rollouts.
5. ``velocity``: test direct deterministic finite-velocity regression.

Do not extend a numerically stable but non-learning run to 1,200 updates.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("fastvideo-finite-transition-v2-complete")

LOCAL_REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = "/root/FastVideo"
MODAL_DATA_ROOT = f"{PROJECT_ROOT}/.modal_data"
MODAL_CACHE_ROOT = f"{PROJECT_ROOT}/.modal_cache"
VIDEOALIGN_ROOT = f"{MODAL_CACHE_ROOT}/VideoReward"
DIFFUSION_NFT_ROOT = f"{MODAL_CACHE_ROOT}/DiffusionNFT"
NUM_GPUS = 4
WANDB_SECRET = "wandb-adamlee00"
HF_SECRET = "hf-adamlee00"

PRESETS = {
    "grpo": (
        "examples/train/configs/rl/wan/"
        "finite_transition_grpo_v2_anyflow_videoalign.yaml"
    ),
    "posterior": (
        "examples/train/configs/rl/wan/"
        "finite_transition_posterior_v2_anyflow_videoalign.yaml"
    ),
    "velocity": (
        "examples/train/configs/rl/wan/"
        "finite_transition_velocity_v2_anyflow_videoalign.yaml"
    ),
    "diagnostic_luminance": (
        "examples/train/configs/rl/wan/"
        "finite_transition_grpo_v2_diagnostic_luminance.yaml"
    ),
    "diagnostic_motion": (
        "examples/train/configs/rl/wan/"
        "finite_transition_grpo_v2_diagnostic_motion.yaml"
    ),
}

IGNORE = [
    ".git/fsmonitor--daemon.ipc",
    ".cache",
    ".cache/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    ".pytest_cache",
    ".ruff_cache",
    "outputs",
    "outputs/**",
    "wandb",
    "wandb/**",
]

data_volume = modal.Volume.from_name("fastvideo-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("fastvideo-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("fastvideo-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
        "ninja-build",
        "cmake",
    )
    .pip_install("uv")
    .add_local_dir(
        str(LOCAL_REPO_ROOT),
        PROJECT_ROOT,
        copy=True,
        ignore=IGNORE,
    )
    .run_commands(
        "cd /root/FastVideo && UV_TORCH_BACKEND=cu130 "
        "uv pip install --system --prerelease=allow -e .",
        "uv pip install --system --no-cache-dir "
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
        "releases/download/v0.9.17/"
        "flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl",
        "uv pip check --system",
        "cd /root/FastVideo && python -m compileall -q "
        "fastvideo/train/methods/rl/common/finite_transition_v2.py "
        "fastvideo/train/methods/rl/rewards/videoalign_audit.py "
        "fastvideo/train/methods/rl/finite_transition_v2.py "
        "fastvideo/train/methods/rl/finite_transition_v2_paired.py "
        "fastvideo/train/methods/rl/finite_transition_v2_exact_paired.py "
        "fastvideo/train/methods/rl/finite_transition_v2_scientific.py "
        "fastvideo/train/methods/rl/finite_transition_v2_final.py "
        "examples/train/compare_finite_transition_paired_runs.py "
        "modal_train_finite_transition_v2_complete.py",
    )
    .env(
        {
            "WANDB_MODE": "online",
            "WANDB_ENTITY": "adamlee00",
            "WANDB__SERVICE_WAIT": "300",
            "TOKENIZERS_PARALLELISM": "false",
            "NUM_GPUS": str(NUM_GPUS),
            "HF_HOME": f"{MODAL_CACHE_ROOT}/huggingface",
            "HF_HUB_CACHE": f"{MODAL_CACHE_ROOT}/huggingface",
            "TRANSFORMERS_CACHE": f"{MODAL_CACHE_ROOT}/huggingface",
            "VIDEOALIGN_CHECKPOINT_PATH": VIDEOALIGN_ROOT,
            "DIFFUSION_NFT_ROOT": DIFFUSION_NFT_ROOT,
            "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


def _complete_qwen_snapshot(cache_root: str) -> str:
    """Return a complete immutable Qwen snapshot prepared in the shared cache."""
    import json

    snapshots_root = (
        Path(cache_root)
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
        "No complete prepared Qwen2-VL snapshot was found under "
        f"{snapshots_root}; missing={missing_by_snapshot}"
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
    preset: str = "diagnostic_luminance",
    max_train_steps: int = 0,
    validation_prompts: int = 0,
    validation_every: int = 0,
    validation_samples_per_prompt: int = 0,
    seed: int = 42,
    comparison_id: str = "",
    run_name: str = "",
    learning_rate: float = 0.0,
    target_kl: float = -1.0,
    rollout_groups_per_update: int = 0,
    group_size: int = 0,
    behavior_policy: str = "",
    max_train_prompts: str = "512",
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

    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}"
        )
    if behavior_policy and behavior_policy not in {"on_policy", "frozen_base"}:
        raise ValueError("behavior_policy must be on_policy or frozen_base")

    repo = Path(PROJECT_ROOT)
    config_path = PRESETS[preset]
    preset_cfg = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))
    preset_method = preset_cfg["method"]
    preset_data = preset_cfg["training"]["data"]
    preset_optimizer = preset_cfg["training"]["optimizer"]
    preset_loop = preset_cfg["training"]["loop"]
    preset_validation = preset_method["validation"]
    preset_evaluation = preset_method["evaluation"]

    max_train_steps = int(max_train_steps or preset_loop["max_train_steps"])
    validation_prompts = int(
        validation_prompts or preset_validation["num_prompts"]
    )
    validation_every = int(
        validation_every or preset_validation["every_steps"]
    )
    validation_samples_per_prompt = int(
        validation_samples_per_prompt
        or preset_evaluation.get("samples_per_prompt", 1)
    )
    learning_rate = float(
        learning_rate if learning_rate > 0 else preset_optimizer["learning_rate"]
    )
    group_size = int(group_size or preset_method["group_size"])
    rollout_groups_per_update = int(
        rollout_groups_per_update
        or preset_method["rollout_groups_per_update"]
    )
    behavior_policy = behavior_policy or str(
        preset_method.get("behavior_policy", "on_policy")
    )

    num_frames = int(preset_data["num_frames"])
    num_height = int(preset_data["num_height"])
    num_width = int(preset_data["num_width"])
    lora_rank = int(preset_cfg["models"]["student"]["lora"]["rank"])

    if group_size % NUM_GPUS != 0:
        raise ValueError("group_size must be divisible by the four Modal GPUs")
    if rollout_groups_per_update <= 0:
        raise ValueError("rollout_groups_per_update must be positive")
    if smoke:
        max_train_steps = 2
        validation_prompts = min(validation_prompts, 8)
        validation_every = 1
        validation_samples_per_prompt = 1
        rollout_groups_per_update = 1
        group_size = 4
        max_train_prompts = "32"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or f"ftv2_s{seed}_{timestamp}"
    run_name = run_name.strip() or f"anyflow_{preset}_v2_s{seed}_{timestamp}"
    output_dir = f"{PROJECT_ROOT}/outputs/finite_transition_v2/{run_name}"
    config_dir = Path(
        f"{PROJECT_ROOT}/outputs/finite_transition_v2_configs/{run_name}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_RUN_GROUP"] = comparison_id
    os.environ["WANDB_JOB_TYPE"] = preset
    os.environ["WANDB_TAGS"] = (
        f"finite-transition-v2,final,{preset},{behavior_policy},seed-{seed}"
    )

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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "fastvideo/tests/train/methods/test_anyflow_schedule.py",
            "fastvideo/tests/train/methods/test_local_asfmc.py",
            "fastvideo/tests/train/methods/test_finite_transition_v2_core.py",
            "fastvideo/tests/train/methods/test_finite_transition_v2_method.py",
            "fastvideo/tests/train/methods/test_videoalign_audit.py",
            "fastvideo/tests/train/methods/test_compare_finite_transition_paired_runs.py",
        ],
        cwd=repo,
        check=True,
    )

    uses_videoalign = any(
        str(name).startswith("videoalign_")
        for name in {
            *preset_method.get("reward_fn", {}).get("rewards", {}),
            *preset_method.get("validation_reward_fn", {}).get("rewards", {}),
        }
    )
    if skip_preprocess and uses_videoalign:
        os.environ["VIDEOALIGN_BASE_MODEL_PATH"] = _complete_qwen_snapshot(
            os.environ["HF_HUB_CACHE"]
        )

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
        "world-r1-enhanced-dynamic",
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

    parent_objective = (
        "flowmap_grpo"
        if preset in {"grpo", "diagnostic_luminance", "diagnostic_motion"}
        else "posterior_projection"
    )
    preparation_reward = "mean_luminance" if preset == "diagnostic_luminance" else "videoalign_mq"
    prep_cmd = [
        sys.executable,
        "-m",
        "examples.train.prepare_finite_transition_posterior_assets",
        "--repo-root",
        str(repo),
        "--config",
        config_path,
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
        preparation_reward,
        "--objective",
        parent_objective,
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
        "finite-transition-v2-wan",
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

    def deep_merge(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            result = dict(left)
            for key, value in right.items():
                result[key] = deep_merge(result.get(key), value)
            return result
        return right

    data_path = generated["training"]["data"]["data_path"]
    validation_path = generated["method"]["validation"]["data_path"]
    merged = deep_merge(generated, preset_cfg)
    merged["method"]["_target_"] = (
        "fastvideo.train.methods.rl.finite_transition_v2_final."
        "FiniteTransitionV2FinalMethod"
    )
    merged["method"]["behavior_policy"] = behavior_policy
    merged["method"]["group_size"] = group_size
    merged["method"]["rollout_groups_per_update"] = (
        rollout_groups_per_update
    )
    merged["training"]["data"]["data_path"] = data_path
    merged["training"]["data"]["train_batch_size"] = group_size // NUM_GPUS
    merged["method"]["validation"]["data_path"] = validation_path
    merged["method"]["validation"]["num_prompts"] = validation_prompts
    merged["method"]["validation"]["every_steps"] = validation_every
    merged["method"]["evaluation"]["samples_per_prompt"] = (
        validation_samples_per_prompt
    )
    merged["training"]["loop"]["max_train_steps"] = max_train_steps
    merged["training"]["optimizer"]["learning_rate"] = learning_rate
    merged["training"]["checkpoint"]["output_dir"] = output_dir
    merged["training"]["tracker"]["run_name"] = run_name
    if target_kl >= 0:
        merged["method"]["target_post_update_kl"] = float(target_kl)

    final_config = config_dir / "resolved_v2_final.yaml"
    final_config.write_text(
        yaml.safe_dump(merged, sort_keys=False),
        encoding="utf-8",
    )
    data_volume.commit()
    cache_volume.commit()
    runs_volume.commit()

    result: dict[str, str | int | float | bool] = {
        "preset": preset,
        "comparison_id": comparison_id,
        "run_name": run_name,
        "run_config": str(final_config),
        "output_dir": output_dir,
        "max_train_steps": max_train_steps,
        "validation_prompts": validation_prompts,
        "validation_samples_per_prompt": validation_samples_per_prompt,
        "learning_rate": learning_rate,
        "target_kl": float(merged["method"]["target_post_update_kl"]),
        "group_size": group_size,
        "rollout_groups_per_update": rollout_groups_per_update,
        "reward_videos_per_update": group_size * rollout_groups_per_update,
        "behavior_policy": behavior_policy,
        "prepare_only": bool(prepare_only),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "modal_launch_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runs_volume.commit()
    if prepare_only:
        return result

    process = subprocess.Popen(
        ["bash", "examples/train/run.sh", str(final_config)],
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
                print("Committed v2 run/cache volumes.", flush=True)
    finally:
        runs_volume.commit()
        cache_volume.commit()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)
    return result


@app.local_entrypoint()
def main(
    preset: str = "diagnostic_luminance",
    max_train_steps: int = 0,
    validation_prompts: int = 0,
    validation_every: int = 0,
    validation_samples_per_prompt: int = 0,
    seed: int = 42,
    comparison_id: str = "",
    learning_rate: float = 0.0,
    target_kl: float = -1.0,
    rollout_groups_per_update: int = 0,
    group_size: int = 0,
    paired: bool = False,
    lr_sweep: bool = False,
    smoke: bool = False,
) -> None:
    from datetime import datetime, timezone

    if paired and lr_sweep:
        raise ValueError("paired and lr-sweep are mutually exclusive")
    if (paired or lr_sweep) and preset not in {"grpo", "posterior"}:
        # The flag controls the mode; the explicit preset is otherwise ignored.
        preset = "grpo"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or f"ftv2_final_s{seed}_{timestamp}"
    common = dict(
        validation_prompts=validation_prompts,
        validation_samples_per_prompt=validation_samples_per_prompt,
        seed=seed,
        comparison_id=comparison_id,
        rollout_groups_per_update=rollout_groups_per_update,
        group_size=group_size,
        smoke=smoke,
    )

    if lr_sweep:
        train.remote(
            preset="grpo",
            max_train_steps=20,
            validation_every=10,
            learning_rate=2.0e-6,
            target_kl=0.0,
            behavior_policy="on_policy",
            run_name=f"prepare_ftv2_lr_s{seed}_{timestamp}",
            prepare_only=True,
            **common,
        )
        handles = [
            train.spawn(
                preset="grpo",
                max_train_steps=20,
                validation_every=10,
                learning_rate=lr,
                target_kl=0.0,
                behavior_policy="on_policy",
                run_name=(
                    f"anyflow_grpo_v2_lr_{lr:.0e}_s{seed}_{timestamp}"
                ),
                skip_preprocess=True,
                **common,
            )
            for lr in (2.0e-6, 2.0e-5, 6.0e-5)
        ]
        for handle in handles:
            handle.get()
        return

    if paired:
        # The preparation job materializes one deterministic split and the full
        # Qwen checkpoint before concurrent arms begin.
        train.remote(
            preset="grpo",
            max_train_steps=max_train_steps,
            validation_every=validation_every,
            learning_rate=learning_rate,
            target_kl=target_kl,
            behavior_policy="frozen_base",
            run_name=f"prepare_ftv2_pair_s{seed}_{timestamp}",
            prepare_only=True,
            **common,
        )
        handles = [
            train.spawn(
                preset=arm,
                max_train_steps=max_train_steps,
                validation_every=validation_every,
                learning_rate=learning_rate,
                target_kl=target_kl,
                behavior_policy="frozen_base",
                run_name=f"anyflow_{arm}_v2_s{seed}_{timestamp}",
                skip_preprocess=True,
                **common,
            )
            for arm in ("grpo", "posterior")
        ]
        results = []
        errors = []
        for handle in handles:
            try:
                results.append(handle.get())
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
        if errors:
            raise RuntimeError(
                "Finite-transition v2 arm failure: " + " | ".join(errors)
            )
        print("Paired arms completed with exact frozen-base rollouts:")
        for result in results:
            print(result)
        print(
            "Compare matching *_samples.json artifacts with "
            "examples/train/compare_finite_transition_paired_runs.py"
        )
        return

    train.remote(
        preset=preset,
        max_train_steps=max_train_steps,
        validation_every=validation_every,
        validation_samples_per_prompt=validation_samples_per_prompt,
        learning_rate=learning_rate,
        target_kl=target_kl,
        rollout_groups_per_update=rollout_groups_per_update,
        group_size=group_size,
        behavior_policy="",
        run_name=f"anyflow_{preset}_v2_s{seed}_{timestamp}",
        comparison_id=comparison_id,
        seed=seed,
        smoke=smoke,
    )
