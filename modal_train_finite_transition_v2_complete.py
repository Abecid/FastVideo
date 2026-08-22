"""Self-contained final Modal launcher for finite-transition v2.

This is the authoritative launcher. It prepares assets, validates the runtime,
merges the selected scientific preset, forces the final audited/paired method
entry point, runs training, waits for all comparison arms, and persists outputs.
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
        "fastvideo/train/methods/rl/finite_transition_v2_scientific.py "
        "fastvideo/train/methods/rl/finite_transition_v2_final.py "
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
    preset: str = "diagnostic_motion",
    max_train_steps: int = 30,
    validation_prompts: int = 64,
    validation_every: int = 5,
    seed: int = 42,
    comparison_id: str = "",
    run_name: str = "",
    learning_rate: float = 0.0,
    target_kl: float = -1.0,
    rollout_groups_per_update: int = 0,
    max_train_prompts: str = "512",
    prepare_only: bool = False,
    skip_preprocess: bool = False,
) -> dict[str, str | int | float | bool]:
    from datetime import datetime, timezone
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    import yaml

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or f"ftv2_s{seed}_{timestamp}"
    run_name = run_name.strip() or f"anyflow_{preset}_v2_s{seed}_{timestamp}"
    repo = Path(PROJECT_ROOT)
    output_dir = f"{PROJECT_ROOT}/outputs/finite_transition_v2/{run_name}"
    config_dir = Path(
        f"{PROJECT_ROOT}/outputs/finite_transition_v2_configs/{run_name}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_RUN_GROUP"] = comparison_id
    os.environ["WANDB_JOB_TYPE"] = preset
    os.environ["WANDB_TAGS"] = f"finite-transition-v2,final,{preset},seed-{seed}"

    subprocess.run(["nvidia-smi"], check=True)
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
        ],
        cwd=repo,
        check=True,
    )

    config_path = PRESETS[preset]
    parent_objective = (
        "flowmap_grpo"
        if preset in {"grpo", "diagnostic_motion"}
        else "posterior_projection"
    )
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
        "videoalign_mq",
        "--objective",
        parent_objective,
        "--max-train-prompts",
        str(max_train_prompts),
        "--validation-prompts",
        str(validation_prompts),
        "--validation-every",
        str(validation_every),
        "--validation-samples-per-prompt",
        "2",
        "--validation-log-videos",
        "8",
        "--group-size",
        "4",
        "--target-ess-ratio",
        "0.5",
        "--lora-rank",
        "64",
        "--learning-rate",
        str(learning_rate if learning_rate > 0 else 2.0e-5),
        "--max-train-steps",
        str(max_train_steps),
        "--num-frames",
        "81",
        "--num-height",
        "480",
        "--num-width",
        "832",
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
    preset_cfg = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))

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
    merged["training"]["data"]["data_path"] = data_path
    merged["method"]["validation"]["data_path"] = validation_path
    merged["method"]["validation"]["num_prompts"] = int(validation_prompts)
    merged["method"]["validation"]["every_steps"] = int(validation_every)
    merged["method"]["evaluation"]["samples_per_prompt"] = 2
    merged["training"]["loop"]["max_train_steps"] = int(max_train_steps)
    merged["training"]["checkpoint"]["output_dir"] = output_dir
    merged["training"]["tracker"]["run_name"] = run_name
    if learning_rate > 0:
        merged["training"]["optimizer"]["learning_rate"] = float(learning_rate)
    if target_kl >= 0:
        merged["method"]["target_post_update_kl"] = float(target_kl)
    if rollout_groups_per_update > 0:
        merged["method"]["rollout_groups_per_update"] = int(
            rollout_groups_per_update
        )

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
        "max_train_steps": int(max_train_steps),
        "validation_prompts": int(validation_prompts),
        "learning_rate": float(
            merged["training"]["optimizer"]["learning_rate"]
        ),
        "target_kl": float(merged["method"]["target_post_update_kl"]),
        "rollout_groups_per_update": int(
            merged["method"]["rollout_groups_per_update"]
        ),
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
    preset: str = "diagnostic_motion",
    max_train_steps: int = 30,
    validation_prompts: int = 64,
    validation_every: int = 5,
    seed: int = 42,
    comparison_id: str = "",
    learning_rate: float = 0.0,
    target_kl: float = -1.0,
    rollout_groups_per_update: int = 0,
    paired: bool = False,
    lr_sweep: bool = False,
) -> None:
    from datetime import datetime, timezone

    comparison_id = comparison_id.strip() or (
        f"ftv2_final_s{seed}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    common = dict(
        validation_prompts=validation_prompts,
        seed=seed,
        comparison_id=comparison_id,
        rollout_groups_per_update=rollout_groups_per_update,
    )

    if lr_sweep:
        # Prepare and commit the shared data/cache once before concurrent arms.
        train.remote(
            preset="grpo",
            max_train_steps=20,
            validation_every=10,
            learning_rate=2.0e-6,
            target_kl=0.0,
            run_name=f"prepare_ftv2_lr_s{seed}",
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
                run_name=f"anyflow_grpo_v2_lr_{lr:.0e}_s{seed}",
                skip_preprocess=True,
                **common,
            )
            for lr in (2.0e-6, 2.0e-5, 6.0e-5)
        ]
        for handle in handles:
            handle.get()
        return

    if paired:
        train.remote(
            preset="grpo",
            max_train_steps=max_train_steps,
            validation_every=validation_every,
            learning_rate=learning_rate,
            target_kl=target_kl,
            run_name=f"prepare_ftv2_pair_s{seed}",
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
                run_name=f"anyflow_{arm}_v2_s{seed}",
                skip_preprocess=True,
                **common,
            )
            for arm in ("grpo", "posterior")
        ]
        errors = []
        for handle in handles:
            try:
                handle.get()
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
        if errors:
            raise RuntimeError("Finite-transition v2 arm failure: " + " | ".join(errors))
        return

    train.remote(
        preset=preset,
        max_train_steps=max_train_steps,
        validation_every=validation_every,
        learning_rate=learning_rate,
        target_kl=target_kl,
        run_name=f"anyflow_{preset}_v2_s{seed}",
        **common,
    )
