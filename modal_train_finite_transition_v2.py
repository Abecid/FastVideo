"""Launch statistically scaled AnyFlow finite-transition v2 experiments.

Presets:

* ``grpo``: multi-transition accumulated Flow-Map-GRPO baseline.
* ``posterior``: identical rollouts with global-temperature posterior weights.
* ``velocity``: shared-state posterior-mean finite-velocity regression.
* ``diagnostic_motion``: short temporal-L1 learnability gate.

Use ``--paired`` only after the diagnostic gate and the GRPO baseline show a
positive deterministic validation response. Use ``--lr-sweep`` to calibrate
actual post-update KL before committing to a long run.
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_train_finite_transition_posterior import (
    CACHE_MOUNT,
    DATA_MOUNT,
    MODAL_CACHE_ROOT,
    MODAL_DATA_ROOT,
    NUM_GPUS,
    PROJECT_ROOT,
    RUNS_MOUNT,
    VIDEOALIGN_ROOT,
    WANDB_SECRET,
    HF_SECRET,
    cache_volume,
    data_volume,
    image,
    runs_volume,
)

app = modal.App("fastvideo-finite-transition-v2")

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


@app.function(
    image=image,
    gpu=f"H100:{NUM_GPUS}",
    cpu=32,
    memory=131072,
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    volumes={
        DATA_MOUNT: data_volume,
        RUNS_MOUNT: runs_volume,
        CACHE_MOUNT: cache_volume,
    },
    secrets=[
        modal.Secret.from_name(WANDB_SECRET),
        modal.Secret.from_name(HF_SECRET),
    ],
)
def train_v2(
    preset: str = "grpo",
    max_train_steps: int = 100,
    dataset: str = "world-r1-enhanced-dynamic",
    max_train_prompts: str = "512",
    validation_prompts: int = 128,
    validation_every: int = 25,
    seed: int = 42,
    learning_rate: float = 0.0,
    target_kl: float = -1.0,
    rollout_groups_per_update: int = 0,
    run_name: str = "",
    comparison_id: str = "",
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
    if max_train_steps <= 0 or validation_every <= 0:
        raise ValueError("max_train_steps and validation_every must be positive")

    repo = Path(PROJECT_ROOT)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_id = comparison_id.strip() or f"ftv2_s{seed}_{timestamp}"
    run_name = run_name.strip() or f"anyflow_{preset}_s{seed}_{timestamp}"
    output_dir = f"{PROJECT_ROOT}/outputs/finite_transition_v2/{run_name}"
    config_dir = Path(f"{PROJECT_ROOT}/outputs/finite_transition_v2_configs/{run_name}")
    config_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_RUN_GROUP"] = comparison_id
    os.environ["WANDB_JOB_TYPE"] = preset
    os.environ["WANDB_TAGS"] = f"finite-transition-v2,anyflow,{preset},seed-{seed}"

    # Run the focused gate in the real CUDA image before model/reward allocation.
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
        ],
        cwd=repo,
        check=True,
    )

    config_path = PRESETS[preset]
    parent_objective = "flowmap_grpo" if preset in {"grpo", "diagnostic_motion"} else "posterior_projection"
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
        f"{MODAL_CACHE_ROOT}/DiffusionNFT",
        "--videoalign-checkpoint-path",
        VIDEOALIGN_ROOT,
        "--dataset",
        dataset,
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
        "1",
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

    # The asset script writes resolved data/checkpoint paths. Preserve those,
    # then replace the scientific method/config values with the selected v2
    # preset and force paired raw/EMA validation.
    generated_path = Path(str(summary["run_config"]))
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    preset_cfg = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))

    def deep_merge(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            result = dict(left)
            for key, value in right.items():
                result[key] = deep_merge(result.get(key), value)
            return result
        return right

    resolved_data = generated.get("training", {}).get("data", {}).get("data_path")
    resolved_validation = generated.get("method", {}).get("validation", {}).get("data_path")
    merged = deep_merge(generated, preset_cfg)
    merged["method"]["_target_"] = (
        "fastvideo.train.methods.rl.finite_transition_v2_paired."
        "FiniteTransitionV2PairedMethod"
    )
    merged["training"]["data"]["data_path"] = resolved_data
    merged["method"]["validation"]["data_path"] = resolved_validation
    merged["training"]["loop"]["max_train_steps"] = int(max_train_steps)
    merged["method"]["validation"]["every_steps"] = int(validation_every)
    merged["method"]["validation"]["num_prompts"] = int(validation_prompts)
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

    final_config = config_dir / "resolved_v2.yaml"
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

    subprocess.run(
        ["bash", "examples/train/run.sh", str(final_config)],
        cwd=repo,
        check=True,
    )
    runs_volume.commit()
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    preset: str = "grpo",
    max_train_steps: int = 100,
    validation_prompts: int = 128,
    validation_every: int = 25,
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
        f"ftv2_s{seed}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    common = dict(
        max_train_steps=max_train_steps,
        validation_prompts=validation_prompts,
        validation_every=validation_every,
        seed=seed,
        comparison_id=comparison_id,
        target_kl=target_kl,
        rollout_groups_per_update=rollout_groups_per_update,
    )

    if lr_sweep:
        # Disable the controller so the first 20 updates reveal raw LR->KL
        # calibration. Compare raw deterministic validation and post-update KL.
        for lr in (2.0e-6, 2.0e-5, 6.0e-5):
            train_v2.spawn(
                preset="grpo",
                max_train_steps=20,
                validation_every=10,
                validation_prompts=min(validation_prompts, 64),
                seed=seed,
                comparison_id=comparison_id,
                learning_rate=lr,
                target_kl=0.0,
                rollout_groups_per_update=(
                    rollout_groups_per_update or 2
                ),
                run_name=f"anyflow_grpo_lr_{lr:.0e}_s{seed}",
            )
        return

    if paired:
        if preset not in {"grpo", "posterior"}:
            raise ValueError("paired mode compares grpo and posterior presets")
        # Deterministic prep is safe to repeat, but prepare one arm first so the
        # shared parquet and model/reward caches are fully committed.
        train_v2.remote(
            preset="grpo",
            prepare_only=True,
            run_name=f"prepare_ftv2_s{seed}",
            **common,
        )
        train_v2.spawn(
            preset="grpo",
            skip_preprocess=True,
            run_name=f"anyflow_grpo_v2_s{seed}",
            learning_rate=learning_rate,
            **common,
        )
        train_v2.spawn(
            preset="posterior",
            skip_preprocess=True,
            run_name=f"anyflow_posterior_v2_s{seed}",
            learning_rate=learning_rate,
            **common,
        )
        return

    train_v2.remote(
        preset=preset,
        learning_rate=learning_rate,
        **common,
    )
