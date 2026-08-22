"""Final one-command launcher for finite-transition v2.

Run order:

1. ``--preset diagnostic_motion``: prove the substrate can move deterministic
   raw/EMA validation under an easy measurable reward.
2. ``--lr-sweep``: calibrate learning rate against post-update KL.
3. ``--preset grpo``: establish a positive VideoAlign baseline.
4. ``--paired``: only then compare GRPO and posterior weighting.
5. ``--preset velocity``: test deterministic finite-velocity regression if the
   two likelihood objectives remain tied.
"""

from __future__ import annotations

import modal

from modal_train_finite_transition_posterior import (
    HF_SECRET,
    MODAL_CACHE_ROOT,
    MODAL_DATA_ROOT,
    NUM_GPUS,
    PROJECT_ROOT,
    WANDB_SECRET,
    cache_volume,
    data_volume,
    image,
    runs_volume,
)
from modal_train_finite_transition_v2_run import train_v2 as prepare_v2

app = modal.App("fastvideo-finite-transition-v2-final")


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
def execute_final(
    run_config: str,
    *,
    run_name: str,
    comparison_id: str,
    job_type: str,
) -> dict[str, str]:
    import os
    from pathlib import Path
    import subprocess

    import yaml

    source = Path(run_config)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["method"]["_target_"] = (
        "fastvideo.train.methods.rl.finite_transition_v2_final."
        "FiniteTransitionV2FinalMethod"
    )
    # Two fixed seeds are required for a non-degenerate diversity diagnostic.
    # The paired reward statistic itself is still prompt-level.
    raw["method"]["evaluation"]["samples_per_prompt"] = 2
    raw["training"]["tracker"]["run_name"] = run_name
    final_path = source.with_name("resolved_v2_final.yaml")
    final_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    os.environ["WANDB_RUN_GROUP"] = comparison_id
    os.environ["WANDB_JOB_TYPE"] = job_type
    os.environ["WANDB_TAGS"] = f"finite-transition-v2,final,{job_type}"
    subprocess.run(
        ["bash", "examples/train/run.sh", str(final_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    runs_volume.commit()
    cache_volume.commit()
    return {
        "run_config": str(final_path),
        "run_name": run_name,
        "comparison_id": comparison_id,
        "job_type": job_type,
    }


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

    def prepare(
        preset_name: str,
        *,
        lr: float,
        steps: int,
        every: int,
        prompts: int,
        name: str,
        target: float,
    ):
        return prepare_v2.remote(
            preset=preset_name,
            max_train_steps=steps,
            validation_prompts=prompts,
            validation_every=every,
            seed=seed,
            comparison_id=comparison_id,
            learning_rate=lr,
            target_kl=target,
            rollout_groups_per_update=rollout_groups_per_update,
            run_name=name,
            prepare_only=True,
        )

    if lr_sweep:
        prepared = []
        for lr in (2.0e-6, 2.0e-5, 6.0e-5):
            name = f"anyflow_grpo_v2_lr_{lr:.0e}_s{seed}"
            result = prepare(
                "grpo",
                lr=lr,
                steps=20,
                every=10,
                prompts=min(validation_prompts, 64),
                name=name,
                target=0.0,
            )
            prepared.append((result, name, lr))
        handles = [
            execute_final.spawn(
                result["run_config"],
                run_name=name,
                comparison_id=comparison_id,
                job_type=f"grpo_lr_{lr:.0e}",
            )
            for result, name, lr in prepared
        ]
        for handle in handles:
            handle.get()
        return

    if paired:
        arms = []
        for arm in ("grpo", "posterior"):
            name = f"anyflow_{arm}_v2_s{seed}"
            result = prepare(
                arm,
                lr=learning_rate,
                steps=max_train_steps,
                every=validation_every,
                prompts=validation_prompts,
                name=name,
                target=target_kl,
            )
            arms.append((arm, name, result))
        handles = [
            execute_final.spawn(
                result["run_config"],
                run_name=name,
                comparison_id=comparison_id,
                job_type=arm,
            )
            for arm, name, result in arms
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

    name = f"anyflow_{preset}_v2_s{seed}"
    result = prepare(
        preset,
        lr=learning_rate,
        steps=max_train_steps,
        every=validation_every,
        prompts=validation_prompts,
        name=name,
        target=target_kl,
    )
    execute_final.remote(
        result["run_config"],
        run_name=name,
        comparison_id=comparison_id,
        job_type=preset,
    )
