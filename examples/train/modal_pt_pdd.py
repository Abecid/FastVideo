# pyright: reportAttributeAccessIssue=false
"""Launch Posterior-Tilted Parallel Decoding Distillation on Modal.

The launcher deliberately pins the released AnyFlow source revision and uses the
scientific configuration checked into this branch. It performs CPU unit tests,
loads the released AnyFlow-Wan checkpoint, runs an official-sampler parity
preflight, logs the untouched four-step base model, and only then begins PT-PDD.

Examples:

    # Safest first command: no optimization, only inference/reward parity.
    modal run examples/train/modal_pt_pdd.py --preflight-only

    # Two-update engineering smoke test. Not a scientific result.
    modal run examples/train/modal_pt_pdd.py --smoke

    # First matched scientific pilot. The target is absolute and auto-resumes.
    modal run examples/train/modal_pt_pdd.py --max-steps 50

    # Full AnyFlow-length experiment; rerun with increasing targets if a single
    # 24-hour Modal allocation is insufficient (e.g. 200, 400, ..., 1200).
    modal run examples/train/modal_pt_pdd.py --max-steps 1200

    # Direct Flow-Map GRPO ablation using the same base/checks/evaluation.
    modal run examples/train/modal_pt_pdd.py \
        --max-steps 50 --objective flowmap_grpo \
        --run-name wan_anyflow_flowmap_grpo_videoalign_mq
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import modal


FASTVIDEO_REPOSITORY = "https://github.com/Abecid/FastVideo.git"
DEFAULT_BRANCH = "agent/posterior-tilted-parallel-decoding-distillation"
ANYFLOW_REPOSITORY = "https://github.com/NVlabs/AnyFlow.git"
# Exact source revision used when auditing the released AnyFlow config/code.
ANYFLOW_COMMIT = "d2acf7373a45173082ec47eb16553a373b10f856"
DEFAULT_CONFIG = "examples/train/configs/pt_pdd/wan_anyflow_videoalign_mq.yaml"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"

IMAGE_REF = os.environ.get("FASTVIDEO_MODAL_IMAGE", DEFAULT_IMAGE)
GPU = os.environ.get("PT_PDD_MODAL_GPU", "H100:4")
SECRET_NAME = os.environ.get("PT_PDD_MODAL_SECRET", "fastvideo-training")

image = (
    modal.Image.from_registry(IMAGE_REF, add_python="3.12")
    .apt_install("git", "ffmpeg")
    .run_commands("python -m pip install --upgrade uv")
)

app = modal.App("fastvideo-pt-pdd")
hf_cache = modal.Volume.from_name("fastvideo-pt-pdd-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("fastvideo-pt-pdd-runs", create_if_missing=True)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _clone_exact_commit(
    repository: str,
    commit: str,
    destination: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    _run(["git", "init", str(destination)], cwd=cwd, env=env)
    _run(
        ["git", "-C", str(destination), "remote", "add", "origin", repository],
        cwd=cwd,
        env=env,
    )
    _run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit],
        cwd=cwd,
        env=env,
    )
    _run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        cwd=cwd,
        env=env,
    )
    resolved = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        cwd=cwd,
        env=env,
        text=True,
    ).strip()
    if resolved != commit:
        raise RuntimeError(
            f"AnyFlow source mismatch: requested {commit}, checked out {resolved}"
        )


@app.function(
    image=image,
    gpu=GPU,
    cpu=32,
    memory=131_072,
    timeout=86_400,
    startup_timeout=4_800,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/runs": runs,
    },
)
def train(
    *,
    branch: str = DEFAULT_BRANCH,
    max_steps: int = 50,
    objective: str = "posterior_tilted_regression",
    reward_key: str = "videoalign_mq",
    dataset_profile: str = "world_r1_enhanced_dynamic",
    validation_profile: str = "world_r1_enhanced_test",
    max_train_prompts: int = 515,
    validation_count: int = 16,
    run_name: str = "wan_anyflow_pt_pdd_videoalign_mq",
    resume: str = "auto",
    preflight_only: bool = False,
    smoke: bool = False,
) -> dict[str, str | int | bool]:
    """Run one resumable PT-PDD target on four H100s.

    Scientific model, optimizer, video-shape, flow-map, LoRA, stochastic-step,
    group-size, and validation settings remain locked in the checked-in YAML.
    The launcher exposes only experiment identity, target update, reward family,
    and matched objective ablations.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if objective not in {
        "posterior_tilted_regression",
        "reference_regression",
        "posterior_distillation",
        "flowmap_grpo",
    }:
        raise ValueError(
            "objective must be posterior_tilted_regression, reference_regression, "
            "posterior_distillation, or flowmap_grpo"
        )
    if reward_key not in {"videoalign_mq", "videoalign_vq", "videoalign_ta"}:
        raise ValueError("reward_key must be one VideoAlign component")
    allowed_profiles = {
        "world_r1_enhanced_dynamic",
        "world_r1_enhanced_train",
        "world_r1_enhanced_test",
        "world_r1_final_dynamic",
        "world_r1_final_train",
        "world_r1_final_test",
        "vimix_public_sample",
        "vidprom_unique",
    }
    if dataset_profile not in allowed_profiles:
        raise ValueError(f"unknown dataset_profile {dataset_profile!r}")
    if validation_profile not in allowed_profiles:
        raise ValueError(f"unknown validation_profile {validation_profile!r}")
    if max_train_prompts <= 0 or validation_count <= 0:
        raise ValueError("prompt counts must be positive")
    if not run_name.strip():
        raise ValueError("run_name must be non-empty")

    workspace = Path("/workspace")
    fastvideo = workspace / "FastVideo"
    anyflow = workspace / "AnyFlow"
    subprocess.run(["rm", "-rf", str(fastvideo), str(anyflow)], check=True)
    workspace.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Do not force an optional Hugging Face transfer backend that is not
    # installed in the base image; this was a previous Modal failure mode.
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.update(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "ANYFLOW_ROOT": str(anyflow),
            "VIDEOALIGN_CHECKPOINT_PATH": "/runs/cache/pt_pdd/VideoReward",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": (
                "online" if (env.get("WANDB_API_KEY") or "").strip() else "offline"
            ),
            "PYTHONPATH": f"{fastvideo}:{anyflow}",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29621",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
    )

    try:
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                FASTVIDEO_REPOSITORY,
                str(fastvideo),
            ],
            cwd=workspace,
            env=env,
        )
        _clone_exact_commit(
            ANYFLOW_REPOSITORY,
            ANYFLOW_COMMIT,
            anyflow,
            cwd=workspace,
            env=env,
        )

        _run(["uv", "pip", "install", "--system", "-e", "."], cwd=fastvideo, env=env)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--system",
                "-r",
                "examples/train/requirements-pt-pdd.txt",
            ],
            cwd=fastvideo,
            env=env,
        )

        # Fast, deterministic checks before allocating hours to a GPU run.
        _run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "fastvideo/tests/train/methods/test_pt_pdd_core.py",
            ],
            cwd=fastvideo,
            env=env,
        )
        _run(
            [
                "python",
                "-c",
                (
                    "from far.models.transformer_far_wan_model import "
                    "FAR_Wan_Transformer3DModel; "
                    "from far.schedulers.scheduling_flowmap_euler_discrete import "
                    "FlowMapDiscreteScheduler; "
                    "from fastvideo.train.methods.rl.rewards import "
                    "build_multi_reward_scorer; print('PT-PDD imports OK')"
                ),
            ],
            cwd=fastvideo,
            env=env,
        )

        output_dir = Path("/runs/outputs") / run_name
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "4",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            env["MASTER_PORT"],
            "examples/train/pt_pdd/train_anyflow.py",
            "--config",
            DEFAULT_CONFIG,
            "--output-dir",
            str(output_dir),
            "--run-name",
            run_name,
            "--max-train-steps",
            str(max_steps),
            "--objective",
            objective,
            "--reward-key",
            reward_key,
            "--dataset-profile",
            dataset_profile,
            "--validation-profile",
            validation_profile,
            "--max-train-prompts",
            str(max_train_prompts),
            "--validation-count",
            str(validation_count),
            "--resume",
            resume,
        ]
        if preflight_only:
            command.append("--preflight-only")
        if smoke:
            command.append("--smoke")
        _run(command, cwd=fastvideo, env=env)

        summary: dict[str, str | int | bool] = {
            "branch": branch,
            "anyflow_commit": ANYFLOW_COMMIT,
            "run_name": run_name,
            "output_dir": str(output_dir),
            "max_steps": int(max_steps),
            "objective": objective,
            "reward_key": reward_key,
            "dataset_profile": dataset_profile,
            "validation_profile": validation_profile,
            "max_train_prompts": int(max_train_prompts),
            "validation_count": int(validation_count),
            "resume": resume,
            "preflight_only": bool(preflight_only),
            "smoke": bool(smoke),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        # Persist logs/checkpoints even when Python raises after training starts.
        try:
            runs.commit()
            hf_cache.commit()
        except AttributeError:
            # Older Modal clients auto-commit mounted volumes on clean exit.
            pass


@app.local_entrypoint()
def main(
    branch: str = DEFAULT_BRANCH,
    max_steps: int = 50,
    objective: str = "posterior_tilted_regression",
    reward_key: str = "videoalign_mq",
    dataset_profile: str = "world_r1_enhanced_dynamic",
    validation_profile: str = "world_r1_enhanced_test",
    max_train_prompts: int = 515,
    validation_count: int = 16,
    run_name: str = "wan_anyflow_pt_pdd_videoalign_mq",
    resume: str = "auto",
    preflight_only: bool = False,
    smoke: bool = False,
) -> None:
    result = train.remote(
        branch=branch,
        max_steps=max_steps,
        objective=objective,
        reward_key=reward_key,
        dataset_profile=dataset_profile,
        validation_profile=validation_profile,
        max_train_prompts=max_train_prompts,
        validation_count=validation_count,
        run_name=run_name,
        resume=resume,
        preflight_only=preflight_only,
        smoke=smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
