# pyright: reportAttributeAccessIssue=false
"""Launch Reward-Tilted Flow Distillation on Modal.

Examples:

    modal run examples/train/modal_rtfd.py --max-steps 5 --num-frames 17
    modal run examples/train/modal_rtfd.py --max-steps 100 --uniform-mix 0.25
    modal run examples/train/modal_rtfd.py --max-steps 100 --uniform-mix 1.0 \
        --run-name wan2.1_rtfd_uniform_baseline
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import modal


REPOSITORY = "https://github.com/Abecid/FastVideo.git"
DEFAULT_BRANCH = "agent/reward-tilted-flow-distillation"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"

IMAGE_REF = os.environ.get("FASTVIDEO_MODAL_IMAGE", DEFAULT_IMAGE)
GPU = os.environ.get("RTFD_MODAL_GPU", "H100:4")
SECRET_NAME = os.environ.get("RTFD_MODAL_SECRET", "fastvideo-training")

image = (
    modal.Image.from_registry(IMAGE_REF, add_python="3.12")
    .apt_install("git", "ffmpeg")
    .run_commands("python -m pip install --upgrade uv")
)

app = modal.App("fastvideo-rtfd")
hf_cache = modal.Volume.from_name("fastvideo-rtfd-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("fastvideo-rtfd-runs", create_if_missing=True)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


@app.function(
    image=image,
    gpu=GPU,
    cpu=32,
    memory=131072,
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
    max_steps: int = 100,
    teacher_steps: int = 16,
    student_steps: int = 4,
    trajectories_per_prompt: int = 4,
    transition_batch_size: int = 1,
    ess_ratio: float = 0.60,
    uniform_mix: float = 0.25,
    num_frames: int = 49,
    height: int = 448,
    width: int = 832,
    max_prompts: str = "64",
    run_name: str = "wan2.1_rtfd_videoalign",
    skip_preprocess: bool = False,
) -> dict[str, str | int | float]:
    if teacher_steps <= 0 or student_steps <= 0:
        raise ValueError("teacher_steps and student_steps must be positive")
    if trajectories_per_prompt <= 0 or transition_batch_size <= 0:
        raise ValueError("trajectory and transition batch sizes must be positive")
    if not 0.0 < ess_ratio <= 1.0:
        raise ValueError("ess_ratio must lie in (0, 1]")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must lie in [0, 1]")

    workspace = Path("/workspace")
    repo = workspace / "FastVideo"
    subprocess.run(["rm", "-rf", str(repo)], check=True)
    workspace.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VIDEOALIGN_CHECKPOINT_PATH": "/runs/cache/rtfd/VideoReward",
            "WANDB_MODE": "online" if (env.get("WANDB_API_KEY") or "").strip() else "offline",
            "NUM_GPUS": "4",
            "MASTER_PORT": "29571",
        }
    )

    _run(
        ["git", "clone", "--depth", "1", "--branch", branch, REPOSITORY, str(repo)],
        cwd=workspace,
        env=env,
    )
    _run(["uv", "pip", "install", "-e", "."], cwd=repo, env=env)
    _run(
        ["uv", "pip", "install", "-r", "examples/train/requirements-dmdr.txt"],
        cwd=repo,
        env=env,
    )

    output_dir = Path("/runs/outputs") / run_name
    config_dir = Path("/runs/configs") / run_name
    prep = [
        "python",
        "examples/train/prepare_dmdr_assets.py",
        "--config",
        "examples/train/configs/reward_tilted_flow/wan/rtfd_videoalign.yaml",
        "--data-root",
        "/runs/data/rtfd",
        "--cache-root",
        "/runs/cache/rtfd",
        "--output-dir",
        str(output_dir),
        "--run-config-dir",
        str(config_dir),
        "--dataset",
        "world-r1-enhanced",
        "--max-prompts",
        str(max_prompts),
        "--num-frames",
        str(num_frames),
        "--num-height",
        str(height),
        "--num-width",
        str(width),
        "--preprocess-num-gpus",
        "1",
        "--num-gpus",
        "4",
        "--hsdp-shard-dim",
        "4",
        "--max-train-steps",
        str(max_steps),
        "--gradient-accumulation-steps",
        "1",
        "--num-samples-per-prompt",
        str(trajectories_per_prompt),
        "--num-batches-per-epoch",
        "1",
        "--collection-batch-size",
        "1",
        "--inner-epochs",
        "1",
        "--train-batch-size",
        "1",
        "--cold-start-steps",
        "0",
        "--dynamic-step",
        "0",
        "--guidance-update-ratio",
        "1",
        "--sample-num-steps",
        str(teacher_steps),
        "--sample-flow-shift",
        "8.0",
        "--sample-guidance-scale",
        "6.0",
        "--validation-every-steps",
        "5",
        "--validation-num-steps",
        str(student_steps),
        "--validation-num-prompts",
        "4",
        "--validation-batch-size",
        "1",
        "--project-name",
        "rtfd_wan",
        "--run-name",
        run_name,
        "--check-rewards",
        "--json",
    ]
    if skip_preprocess:
        prep.append("--skip-preprocess")
    _run(prep, cwd=repo, env=env)

    generated = config_dir / "dmdr_wan_run.yaml"
    resolved = config_dir / "rtfd_wan_run.yaml"
    patch_script = f"""
from pathlib import Path
import yaml

source = Path({str(generated)!r})
target = Path({str(resolved)!r})
cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
method = cfg['method']
method['student_num_steps'] = {student_steps!r}
method['student_guidance_scale'] = 1.0
method['trajectories_per_prompt'] = {trajectories_per_prompt!r}
method['transition_batch_size'] = {transition_batch_size!r}
method['reward_ess_ratio'] = {ess_ratio!r}
method['uniform_mix'] = {uniform_mix!r}
method['reward_bisection_steps'] = 32
for stale in (
    'cold_start_steps', 'dynamic_step', 'guidance_update_ratio',
    'sample_train_batch_size', 'train_batch_size', 'num_batches_per_epoch',
    'num_inner_epochs', 'num_video_per_prompt',
):
    method.pop(stale, None)
method['validation']['num_steps'] = {student_steps!r}
cfg['training']['loop']['gradient_accumulation_steps'] = 1
cfg['training']['checkpoint']['output_dir'] = {str(output_dir)!r}
cfg['training']['tracker']['project_name'] = 'rtfd_wan'
cfg['training']['tracker']['run_name'] = {run_name!r}
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
print(target)
"""
    _run(["python", "-c", patch_script], cwd=repo, env=env)
    _run(["bash", "examples/train/run.sh", str(resolved)], cwd=repo, env=env)

    try:
        runs.commit()
        hf_cache.commit()
    except AttributeError:
        pass

    summary: dict[str, str | int | float] = {
        "branch": branch,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "resolved_config": str(resolved),
        "max_steps": max_steps,
        "teacher_steps": teacher_steps,
        "student_steps": student_steps,
        "trajectories_per_prompt": trajectories_per_prompt,
        "ess_ratio": ess_ratio,
        "uniform_mix": uniform_mix,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main(
    branch: str = DEFAULT_BRANCH,
    max_steps: int = 100,
    teacher_steps: int = 16,
    student_steps: int = 4,
    trajectories_per_prompt: int = 4,
    transition_batch_size: int = 1,
    ess_ratio: float = 0.60,
    uniform_mix: float = 0.25,
    num_frames: int = 49,
    height: int = 448,
    width: int = 832,
    max_prompts: str = "64",
    run_name: str = "wan2.1_rtfd_videoalign",
    skip_preprocess: bool = False,
) -> None:
    result = train.remote(
        branch=branch,
        max_steps=max_steps,
        teacher_steps=teacher_steps,
        student_steps=student_steps,
        trajectories_per_prompt=trajectories_per_prompt,
        transition_batch_size=transition_batch_size,
        ess_ratio=ess_ratio,
        uniform_mix=uniform_mix,
        num_frames=num_frames,
        height=height,
        width=width,
        max_prompts=max_prompts,
        run_name=run_name,
        skip_preprocess=skip_preprocess,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
