"""Launch AnyFlow finite-transition posterior alignment on Modal.

The scientific comparison changes only ``method.objective``:

- ``posterior_projection``: proposed centered forward-KL projection.
- ``flowmap_grpo``: matched clipped likelihood-ratio baseline.

Both runs use the same held-out prompt split, ASFMC branches, rewards, optimizer,
validation seeds, and deterministic 4-step AnyFlow inference.
"""

from __future__ import annotations

import modal

app = modal.App("fastvideo-finite-transition-posterior")

PROJECT_ROOT = "/root/FastVideo"
CONFIG_PATH = (
    "examples/train/configs/rl/wan/"
    "finite_transition_posterior_anyflow_videoalign.yaml"
)
MODAL_DATA_ROOT = f"{PROJECT_ROOT}/.modal_data"
MODAL_CACHE_ROOT = f"{PROJECT_ROOT}/.modal_cache"
OUTPUT_ROOT = f"{PROJECT_ROOT}/outputs/finite_transition_posterior"
VIDEOALIGN_ROOT = f"{MODAL_CACHE_ROOT}/VideoReward"
DIFFUSION_NFT_ROOT = f"{MODAL_CACHE_ROOT}/DiffusionNFT"
WANDB_SECRET = "wandb-adamlee00"
HF_SECRET = "hf-adamlee00"
NUM_GPUS = 4
GPU_TYPE = "H100"

CONTEXT_IGNORE = [
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
        ".",
        PROJECT_ROOT,
        copy=True,
        ignore=CONTEXT_IGNORE,
    )
    .run_commands(
        "cd /root/FastVideo && UV_TORCH_BACKEND=cu130 "
        "uv pip install --system --prerelease=allow -e .",
        "uv pip install --system --no-cache-dir "
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
        "releases/download/v0.9.17/"
        "flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl",
        "uv pip check --system",
        "python -c 'import torch, flash_attn; "
        "print(torch.__version__, torch.version.cuda, flash_attn.__version__)'",
        "cd /root/FastVideo && python -c '"
        "from fastvideo.train.methods.rl.rewards.hpsv3 import "
        "_patch_hpsv3_state_dict_loader, _patch_transformers_video_input_alias; "
        "_patch_transformers_video_input_alias(); "
        "_patch_hpsv3_state_dict_loader(); "
        "from fastvideo.train.methods.rl.rewards.videoalign import "
        "_patch_videoalign_modules; _patch_videoalign_modules(); "
        "print(\"reward runtimes ok\")'",
    )
    .env(
        {
            "WANDB_MODE": "online",
            "WANDB_ENTITY": "adamlee00",
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
    objective: str = "posterior_projection",
    max_train_steps: int = 1200,
    dataset: str = "world-r1-enhanced-dynamic",
    reward: str = "videoalign_mq",
    max_train_prompts: str = "512",
    validation_prompts: int = 64,
    validation_every: int = 100,
    validation_samples_per_prompt: int = 2,
    validation_log_videos: int = 8,
    group_size: int = 4,
    target_ess_ratio: float = 0.5,
    lora_rank: int = 256,
    learning_rate: float = 2.0e-6,
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    smoke: bool = False,
    prepare_only: bool = False,
    skip_preprocess: bool = False,
) -> dict[str, str | int | float | bool]:
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    import subprocess
    import sys

    if objective not in {"posterior_projection", "flowmap_grpo"}:
        raise ValueError(
            "objective must be posterior_projection or flowmap_grpo"
        )
    if group_size % NUM_GPUS != 0:
        raise ValueError("group_size must be divisible by the four Modal GPUs")
    if smoke:
        max_train_steps = 2
        validation_prompts = min(validation_prompts, 8)
        validation_every = 1
        validation_samples_per_prompt = 2
        validation_log_videos = min(validation_log_videos, 4)
        lora_rank = min(lora_rank, 64)
        num_frames = 17
        height = 256
        width = 448
        max_train_prompts = "32"

    repo = Path(PROJECT_ROOT)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"anyflow_ftp_{objective}_{reward}_"
        f"f{num_frames}_{timestamp}"
    )
    output_dir = f"{OUTPUT_ROOT}/{run_name}"
    run_config_dir = f"{PROJECT_ROOT}/outputs/ftp_run_configs/{run_name}"

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from flash_attn import flash_attn_func; "
            "q=torch.randn((1,128,8,64),device='cuda',dtype=torch.bfloat16,requires_grad=True); "
            "flash_attn_func(q,q,q).sum().backward(); print('flash attention ok')",
        ],
        check=True,
    )

    prep_cmd = [
        sys.executable,
        "examples/train/prepare_finite_transition_posterior_assets.py",
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
        run_config_dir,
        "--diffusion-nft-root",
        DIFFUSION_NFT_ROOT,
        "--videoalign-checkpoint-path",
        VIDEOALIGN_ROOT,
        "--dataset",
        dataset,
        "--reward",
        reward,
        "--objective",
        objective,
        "--max-train-prompts",
        str(max_train_prompts),
        "--validation-prompts",
        str(validation_prompts),
        "--validation-every",
        str(validation_every),
        "--validation-samples-per-prompt",
        str(validation_samples_per_prompt),
        "--validation-log-videos",
        str(validation_log_videos),
        "--group-size",
        str(group_size),
        "--target-ess-ratio",
        str(target_ess_ratio),
        "--lora-rank",
        str(lora_rank),
        "--learning-rate",
        str(learning_rate),
        "--max-train-steps",
        str(max_train_steps),
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
        "finite-transition-posterior-wan",
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
    data_volume.commit()
    cache_volume.commit()

    result: dict[str, str | int | float | bool] = {
        "objective": objective,
        "run_name": run_name,
        "output_dir": output_dir,
        "run_config": str(summary["run_config"]),
        "train_prompt_count": int(summary["train_prompt_count"]),
        "validation_prompt_count": int(
            summary["validation_prompt_count"]
        ),
        "num_frames": num_frames,
        "group_size": group_size,
        "target_ess_ratio": target_ess_ratio,
        "smoke": smoke,
        "prepare_only": prepare_only,
    }
    if prepare_only:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result

    subprocess.run(
        [
            "bash",
            "examples/train/run.sh",
            str(summary["run_config"]),
        ],
        cwd=repo,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )
    runs_volume.commit()
    cache_volume.commit()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(
    objective: str = "posterior_projection",
    max_train_steps: int = 1200,
    dataset: str = "world-r1-enhanced-dynamic",
    reward: str = "videoalign_mq",
    max_train_prompts: str = "512",
    validation_prompts: int = 64,
    validation_every: int = 100,
    validation_samples_per_prompt: int = 2,
    validation_log_videos: int = 8,
    group_size: int = 4,
    target_ess_ratio: float = 0.5,
    lora_rank: int = 256,
    learning_rate: float = 2.0e-6,
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    smoke: bool = False,
    prepare_only: bool = False,
    skip_preprocess: bool = False,
    paired: bool = False,
) -> None:
    kwargs = dict(
        max_train_steps=max_train_steps,
        dataset=dataset,
        reward=reward,
        max_train_prompts=max_train_prompts,
        validation_prompts=validation_prompts,
        validation_every=validation_every,
        validation_samples_per_prompt=validation_samples_per_prompt,
        validation_log_videos=validation_log_videos,
        group_size=group_size,
        target_ess_ratio=target_ess_ratio,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        smoke=smoke,
        prepare_only=prepare_only,
        skip_preprocess=skip_preprocess,
    )
    if paired:
        # Materialize and commit the deterministic split once before the two
        # scientific jobs start. Otherwise simultaneous preprocess jobs can
        # race on the shared Modal volume.
        prep_kwargs = dict(kwargs)
        prep_kwargs["prepare_only"] = True
        prep_kwargs["skip_preprocess"] = False
        train.remote(objective="posterior_projection", **prep_kwargs)

        run_kwargs = dict(kwargs)
        run_kwargs["prepare_only"] = False
        run_kwargs["skip_preprocess"] = True
        train.spawn(objective="posterior_projection", **run_kwargs)
        train.spawn(objective="flowmap_grpo", **run_kwargs)
        return
    train.spawn(objective=objective, **kwargs)
