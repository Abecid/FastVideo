# SPDX-License-Identifier: Apache-2.0
"""Fail-fast preflight for the AnyFlow finite-transition experiment.

The check intentionally downloads only small repository metadata/config files.
The actual model weights, text embeddings, reward checkpoint and parquet data are
materialized by the normal preparation/training path and cached on Modal volumes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.train.prepare_diffusion_nft_assets import load_prompts
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import resolve_target

DEFAULT_MODEL_ID = "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_CONFIG = (
    "examples/train/configs/rl/wan/"
    "finite_transition_posterior_anyflow_videoalign.yaml"
)


def _read_hub_json(repo_id: str, filename: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"{repo_id}/{filename} is not a JSON object")
    return raw


def check_anyflow_model_contract(model_id: str) -> dict[str, Any]:
    """Verify that *model_id* is the released two-time AnyFlow checkpoint."""
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(model_id))
    required_files = {
        "model_index.json",
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "scheduler/scheduler_config.json",
    }
    missing = sorted(required_files - files)
    if missing:
        raise RuntimeError(
            f"AnyFlow checkpoint {model_id!r} is missing required files: {missing}"
        )
    if not any(
        name.startswith("transformer/") and name.endswith(".safetensors")
        for name in files
    ):
        raise RuntimeError(f"AnyFlow checkpoint {model_id!r} has no transformer weights")

    model_index = _read_hub_json(model_id, "model_index.json")
    pipeline_class = str(model_index.get("_class_name", ""))
    if "anyflow" not in pipeline_class.lower():
        raise RuntimeError(
            f"Expected an AnyFlow pipeline, got _class_name={pipeline_class!r}"
        )

    transformer = _read_hub_json(model_id, "transformer/config.json")
    gate = transformer.get(
        "gate_value",
        transformer.get("r_embedder_gate_value"),
    )
    deltatime_type = transformer.get(
        "deltatime_type",
        transformer.get("r_embedder_deltatime_type"),
    )
    if gate is None:
        raise RuntimeError(
            "AnyFlow transformer config is missing the dual-timestep gate value"
        )
    if abs(float(gate) - 0.25) > 1.0e-6:
        raise RuntimeError(
            f"Expected released AnyFlow gate_value=0.25, got {gate!r}"
        )
    if str(deltatime_type).strip().lower() != "r":
        raise RuntimeError(
            "Expected released AnyFlow deltatime_type='r', got "
            f"{deltatime_type!r}"
        )

    return {
        "model_id": model_id,
        "pipeline_class": pipeline_class,
        "transformer_class": transformer.get("_class_name"),
        "gate_value": float(gate),
        "deltatime_type": str(deltatime_type),
        "repo_file_count": len(files),
    }


def check_fastvideo_config(config_path: Path) -> dict[str, Any]:
    cfg = load_run_config(str(config_path))
    method_target = str(cfg.method["_target_"])
    method_cls = resolve_target(method_target)
    pipeline = cfg.training.pipeline_config
    if pipeline is None:
        raise RuntimeError("FTPP config did not resolve a pipeline configuration")
    arch = pipeline.dit_config.arch_config
    if not bool(getattr(arch, "r_embedder", False)):
        raise RuntimeError("FTPP config must enable AnyFlow r_embedder")
    gate = float(getattr(arch, "r_embedder_gate_value", float("nan")))
    if abs(gate - 0.25) > 1.0e-6:
        raise RuntimeError(f"FTPP config resolved an invalid r-embedder gate: {gate}")
    deltatime_type = str(getattr(arch, "r_embedder_deltatime_type", ""))
    if deltatime_type != "r":
        raise RuntimeError(
            "FTPP config must use r_embedder_deltatime_type='r', got "
            f"{deltatime_type!r}"
        )
    return {
        "config": str(config_path),
        "method_target": method_target,
        "method_class": method_cls.__name__,
        "flow_shift": float(pipeline.flow_shift),
        "r_embedder_gate_value": gate,
        "r_embedder_deltatime_type": deltatime_type,
    }


def check_prompt_source(
    dataset: str,
    *,
    diffusion_nft_root: Path,
    validation_prompts: int,
    num_gpus: int,
) -> dict[str, Any]:
    prompts, source = load_prompts(
        dataset,
        diffusion_nft_root=diffusion_nft_root,
    )
    unique = list(
        dict.fromkeys(str(prompt).strip() for prompt in prompts if str(prompt).strip())
    )
    minimum = int(validation_prompts) + int(num_gpus) + 1
    if len(unique) < minimum:
        raise RuntimeError(
            f"Prompt source {source!r} has {len(unique)} unique prompts; "
            f"need at least {minimum} for a disjoint split"
        )
    return {
        "dataset": dataset,
        "source": source,
        "unique_prompt_count": len(unique),
        "first_prompt": unique[0][:200],
    }


def check_credentials(*, require_wandb: bool) -> dict[str, bool]:
    wandb_present = bool(os.environ.get("WANDB_API_KEY"))
    hf_present = bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if require_wandb and not wandb_present:
        raise RuntimeError(
            "WANDB_API_KEY is missing. The Modal secret 'wandb-adamlee00' "
            "must expose WANDB_API_KEY."
        )
    return {
        "wandb_api_key_present": wandb_present,
        "huggingface_token_present": hf_present,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", default="world-r1-enhanced-dynamic")
    parser.add_argument("--diffusion-nft-root", type=Path, default=Path(".cache/DiffusionNFT"))
    parser.add_argument("--validation-prompts", type=int, default=64)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if not args.config.is_absolute():
        args.config = (args.repo_root / args.config).resolve()
    if not args.diffusion_nft_root.is_absolute():
        args.diffusion_nft_root = (
            args.repo_root / args.diffusion_nft_root
        ).resolve()
    if args.validation_prompts <= 0 or args.num_gpus <= 0:
        raise ValueError("validation-prompts and num-gpus must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary = {
        "credentials": check_credentials(require_wandb=args.require_wandb),
        "model": check_anyflow_model_contract(args.model_id),
        "config": check_fastvideo_config(args.config),
        "prompts": check_prompt_source(
            args.dataset,
            diffusion_nft_root=args.diffusion_nft_root,
            validation_prompts=args.validation_prompts,
            num_gpus=args.num_gpus,
        ),
    }
    print("Finite-transition posterior preflight passed:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
