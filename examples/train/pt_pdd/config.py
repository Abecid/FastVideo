# SPDX-License-Identifier: Apache-2.0
"""Configuration and prompt profiles for the PT-PDD AnyFlow experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

import yaml

from examples.train.pt_pdd.core import PosteriorPolicyConfig
from examples.train.vptd import train_anyflow as base

PROMPT_PROFILES: dict[str, dict[str, Any]] = {
    "world_r1_enhanced_dynamic": {
        "dataset": "microsoft/World-R1",
        "subset": "enhanced",
        "split": "dynamic",
        "text_field": "prompt",
        "streaming": False,
    },
    "world_r1_enhanced_train": {
        "dataset": "microsoft/World-R1",
        "subset": "enhanced",
        "split": "train",
        "text_field": "prompt",
        "streaming": False,
    },
    "world_r1_enhanced_test": {
        "dataset": "microsoft/World-R1",
        "subset": "enhanced",
        "split": "test",
        "text_field": "prompt",
        "streaming": False,
    },
    "world_r1_final_dynamic": {
        "dataset": "microsoft/World-R1",
        "subset": "final",
        "split": "dynamic",
        "text_field": "prompt",
        "streaming": False,
    },
    "world_r1_final_train": {
        "dataset": "microsoft/World-R1",
        "subset": "final",
        "split": "train",
        "text_field": "prompt",
        "streaming": False,
    },
    "world_r1_final_test": {
        "dataset": "microsoft/World-R1",
        "subset": "final",
        "split": "test",
        "text_field": "prompt",
        "streaming": False,
    },
    # PDD reports data-free Wan training with ViMix prompts. The current public
    # Hub repository is a small sample, so this profile is a distribution-shift
    # probe rather than the main training source.
    "vimix_public_sample": {
        "dataset": "TimingYang/ViMix-14M",
        "subset": None,
        "split": "train",
        "text_field": "caption_middle_en",
        "streaming": False,
    },
    # VidProM is large; streaming and an explicit cap avoid materializing the
    # complete prompt gallery during pilot experiments.
    "vidprom_unique": {
        "dataset": "WenhaoWang/VidProM",
        "subset": "VidProM_unique",
        "split": "train",
        "text_field": "prompt",
        "streaming": True,
    },
}

_ALLOWED_OBJECTIVES = {
    "posterior_tilted_regression",
    "reference_regression",
    "posterior_distillation",
    "flowmap_grpo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/train/configs/pt_pdd/wan_anyflow_videoalign_mq.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--reward-key")
    parser.add_argument("--dataset-profile")
    parser.add_argument("--validation-profile")
    parser.add_argument("--max-train-prompts", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--objective", choices=tuple(sorted(_ALLOWED_OBJECTIVES)))
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run official AnyFlow parity plus base-model evaluation and exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Infrastructure-only two-update run with reduced video shape. "
            "It is not a scientific comparison."
        ),
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = str(args.output_dir)
    if args.run_name:
        cfg["experiment"]["run_name"] = args.run_name
    if args.max_train_steps is not None:
        cfg["experiment"]["max_train_steps"] = int(args.max_train_steps)
    if args.reward_key:
        cfg["reward"]["optimize"] = args.reward_key
    if args.dataset_profile:
        cfg["prompts"]["train"]["profile"] = args.dataset_profile
    if args.validation_profile:
        cfg["prompts"]["validation"]["profile"] = args.validation_profile
    if args.max_train_prompts is not None:
        cfg["prompts"]["train"]["max_count"] = int(args.max_train_prompts)
    if args.validation_count is not None:
        cfg["prompts"]["validation"]["count"] = int(args.validation_count)
    if args.objective:
        cfg["posterior_policy"]["objective"] = args.objective
    if args.eval_every is not None:
        cfg["experiment"]["eval_every"] = int(args.eval_every)
    if args.save_every is not None:
        cfg["experiment"]["save_every"] = int(args.save_every)
    if args.resume is not None:
        cfg["experiment"]["resume"] = args.resume

    cfg["experiment"]["smoke_only"] = bool(args.smoke)
    cfg["experiment"]["preflight_only"] = bool(args.preflight_only)
    if args.smoke:
        cfg["experiment"]["max_train_steps"] = 2
        cfg["experiment"]["eval_every"] = 1
        cfg["experiment"]["save_every"] = 1
        cfg["experiment"]["run_name"] = (
            f"{cfg['experiment']['run_name']}_smoke"
        )
        cfg["model"]["num_frames"] = 17
        cfg["model"]["height"] = 256
        cfg["model"]["width"] = 448
        cfg["prompts"]["validation"]["count"] = 4
        cfg["prompts"]["qualitative_count"] = 4
        cfg["prompts"]["train"]["max_count"] = 16

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Reuse VPTD's locked AnyFlow checks, then validate PT-PDD additions."""

    prompts = cfg["prompts"]
    train_cfg = prompts["train"]
    validation_cfg = prompts["validation"]
    train_profile = str(train_cfg.get("profile", ""))
    validation_profile = str(validation_cfg.get("profile", ""))
    if train_profile not in PROMPT_PROFILES:
        raise ValueError(
            f"unknown train profile {train_profile!r}; "
            f"choose from {sorted(PROMPT_PROFILES)}"
        )
    if validation_profile not in PROMPT_PROFILES:
        raise ValueError(
            f"unknown validation profile {validation_profile!r}; "
            f"choose from {sorted(PROMPT_PROFILES)}"
        )
    if int(train_cfg.get("max_count", 0)) <= 0:
        raise ValueError("prompts.train.max_count must be positive")
    validation_count = int(validation_cfg.get("count", 0))
    qualitative_count = int(prompts.get("qualitative_count", 0))
    if validation_count < qualitative_count:
        raise ValueError("validation count must be >= qualitative_count")

    objective = str(cfg["posterior_policy"].get("objective", ""))
    if objective not in _ALLOWED_OBJECTIVES:
        raise ValueError(f"unsupported posterior_policy.objective {objective!r}")

    # Map the nested prompt profile to the legacy VPTD schema and use a legacy
    # objective solely while running its audited model/schedule/optimizer checks.
    legacy = deepcopy(cfg)
    profile = PROMPT_PROFILES[train_profile]
    legacy["prompts"] = {
        "dataset": profile["dataset"],
        "subset": profile["subset"],
        "split": profile["split"],
        "text_field": profile["text_field"],
        "validation_count": validation_count,
        "qualitative_count": qualitative_count,
    }
    legacy["posterior_policy"]["objective"] = "posterior_distillation"
    base.validate_config(legacy)

    policy = cfg["posterior_policy"]
    PosteriorPolicyConfig(
        stochastic_steps=int(policy["stochastic_steps"]),
        group_size=int(policy["group_size"]),
        target_ess_ratio=float(policy["target_ess_ratio"]),
        clip_range=float(policy["clip_range"]),
        advantage_clip=float(policy["advantage_clip"]),
        advantage_epsilon=float(policy["advantage_epsilon"]),
    ).validate()


def _load_prompt_profile(
    profile_name: str,
    *,
    max_count: int,
    seed: int,
    shuffle: bool,
) -> list[str]:
    from datasets import load_dataset

    profile = PROMPT_PROFILES[profile_name]
    kwargs: dict[str, Any] = {
        "split": profile["split"],
        "streaming": bool(profile["streaming"]),
    }
    subset = profile.get("subset")
    if subset is None:
        rows = load_dataset(profile["dataset"], **kwargs)
    else:
        rows = load_dataset(profile["dataset"], subset, **kwargs)
    if bool(profile["streaming"]) and shuffle:
        rows = rows.shuffle(seed=int(seed))

    field = str(profile["text_field"])
    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        text = str(row.get(field, "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        if len(prompts) >= int(max_count):
            break
    if shuffle and not bool(profile["streaming"]):
        random.Random(int(seed)).shuffle(prompts)
    if not prompts:
        raise RuntimeError(f"prompt profile {profile_name!r} yielded no prompts")
    return prompts


def prepare_prompt_split(
    cfg: dict[str, Any],
    info: base.DistInfo,
) -> tuple[list[str], list[str]]:
    """Materialize explicit train and held-out prompt profiles reproducibly."""

    output_dir = Path(cfg["experiment"]["output_dir"])
    split_path = output_dir / "prompt_split.json"
    prompts_cfg = cfg["prompts"]
    train_cfg = prompts_cfg["train"]
    validation_cfg = prompts_cfg["validation"]
    signature = {
        "train_profile": str(train_cfg["profile"]),
        "train_max_count": int(train_cfg["max_count"]),
        "validation_profile": str(validation_cfg["profile"]),
        "validation_count": int(validation_cfg["count"]),
        "seed": int(cfg["experiment"]["seed"]),
    }

    if info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if split_path.exists():
            existing = json.loads(split_path.read_text(encoding="utf-8"))
            if existing.get("signature") != signature:
                raise RuntimeError(
                    "existing prompt_split.json uses different dataset settings; "
                    "use a new run_name/output_dir"
                )
        else:
            train_prompts = _load_prompt_profile(
                signature["train_profile"],
                max_count=signature["train_max_count"],
                seed=signature["seed"],
                shuffle=True,
            )
            validation_prompts = _load_prompt_profile(
                signature["validation_profile"],
                max_count=signature["validation_count"],
                seed=signature["seed"],
                shuffle=False,
            )
            validation_set = set(validation_prompts)
            train_prompts = [
                prompt for prompt in train_prompts if prompt not in validation_set
            ]
            if len(train_prompts) < info.world_size:
                raise RuntimeError(
                    "training profile did not yield enough non-overlapping prompts "
                    f"for {info.world_size} ranks"
                )
            split_path.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "train": train_prompts,
                        "validation": validation_prompts,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    base.barrier(info)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return list(payload["train"]), list(payload["validation"])
