# SPDX-License-Identifier: Apache-2.0
"""Audit VideoAlign checkpoint coverage and deterministic calibration clips."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def synthetic_clips(
    *,
    frames: int = 16,
    height: int = 224,
    width: int = 224,
) -> tuple[torch.Tensor, list[str]]:
    static = torch.full((3, frames, height, width), 0.25)

    moving = torch.zeros((3, frames, height, width))
    square = max(8, min(height, width) // 8)
    for index in range(frames):
        x = int((width - square) * index / max(frames - 1, 1))
        y = (height - square) // 2
        moving[:, index, y : y + square, x : x + square] = 1.0

    flicker = torch.zeros((3, frames, height, width))
    flicker[:, 1::2] = 1.0
    media = torch.stack((static, moving, flicker), dim=0)
    prompts = [
        "A static gray scene.",
        "A white square moves smoothly from left to right.",
        "A scene rapidly flickers between black and white.",
    ]
    return media, prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-overall-coverage", type=float, default=0.97)
    parser.add_argument("--minimum-component-coverage", type=float, default=0.95)
    parser.add_argument("--require-reward-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-motion-order", action="store_true")
    parser.add_argument("--repeatability-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.checkpoint_path is not None:
        checkpoint = args.checkpoint_path.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = str(checkpoint)

    from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer
    from fastvideo.train.methods.rl.rewards.videoalign_audit import (
        audit_videoalign_checkpoint,
        repeatability_probe,
    )

    reward_map = {
        "videoalign_mq_audited": 1.0,
        "videoalign_vq_audited": 1.0,
        "videoalign_ta_audited": 1.0,
    }
    scorer = build_multi_reward_scorer(
        reward_map,
        backend="genrl",
        device=torch.device(device),
    )
    coverage = audit_videoalign_checkpoint(
        device=device,
        checkpoint_path=os.environ.get("VIDEOALIGN_CHECKPOINT_PATH"),
        minimum_checkpoint_numel_coverage=args.minimum_overall_coverage,
        minimum_component_coverage=args.minimum_component_coverage,
        require_reward_head=args.require_reward_head,
    )
    repeatability = repeatability_probe(
        scorer,
        device=device,
        tolerance=args.repeatability_tolerance,
    )

    media, prompts = synthetic_clips()
    scores = scorer(media.to(device), prompts)
    values = {
        name: [float(item) for item in tensor.detach().float().cpu()]
        for name, tensor in scores.items()
    }
    for name, items in values.items():
        if not torch.isfinite(torch.tensor(items)).all():
            raise RuntimeError(f"VideoAlign returned non-finite {name}: {items}")

    if args.require_motion_order:
        mq = values["videoalign_mq_audited"]
        if not mq[1] > mq[0]:
            raise RuntimeError(
                "VideoAlign MQ calibration did not rank smooth motion above "
                f"the static clip: moving={mq[1]:.6f}, static={mq[0]:.6f}"
            )

    summary = {
        "device": str(device),
        "checkpoint_path": os.environ.get("VIDEOALIGN_CHECKPOINT_PATH"),
        "clips": ["static", "smooth_motion", "flicker"],
        "scores": values,
        "coverage": coverage,
        "repeatability": repeatability,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print("VideoAlign audit passed:")
    print(payload, end="")
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
