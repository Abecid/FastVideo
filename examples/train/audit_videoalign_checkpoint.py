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
    parser.add_argument("--minimum-overall-coverage", type=float, default=0.90)
    parser.add_argument("--minimum-head-coverage", type=float, default=0.99)
    parser.add_argument("--require-motion-order", action="store_true")
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

    os.environ["VIDEOALIGN_MIN_OVERALL_COVERAGE"] = str(
        args.minimum_overall_coverage
    )
    os.environ["VIDEOALIGN_MIN_HEAD_COVERAGE"] = str(
        args.minimum_head_coverage
    )

    from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer
    from fastvideo.train.methods.rl.rewards.videoalign_audit import (
        videoalign_coverage_summary,
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
    media, prompts = synthetic_clips()
    media = media.to(device)
    scores = scorer(media, prompts)
    values = {
        name: [float(item) for item in tensor.detach().float().cpu()]
        for name, tensor in scores.items()
    }
    for name, items in values.items():
        if not all(torch.isfinite(torch.tensor(items))):
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
        "coverage": videoalign_coverage_summary(),
    }
    print("VideoAlign audit passed:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
