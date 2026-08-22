# SPDX-License-Identifier: Apache-2.0
"""Compare two exact finite-transition validation artifacts.

The tool verifies literal prompt/seed identity, computes left-minus-right values
for every sample, averages fixed seeds within each prompt, and bootstraps prompts.
This avoids treating multiple seeds from one prompt as independent evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from fastvideo.train.methods.rl.common.reward_statistics import paired_summary

IDENTITY_KEYS = {"sample_key", "sample_seed", "prompt_index"}


def load_artifact(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} does not contain a metrics mapping")
    return raw


def _prompt_means(
    values: list[float],
    prompt_indices: list[int | float],
) -> torch.Tensor:
    if len(values) != len(prompt_indices):
        raise ValueError("metric and prompt-index vectors have different lengths")
    grouped: dict[int, list[float]] = defaultdict(list)
    for prompt_index, value in zip(prompt_indices, values, strict=True):
        grouped[int(prompt_index)].append(float(value))
    if len(grouped) < 2:
        raise ValueError("cross-arm comparison requires at least two prompts")
    return torch.tensor(
        [
            sum(grouped[index]) / len(grouped[index])
            for index in sorted(grouped)
        ],
        dtype=torch.float64,
    )


def compare(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    left_metrics = dict(left["metrics"])
    right_metrics = dict(right["metrics"])
    for key in IDENTITY_KEYS:
        if key in left_metrics or key in right_metrics:
            if left_metrics.get(key) != right_metrics.get(key):
                raise RuntimeError(
                    f"paired objective artifacts disagree on {key}"
                )

    prompt_indices = left_metrics.get("prompt_index")
    if not isinstance(prompt_indices, list):
        raise ValueError("exact artifacts must contain sample-level prompt_index")

    summaries: dict[str, dict[str, float]] = {}
    for metric_offset, name in enumerate(
        sorted(set(left_metrics) & set(right_metrics))
    ):
        if name in IDENTITY_KEYS:
            continue
        left_values = left_metrics[name]
        right_values = right_metrics[name]
        if not isinstance(left_values, list) or not isinstance(
            right_values,
            list,
        ):
            continue
        if len(left_values) != len(right_values) or len(left_values) < 2:
            continue
        left_prompt = _prompt_means(left_values, prompt_indices)
        right_prompt = _prompt_means(right_values, prompt_indices)
        summary = paired_summary(
            left_prompt,
            right_prompt,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=int(seed) + int(metric_offset),
        )
        summary["sample_count"] = float(len(left_values))
        summary["prompt_count"] = float(left_prompt.numel())
        summaries[name] = summary

    return {
        "left_iteration": int(left.get("iteration", -1)),
        "right_iteration": int(right.get("iteration", -1)),
        "left_mode": left.get("mode"),
        "right_mode": right.get("mode"),
        "orientation": "left_minus_right",
        "bootstrap_unit": "prompt",
        "metrics": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        load_artifact(args.left),
        load_artifact(args.right),
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
