# SPDX-License-Identifier: Apache-2.0
"""Compare two finite-transition validation artifacts with paired statistics.

Each input is a JSON file produced under
``<output_dir>/paired_validation/step_XXXXXX_{raw,ema}.json``. The tool verifies
that prompt/seed identities match before computing left-minus-right bootstrap
confidence intervals for every shared scalar metric.
"""

from __future__ import annotations

import argparse
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

    summaries: dict[str, dict[str, float]] = {}
    for name in sorted(set(left_metrics) & set(right_metrics)):
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
        summaries[name] = paired_summary(
            torch.tensor(left_values),
            torch.tensor(right_values),
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )

    return {
        "left_iteration": int(left.get("iteration", -1)),
        "right_iteration": int(right.get("iteration", -1)),
        "left_mode": left.get("mode"),
        "right_mode": right.get("mode"),
        "orientation": "left_minus_right",
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
