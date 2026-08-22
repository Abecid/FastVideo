# SPDX-License-Identifier: Apache-2.0
"""Compare audited FastVideo VideoAlign scores with an upstream reference.

Create the reference once in an upstream-supported VideoAlign environment using
the same videos and prompts. This script validates absolute score drift and rank
correlation for the exact MQ/VQ/TA wrappers used by the v2 experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.train.methods.rl.rewards.videoalign_audit import (
    AuditedVideoAlignMotionQualityScorer,
    AuditedVideoAlignTextAlignmentScorer,
    AuditedVideoAlignVisualQualityScorer,
)


def _read_video(path: Path) -> torch.Tensor:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(3, 0, 1, 2)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--max-abs-error", type=float, default=1.0e-4)
    parser.add_argument("--min-rank-correlation", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_path is not None:
        import os

        os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = str(
            args.checkpoint_path.expanduser().resolve()
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a non-empty JSON list")

    scorers = {
        "MQ": AuditedVideoAlignMotionQualityScorer(device=args.device),
        "VQ": AuditedVideoAlignVisualQualityScorer(device=args.device),
        "TA": AuditedVideoAlignTextAlignmentScorer(device=args.device),
    }
    rows = []
    for index, item in enumerate(manifest):
        video_path = Path(item["video"]).expanduser().resolve()
        prompt = str(item.get("prompt", ""))
        media = _read_video(video_path).unsqueeze(0)
        row: dict[str, Any] = {
            "index": int(index),
            "video": str(video_path),
            "prompt": prompt,
        }
        for name, scorer in scorers.items():
            row[name] = float(scorer(media, [prompt]).item())
        rows.append(row)

    report: dict[str, Any] = {"current": rows}
    failures = []
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        reference_rows = reference.get("current", reference)
        if not isinstance(reference_rows, list) or len(reference_rows) != len(rows):
            raise ValueError("reference row count does not match manifest")
        comparisons = {}
        for name in scorers:
            current_values = np.asarray(
                [row[name] for row in rows],
                dtype=np.float64,
            )
            reference_values = np.asarray(
                [float(row[name]) for row in reference_rows],
                dtype=np.float64,
            )
            max_error = float(
                np.max(np.abs(current_values - reference_values))
            )
            rank_corr = _pearson(
                _rank(current_values),
                _rank(reference_values),
            )
            comparisons[name] = {
                "max_abs_error": max_error,
                "rank_correlation": rank_corr,
            }
            if max_error > args.max_abs_error:
                failures.append(
                    f"{name} max abs error {max_error:.6g} > "
                    f"{args.max_abs_error:.6g}"
                )
            if rank_corr < args.min_rank_correlation:
                failures.append(
                    f"{name} rank correlation {rank_corr:.6g} < "
                    f"{args.min_rank_correlation:.6g}"
                )
        report["comparison"] = comparisons

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(
            "VideoAlign calibration parity failed: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
