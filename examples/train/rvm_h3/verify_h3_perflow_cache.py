# SPDX-License-Identifier: Apache-2.0
"""Validate an immutable H3 cache and its deterministic PeRFlow top-q view."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastvideo.dataset.h3_perflow_cache import select_h3_teacher_entries
from fastvideo.dataset.h3_rest_cache import validate_h3_rest_cache


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"{path}:{line_number} must contain one JSON object"
            )
        entries.append(value)
    if not entries:
        raise ValueError(f"{path} contains no trajectory entries")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir")
    parser.add_argument("--selected-per-prompt", type=int, default=2)
    parser.add_argument("--ranking-key", default="mixed_advantage")
    parser.add_argument("--expect-samples-per-prompt", type=int, default=None)
    parser.add_argument("--expect-prompts", type=int, default=None)
    parser.add_argument("--expect-cache-fingerprint", default=None)
    parser.add_argument(
        "--require-reward-name",
        action="append",
        default=[],
        help="Reward component that must be present; repeat as needed.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip payload SHA-256 reads; manifest hashes and byte sizes remain checked.",
    )
    parser.add_argument(
        "--student-timesteps",
        nargs="+",
        type=float,
        default=[1000, 750, 500, 250, 0],
    )
    args = parser.parse_args()

    root = Path(args.cache_dir).expanduser().resolve()
    cache_summary = validate_h3_rest_cache(
        root,
        verify_file_hashes=not args.metadata_only,
        expected_student_timesteps=args.student_timesteps,
    )
    metadata = _read_json(root / "metadata.json")
    manifest = _read_manifest(root / "manifest.jsonl")

    observed_k = metadata.get("samples_per_prompt")
    if args.expect_samples_per_prompt is not None:
        if observed_k != int(args.expect_samples_per_prompt):
            raise ValueError(
                "H3 cache candidate count mismatch: "
                f"cache={observed_k}, expected={args.expect_samples_per_prompt}"
            )
    if args.expect_prompts is not None:
        if cache_summary.num_prompts != int(args.expect_prompts):
            raise ValueError(
                "H3 cache prompt count mismatch: "
                f"cache={cache_summary.num_prompts}, expected={args.expect_prompts}"
            )
    if args.expect_cache_fingerprint is not None:
        expected = str(args.expect_cache_fingerprint).strip().lower()
        if cache_summary.fingerprint.lower() != expected:
            raise ValueError(
                "H3 cache fingerprint mismatch: "
                f"cache={cache_summary.fingerprint}, expected={expected}"
            )

    required_rewards = {
        str(name).strip().lower()
        for name in args.require_reward_name
        if str(name).strip()
    }
    missing_rewards = sorted(required_rewards - set(cache_summary.reward_names))
    if missing_rewards:
        raise ValueError(
            f"H3 cache is missing required reward components: {missing_rewards}"
        )

    selected, selection_summary = select_h3_teacher_entries(
        manifest,
        selected_per_prompt=args.selected_per_prompt,
        ranking_key=args.ranking_key,
        num_segments=cache_summary.num_segments,
    )
    expected_selected = (
        cache_summary.num_prompts * int(args.selected_per_prompt)
    )
    if len(selected) != expected_selected:
        raise RuntimeError(
            "Internal PeRFlow selection count mismatch: "
            f"selected={len(selected)}, expected={expected_selected}"
        )

    print(
        json.dumps(
            {
                "cache": asdict(cache_summary),
                "selection": asdict(selection_summary),
                "verification": {
                    "payload_hashes_checked": not args.metadata_only,
                    "required_reward_names": sorted(required_rewards),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
