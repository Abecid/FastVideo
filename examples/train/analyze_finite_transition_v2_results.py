# SPDX-License-Identifier: Apache-2.0
"""Analyze paired raw/EMA finite-transition v2 results.

The v2 evaluator writes one JSON file per raw/EMA checkpoint under
``<output_dir>/paired_validation``. This script computes paired
current-vs-baseline intervals, optionally compares two objective runs on exactly
matching prompt indices, and exports compact JSON/CSV summaries. W&B histories
can also be downloaded for reward-vs-step and reward-vs-GPU-hour curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

_FILE_RE = re.compile(r"(?P<variant>raw|ema)_step_(?P<step>\d+)\.json$")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected a row list in {path}")
    return [dict(row) for row in raw]


def load_run(run_dir: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    root = run_dir / "paired_validation"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing paired validation directory: {root}")
    result: dict[str, dict[int, list[dict[str, Any]]]] = {"raw": {}, "ema": {}}
    for path in sorted(root.glob("*_step_*.json")):
        match = _FILE_RE.match(path.name)
        if match is None:
            continue
        result[match.group("variant")][int(match.group("step"))] = _load_rows(path)
    if not result["raw"] and not result["ema"]:
        raise RuntimeError(f"No paired validation files found under {root}")
    return result


def _numeric_metrics(rows: list[dict[str, Any]]) -> list[str]:
    ignored = {"prompt_index", "iteration", "variant"}
    names = set()
    for row in rows:
        for key, value in row.items():
            if key in ignored or key.startswith("delta/"):
                continue
            if isinstance(value, (int, float)):
                names.add(key)
    return sorted(names)


def _bootstrap(values: np.ndarray, *, seed: int, samples: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_run(
    data: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant, by_step in data.items():
        if not by_step:
            continue
        baseline_step = min(by_step)
        baseline_rows = by_step[baseline_step]
        baseline_map = {int(row["prompt_index"]): row for row in baseline_rows}
        for step, rows in sorted(by_step.items()):
            current_map = {int(row["prompt_index"]): row for row in rows}
            indices = sorted(set(baseline_map) & set(current_map))
            if not indices:
                continue
            metrics = _numeric_metrics(rows)
            for metric_offset, metric in enumerate(metrics):
                current = np.asarray(
                    [float(current_map[index][metric]) for index in indices],
                    dtype=np.float64,
                )
                baseline = np.asarray(
                    [float(baseline_map[index][metric]) for index in indices],
                    dtype=np.float64,
                )
                delta = current - baseline
                low, high = _bootstrap(
                    delta,
                    seed=baseline_step * 1009 + step * 97 + metric_offset,
                )
                summaries.append(
                    {
                        "run": label,
                        "variant": variant,
                        "step": int(step),
                        "baseline_step": int(baseline_step),
                        "metric": metric,
                        "num_prompts": len(indices),
                        "mean": float(current.mean()),
                        "baseline_mean": float(baseline.mean()),
                        "paired_delta": float(delta.mean()),
                        "paired_std": float(delta.std(ddof=0)),
                        "paired_sem": float(delta.std(ddof=0) / np.sqrt(len(delta))),
                        "paired_ci95_low": low,
                        "paired_ci95_high": high,
                    }
                )
    return summaries


def compare_runs(
    left: dict[str, dict[int, list[dict[str, Any]]]],
    right: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    left_label: str,
    right_label: str,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for variant in ("raw", "ema"):
        steps = sorted(set(left[variant]) & set(right[variant]))
        for step in steps:
            left_map = {
                int(row["prompt_index"]): row
                for row in left[variant][step]
            }
            right_map = {
                int(row["prompt_index"]): row
                for row in right[variant][step]
            }
            indices = sorted(set(left_map) & set(right_map))
            if not indices:
                continue
            metrics = sorted(
                set(_numeric_metrics(left[variant][step]))
                & set(_numeric_metrics(right[variant][step]))
            )
            for offset, metric in enumerate(metrics):
                delta = np.asarray(
                    [
                        float(left_map[index][metric])
                        - float(right_map[index][metric])
                        for index in indices
                    ],
                    dtype=np.float64,
                )
                low, high = _bootstrap(delta, seed=step * 997 + offset)
                comparisons.append(
                    {
                        "left": left_label,
                        "right": right_label,
                        "variant": variant,
                        "step": int(step),
                        "metric": metric,
                        "num_prompts": len(indices),
                        "paired_difference": float(delta.mean()),
                        "paired_sem": float(delta.std(ddof=0) / np.sqrt(len(delta))),
                        "paired_ci95_low": low,
                        "paired_ci95_high": high,
                    }
                )
    return comparisons


def fetch_wandb_history(run_path: str) -> list[dict[str, Any]]:
    import wandb

    run = wandb.Api().run(run_path)
    keys = [
        "_step",
        "ftv2/post_update_approx_kl",
        "ftv2/loss_scale_after",
        "ftv2/deterministic_preference_alignment",
        "ftv2/reward_mean",
        "ftv2/cumulative_gpu_hours",
        "validation/primary_paired_delta",
        "validation_raw/primary_paired_delta",
        "validation/primary_paired_ci95_low",
        "validation/primary_paired_ci95_high",
    ]
    rows = []
    for row in run.scan_history(keys=keys):
        rows.append({key: row.get(key) for key in keys})
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--wandb-run", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ftv2_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_dir and not args.wandb_run:
        raise ValueError("Provide at least one --run-dir or --wandb-run")
    labels = list(args.label)
    while len(labels) < len(args.run_dir):
        labels.append(args.run_dir[len(labels)].name)

    loaded = [load_run(path.resolve()) for path in args.run_dir]
    summaries = []
    for label, data in zip(labels, loaded):
        summaries.extend(summarize_run(data, label=label))
    comparisons = []
    if len(loaded) == 2:
        comparisons = compare_runs(
            loaded[0],
            loaded[1],
            left_label=labels[0],
            right_label=labels[1],
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(summaries, output / "paired_summary.csv")
    (output / "objective_comparison.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(comparisons, output / "objective_comparison.csv")

    wandb_history = {
        run_path: fetch_wandb_history(run_path)
        for run_path in args.wandb_run
    }
    (output / "wandb_history.json").write_text(
        json.dumps(wandb_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output),
        "summary_rows": len(summaries),
        "comparison_rows": len(comparisons),
        "wandb_runs": list(wandb_history),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
