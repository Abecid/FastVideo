# SPDX-License-Identifier: Apache-2.0
"""Final scientific entry point for finite-transition v2.

This class adds rank-safe VideoAlign auditing, clean raw/EMA paired metric
namespaces, and a paired aggregate success gate on top of the v2 optimizer and
paired evaluator.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.finite_transition_v2_scientific import (
    FiniteTransitionV2ScientificMethod,
)
from fastvideo.train.methods.rl.rewards.videoalign_audit import (
    audit_videoalign_checkpoint,
    repeatability_probe,
    write_audit_report,
)


class FiniteTransitionV2FinalMethod(FiniteTransitionV2ScientificMethod):
    """Audited v2 optimization with paired raw/EMA validation."""

    def on_train_start(self) -> None:
        # The base v2 class can perform the same audit, but it would let every
        # distributed rank write the same JSON path. Disable that block during
        # parent initialization, then perform the audit once per rank and write
        # the report only from rank zero.
        enabled = self._videoalign_audit_enabled
        self._videoalign_audit_enabled = False
        try:
            super().on_train_start()
        finally:
            self._videoalign_audit_enabled = enabled
        if not enabled:
            return

        checkpoint_path = os.environ.get("VIDEOALIGN_CHECKPOINT_PATH")
        report = audit_videoalign_checkpoint(
            device=self.student.device,
            checkpoint_path=checkpoint_path,
            minimum_checkpoint_numel_coverage=(
                self._videoalign_audit_min_coverage
            ),
            minimum_component_coverage=(
                self._videoalign_audit_min_component
            ),
            require_reward_head=self._videoalign_require_head,
        )
        if self._rank() == 0:
            write_audit_report(
                report,
                Path(self.training_config.checkpoint.output_dir)
                / "videoalign_checkpoint_audit.json",
            )
        overall = report["overall"]
        self._startup_metrics.update(
            {
                "audit/videoalign_checkpoint_numel_coverage": float(
                    overall["numel_ratio"]
                ),
                "audit/videoalign_checkpoint_tensor_coverage": float(
                    overall["tensor_ratio"]
                ),
                "audit/videoalign_unmatched_keys": float(
                    report["unmatched_key_count"]
                ),
            }
        )
        for name, component in report.get("components", {}).items():
            self._startup_metrics[
                f"audit/videoalign_{name}_numel_coverage"
            ] = float(component["numel_ratio"])

        if self._reward_scorer is None:
            raise RuntimeError("reward scorer was not initialized")
        repeat = repeatability_probe(
            self._reward_scorer,
            device=self.student.device,
            tolerance=self._videoalign_repeat_tolerance,
        )
        self._startup_metrics.update(
            {f"audit/videoalign_{key}": float(value) for key, value in repeat.items()}
        )

    @staticmethod
    def _variant_metrics(
        metrics: dict[str, LogScalar],
        variant: str,
    ) -> dict[str, LogScalar]:
        prefixes = (
            "validation_paired_ci95_high/",
            "validation_paired_ci95_low/",
            "validation_paired_delta/",
            "validation_paired_std/",
            "validation_paired_sem/",
            "validation_baseline/",
            "validation_success/",
            "validation_delta/",
            "validation_std/",
            "validation_sem/",
            "validation/",
        )
        result: dict[str, LogScalar] = {}
        for key, value in metrics.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    stem = prefix[:-1]
                    result[f"{stem}_{variant}/{key[len(prefix):]}"] = value
                    break
            else:
                result[f"validation_{variant}_meta/{key}"] = value
        return result

    def on_validation_begin(
        self,
        iteration: int = 0,
    ) -> dict[str, LogScalar]:
        metrics = super().on_validation_begin(iteration)
        if not metrics:
            return metrics

        paired_primary = float(metrics.get("validation_success/primary_paired", 0.0))
        heldout = float(metrics.get("validation_success/heldout_retained", 0.0))
        motion = float(metrics.get("validation_success/motion_retained", 0.0))
        diversity = float(metrics.get("validation_success/diversity_retained", 0.0))
        metrics["validation_success/all_paired"] = float(
            paired_primary > 0.5
            and heldout > 0.5
            and motion > 0.5
            and diversity > 0.5
        )

        raw_primary = float(
            metrics.get("validation_success_raw/primary_paired", 0.0)
        )
        raw_heldout = float(
            metrics.get("validation_success_raw/heldout_retained", 0.0)
        )
        raw_motion = float(
            metrics.get("validation_success_raw/motion_retained", 0.0)
        )
        raw_diversity = float(
            metrics.get("validation_success_raw/diversity_retained", 0.0)
        )
        metrics["validation_success_raw/all_paired"] = float(
            raw_primary > 0.5
            and raw_heldout > 0.5
            and raw_motion > 0.5
            and raw_diversity > 0.5
        )
        return metrics


__all__ = ["FiniteTransitionV2FinalMethod"]
