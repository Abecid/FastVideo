# SPDX-License-Identifier: Apache-2.0
"""Scientific finite-transition v2 entry point.

Raw and EMA validation dispatch through the exact prompt-seed evaluator. Paired
confidence intervals use prompt-level means, while every individual seed value
is persisted for objective-arm comparisons and debugging.
"""

from __future__ import annotations

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.finite_transition_v2_exact_paired import (
    FiniteTransitionV2ExactPairedMethod,
)


class FiniteTransitionV2ScientificMethod(FiniteTransitionV2ExactPairedMethod):
    """V2 optimization with exact fixed-seed raw/EMA validation."""

    def on_validation_begin(
        self,
        iteration: int = 0,
    ) -> dict[str, LogScalar]:
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}
        metrics: dict[str, LogScalar] = {}
        if self._validate_raw_model:
            with self._validation_state_context("raw"):
                raw = self._run_validation(iteration)
            metrics.update(self._variant_metrics(raw, "raw"))
        if self._validate_ema_model:
            with self._validation_state_context("ema"), self._ema_context():
                ema = self._run_validation(iteration)
            metrics.update(ema)
            metrics.update(self._variant_metrics(ema, "ema"))
        return metrics


__all__ = ["FiniteTransitionV2ScientificMethod"]
