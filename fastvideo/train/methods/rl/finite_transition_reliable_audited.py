# SPDX-License-Identifier: Apache-2.0
"""Final reliable finite-transition entry point.

Training can optimize one reward head while held-out validation scores a broader
set. This avoids running three Qwen reward passes for every rollout merely to log
VQ/TA diagnostics, while preserving audited multi-head evaluation.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.finite_transition_reliable_calibrated import (
    CalibratedReliableFiniteTransitionMethod,
)
from fastvideo.train.methods.rl.rewards import (
    build_multi_reward_scorer,
    normalize_reward_weights,
)


class AuditedReliableFiniteTransitionMethod(
    CalibratedReliableFiniteTransitionMethod
):
    """Calibrated method with separate training and validation reward stacks."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config
        raw_validation_reward = mcfg.get("validation_reward_fn")
        if raw_validation_reward is None:
            self._validation_reward_fn_config = dict(self._reward_fn_config)
            self._validation_reward_backend = self._reward_backend
        else:
            (
                self._validation_reward_fn_config,
                inline_backend,
            ) = normalize_reward_weights(raw_validation_reward)
            self._validation_reward_backend = str(
                mcfg.get(
                    "validation_reward_backend",
                    inline_backend or self._reward_backend,
                )
                or self._reward_backend
            ).strip().lower()
        self._validation_reward_scorer: Any | None = None

    def on_train_start(self) -> None:
        super().on_train_start()
        if (
            self._validation_reward_fn_config == self._reward_fn_config
            and self._validation_reward_backend == self._reward_backend
        ):
            self._validation_reward_scorer = self._reward_scorer
        else:
            self._validation_reward_scorer = build_multi_reward_scorer(
                self._validation_reward_fn_config,
                backend=self._validation_reward_backend,
                device=self.student.device,
            )

    def _score_validation_media(
        self,
        media: torch.Tensor,
        prompts: list[str],
    ) -> dict[str, torch.Tensor]:
        scorer = self._validation_reward_scorer or self._reward_scorer
        if scorer is None:
            raise RuntimeError("validation reward scorer is not initialized")
        rewards = scorer(media, prompts)
        return {
            key: value.to(
                device=self.student.device,
                dtype=torch.float32,
            ).reshape(-1)
            for key, value in rewards.items()
        }

    def on_validation_begin(
        self,
        iteration: int = 0,
    ) -> dict[str, LogScalar]:
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}

        original_scorer = self._reward_scorer
        if self._validation_reward_scorer is not None:
            self._reward_scorer = self._validation_reward_scorer
        try:
            metrics = super().on_validation_begin(iteration)
        finally:
            self._reward_scorer = original_scorer

        if any(
            name.startswith("videoalign_")
            for name in self._validation_reward_fn_config
        ):
            from fastvideo.train.methods.rl.rewards.videoalign_audit import (
                videoalign_coverage_summary,
            )

            summary = videoalign_coverage_summary()
            metrics["reward_audit/best_overall_coverage"] = float(
                summary["best_overall_coverage"]
            )
            for category, values in summary["aggregate"].items():
                coverage = values.get("coverage", float("nan"))
                metrics[f"reward_audit/{category}_coverage"] = float(coverage)
                metrics[f"reward_audit/{category}_matched"] = float(
                    values.get("matched", 0)
                )
                metrics[f"reward_audit/{category}_total"] = float(
                    values.get("total", 0)
                )
        return metrics


__all__ = ["AuditedReliableFiniteTransitionMethod"]
