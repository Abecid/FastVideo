# SPDX-License-Identifier: Apache-2.0
"""Audited and batched VideoAlign reward wrappers.

The first finite-transition runs relied on compatibility key remapping between
VideoAlign's Qwen2-VL checkpoint and a newer Transformers runtime. These wrappers
record shape-compatible checkpoint coverage, fail before training if no reward
head is demonstrably loaded, and preserve upstream reward preprocessing:
Motion Quality and Visual Quality use an empty text prompt, while Text Alignment
uses the actual generation prompt.
"""

from __future__ import annotations

from importlib import import_module
import os
from typing import Any

import torch

from fastvideo.train.methods.rl.rewards import videoalign as _videoalign
from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array

_AUDIT_INSTALLED = False
_AUDIT_REPORTS: list[dict[str, Any]] = []


def _category(key: str) -> str:
    lowered = key.lower()
    if "lora" in lowered or "adapter" in lowered:
        return "adapter"
    if any(
        token in lowered
        for token in ("reward", "score", "head", "rm_head")
    ):
        return "head"
    return "base"


def _wrap_load_state_dict(cls: Any, *, label: str) -> None:
    if getattr(cls, "_fastvideo_coverage_audit", False):
        return
    original = cls.load_state_dict

    def audited_load_state_dict(
        self,
        state_dict,
        strict=True,
        assign=False,
    ):
        remapped = _videoalign._remap_qwen2vl_state_dict_keys(
            dict(state_dict)
        )
        model_state = self.state_dict()
        totals = {"base": 0, "adapter": 0, "head": 0}
        matched = {"base": 0, "adapter": 0, "head": 0}
        mismatched: list[str] = []
        for key, value in remapped.items():
            category = _category(key)
            totals[category] += 1
            target = model_state.get(key)
            if (
                target is not None
                and hasattr(value, "shape")
                and tuple(target.shape) == tuple(value.shape)
            ):
                matched[category] += 1
            else:
                mismatched.append(key)
        result = original(
            self,
            remapped,
            strict=strict,
            assign=assign,
        )
        _AUDIT_REPORTS.append(
            {
                "label": label,
                "totals": totals,
                "matched": matched,
                "total": sum(totals.values()),
                "matched_total": sum(matched.values()),
                "mismatched_sample": mismatched[:32],
                "missing_keys": list(
                    getattr(result, "missing_keys", [])
                )[:64],
                "unexpected_keys": list(
                    getattr(result, "unexpected_keys", [])
                )[:64],
            }
        )
        return result

    cls.load_state_dict = audited_load_state_dict
    cls._fastvideo_coverage_audit = True


def install_videoalign_coverage_audit() -> None:
    global _AUDIT_INSTALLED
    if _AUDIT_INSTALLED:
        return
    _videoalign._patch_videoalign_modules()
    reward_model_mod = import_module("reward_model")
    _wrap_load_state_dict(
        reward_model_mod.Qwen2VLRewardModelBT,
        label="reward_model",
    )
    try:
        peft_mod = import_module("peft")
    except ImportError:
        peft_mod = None
    if peft_mod is not None:
        _wrap_load_state_dict(peft_mod.PeftModel, label="peft")
    _AUDIT_INSTALLED = True


def videoalign_coverage_summary() -> dict[str, Any]:
    aggregate = {
        "base": {"total": 0, "matched": 0},
        "adapter": {"total": 0, "matched": 0},
        "head": {"total": 0, "matched": 0},
    }
    for report in _AUDIT_REPORTS:
        for category in aggregate:
            aggregate[category]["total"] += int(
                report["totals"][category]
            )
            aggregate[category]["matched"] += int(
                report["matched"][category]
            )
    for values in aggregate.values():
        total = int(values["total"])
        values["coverage"] = (
            float(values["matched"]) / float(total)
            if total > 0
            else float("nan")
        )
    best_overall = max(
        (
            float(report["matched_total"])
            / max(float(report["total"]), 1.0)
            for report in _AUDIT_REPORTS
            if int(report["total"]) > 0
        ),
        default=0.0,
    )
    return {
        "reports": list(_AUDIT_REPORTS),
        "aggregate": aggregate,
        "best_overall_coverage": best_overall,
    }


def assert_videoalign_checkpoint_coverage(
    model: Any,
    *,
    minimum_overall: float | None = None,
    minimum_head: float | None = None,
) -> dict[str, Any]:
    """Fail if compatibility remapping did not load a real reward head."""
    if minimum_overall is None:
        minimum_overall = float(
            os.environ.get(
                "VIDEOALIGN_MIN_OVERALL_COVERAGE",
                "0.90",
            )
        )
    if minimum_head is None:
        minimum_head = float(
            os.environ.get("VIDEOALIGN_MIN_HEAD_COVERAGE", "0.99")
        )
    summary = videoalign_coverage_summary()
    if not summary["reports"]:
        raise RuntimeError(
            "VideoAlign checkpoint coverage audit saw no state-dict loads"
        )
    if float(summary["best_overall_coverage"]) < float(minimum_overall):
        raise RuntimeError(
            "VideoAlign checkpoint coverage is too low: "
            f"{summary['best_overall_coverage']:.4f} < "
            f"{minimum_overall:.4f}"
        )

    head = summary["aggregate"]["head"]
    if int(head["total"]) <= 0:
        raise RuntimeError(
            "VideoAlign checkpoint audit found no reward-head parameters"
        )
    if float(head["coverage"]) < float(minimum_head):
        raise RuntimeError(
            "VideoAlign reward-head coverage is too low: "
            f"{head['coverage']:.4f} < {minimum_head:.4f}"
        )

    nonfinite = []
    sampled_base = 0
    for name, parameter in model.named_parameters():
        category = _category(name)
        should_check = category in {"adapter", "head"} or sampled_base < 8
        if not should_check:
            continue
        if category == "base":
            sampled_base += 1
        if not torch.isfinite(parameter.detach()).all():
            nonfinite.append(name)
            if len(nonfinite) >= 16:
                break
    if nonfinite:
        raise RuntimeError(
            "VideoAlign contains non-finite audited parameters after load: "
            f"{nonfinite}"
        )
    return summary


class _AuditedScorerMixin:
    _coverage_checked = False

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        install_videoalign_coverage_audit()
        inferencer = _videoalign._get_inferencer(
            self.device,
            self.checkpoint_path,
        )
        images_np = media_to_uint8_array(media)
        paths: list[str] = []
        reward_prompts: list[str] = []
        try:
            for sample_index, sample in enumerate(images_np):
                frames = sample[None] if sample.ndim == 3 else sample
                paths.append(
                    _videoalign._save_video_to_temp(self._frames(frames))
                )
                reward_prompts.append(self._prompt(prompts, sample_index))
            # VideoAlign's inferencer supports a real batch. The old wrapper
            # invoked Qwen separately for every video, which made larger rollout
            # groups unnecessarily expensive.
            results = inferencer.reward(
                paths,
                reward_prompts,
                use_norm=True,
            )
            scores = [
                float(result.get(self.score_key, 0.0))
                for result in results
            ]
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

        if not self._coverage_checked:
            assert_videoalign_checkpoint_coverage(inferencer.model)
            self._coverage_checked = True
        return torch.tensor(
            scores,
            device=self.device,
            dtype=torch.float32,
        )


class AuditedVideoAlignMotionQualityScorer(
    _AuditedScorerMixin,
    _videoalign.VideoAlignMotionQualityScorer,
):
    def _prompt(self, prompts, index: int) -> str:
        del prompts, index
        return ""


class AuditedVideoAlignVisualQualityScorer(
    _AuditedScorerMixin,
    _videoalign.VideoAlignVisualQualityScorer,
):
    def _prompt(self, prompts, index: int) -> str:
        del prompts, index
        return ""


class AuditedVideoAlignTextAlignmentScorer(
    _AuditedScorerMixin,
    _videoalign.VideoAlignTextAlignmentScorer,
):
    pass


__all__ = [
    "AuditedVideoAlignMotionQualityScorer",
    "AuditedVideoAlignTextAlignmentScorer",
    "AuditedVideoAlignVisualQualityScorer",
    "assert_videoalign_checkpoint_coverage",
    "install_videoalign_coverage_audit",
    "videoalign_coverage_summary",
]
