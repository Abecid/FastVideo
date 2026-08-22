# SPDX-License-Identifier: Apache-2.0
"""VideoAlign checkpoint auditing and efficient reward wrappers.

This module serves two purposes:

1. audit the final VideoAlign runtime against the checkpoint tensors actually
   present on disk, including adapter and reward-head coverage; and
2. provide batched scorers that preserve the upstream MQ/VQ/TA preprocessing
   contract.

The first finite-transition runs used compatibility key remapping between an
older VideoAlign/Qwen2-VL checkpoint and a newer Transformers runtime. A finite
reward value is not enough evidence that the adapter and reward head loaded, so
all scientific launchers fail fast on incomplete coverage.
"""

from __future__ import annotations

from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.train.methods.rl.rewards import videoalign as _videoalign
from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array

_AUDIT_INSTALLED = False
_AUDIT_REPORTS: list[dict[str, Any]] = []


def _category(key: str) -> str:
    lowered = key.lower()
    if "lora" in lowered or "adapter" in lowered:
        return "adapter"
    if any(token in lowered for token in ("rm_head", "reward_head", "score_head")):
        return "reward_head"
    if "reward" in lowered or lowered.endswith(".head.weight"):
        return "reward_head"
    return "base"


def _wrap_load_state_dict(cls: Any, *, label: str) -> None:
    """Record shape-compatible coverage for compatibility load calls."""
    if getattr(cls, "_fastvideo_coverage_audit", False):
        return
    original = cls.load_state_dict

    def audited_load_state_dict(self, state_dict, strict=True, assign=False):
        remapped = _videoalign._remap_qwen2vl_state_dict_keys(dict(state_dict))
        model_state = self.state_dict()
        totals = {"base": 0, "adapter": 0, "reward_head": 0}
        matched = {"base": 0, "adapter": 0, "reward_head": 0}
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
        result = original(self, remapped, strict=strict, assign=assign)
        _AUDIT_REPORTS.append(
            {
                "label": label,
                "totals": totals,
                "matched": matched,
                "total": sum(totals.values()),
                "matched_total": sum(matched.values()),
                "mismatched_sample": mismatched[:32],
                "missing_keys": list(getattr(result, "missing_keys", []))[:64],
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


def _latest_checkpoint_dir(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for candidate in root.glob("checkpoint-*"):
        try:
            step = int(candidate.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, candidate))
    return max(candidates, default=(-1, root), key=lambda item: item[0])[1]


def _torch_load_mapping(path: Path) -> dict[str, torch.Tensor]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise RuntimeError(f"Checkpoint {path} is not a state-dict mapping")
    return {
        str(key): value
        for key, value in raw.items()
        if torch.is_tensor(value)
    }


def _insert_adapter_name(
    state_dict: dict[str, torch.Tensor],
    *,
    adapter_name: str = "default",
    parameter_prefix: str = "lora_",
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix, 1)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(
                    suffix_to_replace,
                    f"{adapter_name}.{suffix_to_replace}",
                )
            else:
                key = f"{key}.{adapter_name}"
        result[key] = value
    return result


def _load_checkpoint_tensors(root: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    checkpoint = _latest_checkpoint_dir(root)
    full_path = checkpoint / "model.pth"
    if full_path.is_file():
        state = _torch_load_mapping(full_path)
    else:
        adapter_path = checkpoint / "adapter_model.safetensors"
        non_lora_path = checkpoint / "non_lora_state_dict.pth"
        if not adapter_path.is_file() or not non_lora_path.is_file():
            raise FileNotFoundError(
                "VideoAlign checkpoint must contain model.pth or both "
                f"adapter_model.safetensors and non_lora_state_dict.pth: {checkpoint}"
            )
        from safetensors.torch import load_file

        adapter = _insert_adapter_name(load_file(str(adapter_path), device="cpu"))
        non_lora = _torch_load_mapping(non_lora_path)
        state = {**non_lora, **adapter}
    return checkpoint, _videoalign._remap_qwen2vl_state_dict_keys(state)


def _normalized_key(key: str) -> str:
    normalized = str(key)
    for prefix in ("module.", "base_model.model."):
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def _coverage_report(
    model: Any,
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model_state = model.state_dict()
    normalized_model: dict[str, list[str]] = {}
    for model_key in model_state:
        normalized_model.setdefault(_normalized_key(model_key), []).append(model_key)

    component_totals = {
        name: {"tensor_total": 0, "tensor_matched": 0, "numel_total": 0, "numel_matched": 0}
        for name in ("base", "adapter", "reward_head")
    }
    unmatched: list[str] = []
    matched_keys: dict[str, str] = {}

    for checkpoint_key, value in checkpoint_state.items():
        component = _category(checkpoint_key)
        stats = component_totals[component]
        stats["tensor_total"] += 1
        stats["numel_total"] += int(value.numel())

        candidates = []
        if checkpoint_key in model_state:
            candidates.append(checkpoint_key)
        candidates.extend(normalized_model.get(_normalized_key(checkpoint_key), []))
        selected = next(
            (
                key
                for key in dict.fromkeys(candidates)
                if tuple(model_state[key].shape) == tuple(value.shape)
            ),
            None,
        )
        if selected is None:
            unmatched.append(checkpoint_key)
            continue
        stats["tensor_matched"] += 1
        stats["numel_matched"] += int(value.numel())
        matched_keys[checkpoint_key] = selected

    def finalize(raw: dict[str, int]) -> dict[str, float | int]:
        tensor_total = int(raw["tensor_total"])
        numel_total = int(raw["numel_total"])
        return {
            **raw,
            "tensor_ratio": (
                float(raw["tensor_matched"]) / float(tensor_total)
                if tensor_total
                else float("nan")
            ),
            "numel_ratio": (
                float(raw["numel_matched"]) / float(numel_total)
                if numel_total
                else float("nan")
            ),
        }

    components = {
        name: finalize(values)
        for name, values in component_totals.items()
    }
    overall_raw = {
        key: sum(int(component[key]) for component in component_totals.values())
        for key in ("tensor_total", "tensor_matched", "numel_total", "numel_matched")
    }
    return {
        "overall": finalize(overall_raw),
        "components": components,
        "unmatched_key_count": len(unmatched),
        "unmatched_keys_sample": unmatched[:128],
        "matched_key_count": len(matched_keys),
    }


def audit_videoalign_checkpoint(
    *,
    device: torch.device | str,
    checkpoint_path: str | os.PathLike[str] | None = None,
    minimum_checkpoint_numel_coverage: float = 0.97,
    minimum_component_coverage: float = 0.95,
    require_reward_head: bool = True,
) -> dict[str, Any]:
    """Audit the tensors actually loaded by the final VideoAlign runtime."""
    install_videoalign_coverage_audit()
    root = Path(
        checkpoint_path
        or os.environ.get("VIDEOALIGN_CHECKPOINT_PATH", "")
    ).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"VideoAlign checkpoint path does not exist: {root}")

    inferencer = _videoalign._get_inferencer(device, str(root))
    checkpoint_dir, checkpoint_state = _load_checkpoint_tensors(root)
    report = _coverage_report(inferencer.model, checkpoint_state)
    report.update(
        {
            "checkpoint_root": str(root),
            "checkpoint_dir": str(checkpoint_dir),
            "runtime_model_class": type(inferencer.model).__name__,
        }
    )

    overall = report["overall"]
    if float(overall["numel_ratio"]) < float(minimum_checkpoint_numel_coverage):
        raise RuntimeError(
            "VideoAlign checkpoint numel coverage is too low: "
            f"{overall['numel_ratio']:.6f} < {minimum_checkpoint_numel_coverage:.6f}"
        )
    for name, component in report["components"].items():
        if int(component["numel_total"]) == 0:
            continue
        if float(component["numel_ratio"]) < float(minimum_component_coverage):
            raise RuntimeError(
                f"VideoAlign {name} coverage is too low: "
                f"{component['numel_ratio']:.6f} < {minimum_component_coverage:.6f}"
            )
    reward_head = report["components"]["reward_head"]
    if require_reward_head and int(reward_head["numel_total"]) <= 0:
        raise RuntimeError("VideoAlign checkpoint contains no detected reward-head tensors")

    nonfinite: list[str] = []
    sampled_base = 0
    for name, parameter in inferencer.model.named_parameters():
        category = _category(name)
        should_check = category in {"adapter", "reward_head"} or sampled_base < 8
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
    return report


def write_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_score_dict(result: Any) -> dict[str, torch.Tensor]:
    if torch.is_tensor(result):
        return {"score": result.detach().float().reshape(-1)}
    if isinstance(result, dict):
        return {
            str(key): torch.as_tensor(value).detach().float().reshape(-1)
            for key, value in result.items()
            if torch.is_tensor(value) or isinstance(value, (float, int, list, tuple))
        }
    raise TypeError(f"Unsupported reward output type: {type(result).__name__}")


def repeatability_probe(
    scorer: Any,
    *,
    device: torch.device | str,
    tolerance: float = 1.0e-6,
) -> dict[str, float]:
    """Run the same deterministic synthetic video twice and compare outputs."""
    frames = 16
    height = width = 224
    media = torch.zeros((1, 3, frames, height, width), dtype=torch.float32)
    size = 32
    for index in range(frames):
        x = int((width - size) * index / max(frames - 1, 1))
        y = (height - size) // 2
        media[:, :, index, y:y + size, x:x + size] = 1.0
    media = media.to(device)
    prompts = ["A white square moves smoothly from left to right."]
    first = _as_score_dict(scorer(media, prompts))
    second = _as_score_dict(scorer(media, prompts))
    common = sorted(set(first) & set(second))
    if not common:
        raise RuntimeError("VideoAlign repeatability probe produced no common scores")
    deltas = []
    magnitudes = []
    for key in common:
        if first[key].shape != second[key].shape:
            raise RuntimeError(f"VideoAlign repeatability shape changed for {key}")
        if not torch.isfinite(first[key]).all() or not torch.isfinite(second[key]).all():
            raise RuntimeError(f"VideoAlign returned non-finite values for {key}")
        deltas.append((first[key] - second[key]).abs().max())
        magnitudes.append(torch.maximum(first[key].abs().max(), second[key].abs().max()))
    maximum_delta = float(torch.stack(deltas).max())
    if maximum_delta > float(tolerance):
        raise RuntimeError(
            "VideoAlign repeatability drift exceeded tolerance: "
            f"{maximum_delta:.8g} > {tolerance:.8g}"
        )
    return {
        "repeat_delta_max": maximum_delta,
        "repeat_score_abs_max": float(torch.stack(magnitudes).max()),
        "repeat_score_count": float(len(common)),
    }


def videoalign_coverage_summary() -> dict[str, Any]:
    aggregate = {
        "base": {"total": 0, "matched": 0},
        "adapter": {"total": 0, "matched": 0},
        "reward_head": {"total": 0, "matched": 0},
    }
    for report in _AUDIT_REPORTS:
        for category in aggregate:
            aggregate[category]["total"] += int(report["totals"][category])
            aggregate[category]["matched"] += int(report["matched"][category])
    for values in aggregate.values():
        total = int(values["total"])
        values["coverage"] = (
            float(values["matched"]) / float(total)
            if total > 0
            else float("nan")
        )
    best_overall = max(
        (
            float(report["matched_total"]) / max(float(report["total"]), 1.0)
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
            os.environ.get("VIDEOALIGN_MIN_OVERALL_COVERAGE", "0.90")
        )
    if minimum_head is None:
        minimum_head = float(
            os.environ.get("VIDEOALIGN_MIN_HEAD_COVERAGE", "0.99")
        )
    summary = videoalign_coverage_summary()
    if not summary["reports"]:
        direct = audit_videoalign_checkpoint(
            device=next(model.parameters()).device,
            minimum_checkpoint_numel_coverage=minimum_overall,
            minimum_component_coverage=minimum_head,
            require_reward_head=True,
        )
        components = {
            name: {
                "total": int(value["tensor_total"]),
                "matched": int(value["tensor_matched"]),
                "coverage": float(value["tensor_ratio"]),
            }
            for name, value in direct["components"].items()
        }
        return {
            "reports": [],
            "aggregate": components,
            "best_overall_coverage": float(direct["overall"]["numel_ratio"]),
            "direct": direct,
        }
    if float(summary["best_overall_coverage"]) < float(minimum_overall):
        raise RuntimeError(
            "VideoAlign compatibility-load coverage is too low: "
            f"{summary['best_overall_coverage']:.4f} < {minimum_overall:.4f}"
        )
    head = summary["aggregate"]["reward_head"]
    if int(head["total"]) <= 0:
        raise RuntimeError("VideoAlign compatibility audit found no reward head")
    if float(head["coverage"]) < float(minimum_head):
        raise RuntimeError(
            "VideoAlign reward-head coverage is too low: "
            f"{head['coverage']:.4f} < {minimum_head:.4f}"
        )
    return summary


class _AuditedScorerMixin:
    _coverage_checked = False

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        install_videoalign_coverage_audit()
        inferencer = _videoalign._get_inferencer(self.device, self.checkpoint_path)
        images_np = media_to_uint8_array(media)
        paths: list[str] = []
        reward_prompts: list[str] = []
        try:
            for sample_index, sample in enumerate(images_np):
                frames = sample[None] if sample.ndim == 3 else sample
                paths.append(_videoalign._save_video_to_temp(self._frames(frames)))
                reward_prompts.append(self._prompt(prompts, sample_index))
            results = inferencer.reward(paths, reward_prompts, use_norm=True)
            scores = [float(result.get(self.score_key, 0.0)) for result in results]
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        if not self._coverage_checked:
            assert_videoalign_checkpoint_coverage(inferencer.model)
            self._coverage_checked = True
        return torch.tensor(scores, device=self.device, dtype=torch.float32)


class AuditedVideoAlignMotionQualityScorer(
    _AuditedScorerMixin,
    _videoalign.VideoAlignMotionQualityScorer,
):
    def _prompt(self, prompts, index: int) -> str:
        del prompts, index
        return ""

    def _frames(self, frames: np.ndarray) -> np.ndarray:
        # Match the upstream FastVideo/GenRL VideoAlign wrapper exactly.
        if frames.ndim == 4 and frames.shape[-1] == 3:
            gray = np.mean(frames, axis=-1, keepdims=True)
            return np.repeat(gray.astype(np.uint8), 3, axis=-1)
        return frames


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
    "audit_videoalign_checkpoint",
    "install_videoalign_coverage_audit",
    "repeatability_probe",
    "videoalign_coverage_summary",
    "write_audit_report",
]
