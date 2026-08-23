# SPDX-License-Identifier: Apache-2.0
"""V2-specific asset preparation wrapper.

The original FTPP asset helper predates audited VideoAlign reward names and
classifies unknown native debug rewards as external DiffusionNFT rewards. This
wrapper preserves its deterministic train/validation split and config writing
while resolving the actual v2 reward backends correctly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from examples.train import prepare_finite_transition_posterior_assets as _base

_AUDITED_VIDEOALIGN = {
    "videoalign_mq_audited",
    "videoalign_vq_audited",
    "videoalign_ta_audited",
}
_NATIVE_DEBUG = {
    "mean_luminance",
    "jpeg_compressibility",
    "jpeg_incompressibility",
}

_ATOMIC_METHOD_FIELDS = (
    "reward_backend",
    "reward_fn",
    "optimize_reward",
    "validation_reward_backend",
    "validation_reward_fn",
    "videoalign_audit",
)


def merge_generated_with_preset(
    generated: dict[str, Any],
    preset: dict[str, Any],
) -> dict[str, Any]:
    """Merge prepared paths with a v2 preset without combining scorer maps."""

    def deep_merge(left: Any, right: Any) -> Any:
        if isinstance(left, dict) and isinstance(right, dict):
            result = deepcopy(left)
            for key, value in right.items():
                result[key] = deep_merge(result.get(key), value)
            return result
        return deepcopy(right)

    merged = deep_merge(generated, preset)
    preset_method = preset.get("method", {})
    merged_method = merged.setdefault("method", {})
    for key in _ATOMIC_METHOD_FIELDS:
        if key in preset_method:
            merged_method[key] = deepcopy(preset_method[key])
    return merged


def resolve_reward_setup(
    reward: str,
) -> tuple[dict[str, float], str, str]:
    normalized = str(reward).strip().lower()
    if normalized in _AUDITED_VIDEOALIGN:
        return {normalized: 1.0}, "genrl", normalized
    if normalized in _NATIVE_DEBUG:
        return {normalized: 1.0}, "auto", normalized
    return _base.resolve_reward_setup(normalized)


def main() -> None:
    _base.resolve_reward_setup = resolve_reward_setup
    _base.main()


if __name__ == "__main__":
    main()
