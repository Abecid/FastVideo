# SPDX-License-Identifier: Apache-2.0
"""V2-specific asset preparation wrapper.

The original FTPP asset helper predates audited VideoAlign reward names and
classifies unknown native debug rewards as external DiffusionNFT rewards. This
wrapper preserves its deterministic train/validation split and config writing
while resolving the actual v2 reward backends correctly.
"""

from __future__ import annotations

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
