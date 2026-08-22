# SPDX-License-Identifier: Apache-2.0
"""Final preparation entry point for reliable finite-transition runs.

The base asset module contains the generic split/config machinery. This entry
point fixes the scientific reward contract: expensive online rollouts optimize
only audited VideoAlign MQ, while the source YAML retains audited MQ/VQ/TA for
held-out validation. The sanity recipe remains mean luminance.
"""

from __future__ import annotations

from examples.train import prepare_finite_transition_reliable_assets as _impl


def _training_reward_setup(
    recipe: str,
) -> tuple[dict[str, float], str, str]:
    if recipe == "sanity":
        return {"mean_luminance": 1.0}, "auto", "mean_luminance"
    return (
        {"videoalign_mq_audited": 1.0},
        "genrl",
        "videoalign_mq_audited",
    )


def main() -> None:
    _impl.reward_setup = _training_reward_setup
    _impl.main()


if __name__ == "__main__":
    main()
