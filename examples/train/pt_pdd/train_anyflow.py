# SPDX-License-Identifier: Apache-2.0
"""Posterior-Tilted PDD target correction on released AnyFlow-Wan.

FastGen-PDD's training code is not public yet. This runner tests the unique
PT-PDD target correction on NVIDIA's released, competent four-step AnyFlow
model while reusing the audited VPTD runtime for sampler parity, reward loading,
checkpointing, EMA, validation, and W&B.
"""

from __future__ import annotations

from examples.train.pt_pdd.config import (
    PROMPT_PROFILES,
    load_config,
    parse_args,
    prepare_prompt_split,
    validate_config,
)
from examples.train.pt_pdd.core import provenance_table
from examples.train.pt_pdd.objective import train_one_step
from examples.train.vptd import train_anyflow as base


def install_overrides() -> None:
    """Install PT-PDD-specific hooks into the audited VPTD runtime module."""

    base.parse_args = parse_args
    base.load_config = load_config
    base.validate_config = validate_config
    base.prepare_prompt_split = prepare_prompt_split
    base.train_one_step = train_one_step
    base.provenance_table = provenance_table


def main() -> None:
    install_overrides()
    base.main()


__all__ = [
    "PROMPT_PROFILES",
    "install_overrides",
    "load_config",
    "main",
    "parse_args",
    "prepare_prompt_split",
    "train_one_step",
    "validate_config",
]


if __name__ == "__main__":
    main()
