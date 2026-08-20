# SPDX-License-Identifier: Apache-2.0
"""RL training methods."""

from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod
from fastvideo.train.methods.rl.finite_transition_posterior import (
    FiniteTransitionPosteriorMethod,
)

__all__ = [
    "DiffusionNFTMethod",
    "FiniteTransitionPosteriorMethod",
]
