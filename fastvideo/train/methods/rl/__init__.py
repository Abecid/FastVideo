# SPDX-License-Identifier: Apache-2.0
"""RL training methods."""

from fastvideo.train.methods.rl.dmdr import DMDRMethod
from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod

__all__ = ["DiffusionNFTMethod", "DMDRMethod"]
