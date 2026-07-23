# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.knowledge_distillation.kd import (
    KDCausalMethod,
    KDMethod,
)

__all__ = [
    "KDCausalMethod",
    "KDMethod",
    "RTFDMethod",
    "RewardTiltedFlowDistillationMethod",
]


def __getattr__(name: str) -> object:
    if name in {"RTFDMethod", "RewardTiltedFlowDistillationMethod"}:
        from fastvideo.train.methods.knowledge_distillation.reward_tilted_flow import (
            RTFDMethod,
            RewardTiltedFlowDistillationMethod,
        )
        if name == "RTFDMethod":
            return RTFDMethod
        return RewardTiltedFlowDistillationMethod
    raise AttributeError(name)
