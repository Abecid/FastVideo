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
    "RTRFDMethod",
    "RewardTiltedReflowDistillationMethod",
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
    if name in {"RTRFDMethod", "RewardTiltedReflowDistillationMethod"}:
        from fastvideo.train.methods.knowledge_distillation.reward_tilted_reflow import (
            RTRFDMethod,
            RewardTiltedReflowDistillationMethod,
        )
        if name == "RTRFDMethod":
            return RTRFDMethod
        return RewardTiltedReflowDistillationMethod
    raise AttributeError(name)
