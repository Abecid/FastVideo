# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "H3PeRFlowMethod",
    "H3RESTMethod",
    "KDCausalMethod",
    "KDMethod",
]


def __getattr__(name: str) -> object:
    if name == "H3PeRFlowMethod":
        from fastvideo.train.methods.knowledge_distillation.h3_perflow import (
            H3PeRFlowMethod,
        )
        return H3PeRFlowMethod
    if name == "H3RESTMethod":
        from fastvideo.train.methods.knowledge_distillation.h3_rest import (
            H3RESTMethod,
        )
        return H3RESTMethod
    if name in {"KDCausalMethod", "KDMethod"}:
        from fastvideo.train.methods.knowledge_distillation.kd import (
            KDCausalMethod,
            KDMethod,
        )
        return {
            "KDCausalMethod": KDCausalMethod,
            "KDMethod": KDMethod,
        }[name]
    raise AttributeError(name)
