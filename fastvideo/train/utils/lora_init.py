# SPDX-License-Identifier: Apache-2.0
"""Strict loading of exported FastVideo LoRA tensors for continued training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from fastvideo.layers.lora.linear import BaseLayerWithLoRA


@dataclass(frozen=True, slots=True)
class TrainingLoraLoadSummary:
    path: str
    sha256: str
    layer_count: int
    tensor_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_parameter_tensor(parameter: torch.nn.Parameter) -> torch.Tensor:
    if isinstance(parameter, DTensor):
        local = getattr(parameter, "_local_tensor", None)
        return parameter.to_local() if local is None else local
    return parameter


@torch.no_grad()
def load_training_lora_weights(
    transformer: torch.nn.Module,
    path: str | Path,
) -> TrainingLoraLoadSummary:
    """Strictly restore an exported FastVideo LoRA without merging it.

    The adapter remains trainable and unmerged, so it can initialize a later
    PeRFlow or RVM stage while the frozen base checkpoint stays unchanged.
    """
    adapter_path = Path(path).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(
            f"Training LoRA adapter is missing at {adapter_path}"
        )

    modules = {
        name: module
        for name, module in transformer.named_modules()
        if isinstance(module, BaseLayerWithLoRA)
    }
    if not modules:
        raise RuntimeError("No FastVideo LoRA layers exist on the transformer")

    expected_keys = {
        f"{name}.{suffix}"
        for name in modules
        for suffix in ("lora_A", "lora_B", "lora_alpha")
    }
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        observed_keys = set(handle.keys())
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        if missing or extra:
            raise ValueError(
                "Training LoRA key mismatch: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )

        for name, module in modules.items():
            if module.merged:
                raise RuntimeError(
                    f"Cannot initialize merged LoRA layer {name!r}"
                )
            if module.lora_A is None or module.lora_B is None:
                raise RuntimeError(
                    f"LoRA layer {name!r} has no trainable A/B parameters"
                )
            source_a = handle.get_tensor(f"{name}.lora_A")
            source_b = handle.get_tensor(f"{name}.lora_B")
            source_alpha = handle.get_tensor(
                f"{name}.lora_alpha"
            ).reshape(-1)
            if source_alpha.numel() != 1:
                raise ValueError(
                    f"LoRA alpha for {name!r} must contain one value"
                )
            alpha = int(source_alpha.item())
            if alpha != int(module.lora_alpha or module.lora_rank or 0):
                raise ValueError(
                    f"LoRA alpha mismatch for {name!r}: "
                    f"adapter={alpha}, configured={module.lora_alpha}"
                )

            target_a = _local_parameter_tensor(module.lora_A)
            target_b = _local_parameter_tensor(module.lora_B)
            if tuple(source_a.shape) != tuple(target_a.shape):
                raise ValueError(
                    f"LoRA A shape mismatch for {name!r}: "
                    f"adapter={tuple(source_a.shape)}, "
                    f"configured={tuple(target_a.shape)}"
                )
            if tuple(source_b.shape) != tuple(target_b.shape):
                raise ValueError(
                    f"LoRA B shape mismatch for {name!r}: "
                    f"adapter={tuple(source_b.shape)}, "
                    f"configured={tuple(target_b.shape)}"
                )
            target_a.copy_(
                source_a.to(
                    device=target_a.device,
                    dtype=target_a.dtype,
                )
            )
            target_b.copy_(
                source_b.to(
                    device=target_b.device,
                    dtype=target_b.dtype,
                )
            )
            module.disable_lora = False

    return TrainingLoraLoadSummary(
        path=str(adapter_path),
        sha256=_sha256_file(adapter_path),
        layer_count=len(modules),
        tensor_count=len(expected_keys),
    )


__all__ = ["TrainingLoraLoadSummary", "load_training_lora_weights"]
