# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.train.utils.lora_init import load_training_lora_weights


class _ToyLoRA(torch.nn.Module):
    def __init__(self, *, alpha: int = 4) -> None:
        super().__init__()
        self.block = BaseLayerWithLoRA(
            torch.nn.Linear(3, 2, bias=False),
            lora_rank=2,
            lora_alpha=alpha,
            training_mode=True,
        )


def _write_adapter(
    path: Path,
    *,
    alpha: int = 4,
    extra: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    tensors = {
        "block.lora_A": tensor_a,
        "block.lora_B": tensor_b,
        "block.lora_alpha": torch.tensor(alpha, dtype=torch.int64),
    }
    if extra:
        tensors["unexpected"] = torch.zeros(1)
    save_file(tensors, str(path))
    return tensor_a, tensor_b


def test_training_lora_load_is_strict_unmerged_and_trainable(
    tmp_path: Path,
) -> None:
    model = _ToyLoRA()
    path = tmp_path / "adapter.safetensors"
    expected_a, expected_b = _write_adapter(path)

    summary = load_training_lora_weights(model, path)

    torch.testing.assert_close(model.block.lora_A, expected_a)
    torch.testing.assert_close(model.block.lora_B, expected_b)
    assert model.block.lora_A.requires_grad
    assert model.block.lora_B.requires_grad
    assert not model.block.merged
    assert not model.block.disable_lora
    assert summary.layer_count == 1
    assert summary.tensor_count == 3
    assert len(summary.sha256) == 64


def test_training_lora_load_rejects_key_and_alpha_mismatch(
    tmp_path: Path,
) -> None:
    model = _ToyLoRA()

    extra_path = tmp_path / "extra.safetensors"
    _write_adapter(extra_path, extra=True)
    with pytest.raises(ValueError, match="key mismatch"):
        load_training_lora_weights(model, extra_path)

    alpha_path = tmp_path / "alpha.safetensors"
    _write_adapter(alpha_path, alpha=8)
    with pytest.raises(ValueError, match="alpha mismatch"):
        load_training_lora_weights(model, alpha_path)


def test_training_lora_load_rejects_shape_mismatch(tmp_path: Path) -> None:
    model = _ToyLoRA()
    path = tmp_path / "shape.safetensors"
    save_file(
        {
            "block.lora_A": torch.zeros(3, 3),
            "block.lora_B": torch.zeros(2, 2),
            "block.lora_alpha": torch.tensor(4, dtype=torch.int64),
        },
        str(path),
    )
    with pytest.raises(ValueError, match="A shape mismatch"):
        load_training_lora_weights(model, path)
