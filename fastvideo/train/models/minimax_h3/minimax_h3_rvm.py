# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 model adapter for LoRA-based reward post-training."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import torch

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn_h3 import (
    MiniMaxH3VSAMetadataBuilder,
)
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
    _h3_vsa_prefix_segments,
)
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import MiniMaxH3DMDModel
from fastvideo.train.utils.lora_init import (
    TrainingLoraLoadSummary,
    load_training_lora_weights,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig

logger = init_logger(__name__)


class MiniMaxH3RVMModel(MiniMaxH3DMDModel):
    """Packed H3 adapter with a separately trainable quality LoRA."""

    def __init__(
        self,
        *,
        lora: LoraConfig | dict[str, Any] | None,
        lora_init_from: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        from fastvideo.train.utils.lora import LoraConfig

        self._lora_config = LoraConfig.coerce(lora)
        if self._lora_config is None or not self._lora_config.enable:
            raise ValueError(
                "MiniMaxH3RVMModel requires models.student.lora.enable=true"
            )
        if not self._enable_lora_if_configured(self.transformer):
            raise RuntimeError("Failed to enable the configured H3 quality LoRA")

        # The frozen 35B base can remain BF16, but optimizer updates of a small
        # LoRA at 1e-5 are easily rounded away in BF16 parameter storage. Keep
        # trainable adapter masters in FP32 before loading a previous stage's
        # exported FP32 tensors; loading into BF16 first would irreversibly
        # quantize the continuation checkpoint.
        for parameter in self.transformer.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)

        self.lora_init_summary: TrainingLoraLoadSummary | None = None
        if lora_init_from is not None and str(lora_init_from).strip():
            self.lora_init_summary = load_training_lora_weights(
                self.transformer,
                str(lora_init_from),
            )
            logger.info(
                "Initialized trainable H3 LoRA from %s (sha256=%s, layers=%d)",
                self.lora_init_summary.path,
                self.lora_init_summary.sha256,
                self.lora_init_summary.layer_count,
            )

    def refresh_vsa_metadata(
        self,
        batch: TrainingBatch,
        *,
        current_timestep: int,
    ) -> None:
        """Build the VSA-H3 mask for the actual four-step rollout index."""
        backend = self.attention_backend_name or envs.FASTVIDEO_ATTENTION_BACKEND
        if backend != "VIDEO_SPARSE_ATTN_H3":
            batch.attn_metadata_vsa = None
            return
        layout = batch.minimax_h3_layout
        if layout is None:
            raise RuntimeError(
                "prepare_batch() must set minimax_h3_layout before VSA metadata"
            )
        patch_size = tuple(self.transformer.patch_size)
        builder = getattr(self, "_rvm_vsa_metadata_builder", None)
        if builder is None:
            builder = self._rvm_vsa_metadata_builder = (
                MiniMaxH3VSAMetadataBuilder()
            )
        batch.current_timestep = int(current_timestep)
        batch.attn_metadata_vsa = builder.build(
            current_timestep=int(current_timestep),
            raw_latent_shape=(
                layout.num_video_latent_frames,
                layout.latent_height,
                layout.latent_width,
            ),
            patch_size=patch_size,
            VSA_sparsity=float(self.training_config.vsa_sparsity),
            prefix_segments=_h3_vsa_prefix_segments(layout, patch_size),
            device=self.device,
        )

    def noise_amounts(
        self,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Public reward-training wrapper around paired scheduler shifts."""
        return self._noise_amounts(timestep)

    @torch.no_grad()
    def decode_latents(self, packed: torch.Tensor) -> torch.Tensor:
        """Decode packed endpoints to uint8 video media ``[B,C,T,H,W]``."""
        decode_batch_size = int(
            os.environ.get(
                "FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE",
                packed.shape[0],
            )
        )
        if decode_batch_size <= 0:
            raise ValueError(
                "FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE must be positive"
            )
        decoded = torch.cat(
            [
                torch.from_numpy(self.decode_vis_latents(chunk))
                for chunk in packed.split(decode_batch_size, dim=0)
            ]
        )
        return decoded.permute(0, 2, 1, 3, 4).contiguous()


__all__ = ["MiniMaxH3RVMModel"]
