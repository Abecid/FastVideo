# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 adapter for reward-filtered PeRFlow cache training."""

from __future__ import annotations

from typing import Any

from fastvideo.dataset.h3_perflow_cache import (
    H3PeRFlowCacheDataset,
    H3PeRFlowSelectionSummary,
    build_h3_perflow_cache_dataloader,
)
from fastvideo.distributed import get_sp_group
from fastvideo.train.models.minimax_h3.minimax_h3_rest import MiniMaxH3RESTModel


class MiniMaxH3PeRFlowModel(MiniMaxH3RESTModel):
    """Trainable FastH3 LoRA fed by a deterministic top-q H3 cache view."""

    def __init__(
        self,
        *,
        selected_per_prompt: int = 2,
        ranking_key: str = "mixed_advantage",
        **kwargs: Any,
    ) -> None:
        if isinstance(selected_per_prompt, bool) or int(selected_per_prompt) <= 0:
            raise ValueError("models.student.selected_per_prompt must be positive")
        normalized_key = str(ranking_key).strip().lower()
        if not normalized_key:
            raise ValueError("models.student.ranking_key must be nonempty")
        self._perflow_selected_per_prompt = int(selected_per_prompt)
        self._perflow_ranking_key = normalized_key
        self.perflow_cache_dataset: H3PeRFlowCacheDataset | None = None
        super().__init__(**kwargs)

    @property
    def perflow_selection_summary(self) -> H3PeRFlowSelectionSummary:
        if self.perflow_cache_dataset is None:
            raise RuntimeError("PeRFlow cache is not initialized")
        return self.perflow_cache_dataset.selection_summary

    @property
    def perflow_selection_fingerprint(self) -> str:
        return self.perflow_selection_summary.fingerprint

    def init_preprocessors(self, training_config: Any) -> None:
        cache_path = training_config.data.data_path
        if not isinstance(cache_path, str) or not cache_path.strip():
            raise ValueError(
                "H3 PeRFlow requires training.data.data_path to be one "
                "completed H3 REST cache directory"
            )
        self.sp_group = get_sp_group()
        dataset, dataloader = build_h3_perflow_cache_dataloader(
            cache_path,
            selected_per_prompt=self._perflow_selected_per_prompt,
            ranking_key=self._perflow_ranking_key,
            batch_size=int(training_config.data.train_batch_size),
            num_data_workers=int(training_config.data.dataloader_num_workers),
            seed=int(training_config.data.seed),
            verify_file_hashes=self._rest_verify_cache_hashes,
            expected_student_timesteps=self._rest_student_timesteps,
        )
        self.perflow_cache_dataset = dataset
        # Preserve the inherited cache fingerprint and diagnostics contract.
        self.rest_cache_dataset = dataset
        self.dataloader = dataloader
        self.start_step = 0


__all__ = ["MiniMaxH3PeRFlowModel"]
