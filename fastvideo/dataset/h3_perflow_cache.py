# SPDX-License-Identifier: Apache-2.0
"""Deterministic reward-filtered views over immutable H3 REST caches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.h3_rest_cache import H3RESTCacheDataset
from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler, passthrough
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import canonical_json_hash

logger = init_logger(__name__)

_SELECTION_SCHEMA = "h3_perflow_topq_v1"


@dataclass(frozen=True, slots=True)
class H3PeRFlowSelectionSummary:
    """Immutable description of one deterministic top-q cache view."""

    fingerprint: str
    ranking_key: str
    samples_per_prompt: int
    selected_per_prompt: int
    num_prompts: int
    num_trajectories: int
    num_segments: int
    num_examples: int
    selected_trajectory_ids: tuple[str, ...]


def _selection_score(entry: Mapping[str, Any], ranking_key: str) -> float:
    """Resolve a finite scalar used only to rank candidates within a prompt."""
    if ranking_key == "mixed_advantage":
        raw = entry.get("mixed_advantage")
    elif ranking_key.startswith("reward_scores."):
        name = ranking_key.removeprefix("reward_scores.")
        values = entry.get("reward_scores")
        raw = values.get(name) if isinstance(values, Mapping) else None
    elif ranking_key.startswith("reward_advantages."):
        name = ranking_key.removeprefix("reward_advantages.")
        values = entry.get("reward_advantages")
        raw = values.get(name) if isinstance(values, Mapping) else None
    else:
        raise ValueError(
            "ranking_key must be 'mixed_advantage', 'reward_scores.<name>', "
            "or 'reward_advantages.<name>'"
        )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"Trajectory {entry.get('trajectory_id')!r} has no numeric {ranking_key!r}"
        )
    score = float(raw)
    if not math.isfinite(score):
        raise ValueError(
            f"Trajectory {entry.get('trajectory_id')!r} has non-finite {ranking_key!r}"
        )
    return score


def select_h3_teacher_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    selected_per_prompt: int,
    ranking_key: str = "mixed_advantage",
    num_segments: int,
) -> tuple[list[dict[str, Any]], H3PeRFlowSelectionSummary]:
    """Select deterministic top-q H3 trajectories and assign equal weights.

    Reward values are used exclusively for ranking. Every retained trajectory
    receives the same positive supervised weight ``1 / selected_per_prompt``.
    """
    if isinstance(selected_per_prompt, bool) or int(selected_per_prompt) <= 0:
        raise ValueError("selected_per_prompt must be a positive integer")
    if isinstance(num_segments, bool) or int(num_segments) <= 0:
        raise ValueError("num_segments must be a positive integer")
    normalized_key = str(ranking_key).strip().lower()
    if not normalized_key:
        raise ValueError("ranking_key must be nonempty")
    if not entries:
        raise ValueError("Cannot select trajectories from an empty H3 cache manifest")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        prompt_id = entry.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("Every H3 trajectory entry must have a nonempty prompt_id")
        groups.setdefault(prompt_id, []).append(entry)

    group_sizes = {len(group) for group in groups.values()}
    if len(group_sizes) != 1:
        raise ValueError(
            "Every prompt must expose the same complete candidate count, got "
            f"{sorted(group_sizes)}"
        )
    samples_per_prompt = group_sizes.pop()
    q = int(selected_per_prompt)
    if q > samples_per_prompt:
        raise ValueError(
            f"selected_per_prompt={q} exceeds samples_per_prompt={samples_per_prompt}"
        )

    selected: list[dict[str, Any]] = []
    fingerprint_groups: list[dict[str, Any]] = []
    equal_weight = 1.0 / float(q)
    expected_candidates = set(range(samples_per_prompt))
    for prompt_id in sorted(groups):
        group = groups[prompt_id]
        candidate_indices: list[int] = []
        ranked: list[tuple[float, int, str, Mapping[str, Any]]] = []
        for entry in group:
            candidate_index = entry.get("candidate_index")
            trajectory_id = entry.get("trajectory_id")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index < 0
            ):
                raise ValueError(
                    f"Prompt {prompt_id!r} has invalid candidate_index={candidate_index!r}"
                )
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError(
                    f"Prompt {prompt_id!r} candidate {candidate_index} has invalid trajectory_id"
                )
            candidate_indices.append(candidate_index)
            score = _selection_score(entry, normalized_key)
            ranked.append((-score, candidate_index, trajectory_id, entry))
        if len(set(candidate_indices)) != len(candidate_indices):
            raise ValueError(f"Prompt {prompt_id!r} contains duplicate candidate indices")
        if set(candidate_indices) != expected_candidates:
            raise ValueError(
                f"Prompt {prompt_id!r} candidate set is incomplete: "
                f"observed={sorted(candidate_indices)}, expected={sorted(expected_candidates)}"
            )

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        prompt_selection: list[dict[str, Any]] = []
        for rank, (negative_score, _candidate_index, _trajectory_id, entry) in enumerate(
            ranked[:q]
        ):
            copied = dict(entry)
            copied["perflow_selection_rank"] = rank
            copied["perflow_selection_score"] = -negative_score
            copied["perflow_selection_weight"] = equal_weight
            selected.append(copied)
            prompt_selection.append(
                {
                    "trajectory_id": copied["trajectory_id"],
                    "candidate_index": copied["candidate_index"],
                    "rank": rank,
                    "score": -negative_score,
                }
            )
        fingerprint_groups.append(
            {
                "prompt_id": prompt_id,
                "selected": prompt_selection,
            }
        )

    fingerprint = canonical_json_hash(
        {
            "schema": _SELECTION_SCHEMA,
            "ranking_key": normalized_key,
            "samples_per_prompt": samples_per_prompt,
            "selected_per_prompt": q,
            "num_segments": int(num_segments),
            "groups": fingerprint_groups,
        }
    )
    summary = H3PeRFlowSelectionSummary(
        fingerprint=fingerprint,
        ranking_key=normalized_key,
        samples_per_prompt=samples_per_prompt,
        selected_per_prompt=q,
        num_prompts=len(groups),
        num_trajectories=len(selected),
        num_segments=int(num_segments),
        num_examples=len(selected) * int(num_segments),
        selected_trajectory_ids=tuple(str(entry["trajectory_id"]) for entry in selected),
    )
    return selected, summary


class H3PeRFlowCacheDataset(H3RESTCacheDataset):
    """Read only the deterministic top-q trajectories from an H3 REST cache."""

    def __init__(
        self,
        cache_dir: str,
        *,
        selected_per_prompt: int,
        ranking_key: str = "mixed_advantage",
        batch_size: int = 1,
        seed: int = 0,
        verify_file_hashes: bool = False,
        expected_student_timesteps: Sequence[int | float] | None = None,
    ) -> None:
        super().__init__(
            cache_dir,
            batch_size=batch_size,
            seed=seed,
            verify_file_hashes=verify_file_hashes,
            expected_student_timesteps=expected_student_timesteps,
        )
        self.entries, self.selection_summary = select_h3_teacher_entries(
            self.entries,
            selected_per_prompt=selected_per_prompt,
            ranking_key=ranking_key,
            num_segments=self.num_segments,
        )
        self.sampler = DP_SP_BatchSampler(
            batch_size=1,
            dataset_size=len(self),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=True,
            seed=int(seed),
        )
        logger.info(
            "Loaded reward-filtered H3 PeRFlow view %s: %d/%d trajectories "
            "per prompt, %d examples, selection=%s",
            self.root,
            self.selection_summary.selected_per_prompt,
            self.selection_summary.samples_per_prompt,
            self.selection_summary.num_examples,
            self.selection_summary.fingerprint,
        )

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        batch = super().__getitems__(indices)
        flat_index = int(indices[0])
        trajectory_index, _segment_index = divmod(flat_index, self.num_segments)
        entry = self.entries[trajectory_index]
        batch["perflow_selection_weight"] = torch.tensor(
            [float(entry["perflow_selection_weight"])], dtype=torch.float32
        )
        batch["perflow_selection_rank"] = torch.tensor(
            [int(entry["perflow_selection_rank"])], dtype=torch.long
        )
        batch["perflow_selection_score"] = torch.tensor(
            [float(entry["perflow_selection_score"])], dtype=torch.float32
        )
        batch["perflow_selection_fingerprint"] = self.selection_summary.fingerprint
        infos = batch.get("info_list")
        if isinstance(infos, list) and infos and isinstance(infos[0], dict):
            infos[0].update(
                {
                    "perflow_selection_rank": int(entry["perflow_selection_rank"]),
                    "perflow_selection_score": float(entry["perflow_selection_score"]),
                    "perflow_selection_weight": float(entry["perflow_selection_weight"]),
                }
            )
        return batch


def build_h3_perflow_cache_dataloader(
    cache_dir: str,
    *,
    selected_per_prompt: int,
    ranking_key: str,
    batch_size: int,
    num_data_workers: int,
    seed: int,
    verify_file_hashes: bool = False,
    expected_student_timesteps: Sequence[int | float] | None = None,
) -> tuple[H3PeRFlowCacheDataset, StatefulDataLoader]:
    dataset = H3PeRFlowCacheDataset(
        cache_dir,
        selected_per_prompt=selected_per_prompt,
        ranking_key=ranking_key,
        batch_size=batch_size,
        seed=seed,
        verify_file_hashes=verify_file_hashes,
        expected_student_timesteps=expected_student_timesteps,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=int(num_data_workers),
        pin_memory=True,
        persistent_workers=int(num_data_workers) > 0,
    )
    return dataset, loader


__all__ = [
    "H3PeRFlowCacheDataset",
    "H3PeRFlowSelectionSummary",
    "build_h3_perflow_cache_dataloader",
    "select_h3_teacher_entries",
]
