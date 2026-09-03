# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fastvideo.dataset.h3_perflow_cache import (
    H3PeRFlowCacheDataset,
    select_h3_teacher_entries,
)
from fastvideo.dataset.h3_rest_cache import H3_REST_CACHE_SCHEMA_VERSION, sha256_file
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import canonical_json_hash


def _entry(prompt: str, candidate: int, score: float) -> dict[str, object]:
    return {
        "trajectory_id": f"{prompt}-c{candidate}",
        "prompt_id": prompt,
        "candidate_index": candidate,
        "mixed_advantage": score,
        "reward_scores": {"quality": score},
        "reward_advantages": {"quality": score},
    }


def test_topq_selection_is_deterministic_and_equal_weighted() -> None:
    entries = [
        _entry("p1", 3, 0.0),
        _entry("p0", 2, 2.0),
        _entry("p1", 0, 4.0),
        _entry("p0", 0, 1.0),
        _entry("p1", 2, 3.0),
        _entry("p0", 3, -1.0),
        _entry("p1", 1, 4.0),
        _entry("p0", 1, 2.0),
    ]
    selected, summary = select_h3_teacher_entries(
        entries,
        selected_per_prompt=2,
        ranking_key="mixed_advantage",
        num_segments=4,
    )

    assert [entry["trajectory_id"] for entry in selected] == [
        "p0-c1",
        "p0-c2",
        "p1-c0",
        "p1-c1",
    ]
    assert [entry["perflow_selection_rank"] for entry in selected] == [0, 1, 0, 1]
    assert all(entry["perflow_selection_weight"] == 0.5 for entry in selected)
    assert summary.samples_per_prompt == 4
    assert summary.selected_per_prompt == 2
    assert summary.num_examples == 16

    reversed_selected, reversed_summary = select_h3_teacher_entries(
        list(reversed(entries)),
        selected_per_prompt=2,
        ranking_key="mixed_advantage",
        num_segments=4,
    )
    assert [entry["trajectory_id"] for entry in reversed_selected] == [
        entry["trajectory_id"] for entry in selected
    ]
    assert reversed_summary.fingerprint == summary.fingerprint


def test_selection_supports_raw_component_and_rejects_bad_groups() -> None:
    entries = [_entry("p0", candidate, float(candidate)) for candidate in range(3)]
    selected, _summary = select_h3_teacher_entries(
        entries,
        selected_per_prompt=1,
        ranking_key="reward_scores.quality",
        num_segments=4,
    )
    assert selected[0]["candidate_index"] == 2

    malformed = [*entries[:-1], _entry("p0", 4, 10.0)]
    with pytest.raises(ValueError, match="candidate set is incomplete"):
        select_h3_teacher_entries(
            malformed,
            selected_per_prompt=1,
            num_segments=4,
        )

    nonfinite = [dict(entry) for entry in entries]
    nonfinite[0]["mixed_advantage"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        select_h3_teacher_entries(
            nonfinite,
            selected_per_prompt=1,
            num_segments=4,
        )


def _write_cache(root: Path) -> None:
    (root / "prompts").mkdir(parents=True)
    (root / "trajectories").mkdir()
    manifest: list[dict[str, object]] = []
    for prompt_index in range(2):
        prompt_id = f"p{prompt_index}"
        prompt_path = root / "prompts" / f"{prompt_id}.pt"
        torch.save(
            {
                "text_embedding": torch.zeros(1, 4, 8, dtype=torch.bfloat16),
                "text_attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            prompt_path,
        )
        scores = [0.0, 3.0, 3.0, 1.0]
        for candidate, score in enumerate(scores):
            trajectory_path = root / "trajectories" / f"{prompt_id}_c{candidate}.pt"
            torch.save(
                {
                    "anchor_states": (
                        torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
                        + prompt_index * 10
                        + candidate
                    ),
                    "anchor_timesteps": torch.tensor(
                        [1000.0, 750.0, 500.0, 250.0, 0.0],
                        dtype=torch.float32,
                    ),
                },
                trajectory_path,
            )
            manifest.append(
                {
                    "trajectory_id": f"{prompt_id}-c{candidate}",
                    "prompt_id": prompt_id,
                    "prompt": "a test prompt",
                    "candidate_index": candidate,
                    "seed": 100 * prompt_index + candidate,
                    "prompt_file": str(prompt_path.relative_to(root)),
                    "prompt_sha256": sha256_file(prompt_path),
                    "prompt_bytes": prompt_path.stat().st_size,
                    "trajectory_file": str(trajectory_path.relative_to(root)),
                    "trajectory_sha256": sha256_file(trajectory_path),
                    "trajectory_bytes": trajectory_path.stat().st_size,
                    "reward_scores": {"quality": score},
                    "reward_advantages": {"quality": score},
                    "mixed_advantage": score,
                }
            )
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    metadata = {
        "schema_version": H3_REST_CACHE_SCHEMA_VERSION,
        "num_prompts": 2,
        "samples_per_prompt": 4,
        "num_trajectories": 8,
        "num_segments": 4,
        "student_timesteps": [1000, 750, 500, 250, 0],
        "reward_names": ["quality"],
        "manifest_sha256": sha256_file(manifest_path),
        "provenance": {"unit_test": True},
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root / "COMPLETE").write_text(metadata["fingerprint"] + "\n", encoding="utf-8")


def test_selected_dataset_exposes_selection_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cache(tmp_path)
    monkeypatch.setattr("fastvideo.dataset.h3_rest_cache.get_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.h3_rest_cache.get_sp_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.h3_rest_cache.get_world_rank", lambda: 0)
    monkeypatch.setattr("fastvideo.dataset.h3_perflow_cache.get_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.h3_perflow_cache.get_sp_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.h3_perflow_cache.get_world_rank", lambda: 0)

    dataset = H3PeRFlowCacheDataset(
        str(tmp_path),
        selected_per_prompt=2,
        seed=4,
        verify_file_hashes=True,
    )
    assert len(dataset) == 2 * 2 * 4
    assert dataset.selection_summary.selected_trajectory_ids == (
        "p0-c1",
        "p0-c2",
        "p1-c1",
        "p1-c2",
    )

    batch = dataset.__getitems__([5])
    assert batch["rest_segment_index"].item() == 1
    assert batch["info_list"][0]["trajectory_id"] == "p0-c2"
    assert batch["perflow_selection_rank"].item() == 1
    assert batch["perflow_selection_weight"].item() == pytest.approx(0.5)
    assert batch["perflow_selection_fingerprint"] == dataset.selection_summary.fingerprint
