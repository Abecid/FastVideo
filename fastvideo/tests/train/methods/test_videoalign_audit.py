# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

import fastvideo.train.methods.rl.rewards.videoalign_audit as audit


class _FakeRewardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3, bias=False)
        self.rm_head = torch.nn.Linear(3, 4, bias=False)
        self.adapter = torch.nn.Linear(3, 3, bias=False)


def test_compatibility_audit_identifies_loaded_reward_head() -> None:
    audit._AUDIT_REPORTS.clear()
    model = _FakeRewardModel()
    audit._wrap_load_state_dict(_FakeRewardModel, label="fake")
    model.load_state_dict(model.state_dict())

    summary = audit.assert_videoalign_checkpoint_coverage(
        model,
        minimum_overall=1.0,
        minimum_head=1.0,
    )
    assert summary["best_overall_coverage"] == 1.0
    assert summary["aggregate"]["reward_head"]["total"] > 0
    assert summary["aggregate"]["reward_head"]["coverage"] == 1.0


def test_direct_coverage_report_separates_model_components() -> None:
    model = _FakeRewardModel()
    checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
    report = audit._coverage_report(model, checkpoint)

    assert report["overall"]["tensor_ratio"] == 1.0
    assert report["components"]["reward_head"]["numel_total"] > 0
    assert report["components"]["reward_head"]["numel_ratio"] == 1.0
    assert report["components"]["adapter"]["numel_total"] > 0


def test_videoalign_prompt_semantics_match_upstream_heads() -> None:
    prompts = ["a red car moves from left to right"]
    mq = audit.AuditedVideoAlignMotionQualityScorer(device="cpu")
    vq = audit.AuditedVideoAlignVisualQualityScorer(device="cpu")
    ta = audit.AuditedVideoAlignTextAlignmentScorer(device="cpu")

    assert mq._prompt(prompts, 0) == ""
    assert vq._prompt(prompts, 0) == ""
    assert ta._prompt(prompts, 0) == prompts[0]


def test_mq_grayscale_matches_upstream_mean_channel_conversion() -> None:
    mq = audit.AuditedVideoAlignMotionQualityScorer(device="cpu")
    frames = np.array([[[[30, 60, 90]]]], dtype=np.uint8)
    converted = mq._frames(frames)

    assert converted.dtype == np.uint8
    assert converted.shape == frames.shape
    assert converted.tolist() == [[[[60, 60, 60]]]]
