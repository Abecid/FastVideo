# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import fastvideo.train.methods.rl.rewards.videoalign_audit as audit


class _FakeRewardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3, bias=False)
        self.rm_head = torch.nn.Linear(3, 4, bias=False)


def test_checkpoint_audit_identifies_loaded_reward_head() -> None:
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
    assert summary["aggregate"]["head"]["total"] > 0
    assert summary["aggregate"]["head"]["coverage"] == 1.0


def test_videoalign_prompt_semantics_match_upstream_heads() -> None:
    prompts = ["a red car moves from left to right"]
    mq = audit.AuditedVideoAlignMotionQualityScorer(device="cpu")
    vq = audit.AuditedVideoAlignVisualQualityScorer(device="cpu")
    ta = audit.AuditedVideoAlignTextAlignmentScorer(device="cpu")

    assert mq._prompt(prompts, 0) == ""
    assert vq._prompt(prompts, 0) == ""
    assert ta._prompt(prompts, 0) == prompts[0]
