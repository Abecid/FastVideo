# SPDX-License-Identifier: Apache-2.0
"""Incremental-KL correction for reliable finite-transition training.

When rollouts come from ``base_adapter_disabled``, the behavior log probability
is a frozen-base likelihood. Comparing the post-update policy to that behavior
would measure cumulative drift, not the size of the latest optimizer step. This
subclass stores the learner's pre-update likelihood and uses it exclusively for
target-KL calibration while retaining the frozen behavior likelihood in the
GRPO importance ratio.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from fastvideo.train.methods.rl.common.finite_transition import (
    clipped_grpo_loss,
    posterior_projection_loss,
)
from fastvideo.train.methods.rl.finite_transition_reliable import (
    ReliableFiniteTransitionMethod,
)


class CalibratedReliableFiniteTransitionMethod(
    ReliableFiniteTransitionMethod
):
    """Reliable finite-transition method with incremental update-KL probes."""

    def _backward_likelihood_transitions(
        self,
        rollout: dict[str, Any],
        *,
        local_weights: torch.Tensor,
        local_advantages: torch.Tensor,
        loss_scale: float,
    ) -> tuple[list[torch.Tensor], dict[str, list[float]]]:
        losses: list[torch.Tensor] = []
        diagnostics: dict[str, list[float]] = defaultdict(list)
        transition_count = len(rollout["transitions"])
        if self._transition_loss_reduction == "mean":
            denominator = self._rollout_groups_per_update * transition_count
        else:
            denominator = self._rollout_groups_per_update

        batch = rollout["batch"]
        for transition in rollout["transitions"]:
            new_log_prob = self._new_transition_log_prob(transition, batch)
            # This reference is the learner immediately before optimizer.step.
            # The GRPO ratio below still uses transition["old_log_prob"], which
            # may intentionally come from a frozen behavior policy.
            transition["pre_update_log_prob"] = new_log_prob.detach()
            if self._objective == "posterior_projection":
                raw_loss, transition_diagnostics = posterior_projection_loss(
                    new_log_prob,
                    local_weights,
                    global_group_size=self._group_size,
                    distributed_world_size=self._world_size(),
                )
            else:
                raw_loss, transition_diagnostics = clipped_grpo_loss(
                    new_log_prob,
                    transition["old_log_prob"],
                    local_advantages,
                    clip_range=self._clip_range,
                )
            self.student.backward(
                raw_loss * float(loss_scale),
                (batch.timesteps, batch.attn_metadata),
                grad_accum_rounds=denominator,
            )
            losses.append(raw_loss.detach())
            for name, value in transition_diagnostics.items():
                diagnostics[name].append(float(value))
                diagnostics[
                    f"transition_{int(transition['index'])}/{name}"
                ].append(float(value))
        return losses, diagnostics

    @torch.no_grad()
    def _post_update_policy_probe(
        self,
        rollout: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deltas = []
        batch = rollout["batch"]
        for transition in rollout["transitions"]:
            post_log_prob = self._new_transition_log_prob(transition, batch)
            reference = transition.get("pre_update_log_prob")
            if reference is None:
                reference = transition["old_log_prob"]
            deltas.append(post_log_prob - reference)
        stacked = torch.stack(deltas, dim=0)
        return 0.5 * stacked.square().mean(), stacked.abs().mean()


__all__ = ["CalibratedReliableFiniteTransitionMethod"]
