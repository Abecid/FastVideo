# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible imports for finite-transition posterior math.

Reusable finite-transition sampling, posterior and diagnostic primitives live in
``fastvideo.train.methods.rl.common.finite_transition``.
"""

from fastvideo.train.methods.rl.common.finite_transition import (
    append_data_endpoint,
    clipped_grpo_loss,
    effective_sample_size,
    endpoint_anchor_parameters,
    gaussian_log_prob_mean,
    group_advantages,
    mean_pairwise_rms,
    posterior_projection_loss,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
)

__all__ = [
    "append_data_endpoint",
    "clipped_grpo_loss",
    "effective_sample_size",
    "endpoint_anchor_parameters",
    "gaussian_log_prob_mean",
    "group_advantages",
    "mean_pairwise_rms",
    "posterior_projection_loss",
    "reward_tilted_weights",
    "sample_diagonal_gaussian",
    "temporal_l1",
    "validate_training_schedule",
]
