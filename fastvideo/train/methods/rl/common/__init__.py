# SPDX-License-Identifier: Apache-2.0
"""Reusable RL training primitives."""

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
from fastvideo.train.methods.rl.common.local_asfmc import (
    local_anchor_gaussian_parameters,
)
from fastvideo.train.methods.rl.common.sampling import (
    DiffusionSampler,
    SamplingConfig,
    SamplingResult,
)
from fastvideo.train.methods.rl.common.prompt_sampling import (
    KRepeatSample,
    distributed_k_repeat_indices,
)
from fastvideo.train.methods.rl.common.validation import (
    RLValidationConfig,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)

__all__ = [
    "DiffusionSampler",
    "KRepeatSample",
    "RLValidationConfig",
    "SamplingConfig",
    "SamplingResult",
    "append_data_endpoint",
    "clipped_grpo_loss",
    "distributed_k_repeat_indices",
    "effective_sample_size",
    "endpoint_anchor_parameters",
    "gaussian_log_prob_mean",
    "group_advantages",
    "local_anchor_gaussian_parameters",
    "mean_pairwise_rms",
    "media_to_video_array",
    "posterior_projection_loss",
    "reward_tilted_weights",
    "sample_diagonal_gaussian",
    "temporal_l1",
    "validate_training_schedule",
    "validation_caption",
    "validation_shard_indices",
]
