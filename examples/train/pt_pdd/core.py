# SPDX-License-Identifier: Apache-2.0
"""Math utilities for Posterior-Tilted Parallel Decoding Distillation.

The released VPTD branch already contains the path-preserving ASFMC posterior,
Feynman--Kac weighting, posterior likelihood baseline, and Flow-Map-GRPO loss.
PT-PDD reuses those audited primitives and adds only the centered finite-velocity
target correction proposed for parallel decoding distillation.
"""

from __future__ import annotations

import torch

from examples.train.vptd.core import (
    PosteriorPolicyConfig,
    append_data_endpoint,
    clipped_grpo_loss,
    effective_sample_size,
    endpoint_anchor_parameters,
    gaussian_log_prob_mean,
    global_group_advantages,
    posterior_distillation_loss,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
    verify_group_partition,
)


def finite_interval_velocity(
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    source_timestep: torch.Tensor | float,
    target_timestep: torch.Tensor | float,
    *,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Recover the mean velocity of one AnyFlow reverse-time transition.

    AnyFlow uses

    ``x_r = x_t - ((t-r)/N) u``.

    A realized same-state posterior transition therefore defines

    ``u = (x_t - x_r) / ((t-r)/N)``.
    """

    if source_state.shape != target_state.shape:
        raise ValueError(
            "source_state and target_state must have the same shape, got "
            f"{source_state.shape} and {target_state.shape}"
        )
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    source = torch.as_tensor(
        source_timestep,
        device=source_state.device,
        dtype=torch.float32,
    )
    target = torch.as_tensor(
        target_timestep,
        device=source_state.device,
        dtype=torch.float32,
    )
    delta = (source - target) / float(num_train_timesteps)
    if torch.any(delta <= 0):
        raise ValueError("source_timestep must be strictly larger than target_timestep")
    while delta.ndim < source_state.ndim:
        delta = delta.unsqueeze(-1)
    return (source_state.float() - target_state.float()) / delta


def centered_velocity_correction(
    candidate_velocities: torch.Tensor,
    local_weights: torch.Tensor,
    *,
    global_group_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return one rank's contribution to the centered reward correction.

    For globally normalized weights ``w_j`` and same-state candidate velocities
    ``g_j``, PT-PDD uses

    ``Delta u = sum_j (w_j - 1/G) g_j``.

    The centering is an exact control variate: an uninformative reward produces
    zero target correction rather than a finite-sample behavior-cloning drift.
    Distributed callers sum the returned local contributions across ranks.
    """

    if candidate_velocities.ndim < 2:
        raise ValueError("candidate_velocities must have shape [B, ...]")
    if (
        local_weights.ndim != 1
        or local_weights.shape[0] != candidate_velocities.shape[0]
    ):
        raise ValueError("local_weights must match candidate batch dimension")
    if global_group_size <= 1:
        raise ValueError("global_group_size must exceed one")
    if torch.any(local_weights < 0):
        raise ValueError("local_weights must be non-negative")

    coefficients = local_weights.detach().float() - (1.0 / float(global_group_size))
    view = [coefficients.shape[0]] + [1] * (candidate_velocities.ndim - 1)
    correction = (
        coefficients.view(*view) * candidate_velocities.float()
    ).sum(dim=0, keepdim=True)
    return correction, {
        "score_coefficient_mass_local": coefficients.sum().detach(),
        "score_coefficient_abs_mean_local": coefficients.abs().mean().detach(),
        "candidate_velocity_rms_local": (
            candidate_velocities.float().square().mean().sqrt().detach()
        ),
    }


def posterior_tilted_regression_loss(
    student_velocity: torch.Tensor,
    reference_velocity: torch.Tensor,
    global_correction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress onto ``reference_velocity + reward_correction``.

    This is the executable AnyFlow proxy for the full PT-PDD target

    ``u_teacher_RK + sum_j (w_j - 1/G) g_j``.

    The complete target is stopped-gradient. Reward changes the finite teacher
    velocity rather than appearing as a second additive objective.
    """

    if not (
        student_velocity.shape
        == reference_velocity.shape
        == global_correction.shape
    ):
        raise ValueError(
            "student_velocity, reference_velocity, and global_correction must "
            "have identical shapes"
        )
    target = (reference_velocity.float() + global_correction.float()).detach()
    residual = student_velocity.float() - target
    loss = residual.square().mean()
    reference_rms = reference_velocity.float().square().mean().sqrt()
    correction_rms = global_correction.float().square().mean().sqrt()
    flat_reference = reference_velocity.float().flatten(1)
    flat_correction = global_correction.float().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_correction,
        flat_reference,
        dim=1,
    ).mean()
    return loss, {
        "posterior_tilted_regression_loss": loss.detach(),
        "reference_velocity_rms": reference_rms.detach(),
        "reward_correction_rms": correction_rms.detach(),
        "reward_correction_to_reference": (
            correction_rms / reference_rms.clamp_min(1.0e-12)
        ).detach(),
        "reward_correction_reference_cosine": cosine.detach(),
        "student_reference_mse": (
            student_velocity.float() - reference_velocity.float().detach()
        ).square().mean().detach(),
    }


def provenance_table() -> dict[str, str]:
    """Machine-readable provenance for the locked experiment defaults."""

    return {
        "base_checkpoint_and_video_shape": (
            "NVlabs/AnyFlow released Wan-1.3B configuration"
        ),
        "flow_shift_and_eval_nfe": (
            "NVlabs/AnyFlow released Wan-1.3B configuration"
        ),
        "lora_optimizer_and_ema": "NVlabs/AnyFlow Wan on-policy configuration",
        "K_clip_advantage_epsilon": "Flow-Map GRPO Appendix C.1",
        "video_group_size": "released Flow-GRPO Wan configuration",
        "endpoint_posterior": (
            "Flow-Map GRPO Eq. (13), exact affine-path conditional"
        ),
        "shared_prefix_transition_credit": "BranchGRPO / Flow-GRPO-Fast",
        "reward_tilt": "Diamond Maps / Feynman-Kac steering",
        "reward_corrected_regression": "RSM, RAM, and AWM",
        "centered_velocity_control_variate": (
            "within-group centering from CRD/AWM/GRPO baselines"
        ),
        "pdd_integration": (
            "replace PDD Runge--Kutta interval target when official code releases"
        ),
    }


__all__ = [
    "PosteriorPolicyConfig",
    "append_data_endpoint",
    "centered_velocity_correction",
    "clipped_grpo_loss",
    "effective_sample_size",
    "endpoint_anchor_parameters",
    "finite_interval_velocity",
    "gaussian_log_prob_mean",
    "global_group_advantages",
    "posterior_distillation_loss",
    "posterior_tilted_regression_loss",
    "provenance_table",
    "reward_tilted_weights",
    "sample_diagonal_gaussian",
    "temporal_l1",
    "validate_training_schedule",
    "verify_group_partition",
]
