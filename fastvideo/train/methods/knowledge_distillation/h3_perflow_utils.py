# SPDX-License-Identifier: Apache-2.0
"""Pure Piecewise Rectified Flow utilities for H3 -> FastH3 distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


LossType = Literal["mse", "huber"]


@dataclass(frozen=True, slots=True)
class PeRFlowSegmentSample:
    """One modality's straight segment evaluated at a query noise amount."""

    state: torch.Tensor
    velocity_target: torch.Tensor
    interpolation_fraction: torch.Tensor
    sigma_query: torch.Tensor
    sigma_delta: torch.Tensor


def _as_batch_scalar(
    value: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(batch_size)
    elif tensor.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or have one value per batch row; "
            f"got {tensor.numel()} values for batch_size={batch_size}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def sample_segment_timestep(
    timestep_current: torch.Tensor | float,
    timestep_next: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one shared base-time fraction inside each cached segment.

    H3 video and audio apply different scheduler shifts to this shared base
    timestep. Callers therefore map the returned query timestep through the
    model's paired ``noise_amounts`` helper before interpolating either
    modality.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    current = _as_batch_scalar(
        timestep_current,
        batch_size=batch_size,
        device=device,
        name="timestep_current",
    )
    next_value = _as_batch_scalar(
        timestep_next,
        batch_size=batch_size,
        device=device,
        name="timestep_next",
    )
    delta = next_value - current
    if bool((delta.abs() <= 1e-8).any()):
        raise ValueError("PeRFlow segment has a zero/degenerate base-time interval")
    fraction = torch.rand(
        (batch_size,),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    query = current + fraction * delta
    return query, fraction


def interpolate_sigma_segment(
    current_state: torch.Tensor,
    next_state: torch.Tensor,
    *,
    sigma_current: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
    sigma_query: torch.Tensor | float,
    eps: float = 1e-8,
    tolerance: float = 1e-5,
) -> PeRFlowSegmentSample:
    """Interpolate one modality in its native rectified-flow coordinate.

    FastH3 predicts ``noise - clean``, the derivative of
    ``x_sigma = (1 - sigma) * clean + sigma * noise`` with respect to sigma.
    Consequently each modality must be straightened in its own shifted sigma
    coordinate even though video and audio share one base query timestep.
    """
    if current_state.shape != next_state.shape or current_state.ndim < 2:
        raise ValueError(
            "PeRFlow anchors must share shape [B, ...], got "
            f"{tuple(current_state.shape)} and {tuple(next_state.shape)}"
        )
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    current = current_state.detach().float()
    next_value = next_state.detach().float()
    if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(next_value).all()):
        raise ValueError("PeRFlow anchor states contain NaN or Inf")

    batch_size = current.shape[0]
    sigma_0 = _as_batch_scalar(
        sigma_current,
        batch_size=batch_size,
        device=current.device,
        name="sigma_current",
    )
    sigma_1 = _as_batch_scalar(
        sigma_next,
        batch_size=batch_size,
        device=current.device,
        name="sigma_next",
    )
    sigma_q = _as_batch_scalar(
        sigma_query,
        batch_size=batch_size,
        device=current.device,
        name="sigma_query",
    )
    sigma_delta = sigma_1 - sigma_0
    if bool((sigma_delta.abs() <= float(eps)).any()):
        raise ValueError(
            "PeRFlow segment has a zero/degenerate sigma interval: "
            f"sigma_current={sigma_0}, sigma_next={sigma_1}"
        )

    fraction = (sigma_q - sigma_0) / sigma_delta
    if bool(((fraction < -float(tolerance)) | (fraction > 1.0 + float(tolerance))).any()):
        raise ValueError(
            "sigma_query lies outside the cached segment: "
            f"fraction={fraction}, sigma_current={sigma_0}, "
            f"sigma_next={sigma_1}, sigma_query={sigma_q}"
        )
    fraction = fraction.clamp(0.0, 1.0)

    view_shape = (batch_size,) + (1,) * (current.ndim - 1)
    fraction_view = fraction.reshape(view_shape)
    delta_view = sigma_delta.reshape(view_shape)
    state_delta = next_value - current
    state = current + fraction_view * state_delta
    target = state_delta / delta_view

    return PeRFlowSegmentSample(
        state=state,
        velocity_target=target.detach(),
        interpolation_fraction=fraction,
        sigma_query=sigma_q,
        sigma_delta=sigma_delta,
    )


def _elementwise_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: LossType,
    huber_delta: float,
) -> torch.Tensor:
    error = prediction.float() - target.detach().float()
    if loss_type == "mse":
        return error.square()
    if loss_type != "huber":
        raise ValueError("loss_type must be one of {'mse', 'huber'}")
    if not math.isfinite(float(huber_delta)) or huber_delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    absolute = error.abs()
    delta = float(huber_delta)
    return torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )


def weighted_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | float | None = None,
    loss_type: LossType = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Return a normalized weighted mean over per-sample regression losses."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError(
            "prediction and target must share shape [B, ...], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(prediction.float()).all()):
        raise ValueError("prediction contains NaN or Inf")
    if not bool(torch.isfinite(target.detach().float()).all()):
        raise ValueError("target contains NaN or Inf")

    elementwise = _elementwise_regression_loss(
        prediction,
        target,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )
    reduce_dims = tuple(range(1, elementwise.ndim))
    per_sample = elementwise.mean(dim=reduce_dims)
    if sample_weight is None:
        return per_sample.mean()

    weight = _as_batch_scalar(
        sample_weight,
        batch_size=prediction.shape[0],
        device=prediction.device,
        name="sample_weight",
    ).detach()
    if bool((weight < 0.0).any()):
        raise ValueError("sample_weight must be nonnegative")
    weight_sum = weight.sum()
    if float(weight_sum) <= 0.0:
        raise ValueError("sample_weight must contain positive total mass")
    return (per_sample * weight).sum() / weight_sum


def compute_h3_perflow_losses(
    prediction: torch.Tensor,
    *,
    video_target: torch.Tensor,
    audio_target: torch.Tensor,
    video_slice: slice,
    audio_slice: slice,
    sample_weight: torch.Tensor | float,
    audio_loss_weight: float = 1.0,
    loss_type: LossType = "mse",
    huber_delta: float = 1.0,
    reference_prediction: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Reduce video/audio PeRFlow losses independently before combining them."""
    if prediction.ndim != 2:
        raise ValueError(f"prediction must have packed shape [B, N], got {prediction.shape}")
    if not math.isfinite(float(audio_loss_weight)) or audio_loss_weight < 0.0:
        raise ValueError("audio_loss_weight must be finite and nonnegative")
    if not math.isfinite(float(anchor_weight)) or anchor_weight < 0.0:
        raise ValueError("anchor_weight must be finite and nonnegative")

    video_prediction = prediction[:, video_slice]
    audio_prediction = prediction[:, audio_slice]
    video_loss = weighted_regression_loss(
        video_prediction,
        video_target,
        sample_weight=sample_weight,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )
    audio_loss = weighted_regression_loss(
        audio_prediction,
        audio_target,
        sample_weight=sample_weight,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )

    anchor_loss = prediction.sum() * 0.0
    if anchor_weight != 0.0:
        if reference_prediction is None or reference_prediction.shape != prediction.shape:
            raise ValueError(
                "reference_prediction must match prediction when anchor_weight is nonzero"
            )
        anchor_loss = weighted_regression_loss(
            prediction,
            reference_prediction.detach(),
            sample_weight=sample_weight,
            loss_type="mse",
        )

    total = (
        video_loss
        + float(audio_loss_weight) * audio_loss
        + float(anchor_weight) * anchor_loss
    )
    return {
        "total_loss": total,
        "video_perflow_loss": video_loss,
        "audio_perflow_loss": audio_loss,
        "function_anchor_loss": anchor_loss,
    }


__all__ = [
    "LossType",
    "PeRFlowSegmentSample",
    "compute_h3_perflow_losses",
    "interpolate_sigma_segment",
    "sample_segment_timestep",
    "weighted_regression_loss",
]
