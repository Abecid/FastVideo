# SPDX-License-Identifier: Apache-2.0
"""Raw/EMA paired held-out validation for finite-transition experiments."""

from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common.finite_transition import (
    mean_pairwise_rms,
    temporal_l1,
)
from fastvideo.train.methods.rl.common.reward_statistics import paired_summary
from fastvideo.train.methods.rl.finite_transition_posterior import (
    _prepare_validation_log_entry,
)


class PairedFiniteTransitionValidationMixin:
    """Adds raw/EMA paired validation and per-prompt JSON artifacts."""

    def _paired_validation_state(self) -> dict[str, Any]:
        if not hasattr(self, "_paired_validation_baselines"):
            self._paired_validation_baselines: dict[
                str,
                dict[str, list[float]],
            ] = {}
        return self._paired_validation_baselines

    def _paired_bootstrap_samples(self) -> int:
        raw = self.method_config.get("paired_validation", {}) or {}
        if not isinstance(raw, dict):
            raise ValueError("method.paired_validation must be a mapping")
        return max(100, int(raw.get("bootstrap_samples", 2000)))

    def _paired_confidence(self) -> float:
        raw = self.method_config.get("paired_validation", {}) or {}
        if not isinstance(raw, dict):
            raise ValueError("method.paired_validation must be a mapping")
        confidence = float(raw.get("confidence", 0.95))
        if not 0.0 < confidence < 1.0:
            raise ValueError("paired-validation confidence must lie in (0, 1)")
        return confidence

    @torch.no_grad()
    def _collect_validation_vectors(
        self,
        *,
        iteration: int,
        mode: str,
        log_samples: bool,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, list[float]],
        list[dict[str, Any]],
    ]:
        self.student.transformer.eval()
        config = self._validation_config
        schedule = self._build_schedule(
            steps=self._eval_map_steps,
            override=self._eval_schedule_override,
            device=self.student.device,
        )
        items = self._get_validation_items()
        local_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        local_masks: list[torch.Tensor] = []
        local_indices: list[torch.Tensor] = []
        local_logs: list[dict[str, Any]] = []

        for start in range(0, len(items), config.batch_size):
            batch_items = items[start : start + config.batch_size]
            repeated_rows: list[dict[str, Any]] = []
            expanded_meta: list[tuple[int, bool]] = []
            for global_index, valid, row in batch_items:
                for _ in range(self._validation_samples_per_prompt):
                    repeated_rows.append(copy.deepcopy(row))
                    expanded_meta.append((global_index, valid))
            raw_batch = self._collate_rows(repeated_rows)
            prepare_generator = torch.Generator(
                device=self.student.device
            ).manual_seed(
                int(config.seed) + 100_000 + self._rank() + start
            )
            batch = self.student.prepare_batch(
                raw_batch,
                generator=prepare_generator,
                latents_source="zeros",
                num_latent_t=config.num_latent_t,
            )
            prompts = self._extract_prompts(raw_batch)
            if batch.latents is None:
                raise RuntimeError("validation batch is missing latent shape")

            noises = []
            for sample_index, (global_index, _valid) in enumerate(
                expanded_meta
            ):
                seed = (
                    int(config.seed)
                    + int(global_index) * 10_000
                    + sample_index % self._validation_samples_per_prompt
                )
                generator = torch.Generator(
                    device=self.student.device
                ).manual_seed(seed)
                noises.append(
                    torch.randn(
                        (1, *batch.latents.shape[1:]),
                        device=self.student.device,
                        dtype=batch.latents.dtype,
                        generator=generator,
                    )
                )
            initial = torch.cat(noises, dim=0)
            endpoint = self._deterministic_rollout(
                initial,
                batch,
                schedule,
            )
            media = self.student.decode_latents(endpoint).detach().cpu()
            rewards = self._score_media(media, prompts)
            motion = temporal_l1(media)

            item_count = len(batch_items)
            samples_per_prompt = self._validation_samples_per_prompt
            valid_mask = torch.tensor(
                [bool(item[1]) for item in batch_items],
                device=self.student.device,
                dtype=torch.bool,
            )
            prompt_indices = torch.tensor(
                [int(item[0]) for item in batch_items],
                device=self.student.device,
                dtype=torch.long,
            )
            local_masks.append(valid_mask)
            local_indices.append(prompt_indices)

            for name, value in rewards.items():
                grouped = value.reshape(item_count, samples_per_prompt)
                local_metrics[f"reward/{name}"].append(grouped.mean(dim=1))
            local_metrics["temporal_l1"].append(
                motion.to(self.student.device)
                .reshape(item_count, samples_per_prompt)
                .mean(dim=1)
            )
            static = (
                motion.to(self.student.device)
                < self._static_temporal_threshold
            ).float().reshape(item_count, samples_per_prompt).mean(dim=1)
            local_metrics["static_sample_ratio"].append(static)

            latent_diversity = []
            video_diversity = []
            for item_index in range(item_count):
                lo = item_index * samples_per_prompt
                hi = lo + samples_per_prompt
                latent_diversity.append(mean_pairwise_rms(endpoint[lo:hi]))
                video_diversity.append(mean_pairwise_rms(media[lo:hi]))
            local_metrics["latent_diversity_rms"].append(
                torch.stack(latent_diversity).to(self.student.device)
            )
            local_metrics["video_diversity_rms"].append(
                torch.stack(video_diversity).to(self.student.device)
            )

            if log_samples:
                for item_index, (global_index, valid, _row) in enumerate(
                    batch_items
                ):
                    if not valid:
                        continue
                    sample_offset = item_index * samples_per_prompt
                    sample_rewards = {
                        name: float(value[sample_offset])
                        for name, value in rewards.items()
                    }
                    sample_rewards["temporal_l1"] = float(
                        motion[sample_offset]
                    )
                    entry = _prepare_validation_log_entry(
                        index=int(global_index),
                        prompt=prompts[sample_offset],
                        media=media[sample_offset],
                        rewards=sample_rewards,
                        max_samples=config.max_samples,
                    )
                    if entry is not None:
                        entry["mode"] = mode
                        local_logs.append(entry)

        if not local_masks:
            return {}, {}, []

        mask = self._all_gather_1d(
            torch.cat(local_masks).float()
        ).bool()
        indices = self._all_gather_1d(
            torch.cat(local_indices).float()
        ).long()
        valid_indices = indices[mask]
        order = torch.argsort(valid_indices)

        scalars: dict[str, torch.Tensor] = {}
        vectors: dict[str, list[float]] = {
            "prompt_index": [
                float(value)
                for value in valid_indices[order].cpu()
            ]
        }
        for name, chunks in local_metrics.items():
            gathered = self._all_gather_1d(
                torch.cat(chunks).to(self.student.device).float()
            )
            values = gathered[mask][order]
            if values.numel() == 0:
                continue
            scalars[f"{mode}/{name}"] = values.mean()
            scalars[f"{mode}_std/{name}"] = values.std(unbiased=False)
            scalars[f"{mode}_sem/{name}"] = (
                values.std(unbiased=False)
                / math.sqrt(float(values.numel()))
            )
            vectors[name] = [float(value) for value in values.cpu()]

        scalars[f"{mode}/num_prompts"] = mask.sum().float()
        scalars[f"{mode}/samples_per_prompt"] = torch.tensor(
            float(self._validation_samples_per_prompt),
            device=self.student.device,
        )
        return scalars, vectors, local_logs

    def _paired_metrics(
        self,
        *,
        mode: str,
        vectors: dict[str, list[float]],
        iteration: int,
    ) -> dict[str, LogScalar]:
        baselines = self._paired_validation_state()
        if mode not in baselines:
            baselines[mode] = {
                key: list(values)
                for key, values in vectors.items()
            }
            return {
                f"paired_{mode}/baseline_initialized": 1.0,
            }

        metrics: dict[str, LogScalar] = {}
        for name, current in vectors.items():
            if name == "prompt_index":
                continue
            baseline = baselines[mode].get(name)
            if baseline is None or len(baseline) != len(current):
                continue
            summary = paired_summary(
                torch.tensor(current),
                torch.tensor(baseline),
                bootstrap_samples=self._paired_bootstrap_samples(),
                confidence=self._paired_confidence(),
                seed=(
                    int(self._validation_config.seed)
                    + int(iteration) * 101
                    + (0 if mode == "raw" else 1)
                ),
            )
            for key, value in summary.items():
                metrics[f"paired_{mode}/{name}/{key}"] = value
        return metrics

    def _write_paired_validation_artifact(
        self,
        *,
        iteration: int,
        mode: str,
        vectors: dict[str, list[float]],
    ) -> None:
        if self._rank() != 0:
            return
        output_dir = Path(
            self.training_config.checkpoint.output_dir
        ) / "paired_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"step_{int(iteration):06d}_{mode}.json"
        path.write_text(
            json.dumps(
                {
                    "iteration": int(iteration),
                    "mode": mode,
                    "metrics": vectors,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if self.tracker is not None:
            self.tracker.log_file(str(path))

    def _paired_success_metrics(
        self,
        *,
        metrics: dict[str, LogScalar],
    ) -> dict[str, LogScalar]:
        primary = f"paired_ema/reward/{self._optimize_reward}"
        primary_delta = float(
            metrics.get(f"{primary}/mean_delta", float("nan"))
        )
        ci_lower = float(
            metrics.get(f"{primary}/ci_lower", float("-inf"))
        )
        primary_success = (
            math.isfinite(primary_delta)
            and primary_delta >= self._success_primary_min_delta
            and ci_lower > 0.0
        )

        motion_delta = float(
            metrics.get(
                "paired_ema/temporal_l1/mean_delta",
                float("nan"),
            )
        )
        baseline_motion = self._paired_validation_state().get(
            "ema",
            {},
        ).get("temporal_l1", [])
        baseline_motion_mean = (
            sum(baseline_motion) / len(baseline_motion)
            if baseline_motion
            else float("nan")
        )
        motion_ratio = (
            (baseline_motion_mean + motion_delta)
            / max(abs(baseline_motion_mean), 1.0e-12)
            if math.isfinite(motion_delta)
            and math.isfinite(baseline_motion_mean)
            else float("nan")
        )

        diversity_delta = float(
            metrics.get(
                "paired_ema/latent_diversity_rms/mean_delta",
                float("nan"),
            )
        )
        baseline_diversity = self._paired_validation_state().get(
            "ema",
            {},
        ).get("latent_diversity_rms", [])
        baseline_diversity_mean = (
            sum(baseline_diversity) / len(baseline_diversity)
            if baseline_diversity
            else float("nan")
        )
        diversity_ratio = (
            (baseline_diversity_mean + diversity_delta)
            / max(abs(baseline_diversity_mean), 1.0e-12)
            if math.isfinite(diversity_delta)
            and math.isfinite(baseline_diversity_mean)
            else float("nan")
        )
        heldout_success = True
        for reward_name, max_drop in self._heldout_max_drop.items():
            delta = float(
                metrics.get(
                    f"paired_ema/reward/{reward_name}/mean_delta",
                    float("nan"),
                )
            )
            retained = math.isfinite(delta) and delta >= -float(max_drop)
            heldout_success = heldout_success and retained
            metrics[
                f"paired_validation_success/heldout_{reward_name}"
            ] = float(retained)

        motion_success = (
            math.isfinite(motion_ratio)
            and motion_ratio >= self._success_min_motion_ratio
        )
        diversity_success = (
            math.isfinite(diversity_ratio)
            and diversity_ratio >= self._success_min_diversity_ratio
        )
        metrics.update(
            {
                "paired_validation/motion_ratio_to_base": motion_ratio,
                "paired_validation/latent_diversity_ratio_to_base": (
                    diversity_ratio
                ),
                "paired_validation_success/primary_reward": float(
                    primary_success
                ),
                "paired_validation_success/motion_retained": float(
                    motion_success
                ),
                "paired_validation_success/diversity_retained": float(
                    diversity_success
                ),
                "paired_validation_success/heldout_retained": float(
                    heldout_success
                ),
                "paired_validation_success/all": float(
                    primary_success
                    and motion_success
                    and diversity_success
                    and heldout_success
                ),
            }
        )
        return metrics

    def on_validation_begin(
        self,
        iteration: int = 0,
    ) -> dict[str, LogScalar]:
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}
        if self._reward_scorer is None:
            raise RuntimeError("reward scorer has not been initialized")

        raw_scalars, raw_vectors, _ = self._collect_validation_vectors(
            iteration=iteration,
            mode="raw",
            log_samples=False,
        )
        raw_metrics: dict[str, LogScalar] = {
            f"validation_{key}": value
            for key, value in raw_scalars.items()
        }
        raw_metrics.update(
            self._paired_metrics(
                mode="raw",
                vectors=raw_vectors,
                iteration=iteration,
            )
        )
        self._write_paired_validation_artifact(
            iteration=iteration,
            mode="raw",
            vectors=raw_vectors,
        )

        with self._ema_context():
            ema_scalars, ema_vectors, ema_logs = (
                self._collect_validation_vectors(
                    iteration=iteration,
                    mode="ema",
                    log_samples=config.log_samples,
                )
            )
        ema_metrics: dict[str, LogScalar] = {
            f"validation_{key}": value
            for key, value in ema_scalars.items()
        }
        ema_metrics.update(
            self._paired_metrics(
                mode="ema",
                vectors=ema_vectors,
                iteration=iteration,
            )
        )
        self._write_paired_validation_artifact(
            iteration=iteration,
            mode="ema",
            vectors=ema_vectors,
        )
        if config.log_samples:
            self._log_validation_samples(ema_logs, iteration)

        metrics = {**raw_metrics, **ema_metrics}
        return self._paired_success_metrics(metrics=metrics)


__all__ = ["PairedFiniteTransitionValidationMixin"]
