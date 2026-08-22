# SPDX-License-Identifier: Apache-2.0
"""Paired fixed-seed validation for finite-transition v2.

The first experiment compared baseline and checkpoints with independent SEMs
even though every validation prompt and seed was fixed.  This subclass keeps the
same generation/evaluation path but stores prompt-level baseline values and logs
paired deltas, paired SEMs, deterministic bootstrap intervals, and JSON records
for every raw and EMA checkpoint.
"""

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
from fastvideo.train.methods.rl.common.finite_transition_v2 import (
    paired_bootstrap_interval,
    paired_difference_statistics,
)
from fastvideo.train.methods.rl.finite_transition_posterior import (
    _prepare_validation_log_entry,
)
from fastvideo.train.methods.rl.finite_transition_v2 import FiniteTransitionV2Method


class FiniteTransitionV2PairedMethod(FiniteTransitionV2Method):
    """Finite-transition v2 with statistically correct paired validation."""

    @torch.no_grad()
    def _run_validation(self, iteration: int) -> dict[str, LogScalar]:
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
            batch_items = items[start:start + config.batch_size]
            repeated_rows: list[dict[str, Any]] = []
            expanded_meta: list[tuple[int, bool]] = []
            for global_index, valid, row in batch_items:
                for _ in range(self._validation_samples_per_prompt):
                    repeated_rows.append(copy.deepcopy(row))
                    expanded_meta.append((global_index, valid))
            raw_batch = self._collate_rows(repeated_rows)
            prepare_generator = torch.Generator(
                device=self.student.device
            ).manual_seed(config.seed + 100_000 + self._rank() + start)
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
            for sample_index, (global_index, _valid) in enumerate(expanded_meta):
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
            endpoint = self._deterministic_rollout(initial, batch, schedule)
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
            index_tensor = torch.tensor(
                [int(item[0]) for item in batch_items],
                device=self.student.device,
                dtype=torch.long,
            )
            local_masks.append(valid_mask)
            local_indices.append(index_tensor)

            for name, value in rewards.items():
                grouped = value.reshape(item_count, samples_per_prompt)
                local_metrics[f"reward/{name}"].append(grouped.mean(dim=1))
            local_metrics["temporal_l1"].append(
                motion.to(self.student.device)
                .reshape(item_count, samples_per_prompt)
                .mean(dim=1)
            )
            local_metrics["static_sample_ratio"].append(
                (
                    motion.to(self.student.device)
                    < self._static_temporal_threshold
                )
                .float()
                .reshape(item_count, samples_per_prompt)
                .mean(dim=1)
            )

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

            if config.log_samples:
                for item_index, (global_index, valid, _row) in enumerate(batch_items):
                    if not valid:
                        continue
                    sample_offset = item_index * samples_per_prompt
                    sample_rewards = {
                        name: float(value[sample_offset])
                        for name, value in rewards.items()
                    }
                    sample_rewards["temporal_l1"] = float(motion[sample_offset])
                    entry = _prepare_validation_log_entry(
                        index=int(global_index),
                        prompt=prompts[sample_offset],
                        media=media[sample_offset],
                        rewards=sample_rewards,
                        max_samples=config.max_samples,
                    )
                    if entry is not None:
                        local_logs.append(entry)

        if not local_masks:
            return {}
        gathered_mask = self._all_gather_1d(torch.cat(local_masks).float()).bool()
        gathered_indices = self._all_gather_1d(torch.cat(local_indices).float()).long()
        valid_indices = gathered_indices[gathered_mask]
        order = torch.argsort(valid_indices)
        ordered_indices = valid_indices[order].cpu()

        summary: dict[str, float] = {}
        metrics: dict[str, LogScalar] = {}
        ordered_values: dict[str, torch.Tensor] = {}
        for name, chunks in local_metrics.items():
            local_values = torch.cat(chunks).to(self.student.device)
            gathered = self._all_gather_1d(local_values.float())
            values = gathered[gathered_mask][order]
            if values.numel() == 0:
                continue
            ordered_values[name] = values.detach().cpu()
            mean = values.mean()
            std = values.std(unbiased=False)
            sem = std / math.sqrt(float(values.numel()))
            metrics[f"validation/{name}"] = mean
            metrics[f"validation_std/{name}"] = std
            metrics[f"validation_sem/{name}"] = sem
            summary[name] = float(mean)
            summary[f"sem/{name}"] = float(sem)

        metrics["validation/num_prompts"] = float(ordered_indices.numel())
        metrics["validation/samples_per_prompt"] = float(
            self._validation_samples_per_prompt
        )
        metrics.update(self._validation_delta_metrics(summary, iteration))
        metrics.update(
            self._paired_validation_metrics(
                iteration=iteration,
                indices=ordered_indices,
                values=ordered_values,
            )
        )

        if config.log_samples:
            self._log_validation_samples(local_logs, iteration)
        return metrics

    def _active_variant_state(self) -> dict[str, Any]:
        return (
            self._raw_validation_state
            if self._active_validation_variant == "raw"
            else self._ema_validation_state
        )

    def _paired_validation_metrics(
        self,
        *,
        iteration: int,
        indices: torch.Tensor,
        values: dict[str, torch.Tensor],
    ) -> dict[str, LogScalar]:
        state = self._active_variant_state()
        baseline = state.setdefault("paired_baseline", {})
        baseline_indices = state.get("paired_indices")
        metrics: dict[str, LogScalar] = {}

        if not baseline:
            state["paired_indices"] = [int(value) for value in indices.tolist()]
            state["paired_baseline"] = {
                name: [float(value) for value in tensor.tolist()]
                for name, tensor in values.items()
            }
            self._write_paired_rows(
                iteration=iteration,
                indices=indices,
                values=values,
                baseline_values=None,
            )
            return metrics

        expected_indices = [int(value) for value in indices.tolist()]
        if baseline_indices != expected_indices:
            raise RuntimeError(
                "fixed paired validation indices changed between checkpoints"
            )

        primary_name = f"reward/{self._optimize_reward}"
        for offset, (name, current) in enumerate(sorted(values.items())):
            raw_baseline = baseline.get(name)
            if raw_baseline is None:
                continue
            base = torch.tensor(raw_baseline, dtype=torch.float32)
            if base.shape != current.shape:
                raise RuntimeError(
                    f"paired validation shape changed for {name}: "
                    f"{tuple(base.shape)} vs {tuple(current.shape)}"
                )
            stats = paired_difference_statistics(current, base)
            low, high = paired_bootstrap_interval(
                stats["delta"],
                confidence=0.95,
                num_bootstrap=2000,
                seed=int(self._validation_config.seed) + iteration * 97 + offset,
            )
            metrics[f"validation_paired_delta/{name}"] = stats["mean"]
            metrics[f"validation_paired_std/{name}"] = stats["std"]
            metrics[f"validation_paired_sem/{name}"] = stats["sem"]
            metrics[f"validation_paired_ci95_low/{name}"] = low
            metrics[f"validation_paired_ci95_high/{name}"] = high
            if name == primary_name:
                practical = float(stats["mean"]) >= self._success_primary_min_delta
                statistically_positive = float(low) > 0.0
                metrics["validation/primary_paired_delta"] = stats["mean"]
                metrics["validation/primary_paired_sem"] = stats["sem"]
                metrics["validation/primary_paired_ci95_low"] = low
                metrics["validation/primary_paired_ci95_high"] = high
                metrics["validation_success/primary_paired"] = float(
                    practical and statistically_positive
                )

        self._write_paired_rows(
            iteration=iteration,
            indices=indices,
            values=values,
            baseline_values=baseline,
        )
        return metrics

    def _write_paired_rows(
        self,
        *,
        iteration: int,
        indices: torch.Tensor,
        values: dict[str, torch.Tensor],
        baseline_values: dict[str, list[float]] | None,
    ) -> None:
        if self._rank() != 0:
            return
        rows = []
        for row_index, prompt_index in enumerate(indices.tolist()):
            row: dict[str, Any] = {
                "prompt_index": int(prompt_index),
                "iteration": int(iteration),
                "variant": self._active_validation_variant,
            }
            for name, tensor in values.items():
                current = float(tensor[row_index])
                row[name] = current
                if baseline_values is not None and name in baseline_values:
                    row[f"delta/{name}"] = current - float(
                        baseline_values[name][row_index]
                    )
            rows.append(row)

        output_dir = Path(self.training_config.checkpoint.output_dir)
        directory = output_dir / "paired_validation"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"{self._active_validation_variant}_step_{int(iteration):06d}.json"
        )
        path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["FiniteTransitionV2PairedMethod"]
