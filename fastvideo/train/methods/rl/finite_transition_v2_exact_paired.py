# SPDX-License-Identifier: Apache-2.0
"""Exact prompt-seed validation records for finite-transition v2.

Statistical tests remain prompt-level: multiple fixed seeds for one prompt are
averaged before the paired bootstrap, avoiding seed-level pseudoreplication.
Every individual prompt-seed reward/motion value is also persisted so two
objective arms can be compared on literally identical generated examples.
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
from fastvideo.train.methods.rl.finite_transition_posterior import (
    _prepare_validation_log_entry,
)
from fastvideo.train.methods.rl.finite_transition_v2_paired import (
    FiniteTransitionV2PairedMethod,
)


class FiniteTransitionV2ExactPairedMethod(FiniteTransitionV2PairedMethod):
    """V2 paired evaluation with exact prompt/seed identity artifacts."""

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

        local_prompt_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        local_sample_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        local_prompt_masks: list[torch.Tensor] = []
        local_prompt_indices: list[torch.Tensor] = []
        local_sample_masks: list[torch.Tensor] = []
        local_sample_keys: list[torch.Tensor] = []
        local_sample_prompt_indices: list[torch.Tensor] = []
        local_sample_seeds: list[torch.Tensor] = []
        local_logs: list[dict[str, Any]] = []

        for start in range(0, len(items), config.batch_size):
            batch_items = items[start : start + config.batch_size]
            repeated_rows: list[dict[str, Any]] = []
            expanded_meta: list[tuple[int, bool, int, int]] = []
            for global_index, valid, row in batch_items:
                for seed_offset in range(self._validation_samples_per_prompt):
                    repeated_rows.append(copy.deepcopy(row))
                    sample_seed = (
                        int(config.seed)
                        + int(global_index) * 10_000
                        + int(seed_offset)
                    )
                    expanded_meta.append(
                        (
                            int(global_index),
                            bool(valid),
                            int(seed_offset),
                            int(sample_seed),
                        )
                    )

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
            for _prompt_index, _valid, _seed_offset, sample_seed in expanded_meta:
                generator = torch.Generator(
                    device=self.student.device
                ).manual_seed(sample_seed)
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
            motion = temporal_l1(media).to(self.student.device)

            item_count = len(batch_items)
            samples_per_prompt = self._validation_samples_per_prompt
            prompt_valid = torch.tensor(
                [bool(item[1]) for item in batch_items],
                device=self.student.device,
                dtype=torch.bool,
            )
            prompt_indices = torch.tensor(
                [int(item[0]) for item in batch_items],
                device=self.student.device,
                dtype=torch.long,
            )
            sample_valid = torch.tensor(
                [bool(meta[1]) for meta in expanded_meta],
                device=self.student.device,
                dtype=torch.bool,
            )
            sample_prompt_indices = torch.tensor(
                [int(meta[0]) for meta in expanded_meta],
                device=self.student.device,
                dtype=torch.long,
            )
            sample_seeds = torch.tensor(
                [int(meta[3]) for meta in expanded_meta],
                device=self.student.device,
                dtype=torch.long,
            )
            # Validation uses only a handful of seeds per prompt. A multiplier
            # of 1,000 is collision-free and safely represented as int64.
            sample_keys = torch.tensor(
                [
                    int(meta[0]) * 1_000 + int(meta[2])
                    for meta in expanded_meta
                ],
                device=self.student.device,
                dtype=torch.long,
            )
            local_prompt_masks.append(prompt_valid)
            local_prompt_indices.append(prompt_indices)
            local_sample_masks.append(sample_valid)
            local_sample_keys.append(sample_keys)
            local_sample_prompt_indices.append(sample_prompt_indices)
            local_sample_seeds.append(sample_seeds)

            for name, value in rewards.items():
                flat = value.to(self.student.device).reshape(-1)
                local_sample_metrics[f"reward/{name}"].append(flat)
                local_prompt_metrics[f"reward/{name}"].append(
                    flat.reshape(item_count, samples_per_prompt).mean(dim=1)
                )

            local_sample_metrics["temporal_l1"].append(motion.reshape(-1))
            local_prompt_metrics["temporal_l1"].append(
                motion.reshape(item_count, samples_per_prompt).mean(dim=1)
            )
            static = (motion < self._static_temporal_threshold).float()
            local_sample_metrics["static_sample"].append(static.reshape(-1))
            local_prompt_metrics["static_sample_ratio"].append(
                static.reshape(item_count, samples_per_prompt).mean(dim=1)
            )

            latent_diversity = []
            video_diversity = []
            for item_index in range(item_count):
                lo = item_index * samples_per_prompt
                hi = lo + samples_per_prompt
                latent_diversity.append(mean_pairwise_rms(endpoint[lo:hi]))
                video_diversity.append(mean_pairwise_rms(media[lo:hi]))
            local_prompt_metrics["latent_diversity_rms"].append(
                torch.stack(latent_diversity).to(self.student.device)
            )
            local_prompt_metrics["video_diversity_rms"].append(
                torch.stack(video_diversity).to(self.student.device)
            )

            if config.log_samples:
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
                        local_logs.append(entry)

        if not local_prompt_masks:
            return {}

        prompt_mask = self._all_gather_1d(
            torch.cat(local_prompt_masks)
        ).bool()
        gathered_prompt_indices = self._all_gather_1d(
            torch.cat(local_prompt_indices)
        ).long()
        valid_prompt_indices = gathered_prompt_indices[prompt_mask]
        prompt_order = torch.argsort(valid_prompt_indices)
        ordered_prompt_indices = valid_prompt_indices[prompt_order].cpu()

        summary: dict[str, float] = {}
        metrics: dict[str, LogScalar] = {}
        ordered_prompt_values: dict[str, torch.Tensor] = {}
        for name, chunks in local_prompt_metrics.items():
            gathered = self._all_gather_1d(
                torch.cat(chunks).to(self.student.device).float()
            )
            values = gathered[prompt_mask][prompt_order]
            if values.numel() == 0:
                continue
            ordered_prompt_values[name] = values.detach().cpu()
            mean = values.mean()
            std = values.std(unbiased=False)
            sem = std / math.sqrt(float(values.numel()))
            metrics[f"validation/{name}"] = mean
            metrics[f"validation_std/{name}"] = std
            metrics[f"validation_sem/{name}"] = sem
            summary[name] = float(mean)
            summary[f"sem/{name}"] = float(sem)

        sample_mask = self._all_gather_1d(
            torch.cat(local_sample_masks)
        ).bool()
        gathered_sample_keys = self._all_gather_1d(
            torch.cat(local_sample_keys)
        ).long()[sample_mask]
        gathered_sample_prompts = self._all_gather_1d(
            torch.cat(local_sample_prompt_indices)
        ).long()[sample_mask]
        gathered_sample_seeds = self._all_gather_1d(
            torch.cat(local_sample_seeds)
        ).long()[sample_mask]
        sample_order = torch.argsort(gathered_sample_keys)
        exact_values: dict[str, list[float | int]] = {
            "sample_key": [
                int(value) for value in gathered_sample_keys[sample_order].cpu()
            ],
            "prompt_index": [
                int(value)
                for value in gathered_sample_prompts[sample_order].cpu()
            ],
            "sample_seed": [
                int(value) for value in gathered_sample_seeds[sample_order].cpu()
            ],
        }
        for name, chunks in local_sample_metrics.items():
            gathered = self._all_gather_1d(
                torch.cat(chunks).to(self.student.device).float()
            )
            values = gathered[sample_mask][sample_order]
            exact_values[name] = [float(value) for value in values.cpu()]

        metrics["validation/num_prompts"] = float(
            ordered_prompt_indices.numel()
        )
        metrics["validation/num_prompt_seed_pairs"] = float(
            gathered_sample_keys.numel()
        )
        metrics["validation/samples_per_prompt"] = float(
            self._validation_samples_per_prompt
        )
        metrics.update(self._validation_delta_metrics(summary, iteration))
        metrics.update(
            self._paired_validation_metrics(
                iteration=iteration,
                indices=ordered_prompt_indices,
                values=ordered_prompt_values,
            )
        )
        self._write_exact_sample_artifact(
            iteration=iteration,
            exact_values=exact_values,
        )

        if config.log_samples:
            self._log_validation_samples(local_logs, iteration)
        return metrics

    def _write_exact_sample_artifact(
        self,
        *,
        iteration: int,
        exact_values: dict[str, list[float | int]],
    ) -> None:
        if self._rank() != 0:
            return
        output_dir = (
            Path(self.training_config.checkpoint.output_dir)
            / "paired_validation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / (
            f"{self._active_validation_variant}_step_"
            f"{int(iteration):06d}_samples.json"
        )
        path.write_text(
            json.dumps(
                {
                    "iteration": int(iteration),
                    "mode": self._active_validation_variant,
                    "metrics": exact_values,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if self.tracker is not None:
            self.tracker.log_file(str(path))


__all__ = ["FiniteTransitionV2ExactPairedMethod"]
