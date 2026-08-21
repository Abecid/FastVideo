# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from fastvideo.train.methods.rl.common.anyflow_schedule import (
    anyflow_inference_schedule,
)


def test_released_anyflow_four_step_grid_shift_five() -> None:
    schedule = anyflow_inference_schedule(
        num_steps=4,
        shift=5.0,
        num_train_timesteps=1000,
        device="cpu",
    )
    expected = torch.tensor(
        [1000.0, 937.5, 833.3333333, 625.0, 0.0],
        dtype=torch.float32,
    )
    assert torch.allclose(schedule, expected, atol=1.0e-4, rtol=0.0)


def test_released_anyflow_five_step_training_grid_shift_five() -> None:
    schedule = anyflow_inference_schedule(
        num_steps=5,
        shift=5.0,
        num_train_timesteps=1000,
        device="cpu",
    )
    expected = torch.tensor(
        [
            1000.0,
            952.3809524,
            882.3529412,
            769.2307692,
            555.5555556,
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(schedule, expected, atol=1.0e-4, rtol=0.0)
    assert float(schedule[-2]) > 30.0


def test_anyflow_grid_is_strictly_descending() -> None:
    schedule = anyflow_inference_schedule(
        num_steps=8,
        shift=5.0,
        num_train_timesteps=1000,
        device="cpu",
    )
    assert schedule.shape == (9,)
    assert torch.all(schedule[:-1] > schedule[1:])
    assert float(schedule[-1]) == 0.0
