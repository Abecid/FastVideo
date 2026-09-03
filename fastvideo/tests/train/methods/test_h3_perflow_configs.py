# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PERFLOW_DIR = (
    _REPO_ROOT
    / "examples/train/configs/knowledge_distillation/minimax_h3"
)
_RVM_DIR = _REPO_ROOT / "examples/train/configs/rl/minimax_h3"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_perflow_configs_preserve_method_and_selection_contract() -> None:
    paths = [
        _PERFLOW_DIR / "h3_perflow_1gpu_smoke.yaml",
        _PERFLOW_DIR / "h3_perflow_8gpu_pilot.yaml",
        _PERFLOW_DIR / "h3_perflow_8gpu_full.yaml",
    ]
    for path in paths:
        config = _load(path)
        student = config["models"]["student"]
        method = config["method"]
        assert student["_target_"] == (
            "fastvideo.train.models.minimax_h3.MiniMaxH3PeRFlowModel"
        )
        assert student["student_timesteps"] == [1000, 750, 500, 250, 0]
        assert student["selected_per_prompt"] == 2
        assert student["ranking_key"] == "mixed_advantage"
        assert method["_target_"] == (
            "fastvideo.train.methods.knowledge_distillation.H3PeRFlowMethod"
        )
        assert method["attn_kind"] == "vsa"
        assert method["loss_type"] == "mse"
        assert method["function_anchor_weight"] == 0.0
        assert config["training"]["data"]["train_batch_size"] == 1
        assert config["training"]["data"]["training_cfg_rate"] == 0.0
        assert config["training"]["vsa_sparsity"] == 0.9


def test_production_perflow_lora_is_function_compatible_with_rvm() -> None:
    perflow_configs = [
        _load(_PERFLOW_DIR / "h3_perflow_8gpu_pilot.yaml"),
        _load(_PERFLOW_DIR / "h3_perflow_8gpu_full.yaml"),
    ]
    rvm_configs = [
        _load(_RVM_DIR / "rvm_h3_8gpu_exact.yaml"),
        _load(_RVM_DIR / "rvm_h3_8gpu_physion_mj.yaml"),
    ]
    expected = {
        "enable": True,
        "rank": 128,
        "alpha": 64,
        "target_modules": ["to_q", "to_k", "to_v", "to_out"],
    }
    for config in [*perflow_configs, *rvm_configs]:
        assert config["models"]["student"]["lora"] == expected


def test_rvm_baselines_remain_on_policy_and_cache_independent() -> None:
    for name in (
        "rvm_h3_8gpu_exact.yaml",
        "rvm_h3_8gpu_physion_mj.yaml",
    ):
        config = _load(_RVM_DIR / name)
        target = str(config["method"]["_target_"])
        assert "RVM" in target
        assert "PeRFlow" not in target
        assert "lora_init_from" not in config["models"]["student"]
        assert config["training"]["data"]["preprocessed_data_type"] == (
            "text_only"
        )
        assert "rest_cache" not in str(config["training"]["data"]["data_path"])
        assert config["method"]["sampling"]["denoising_steps"] == [
            1000,
            750,
            500,
            250,
        ]


def test_full_config_is_one_selected_cache_pass_at_sp4_dp2() -> None:
    config = _load(_PERFLOW_DIR / "h3_perflow_8gpu_full.yaml")
    distributed = config["training"]["distributed"]
    assert distributed == {
        "num_gpus": 8,
        "sp_size": 4,
        "tp_size": 1,
        "hsdp_replicate_dim": 2,
        "hsdp_shard_dim": 4,
    }
    # Production cache contract: 100 prompts x top-2 x 4 segments / DP2.
    assert config["training"]["loop"]["max_train_steps"] == 400


def test_teacher_cache_configs_retain_all_candidates() -> None:
    compact = _load(_PERFLOW_DIR / "h3_rest_cache_4gpu_compact.yaml")
    full = _load(_PERFLOW_DIR / "h3_rest_cache_4gpu_full.yaml")
    assert compact["method"]["samples_per_prompt"] == 2
    assert compact["method"]["max_prompts"] == 1
    assert full["method"]["samples_per_prompt"] == 8
    assert full["method"]["max_prompts"] == 100
    for config in (compact, full):
        assert config["method"]["student_timesteps"] == [
            1000,
            750,
            500,
            250,
            0,
        ]
        assert "selected_per_prompt" not in config["method"]
