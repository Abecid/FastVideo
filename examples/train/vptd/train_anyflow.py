# SPDX-License-Identifier: Apache-2.0
"""Video Posterior Transition Distillation on released AnyFlow-Wan.

The experiment begins from NVIDIA's released, already competent four-step
AnyFlow Wan2.1-1.3B model. It never evaluates raw Wan as a four-step generator.
For each update it:

1. chooses one of four finite flow-map transitions;
2. builds one shared deterministic prefix for the prompt;
3. draws the released video-domain group of four path-preserving endpoint-anchor
   posterior actions at that state;
4. completes each action deterministically and scores the resulting video;
5. forms a Feynman--Kac reward tilt with a target ESS; and
6. projects that tilted conditional posterior back into the AnyFlow LoRA policy
   by weighted maximum likelihood.

The optional ``flowmap_grpo`` objective is a direct likelihood-ratio baseline.
Deterministic four-step AnyFlow inference is always used for validation. Before
training, a parity preflight compares this script's sampler against AnyFlow's
released pipeline implementation and aborts on mismatch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.train.vptd.core import (  # noqa: E402
    PosteriorPolicyConfig,
    append_data_endpoint,
    clipped_grpo_loss,
    endpoint_anchor_parameters,
    gaussian_log_prob_mean,
    global_group_advantages,
    posterior_distillation_loss,
    provenance_table,
    reward_tilted_weights,
    sample_diagonal_gaussian,
    temporal_l1,
    validate_training_schedule,
    verify_group_partition,
)


@dataclass(slots=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(slots=True)
class AnyFlowRuntime:
    transformer: torch.nn.Module
    tokenizer: Any
    text_encoder: torch.nn.Module
    vae: torch.nn.Module
    scheduler: Any
    num_train_timesteps: int
    vae_scale_factor_temporal: int
    vae_scale_factor_spatial: int


class LoRAEMA:
    """EMA over trainable adapter parameters only.

    Decay 0.99 and warmup 200 follow AnyFlow's released Wan on-policy config.
    """

    def __init__(self, model: torch.nn.Module, *, decay: float, warmup_steps: int) -> None:
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.shadow = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("LoRAEMA found no trainable parameters")
        self._stored: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = self.decay if int(step) >= self.warmup_steps else 0.0
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    parameter.detach().float().cpu(), alpha=1.0 - decay
                )

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        if self._stored is not None:
            raise RuntimeError("EMA parameters are already stored")
        self._stored = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name].to(parameter.device, parameter.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self._stored is None:
            raise RuntimeError("EMA restore requires store() first")
        for name, parameter in model.named_parameters():
            if name in self._stored:
                parameter.copy_(self._stored[name].to(parameter.device, parameter.dtype))
        self._stored = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "shadow": {name: value.cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        shadow = state.get("shadow", {})
        missing = sorted(set(self.shadow) - set(shadow))
        if missing:
            raise ValueError(f"EMA checkpoint is missing {len(missing)} parameter(s)")
        for name in self.shadow:
            self.shadow[name].copy_(shadow[name].to(self.shadow[name].device))


@contextmanager
def ema_scope(ema: LoRAEMA, model: torch.nn.Module):
    ema.store(model)
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/train/configs/vptd/wan_anyflow_videoalign_mq.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--reward-key")
    parser.add_argument("--objective", choices=("posterior_distillation", "flowmap_grpo"))
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run official AnyFlow parity plus fixed-prompt base evaluation and exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Infrastructure-only two-update run. It reduces video size and validation "
            "coverage and must not be used as a scientific comparison."
        ),
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = str(args.output_dir)
    if args.run_name:
        cfg["experiment"]["run_name"] = args.run_name
    if args.max_train_steps is not None:
        cfg["experiment"]["max_train_steps"] = int(args.max_train_steps)
    if args.reward_key:
        cfg["reward"]["optimize"] = args.reward_key
    if args.objective:
        cfg["posterior_policy"]["objective"] = args.objective
    if args.eval_every is not None:
        cfg["experiment"]["eval_every"] = int(args.eval_every)
    if args.save_every is not None:
        cfg["experiment"]["save_every"] = int(args.save_every)
    if args.resume is not None:
        cfg["experiment"]["resume"] = args.resume

    cfg["experiment"]["smoke_only"] = bool(args.smoke)
    cfg["experiment"]["preflight_only"] = bool(args.preflight_only)
    if args.smoke:
        cfg["experiment"]["max_train_steps"] = 2
        cfg["experiment"]["eval_every"] = 1
        cfg["experiment"]["save_every"] = 1
        cfg["experiment"]["run_name"] = f"{cfg['experiment']['run_name']}_smoke"
        cfg["posterior_policy"]["group_size"] = 4
        cfg["model"]["num_frames"] = 17
        cfg["model"]["height"] = 256
        cfg["model"]["width"] = 448
        cfg["prompts"]["validation_count"] = 4
        cfg["prompts"]["qualitative_count"] = 4

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    lora = cfg["lora"]
    optimizer = cfg["optimizer"]
    policy = cfg["posterior_policy"]
    experiment = cfg["experiment"]
    prompts = cfg["prompts"]
    smoke = bool(experiment.get("smoke_only", False))

    if (int(model["num_frames"]) - 1) % 4 != 0:
        raise ValueError("Wan num_frames must satisfy (num_frames - 1) % 4 == 0")
    if int(model["height"]) % 16 or int(model["width"]) % 16:
        raise ValueError("AnyFlow Wan height and width must be divisible by 16")
    if int(model["train_map_steps"]) != int(policy["stochastic_steps"]) + 1:
        raise ValueError("train_map_steps must equal K stochastic steps plus one final step")
    if not smoke:
        if int(model["num_frames"]) != 81 or int(model["height"]) != 480 or int(model["width"]) != 832:
            raise ValueError("released AnyFlow Wan-1.3B scientific setup is 81 frames at 480x832")
        if int(model["eval_map_steps"]) != 4:
            raise ValueError("released AnyFlow Wan checkpoint must be evaluated at four steps")
        if float(model["flow_shift"]) != 5.0:
            raise ValueError("released AnyFlow Wan-1.3B uses flow_shift=5.0")
    if float(model["guidance_scale"]) != 1.0:
        raise ValueError("released AnyFlow Wan-1.3B uses guidance_scale=1.0")
    if int(model["fps"]) != 16 and not smoke:
        raise ValueError("released AnyFlow Wan validation uses 16 fps")

    expected_targets = {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
    }
    if int(lora["rank"]) != 256 or int(lora["alpha"]) != 256:
        raise ValueError("released AnyFlow Wan LoRA rank and alpha are both 256")
    if set(lora["target_modules"]) != expected_targets:
        raise ValueError("LoRA target modules must match the released AnyFlow Wan config")
    if float(lora["dropout"]) != 0.0:
        raise ValueError("released AnyFlow Wan LoRA dropout is zero")

    if float(optimizer["learning_rate"]) != 2.0e-6:
        raise ValueError("released AnyFlow Wan on-policy LR is 2e-6")
    if [float(x) for x in optimizer["betas"]] != [0.0, 0.999]:
        raise ValueError("released AnyFlow Wan on-policy Adam betas are [0, 0.999]")
    if float(optimizer["weight_decay"]) != 0.0:
        raise ValueError("released AnyFlow Wan on-policy weight decay is zero")
    if float(optimizer["max_grad_norm"]) != 1.0:
        raise ValueError("released AnyFlow/Flow-Map GRPO gradient clipping is 1.0")
    if float(optimizer["ema_decay"]) != 0.99 or int(optimizer["ema_warmup_steps"]) != 200:
        raise ValueError("released AnyFlow Wan on-policy EMA is 0.99 after 200 warmup steps")

    posterior = PosteriorPolicyConfig(
        stochastic_steps=int(policy["stochastic_steps"]),
        group_size=int(policy["group_size"]),
        target_ess_ratio=float(policy["target_ess_ratio"]),
        clip_range=float(policy["clip_range"]),
        advantage_clip=float(policy["advantage_clip"]),
        advantage_epsilon=float(policy["advantage_epsilon"]),
    )
    posterior.validate()
    if not smoke and int(policy["stochastic_steps"]) != 4:
        raise ValueError("Flow-Map GRPO uses K=4 stochastic transitions")
    if not smoke and int(policy["group_size"]) != 4:
        raise ValueError("released Flow-GRPO Wan uses four videos per prompt")
    if str(policy["objective"]) not in {"posterior_distillation", "flowmap_grpo"}:
        raise ValueError("unsupported posterior_policy.objective")
    if int(policy["inner_epochs"]) != 1:
        raise ValueError("released flow-map/video on-policy recipes use one inner epoch")

    if int(prompts["validation_count"]) < int(prompts["qualitative_count"]):
        raise ValueError("validation_count must be >= qualitative_count")
    if int(prompts["qualitative_count"]) < 5 and not smoke:
        raise ValueError("scientific runs must log at least five validation videos")
    if int(experiment["max_train_steps"]) <= 0:
        raise ValueError("max_train_steps must be positive")


def init_distributed(seed: int) -> DistInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("VPTD requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    random.seed(int(seed) + rank)
    np.random.seed(int(seed) + rank)
    torch.manual_seed(int(seed) + rank)
    torch.cuda.manual_seed_all(int(seed) + rank)
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier(info: DistInfo) -> None:
    if info.world_size > 1:
        dist.barrier()


def broadcast_int(value: int, info: DistInfo) -> int:
    tensor = torch.tensor([int(value)], device=info.device, dtype=torch.long)
    if info.world_size > 1:
        dist.broadcast(tensor, src=0)
    return int(tensor.item())


def all_gather_1d(local: torch.Tensor, info: DistInfo) -> torch.Tensor:
    local = local.contiguous()
    if info.world_size == 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(info.world_size)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=0)


def reduce_mean(value: torch.Tensor, info: DistInfo) -> torch.Tensor:
    reduced = value.detach().float().clone()
    if info.world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= info.world_size
    return reduced


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def rank_print(info: DistInfo, *values: Any) -> None:
    if info.is_main:
        print(*values, flush=True)


def prepare_prompt_split(cfg: dict[str, Any], info: DistInfo) -> tuple[list[str], list[str]]:
    output_dir = Path(cfg["experiment"]["output_dir"])
    split_path = output_dir / "prompt_split.json"
    if info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not split_path.exists():
            from datasets import load_dataset

            p_cfg = cfg["prompts"]
            rows = load_dataset(
                p_cfg["dataset"],
                p_cfg.get("subset"),
                split=p_cfg.get("split", "train"),
            )
            field = str(p_cfg.get("text_field", "prompt"))
            seen: set[str] = set()
            prompts: list[str] = []
            for row in rows:
                text = str(row.get(field, "")).strip()
                if text and text not in seen:
                    seen.add(text)
                    prompts.append(text)
            validation_count = int(p_cfg["validation_count"])
            if len(prompts) <= validation_count:
                raise RuntimeError("prompt dataset is too small for the validation split")
            rng = random.Random(int(cfg["experiment"]["seed"]))
            rng.shuffle(prompts)
            split_path.write_text(
                json.dumps(
                    {
                        "validation": prompts[:validation_count],
                        "train": prompts[validation_count:],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    barrier(info)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return list(payload["train"]), list(payload["validation"])


def ensure_videoalign_checkpoint(info: DistInfo) -> Path:
    root = Path(os.environ.get("VIDEOALIGN_CHECKPOINT_PATH", "/runs/cache/vptd/VideoReward"))
    if info.is_main:
        root.mkdir(parents=True, exist_ok=True)
        has_checkpoint = (root / "model_config.json").exists() and (
            (root / "model.pth").exists()
            or (root / "adapter_model.safetensors").exists()
            or any(root.glob("checkpoint-*/model.pth"))
        )
        if not has_checkpoint:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="KwaiVGI/VideoReward",
                repo_type="model",
                local_dir=str(root),
            )
    barrier(info)
    os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = str(root)
    return root


def load_anyflow_runtime(cfg: dict[str, Any], info: DistInfo) -> AnyFlowRuntime:
    anyflow_root = Path(os.environ.get("ANYFLOW_ROOT", "/workspace/AnyFlow"))
    if not (anyflow_root / "far").is_dir():
        raise RuntimeError(f"ANYFLOW_ROOT does not contain AnyFlow: {anyflow_root}")
    if str(anyflow_root) not in sys.path:
        sys.path.insert(0, str(anyflow_root))

    from diffusers.models import AutoencoderKLWan
    from far.models.transformer_far_wan_model import FAR_Wan_Transformer3DModel
    from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
    from far.utils.lora_util import filter_learnable_module
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, UMT5EncoderModel

    m_cfg = cfg["model"]
    l_cfg = cfg["lora"]
    model_id = str(m_cfg["model_id"])
    dtype = torch.bfloat16

    rank_print(info, f"Loading released AnyFlow checkpoint: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(info.device)
    text_encoder.requires_grad_(False).eval()
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(info.device)
    vae.requires_grad_(False).eval()

    transformer = FAR_Wan_Transformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    targets = filter_learnable_module(transformer, list(l_cfg["target_modules"]))
    if not targets:
        raise RuntimeError("AnyFlow LoRA target filter matched no linear modules")
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=int(l_cfg["rank"]),
            lora_alpha=int(l_cfg["alpha"]),
            lora_dropout=float(l_cfg["dropout"]),
            target_modules=targets,
            bias="none",
        ),
    )
    transformer.enable_gradient_checkpointing()
    enable_input_grads = getattr(transformer, "enable_input_require_grads", None)
    if callable(enable_input_grads):
        enable_input_grads()
    transformer.to(info.device)

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    rank_print(info, f"Trainable VPTD LoRA parameters: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    if info.world_size > 1:
        transformer = DDP(
            transformer,
            device_ids=[info.local_rank],
            output_device=info.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    scheduler = FlowMapDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        shift=float(m_cfg["flow_shift"]),
        weight_type="beta08",
    )
    return AnyFlowRuntime(
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        num_train_timesteps=int(scheduler.config.num_train_timesteps),
        vae_scale_factor_temporal=int(vae.config.scale_factor_temporal),
        vae_scale_factor_spatial=int(vae.config.scale_factor_spatial),
    )


@torch.no_grad()
def encode_prompt(
    runtime: AnyFlowRuntime,
    prompt: str,
    cfg: dict[str, Any],
    info: DistInfo,
) -> torch.Tensor:
    from far.pipelines.pipeline_wan_anyflow import prompt_clean

    max_length = int(cfg["model"]["max_sequence_length"])
    cleaned_prompt = prompt_clean(prompt)
    inputs = runtime.tokenizer(
        [cleaned_prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = inputs.input_ids.to(info.device)
    mask = inputs.attention_mask.to(info.device)
    sequence_length = int(mask[0].sum())
    hidden = runtime.text_encoder(ids, mask).last_hidden_state
    valid = hidden[0, :sequence_length]
    padded = torch.cat(
        [valid, valid.new_zeros(max_length - valid.shape[0], valid.shape[1])], dim=0
    )
    return padded.unsqueeze(0).to(info.device, torch.bfloat16)


def initial_noise(
    runtime: AnyFlowRuntime,
    cfg: dict[str, Any],
    info: DistInfo,
    *,
    seed: int,
) -> torch.Tensor:
    m_cfg = cfg["model"]
    latent_frames = (
        (int(m_cfg["num_frames"]) - 1) // runtime.vae_scale_factor_temporal + 1
    )
    shape = (
        1,
        latent_frames,
        int(unwrap(runtime.transformer).config.in_channels),
        int(m_cfg["height"]) // runtime.vae_scale_factor_spatial,
        int(m_cfg["width"]) // runtime.vae_scale_factor_spatial,
    )
    generator = torch.Generator(device=info.device).manual_seed(int(seed))
    return torch.randn(
        shape,
        generator=generator,
        device=info.device,
        dtype=torch.float32,
    ).to(torch.bfloat16)


def expand_time(time_value: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return time_value.reshape(1, 1).expand(latents.shape[0], latents.shape[1]).contiguous()


def flow_map(
    model: torch.nn.Module,
    latents: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    num_train_timesteps: int,
    enable_grad: bool,
) -> torch.Tensor:
    context = nullcontext() if enable_grad else torch.no_grad()
    with context:
        source = expand_time(source_time, latents)
        target = expand_time(target_time, latents)
        embeds = prompt_embeds.expand(latents.shape[0], -1, -1)
        velocity = model(
            hidden_states=latents,
            timestep=source,
            r_timestep=target,
            encoder_hidden_states=embeds,
            return_dict=False,
            is_causal=False,
        )[0]
        delta = (source_time - target_time) / float(num_train_timesteps)
        while delta.ndim < velocity.ndim:
            delta = delta.unsqueeze(-1)
        # Mirror FlowMapDiscreteScheduler.step exactly: perform scheduler
        # arithmetic in the timestep dtype, then cast back to the model dtype.
        return (latents - delta * velocity).to(velocity.dtype)


def schedule(runtime: AnyFlowRuntime, steps: int, info: DistInfo) -> torch.Tensor:
    runtime.scheduler.set_timesteps(int(steps), device=info.device)
    return append_data_endpoint(runtime.scheduler.timesteps.to(info.device))


def deterministic_rollout(
    runtime: AnyFlowRuntime,
    model: torch.nn.Module,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    time_grid: torch.Tensor,
) -> torch.Tensor:
    current = latents
    for index in range(time_grid.numel() - 1):
        current = flow_map(
            model,
            current,
            time_grid[index],
            time_grid[index + 1],
            prompt_embeds,
            num_train_timesteps=runtime.num_train_timesteps,
            enable_grad=False,
        )
    return current


def branch_policy(
    runtime: AnyFlowRuntime,
    model: torch.nn.Module,
    shared_state: torch.Tensor,
    prompt_embeds: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    *,
    enable_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    deterministic_target = flow_map(
        model,
        shared_state,
        source_time,
        target_time,
        prompt_embeds,
        num_train_timesteps=runtime.num_train_timesteps,
        enable_grad=enable_grad,
    )
    clean_endpoint = flow_map(
        model,
        deterministic_target,
        target_time,
        target_time.new_zeros(()),
        prompt_embeds,
        num_train_timesteps=runtime.num_train_timesteps,
        enable_grad=enable_grad,
    )
    mean, std = endpoint_anchor_parameters(
        clean_endpoint,
        target_time,
        num_train_timesteps=runtime.num_train_timesteps,
    )
    return mean, std, deterministic_target


def complete_from_action(
    runtime: AnyFlowRuntime,
    model: torch.nn.Module,
    action: torch.Tensor,
    prompt_embeds: torch.Tensor,
    time_grid: torch.Tensor,
    branch_index: int,
) -> torch.Tensor:
    current = action
    for index in range(branch_index + 1, time_grid.numel() - 1):
        current = flow_map(
            model,
            current,
            time_grid[index],
            time_grid[index + 1],
            prompt_embeds,
            num_train_timesteps=runtime.num_train_timesteps,
            enable_grad=False,
        )
    return current


@torch.no_grad()
def decode_latents(runtime: AnyFlowRuntime, latents: torch.Tensor) -> torch.Tensor:
    z = latents.permute(0, 2, 1, 3, 4).contiguous().to(runtime.vae.dtype)
    mean = torch.tensor(
        runtime.vae.config.latents_mean, device=z.device, dtype=z.dtype
    ).view(1, -1, 1, 1, 1)
    std_inverse = (
        1.0
        / torch.tensor(runtime.vae.config.latents_std, device=z.device, dtype=z.dtype)
    ).view(1, -1, 1, 1, 1)
    z = z / std_inverse + mean
    video = runtime.vae.decode(z, return_dict=False)[0]
    return (video.float() / 2.0 + 0.5).clamp(0.0, 1.0)


def load_reward_scorer(keys: Sequence[str], cfg: dict[str, Any], info: DistInfo):
    from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer

    return build_multi_reward_scorer(
        {str(key): 1.0 for key in keys},
        backend=str(cfg["reward"].get("backend", "genrl")),
        device=info.device,
    )


def score_video(scorer: Any, video: torch.Tensor, prompt: str) -> dict[str, torch.Tensor]:
    scores = scorer(video.detach().cpu(), [prompt])
    return {name: value.detach().float() for name, value in scores.items()}


def media_to_wandb(video: torch.Tensor) -> np.ndarray:
    return (
        (video[0].detach().float().clamp(0, 1) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 0, 2, 3)
        .cpu()
        .numpy()
    )


def init_wandb(cfg: dict[str, Any], info: DistInfo):
    if not info.is_main:
        return None
    import wandb

    output = Path(cfg["experiment"]["output_dir"]) / "wandb"
    output.mkdir(parents=True, exist_ok=True)
    mode = str(cfg["wandb"].get("mode", "online"))
    if not (os.environ.get("WANDB_API_KEY") or "").strip():
        mode = "offline"
    return wandb.init(
        project=str(cfg["wandb"]["project"]),
        entity=cfg["wandb"].get("entity") or None,
        name=str(cfg["experiment"]["run_name"]),
        dir=str(output),
        mode=mode,
        config={**cfg, "hyperparameter_provenance": provenance_table()},
    )


def log_wandb(run: Any, payload: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(payload, step=int(step))


def parity_preflight(
    runtime: AnyFlowRuntime,
    cfg: dict[str, Any],
    info: DistInfo,
    prompt: str,
    run: Any,
) -> None:
    """Abort unless our deterministic sampler matches released AnyFlow code."""

    if not info.is_main:
        barrier(info)
        return
    from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
    import wandb

    model = unwrap(runtime.transformer)
    model.eval()
    embeds = encode_prompt(runtime, prompt, cfg, info)
    seed = int(cfg["experiment"]["seed"])
    noise = initial_noise(runtime, cfg, info, seed=seed)
    eval_steps = int(cfg["model"]["eval_map_steps"])
    time_grid = schedule(runtime, eval_steps, info)
    custom = deterministic_rollout(runtime, model, noise.clone(), embeds, time_grid)

    pipeline = WanAnyFlowPipeline(
        tokenizer=runtime.tokenizer,
        text_encoder=runtime.text_encoder,
        transformer=model,
        vae=runtime.vae,
        scheduler=runtime.scheduler,
        use_mean_velocity=True,
    ).to(info.device)
    official = pipeline.training_rollout(
        num_inference_steps=eval_steps,
        latents=noise.clone(),
        prompt_embeds=embeds,
        guidance_scale=float(cfg["model"]["guidance_scale"]),
    )
    max_abs = float((custom.float() - official.float()).abs().max().cpu())
    try:
        # Use PyTorch's dtype-specific default tolerances rather than inventing a
        # custom numerical threshold.
        torch.testing.assert_close(custom, official)
    except AssertionError as exc:
        raise RuntimeError(
            "VPTD sampler does not match released AnyFlow inference; "
            f"max_abs_error={max_abs:.6g}"
        ) from exc

    video = decode_latents(runtime, custom)
    if not torch.isfinite(video).all():
        raise RuntimeError("AnyFlow parity preflight decoded non-finite video")
    if int(video.shape[2]) != int(cfg["model"]["num_frames"]):
        raise RuntimeError("AnyFlow parity preflight decoded the wrong frame count")
    log_wandb(
        run,
        {
            "preflight/max_abs_latent_parity_error": max_abs,
            "preflight/temporal_delta_l1": float(temporal_l1(video).mean()),
            "preflight/base_video": wandb.Video(
                media_to_wandb(video),
                fps=int(cfg["model"]["fps"]),
                format="mp4",
                caption=f"released AnyFlow four-step base | {prompt}",
            ),
        },
        0,
    )
    del pipeline, custom, official, video, noise, embeds
    torch.cuda.empty_cache()
    barrier(info)


def validation(
    runtime: AnyFlowRuntime,
    cfg: dict[str, Any],
    info: DistInfo,
    prompts: Sequence[str],
    ema: LoRAEMA,
    run: Any,
    step: int,
) -> None:
    if not info.is_main:
        barrier(info)
        return
    import wandb

    model = unwrap(runtime.transformer)
    model.eval()
    scorer = load_reward_scorer(list(cfg["reward"]["validation"]), cfg, info)
    time_grid = schedule(runtime, int(cfg["model"]["eval_map_steps"]), info)
    metric_lists: dict[str, list[float]] = {}
    videos: list[Any] = []
    qualitative_count = int(cfg["prompts"]["qualitative_count"])
    seed = int(cfg["experiment"]["seed"])

    with ema_scope(ema, model), torch.no_grad():
        for index, prompt in enumerate(prompts):
            embeds = encode_prompt(runtime, prompt, cfg, info)
            initial = initial_noise(runtime, cfg, info, seed=seed + index)
            endpoint = deterministic_rollout(runtime, model, initial, embeds, time_grid)
            media = decode_latents(runtime, endpoint)
            scores = score_video(scorer, media, prompt)
            scores["temporal_delta_l1"] = temporal_l1(media).cpu()
            for name, value in scores.items():
                metric_lists.setdefault(name, []).append(float(value.reshape(-1)[0]))
            if index < qualitative_count:
                caption = " | ".join(
                    f"{name}: {float(value.reshape(-1)[0]):.4f}"
                    for name, value in scores.items()
                )
                videos.append(
                    wandb.Video(
                        media_to_wandb(media),
                        fps=int(cfg["model"]["fps"]),
                        format="mp4",
                        caption=f"{caption} | {prompt[:600]}",
                    )
                )
            del embeds, initial, endpoint, media

    payload: dict[str, Any] = {
        "validation/num_prompts": len(prompts),
        "validation/videos": videos,
    }
    for name, values in metric_lists.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        payload[f"validation/reward/{name}"] = float(tensor.mean())
        payload[f"validation/reward_std/{name}"] = float(tensor.std(unbiased=False))
    log_wandb(run, payload, step)
    del scorer
    torch.cuda.empty_cache()
    barrier(info)


def find_adapter_file(root: Path) -> Path:
    candidates = sorted(root.rglob("adapter_model.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one adapter_model.safetensors under {root}, found {len(candidates)}")
    return candidates[0]


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resolve_auto_resume(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def maybe_resume(
    runtime: AnyFlowRuntime,
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    ema: LoRAEMA,
    info: DistInfo,
) -> int:
    resume = cfg["experiment"].get("resume")
    if resume == "auto":
        resume_path = resolve_auto_resume(Path(cfg["experiment"]["output_dir"]))
        if resume_path is None:
            return 0
    elif resume:
        resume_path = Path(str(resume))
    else:
        return 0

    state_path = resume_path / "training_state.pt" if resume_path.is_dir() else resume_path
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter = load_file(str(find_adapter_file(state_path.parent / "lora")), device="cpu")
    set_peft_model_state_dict(unwrap(runtime.transformer), adapter)
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_device(optimizer, info.device)
    ema.load_state_dict(state["ema"])
    rank_print(info, f"Resumed VPTD from {state_path} at step {state['step']}")
    return int(state["step"])


def save_checkpoint(
    runtime: AnyFlowRuntime,
    cfg: dict[str, Any],
    info: DistInfo,
    optimizer: torch.optim.Optimizer,
    ema: LoRAEMA,
    step: int,
) -> None:
    barrier(info)
    if info.is_main:
        root = Path(cfg["experiment"]["output_dir"]) / f"checkpoint-{step}"
        root.mkdir(parents=True, exist_ok=True)
        model = unwrap(runtime.transformer)
        model.save_pretrained(root / "lora", safe_serialization=True)
        with ema_scope(ema, model):
            model.save_pretrained(root / "ema_lora", safe_serialization=True)
        torch.save(
            {
                "step": int(step),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "config": cfg,
            },
            root / "training_state.pt",
        )
        (root / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    barrier(info)


def train_one_step(
    runtime: AnyFlowRuntime,
    cfg: dict[str, Any],
    info: DistInfo,
    optimizer: torch.optim.Optimizer,
    ema: LoRAEMA,
    scorer: Any,
    prompt: str,
    step: int,
) -> dict[str, float]:
    policy_cfg = cfg["posterior_policy"]
    local_group = verify_group_partition(int(policy_cfg["group_size"]), info.world_size)
    model = runtime.transformer
    model.eval()
    embeds = encode_prompt(runtime, prompt, cfg, info)
    time_grid = schedule(runtime, int(cfg["model"]["train_map_steps"]), info)
    validate_training_schedule(time_grid, stochastic_steps=int(policy_cfg["stochastic_steps"]))

    if info.is_main:
        generator = torch.Generator(device=info.device).manual_seed(
            int(cfg["experiment"]["seed"]) + int(step)
        )
        selected = int(
            torch.randint(
                0,
                int(policy_cfg["stochastic_steps"]),
                (1,),
                generator=generator,
                device=info.device,
            ).item()
        )
    else:
        selected = 0
    branch_index = broadcast_int(selected, info)

    shared_seed = int(cfg["experiment"]["seed"]) + 10_000_000 + int(step)
    shared_state = initial_noise(runtime, cfg, info, seed=shared_seed)
    with torch.no_grad():
        for index in range(branch_index):
            shared_state = flow_map(
                model,
                shared_state,
                time_grid[index],
                time_grid[index + 1],
                embeds,
                num_train_timesteps=runtime.num_train_timesteps,
                enable_grad=False,
            )
        old_mean, old_std, deterministic_target = branch_policy(
            runtime,
            model,
            shared_state,
            embeds,
            time_grid[branch_index],
            time_grid[branch_index + 1],
            enable_grad=False,
        )

    action_chunks: list[torch.Tensor] = []
    log_prob_chunks: list[torch.Tensor] = []
    reward_chunks: list[torch.Tensor] = []
    temporal_chunks: list[torch.Tensor] = []
    for local_index in range(local_group):
        branch_seed = (
            int(cfg["experiment"]["seed"])
            + int(step) * 100_000
            + info.rank * 1_000
            + local_index
        )
        branch_generator = torch.Generator(device=info.device).manual_seed(branch_seed)
        with torch.no_grad():
            action, _ = sample_diagonal_gaussian(
                old_mean, old_std, generator=branch_generator
            )
            old_log_prob = gaussian_log_prob_mean(action, old_mean, old_std)
            endpoint = complete_from_action(
                runtime, model, action, embeds, time_grid, branch_index
            )
            media = decode_latents(runtime, endpoint)
            scores = score_video(scorer, media, prompt)
            reward = scores[str(cfg["reward"]["optimize"])].to(info.device).reshape(-1)
            action_chunks.append(action.detach())
            log_prob_chunks.append(old_log_prob.detach())
            reward_chunks.append(reward.detach())
            temporal_chunks.append(temporal_l1(media).to(info.device).detach())
            del endpoint, media, scores

    actions = torch.cat(action_chunks, dim=0)
    old_log_probs = torch.cat(log_prob_chunks, dim=0)
    local_rewards = torch.cat(reward_chunks, dim=0)
    local_temporal = torch.cat(temporal_chunks, dim=0)
    global_rewards = all_gather_1d(local_rewards, info)
    global_temporal = all_gather_1d(local_temporal, info)
    advantages, reward_mean, reward_std = global_group_advantages(
        global_rewards,
        epsilon=float(policy_cfg["advantage_epsilon"]),
        clip=float(policy_cfg["advantage_clip"]),
    )
    global_weights, temperature, ess = reward_tilted_weights(
        global_rewards,
        target_ess_ratio=float(policy_cfg["target_ess_ratio"]),
    )
    local_start = info.rank * local_group
    local_end = local_start + local_group
    local_advantages = advantages[local_start:local_end].to(info.device)
    local_weights = global_weights[local_start:local_end].to(info.device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    new_mean, new_std, _ = branch_policy(
        runtime,
        model,
        shared_state.detach(),
        embeds.detach(),
        time_grid[branch_index],
        time_grid[branch_index + 1],
        enable_grad=True,
    )
    expanded_mean = new_mean.expand(actions.shape[0], *new_mean.shape[1:])
    expanded_std = new_std.expand(actions.shape[0], *new_std.shape[1:])
    new_log_probs = gaussian_log_prob_mean(actions, expanded_mean, expanded_std)

    if str(policy_cfg["objective"]) == "posterior_distillation":
        loss, diagnostics = posterior_distillation_loss(
            new_log_probs,
            local_weights,
            distributed_world_size=info.world_size,
        )
    else:
        loss, diagnostics = clipped_grpo_loss(
            new_log_probs,
            old_log_probs,
            local_advantages,
            clip_range=float(policy_cfg["clip_range"]),
        )

    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, float(cfg["optimizer"]["max_grad_norm"])
    )
    optimizer.step()
    ema.update(unwrap(model), step)

    action_deviation = (
        actions.float() - deterministic_target.expand_as(actions).float()
    ).square().mean().sqrt()
    safe_weights = global_weights.clamp_min(1.0e-12)
    metrics: dict[str, float] = {
        "train/loss": float(reduce_mean(loss, info)),
        "train/grad_norm": float(
            reduce_mean(torch.as_tensor(grad_norm, device=info.device), info)
        ),
        "train/reward_mean": float(reward_mean),
        "train/reward_std": float(reward_std),
        "train/reward_min": float(global_rewards.min()),
        "train/reward_max": float(global_rewards.max()),
        "train/advantage_abs_mean": float(advantages.abs().mean()),
        "train/zero_std_group": float(
            reward_std < float(policy_cfg["advantage_epsilon"])
        ),
        "train/transition_index": float(branch_index),
        "train/source_timestep": float(time_grid[branch_index]),
        "train/target_timestep": float(time_grid[branch_index + 1]),
        "train/posterior_std": float(old_std.float().mean()),
        "train/posterior_action_deviation": float(reduce_mean(action_deviation, info)),
        "train/temporal_delta_l1": float(global_temporal.mean()),
        "train/group_size": float(global_rewards.numel()),
        "train/reward_temperature": (
            float(temperature) if torch.isfinite(temperature) else 0.0
        ),
        "train/reward_temperature_is_infinite": float(
            not bool(torch.isfinite(temperature))
        ),
        "train/posterior_ess": float(ess),
        "train/posterior_ess_ratio": float(ess / global_rewards.numel()),
        "train/posterior_weight_max": float(global_weights.max()),
        "train/posterior_weight_entropy": float(-(safe_weights * safe_weights.log()).sum()),
        "train/objective_is_distillation": float(
            str(policy_cfg["objective"]) == "posterior_distillation"
        ),
    }
    for name, value in diagnostics.items():
        metrics[f"train/{name}"] = float(reduce_mean(value, info))

    del (
        actions,
        old_log_probs,
        local_rewards,
        global_rewards,
        shared_state,
        embeds,
    )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    info = init_distributed(int(cfg["experiment"]["seed"]))
    output_dir = Path(cfg["experiment"]["output_dir"])
    if info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    barrier(info)

    train_prompts, validation_prompts = prepare_prompt_split(cfg, info)
    checkpoint = ensure_videoalign_checkpoint(info)
    rank_print(info, f"VideoAlign checkpoint: {checkpoint}")
    runtime = load_anyflow_runtime(cfg, info)

    optimizer_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in runtime.transformer.parameters() if p.requires_grad],
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(x) for x in optimizer_cfg["betas"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    ema = LoRAEMA(
        unwrap(runtime.transformer),
        decay=float(optimizer_cfg["ema_decay"]),
        warmup_steps=int(optimizer_cfg["ema_warmup_steps"]),
    )
    start_step = maybe_resume(runtime, cfg, optimizer, ema, info)
    run = init_wandb(cfg, info)
    scorer = load_reward_scorer([str(cfg["reward"]["optimize"])], cfg, info)

    parity_preflight(runtime, cfg, info, validation_prompts[0], run)
    validation(runtime, cfg, info, validation_prompts, ema, run, start_step)
    if bool(cfg["experiment"].get("preflight_only", False)):
        if info.is_main and run is not None:
            run.finish()
        barrier(info)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    max_steps = int(cfg["experiment"]["max_train_steps"])
    save_every = int(cfg["experiment"]["save_every"])
    eval_every = int(cfg["experiment"]["eval_every"])
    log_every = int(cfg["experiment"]["log_every"])
    shuffled = list(train_prompts)
    random.Random(int(cfg["experiment"]["seed"])).shuffle(shuffled)

    for step in range(start_step + 1, max_steps + 1):
        prompt = shuffled[(step - 1) % len(shuffled)]
        started = time.perf_counter()
        metrics = train_one_step(
            runtime, cfg, info, optimizer, ema, scorer, prompt, step
        )
        metrics["train/step_time_sec"] = time.perf_counter() - started
        if info.is_main and step % log_every == 0:
            metrics["train/prompt"] = prompt
            log_wandb(run, metrics, step)
            print(
                f"step={step} loss={metrics['train/loss']:.6f} "
                f"reward={metrics['train/reward_mean']:.4f}±{metrics['train/reward_std']:.4f} "
                f"ess={metrics['train/posterior_ess_ratio']:.3f} "
                f"transition={int(metrics['train/transition_index'])} "
                f"time={metrics['train/step_time_sec']:.1f}s",
                flush=True,
            )
        if step % save_every == 0:
            save_checkpoint(runtime, cfg, info, optimizer, ema, step)
        if step % eval_every == 0:
            validation(runtime, cfg, info, validation_prompts, ema, run, step)

    if max_steps % save_every:
        save_checkpoint(runtime, cfg, info, optimizer, ema, max_steps)
    if max_steps % eval_every:
        validation(runtime, cfg, info, validation_prompts, ema, run, max_steps)
    if info.is_main and run is not None:
        run.finish()
    barrier(info)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
