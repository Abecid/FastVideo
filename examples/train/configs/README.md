# Training Configs

Single-step training configurations organized by method and model.

```
configs/
├── fine_tuning/          # Standard finetuning and DFSFT
├── distribution_matching/  # DMD2 and Self-Forcing
├── knowledge_distillation/ # KD from teacher to student
├── rl/                   # RL methods such as DiffusionNFT and DMDR
└── example.yaml          # Annotated reference config with all fields
```

Each method directory contains per-model subdirectories (e.g. `wan/`, `hunyuan/`).

RL configs under `configs/rl/wan/` include:

- `diffusion_nft_pick_clip.yaml`: single-frame DiffusionNFT reward tuning.
- `dmdr_pick_clip.yaml`: single-frame DMDR algorithm smoke test.
- `dmdr_videoalign.yaml`: video DMDR run with VideoReward/VideoAlign VQ, MQ,
  and TA rewards.

Launch any config with:

```bash
bash examples/train/run.sh examples/train/configs/<method>/<model>/<config>.yaml
```

For multi-step training pipelines, see `examples/train/scenario/`.
