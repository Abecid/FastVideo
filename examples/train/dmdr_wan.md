# DMDR Wan RL Distillation

DMDR jointly trains a few-step student with reward optimization and
distribution-matching regularization. This FastVideo implementation is a
DMDR-style baseline: it combines the existing DiffusionNFT-style grouped reward
objective with a DMD fake-score critic and frozen teacher, while staying inside
the modular `fastvideo/train` RL method structure.

This is not a bit-for-bit reproduction of the public SiT/ImageNet DMDR demo.
The current method does not yet implement the paper's dynamic distribution
guidance or dynamic re-noise sampling schedules; those should be added as
explicit `method` / `method.sampling` knobs once we have a Wan experiment that
needs them.

## Related Work

- DMDR, "Distribution Matching Distillation Meets Reinforcement Learning",
  jointly optimizes reward-tilted distribution matching and RL so the distilled
  few-step generator is not capped by the teacher.
- RTDMD frames the target as a reward-tilted teacher distribution and uses a
  two-stage DMD plus RL recipe for few-step flow generators.
- R_dm / GNDMR treats distribution matching itself as a reward, then combines it
  with external rewards using group normalization.
- AdvDMD uses the DMD2 discriminator as an adversarial reward for few-step
  generators.

## VideoAlign Video Launch

For the 4x A100 video experiment, use the tracked prep script first. It writes
the prompt text file, runs text-only Wan preprocessing, downloads the
`KwaiVGI/VideoReward` checkpoint, optionally preflights the reward model, and
writes a resolved run YAML:

```bash
python examples/train/prepare_dmdr_assets.py \
    --config examples/train/configs/rl/wan/dmdr_videoalign.yaml \
    --data-root data/dmdr \
    --cache-root .cache/dmdr \
    --output-dir outputs/wan2.1_dmdr_videoalign \
    --num-frames 49 \
    --max-train-steps 100 \
    --check-rewards \
    --json
```

Then launch the generated config:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    outputs/dmdr_run_configs/dmdr_wan_run.yaml
```

Our lab's Modal launcher is a convenience wrapper around those two commands:

```bash
conda run --no-capture-output -n fastvideo modal run modal_train_dmdr.py
```

The config uses five roles:

- `student`: trainable few-step generator.
- `old`: EMA-style old policy used by the reward-policy loss.
- `reference`: frozen KL reference for RL stability.
- `teacher`: frozen multi-step teacher used for real-score DMD guidance.
- `critic`: trainable fake-score estimator used by DMD.

The key method weights are `method.rl_loss_weight`,
`method.dmd_loss_weight`, and `method.fake_score_loss_weight`. Start with the
checked-in video values for a smoke run, then tune `dmd_loss_weight` downward if
reward curves improve while visual diversity collapses.

The critical W&B signals are `reward/avg`, `reward/videoalign_*`,
`reward_std_mean`, `zero_std_ratio`, `policy_loss`, `kl_div_loss`, `dmd_loss`,
`fake_score_loss`, `dmdr/optimizer_steps`, and `validation/reward/*`. Validation
runs every `method.validation.every_steps` outer epochs and logs videos when
`method.validation.log_samples` is true.

## Single-Frame Smoke Launch

The PickScore/CLIPScore config remains useful for cheap algorithm debugging:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    examples/train/configs/rl/wan/dmdr_pick_clip.yaml
```
