# DMDR Wan RL Distillation

DMDR jointly trains a few-step student with reward optimization and
distribution-matching regularization. The FastVideo method follows the public
SiT/ImageNet DMDR training loop at the algorithmic level while staying inside
the modular `fastvideo/train` RL structure:

- train a fake-score/guidance estimator every inner step,
- update the few-step student every `method.guidance_update_ratio` inner steps,
- run a DMD-only cold start before reward optimization turns on,
- anneal beta-distributed DMD/guidance timesteps with `method.dynamic_step`,
- cosine-decay the real-score guidance scale during the dynamic phase.

Wan does not currently expose the SiT reference's per-forward LoRA scale hook,
so `method.real_score_guidance_scale` is implemented as decayed CFG guidance on
the real-score path. Treat that as the main remaining architecture-specific
gap to close before claiming bit-for-bit method parity.

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

For a short 4x A100 smoke experiment, use the tracked prep script first. It writes
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
    --cold-start-steps 0 \
    --check-rewards \
    --json
```

Then launch the generated config:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    outputs/dmdr_run_configs/dmdr_wan_run.yaml
```

Provider-specific launchers should remain local convenience wrappers around
those two tracked commands:

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

For a research run, follow the reference two-stage shape:

```bash
python examples/train/prepare_dmdr_assets.py \
    --config examples/train/configs/rl/wan/dmdr_videoalign.yaml \
    --data-root data/dmdr \
    --cache-root .cache/dmdr \
    --output-dir outputs/wan2.1_dmdr_cold_start \
    --num-frames 49 \
    --max-train-steps 20000 \
    --cold-start-steps 20000 \
    --dynamic-step 10000 \
    --guidance-update-ratio 5 \
    --check-rewards \
    --json
```

Resume from the cold-start checkpoint for the reward-active stage, keeping
`cold_start_steps` equal to the global iteration where reward should turn on.
The generated YAML should be the source of truth; Modal, Slurm, or any other
launcher should only call `examples/train/prepare_dmdr_assets.py` and
`examples/train/run.sh`.

The critical W&B signals are `reward/avg`, `reward/videoalign_*`,
`reward_std_mean`, `zero_std_ratio`, `policy_loss`, `kl_div_loss`, `dmd_loss`,
`fake_score_loss`, `dmdr/student_optimizer_steps`,
`dmdr/critic_optimizer_steps`, `dmdr/reward_active`, and
`validation/reward/*`. Validation runs every
`method.validation.every_steps` outer epochs and logs videos when
`method.validation.log_samples` is true.

## Single-Frame Smoke Launch

The PickScore/CLIPScore config remains useful for cheap algorithm debugging:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    examples/train/configs/rl/wan/dmdr_pick_clip.yaml
```
