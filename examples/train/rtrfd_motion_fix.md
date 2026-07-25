# Motion-preserving RTFD follow-up

The first 100-step `wan2.1_rtfd_tilted` run completed, but the generated videos
became nearly static while text alignment improved. This follow-up isolates
three concrete causes in the original endpoint-only experiment.

## Why motion collapsed

1. **The teacher transport coupling was discarded.** The teacher generated
   `x0 = T(z)`, but training paired `x0` with a new independent noise tensor.
   Deterministic MSE flow matching then learned a conditional mean across
   incompatible video motions.

2. **The four-step `flow_shift=8` grid was extremely unbalanced.** Its interval
   weights are approximately `[0.059, 0.141, 0.793, 0.008]`, so one Euler update
   must traverse almost four-fifths of the path.

3. **Raw VQ/MQ/TA scores were summed before reward tilting.** Different reward
   scales and variances can let text alignment improve while motion quality is
   traded away.

The W&B player also used `fps=1`, which made visual motion harder to inspect.

## Default correction

`RewardTiltedReflowDistillationMethod` now uses:

- `coupling_mode: teacher_noise`: pair every teacher endpoint with the exact
  Gaussian noise that generated it;
- `student_flow_shift: 1.0`: use a balanced four-step student schedule while
  leaving the 16-step teacher at shift 8;
- `reward_aggregation: component_zscore`: normalize VQ, MQ, and TA inside each
  prompt before scalarization;
- eight fixed validation prompts and eight W&B videos at 8 fps;
- `temporal_delta_l1` as a cheap model-independent static-video diagnostic.

The original behavior remains available through:

```yaml
coupling_mode: independent
student_flow_shift: 8.0
reward_aggregation: raw_sum
```

## Smoke run

```bash
modal run examples/train/modal_rtfd.py \
  --max-steps 5 \
  --num-frames 17 \
  --height 256 \
  --width 448 \
  --max-prompts 32 \
  --validation-num-prompts 8 \
  --validation-max-log-videos 8 \
  --validation-fps 8 \
  --run-name wan2.1_rtrfd_motion_smoke
```

## First meaningful comparison

Motion-preserving no-reward baseline:

```bash
modal run examples/train/modal_rtfd.py \
  --max-steps 100 \
  --uniform-mix 1.0 \
  --coupling-mode teacher_noise \
  --student-flow-shift 1.0 \
  --reward-aggregation component_zscore \
  --run-name wan2.1_rtrfd_uniform
```

Reward-tilted run:

```bash
modal run examples/train/modal_rtfd.py \
  --max-steps 100 \
  --uniform-mix 0.25 \
  --ess-ratio 0.60 \
  --coupling-mode teacher_noise \
  --student-flow-shift 1.0 \
  --reward-aggregation component_zscore \
  --run-name wan2.1_rtrfd_tilted
```

For strict attribution, also run the two one-factor ablations:

```bash
# Coupling fix only
modal run examples/train/modal_rtfd.py \
  --max-steps 30 \
  --uniform-mix 1.0 \
  --coupling-mode teacher_noise \
  --student-flow-shift 8.0 \
  --reward-aggregation raw_sum \
  --run-name wan2.1_rtrfd_coupling_only

# Schedule fix only, reproducing the independent endpoint coupling
modal run examples/train/modal_rtfd.py \
  --max-steps 30 \
  --uniform-mix 1.0 \
  --coupling-mode independent \
  --student-flow-shift 1.0 \
  --reward-aggregation raw_sum \
  --run-name wan2.1_rtrfd_schedule_only
```

## Metrics to inspect first

- `validation/reward/videoalign_mq`
- `validation/reward/temporal_delta_l1`
- `rtfd/teacher_reward/temporal_delta_l1`
- `rtfd/selection/videoalign_mq_gain`
- `rtfd/schedule/interval_weight_*`
- the eight fixed W&B videos at every validation step

Do not interpret an aggregate reward increase as a success when MQ and
`temporal_delta_l1` both decrease.
