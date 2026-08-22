# Reliable finite-transition RL: implementation and run protocol

Read this before launching a new AnyFlow reward-alignment experiment.

The corrected 200-step FTPP/GRPO run was stable but neither arm improved held-out
VideoAlign MQ. The result did not isolate a posterior-specific failure: both arms
used only four candidate videos, one selected transition, one prompt group per
optimizer update, four-sample reward normalization, a very small effective
update, EMA-only validation, and an unaudited compatibility path for VideoAlign.

This follow-up repairs that shared substrate before another novel-method claim is
attempted.

## What changed

The new method is:

```text
fastvideo/train/methods/rl/finite_transition_reliable.py
```

It supports three objectives:

```text
flowmap_grpo
posterior_projection
finite_velocity_regression
```

The first two share the same full-trajectory rollout implementation. The third
is a genuinely different target-space update and must be treated as a separate
method.

### 1. Full stochastic trajectory

The reliable likelihood recipe uses an AnyFlow five-segment training rollout:

```text
four local-ASFMC stochastic finite transitions
one deterministic completion to x_0
```

Every stochastic transition contributes a log-probability loss. Deterministic
evaluation remains the released four-step AnyFlow sampler.

This is intentional. Exact AnyFlow schedule construction remains mandatory, but
a two-time flow map is designed to generalize across `(t, r)` pairs. The earlier
rule requiring train and evaluation grids to match applies only to the archived
single-transition deployment-grid experiment, not to this fuller Flow-Map-GRPO
baseline.

### 2. More reward evidence per optimizer update

The default reliable run uses:

```text
group size: 8
rollout prompt groups per update: 4
reward-scored videos per update: 32
stochastic transition likelihoods per update: 128
```

Four H100s each hold two candidate trajectories. Prompt groups are processed
sequentially and their gradients are accumulated. Transition graphs are also
backpropagated one at a time, so the implementation does not retain four video
transformer graphs simultaneously.

### 3. Stable reward normalization

The old implementation normalized every four-candidate group by its own noisy
standard deviation. The new default uses:

```text
running per-prompt reward mean
running global reward standard deviation
```

A prompt falls back to the running global mean until it has enough observations.
The global scale falls back to the current group only during initial warmup.

For posterior weighting, the default temperature is proportional to the running
global reward standard deviation. Therefore a nearly flat reward group remains
nearly uniform and causes a weak update. `fixed_ess` remains available only as an
explicit ablation.

### 4. Target-KL update calibration

The old post-update KL was around `1e-7`, which is consistent with both models
barely moving. The reliable method contains a multiplicative loss-scale
controller:

```text
measured KL below target -> increase effective loss scale
measured KL above target -> decrease effective loss scale
```

The default target is `1e-5`. The controller changes scale by at most 2x per
optimizer update and is bounded. This is not claimed as a universal optimal KL;
the launcher provides a calibration sweep over `1e-6`, `1e-5`, and `1e-4`.

### 5. Exact shared behavior for objective-only comparisons

Standalone GRPO uses the current policy for on-policy collection.

The paired launcher instead sets:

```yaml
behavior_policy: base_adapter_disabled
```

for both arms. LoRA is disabled only during rollout generation, so prompts,
source noise, local-ASFMC means, action noise, candidate actions, completed
videos, and rewards remain identical even after the learners diverge. The two
arms differ only in the update coefficients/loss.

This fixed-behavior comparison is off-policy by construction, so it is for
isolating the optimizer—not for claiming it is the best online RL recipe.

### 6. Raw and EMA paired validation

Every validation checkpoint now evaluates:

```text
raw LoRA weights
EMA weights
```

using exact fixed `(prompt_index, sample_seed)` identities. It saves all
per-sample scalar values to JSON and computes:

```text
paired mean delta
paired standard error
paired bootstrap confidence interval
positive-pair fraction
```

The success gate uses the EMA paired MQ delta, requires the absolute practical
threshold, and requires the lower paired confidence bound to exceed zero.

### 7. VideoAlign load audit

The audited reward names are:

```text
videoalign_mq_audited
videoalign_vq_audited
videoalign_ta_audited
```

They record shape-compatible state-dict coverage for the Qwen base, PEFT adapter,
and reward head. Training fails when no reward head was loaded or when coverage
falls below the configured threshold.

The preparation path also runs deterministic synthetic static, smooth-motion,
and flicker clips and records their scores plus the coverage report.

### 8. Deterministic-transfer probe

For single-transition experiments, the code measures whether the deterministic
AnyFlow transition moves in the direction preferred by the reward-weighted
candidate actions:

```text
reliable/deterministic_transfer_cosine
reliable/deterministic_shift_rms
```

This directly tests the assumption that optimizing the auxiliary stochastic
policy changes the deterministic map used at inference.

## Configurations

### Reliable GRPO / likelihood-posterior comparison

```text
examples/train/configs/rl/wan/
  finite_transition_reliable_anyflow_videoalign.yaml
```

Default:

```text
AnyFlow-Wan 1.3B
81 frames, 480x832
LoRA rank/alpha 64/128
group 8
4 rollout groups/update
4 stochastic transitions/trajectory
AdamW 2e-5, betas (0.9, 0.999), weight decay 1e-4
EMA 0.9
target KL 1e-5
```

### Finite-velocity posterior regression

```text
examples/train/configs/rl/wan/
  finite_transition_velocity_anyflow_videoalign.yaml
```

This uses the deployed four-step grid, selects one positive-target transition,
and constructs:

```text
c_j = w_j - 1/G
g_j = (x_t - a_j) / Delta
Delta_u = sum_j c_j g_j
u_target = u_reference + eta * Delta_u
```

`eta` is chosen so the deterministic transition shift has a configured RMS.
The model then receives ordinary finite-velocity regression supervision.

### Easy learnability gate

```text
examples/train/configs/rl/wan/
  finite_transition_reliable_sanity_luminance.yaml
```

This short, lower-resolution run optimizes mean luminance. It is not a research
result. It must show deterministic held-out reward improvement before a failed
VideoAlign run is blamed on subtle reward semantics.

## Required run order

### 1. Pull and run the focused gate

```bash
git switch adam/finite-transition-alignment
git pull --ff-only origin adam/finite-transition-alignment

pytest -q \
  fastvideo/tests/train/methods/test_anyflow_schedule.py \
  fastvideo/tests/train/methods/test_local_asfmc.py \
  fastvideo/tests/train/methods/test_reward_statistics.py \
  fastvideo/tests/train/methods/test_finite_transition_reliable.py \
  fastvideo/tests/train/methods/test_videoalign_audit.py
```

### 2. Real smoke

```bash
modal run modal_train_finite_transition_reliable.py --smoke
```

A smoke proves plumbing only.

### 3. Easy reward sanity gate

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe sanity \
  --comparison-id ftr_luminance_sanity_s42
```

Continue only when raw and EMA deterministic held-out luminance increase without
numerical instability.

### 4. Target-KL calibration

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --calibrate-kl \
  --comparison-id ftr_kl_calibration_s42
```

This launches 20-update GRPO probes targeting:

```text
1e-6
1e-5
1e-4
```

Select the largest target that remains stable and produces a coherent raw-model
deterministic reward response. Do not select solely from training reward.

### 5. Reliable GRPO baseline

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --objective flowmap_grpo \
  --max-train-steps 100 \
  --comparison-id ftr_reliable_grpo_s42
```

Do not compare a novel method until this baseline demonstrates learnability.

### 6. Strict objective-only pair

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --paired \
  --max-train-steps 100 \
  --comparison-id ftr_shared_behavior_pair_s42
```

Both arms use the frozen base behavior policy and therefore receive identical
rollout data.

### 7. Finite-velocity follow-up

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe velocity \
  --max-train-steps 100 \
  --comparison-id ftr_velocity_s42
```

Run this after the baseline works or when likelihood objectives remain tied.

## W&B metrics to inspect

### Training signal

```text
reliable/reward_mean
reliable/reward_std
reliable/reward_selection_gain
reliable/reward_baseline
reliable/reward_scale
reliable/posterior_ess
reliable/posterior_temperature
reliable/stochastic_transitions_per_trajectory
reliable/reward_samples_per_update
```

### Update size and stability

```text
reliable/loss_scale_used
reliable/loss_scale_next
reliable/target_kl
reliable/post_update_approx_kl
reliable/post_update_logprob_delta_abs
reliable/grad_norm
```

### Deterministic transfer

```text
reliable/deterministic_transfer_cosine
reliable/deterministic_shift_rms
```

### Raw/EMA held-out evidence

```text
validation_raw/reward/<reward>
validation_ema/reward/<reward>
paired_raw/reward/<reward>/mean_delta
paired_raw/reward/<reward>/ci_lower
paired_raw/reward/<reward>/ci_upper
paired_ema/reward/<reward>/mean_delta
paired_ema/reward/<reward>/ci_lower
paired_ema/reward/<reward>/ci_upper
paired_validation_success/all
```

The complete prompt-seed values are saved under:

```text
<output_dir>/paired_validation/
```

## Stop criteria

Stop or repair the common substrate when:

- the easy luminance reward does not improve deterministically;
- post-update KL remains below `1e-6` after controller warmup;
- update KL exceeds the selected trust region or gradients become non-finite;
- training reward improves while both raw and EMA paired held-out reward remain
  flat or negative;
- audited VideoAlign coverage or calibration fails;
- deterministic-transfer cosine is consistently non-positive in the
  single-transition probe.

Do not resume the old 200-step FTPP/GRPO checkpoints into this method. The
rollout structure, optimizer state, reward-normalizer state, EMA, and validation
state are different.
