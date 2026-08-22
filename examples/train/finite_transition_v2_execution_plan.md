# Finite-transition v2: execution and decision plan

Read this before launching or repairing the next AnyFlow reward-training run.

The corrected 200-step FTPP/GRPO experiment was numerically stable but neither
arm improved deterministic held-out VideoAlign MQ. The old recipe used only one
prompt group, four candidate videos, one selected transition, learning rate
`2e-6`, noisy group-local normalization, and EMA-only evaluation. Maximum
post-update KL was about `1e-7`: both policies barely moved.

V2 repairs the common substrate before asking which update rule is better.

## Authoritative files

Use only:

```text
modal_train_finite_transition_v2_complete.py
fastvideo/train/methods/rl/finite_transition_v2_final.py
examples/train/configs/rl/wan/finite_transition_*_v2_*.yaml
```

The older `finite_transition_reliable*` and original FTPP files are preserved for
history and ablation compatibility, not for new scientific jobs.

## What v2 changes

### Stronger rollout statistics

The GRPO and posterior presets use:

```text
8 candidate trajectories per prompt group
4 prompt groups per optimizer update
4 stochastic finite transitions per trajectory
32 reward-scored videos per optimizer update
128 transition likelihood records per optimizer update
```

Each transition is recomputed and backpropagated separately, so the code does
not retain four full video-transformer graphs simultaneously.

Training uses the official AnyFlow five-segment map:

```text
4 local-ASFMC stochastic transitions
1 deterministic completion to x_0
```

Deterministic evaluation remains the released four-step AnyFlow sampler.

### Reward statistics that retain confidence

GRPO uses a running prompt baseline and a reward scale pooled across the full
optimizer rollout batch. Posterior weighting uses one global/EMA temperature,
not forced per-group ESS.

A nearly flat group therefore stays nearly uniform and causes a weak centered
posterior update. `group_ess` remains an explicit ablation only.

### Explicit policy-movement calibration

Every optimizer step records:

```text
ftv2/post_update_approx_kl
ftv2/post_update_logprob_delta_abs
ftv2/loss_scale_before
ftv2/loss_scale_after
```

A conservative controller changes the loss scale toward a target KL. The
launcher also provides a controller-disabled learning-rate sweep at:

```text
2e-6
2e-5
6e-5
```

The `1e-5` to `1e-4` KL range is an initial diagnostic band, not a universal
constant. Select a setting from deterministic held-out behavior, not training
reward alone.

### Audited reward loading and upstream preprocessing

Scientific VideoAlign configs use:

```text
videoalign_mq_audited
videoalign_vq_audited
videoalign_ta_audited
```

The audit:

1. reads the actual full or adapter/non-LoRA checkpoint tensors;
2. compares keys, shapes, tensor counts, and parameter counts against the final
   runtime model;
3. reports base, adapter, and reward-head coverage;
4. requires a detected reward head;
5. checks audited parameters for non-finite values; and
6. runs a deterministic repeatability probe.

MQ preprocessing matches the upstream wrapper exactly: mean-channel grayscale
and an empty prompt. VQ uses color and an empty prompt. TA uses color and the
actual generation prompt.

Online training calls only MQ. VQ and TA are held out and evaluated only at fixed
validation checkpoints, eliminating two unnecessary Qwen passes per rollout.

### Raw and EMA paired validation

Every validation checkpoint evaluates both:

```text
raw LoRA weights
EMA weights
```

The evaluator stores exact `(prompt_index, sample_seed)` reward and motion values
under:

```text
<output_dir>/paired_validation/
  raw_step_XXXXXX_samples.json
  ema_step_XXXXXX_samples.json
```

Statistical confidence intervals use prompt-level means across fixed seeds to
avoid treating seeds from one prompt as independent prompts. JSON artifacts keep
every seed value for debugging and cross-arm analysis.

Logged statistics include:

```text
paired mean delta from step zero
paired delta standard deviation and SEM
paired bootstrap 95% interval
raw and EMA practical-success gates
```

Never combine baseline and current SEMs as independent measurements when prompts
and seeds are fixed.

### Literal shared rollouts for objective-only comparison

Standalone GRPO is on-policy:

```yaml
behavior_policy: on_policy
```

The `--paired` launcher forces both GRPO and posterior arms to:

```yaml
behavior_policy: frozen_base
```

LoRA is disabled only during rollout collection. With identical seeds, both
learners receive the same prompts, source noise, ASFMC means, actions, completed
videos, and rewards even after learner weights diverge. This is an off-policy
objective-isolation experiment, not a claim that frozen behavior is the best
online algorithm.

### Stochastic-to-deterministic transfer diagnostics

For a probe transition, v2 logs:

```text
ftv2/preferred_action_shift_rms
ftv2/deterministic_map_shift_rms
ftv2/deterministic_preference_alignment
```

The alignment is the cosine between the reward-preferred next-state direction
and the actual post-update shift of AnyFlow's deterministic finite map. Positive
within-group selection is not useful when deterministic alignment is zero or
negative.

## Objectives

### `grpo`

Resource-aware multi-transition Flow-Map GRPO using running reward statistics,
32 rollout videos per update, and target-KL control.

### `posterior`

Identical rollout and trust-region substrate, but uses centered global-
temperature Boltzmann weights instead of running-baseline advantages. This is the
fair test of the original posterior-weighting idea.

### `velocity`

Uses frozen-base shared-state branches on one deployed transition. Candidate
next states become finite velocities:

```text
g_j = (x_t - a_j) / Delta
Delta_u = sum_j (w_j - 1/G) g_j
u_target = u_behavior + eta * Delta_u
```

`eta` is capped by the desired deterministic next-state RMS change. The student
receives ordinary stopped velocity regression, directly changing the finite map
used at deterministic inference.

### `diagnostic_luminance`

Cheap 17-frame, 256x448 systems gate using mean luminance as both training and
held-out reward. It has no VideoAlign dependency. This must learn before subtle
VideoAlign failures are blamed on reward semantics.

### `diagnostic_motion`

Optional full-resolution systems gate optimizing decoded temporal L1 while
logging audited VideoAlign MQ and holding VQ/TA out. Run it only after the cheaper
luminance gate.

## Required execution order

### 1. Pull and run a real smoke

```bash
git fetch origin
git switch adam/finite-transition-alignment
git pull --ff-only origin adam/finite-transition-alignment

modal run modal_train_finite_transition_v2_complete.py --smoke
```

A smoke proves environment, distributed, reward, optimizer, validation, and W&B
plumbing only.

### 2. Easy deterministic learnability gate

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset diagnostic_luminance \
  --max-train-steps 30 \
  --validation-every 5 \
  --comparison-id ftv2_luminance_gate_s42
```

Required outcome:

- raw deterministic held-out luminance increases;
- EMA follows with expected lag;
- post-update KL leaves the `1e-7` regime without NaN/collapse;
- motion and diversity remain inspectable.

If this fails, stop. Audit ASFMC, log-probability scale, LoRA gradients, and
deterministic transfer before loading VideoAlign.

### 3. Learning-rate / KL calibration

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --lr-sweep \
  --validation-prompts 128 \
  --comparison-id ftv2_grpo_lr_s42
```

The target-KL controller is disabled for these 20-update probes. Choose the
largest stable learning rate with coherent raw deterministic behavior.

### 4. Establish a GRPO baseline that learns

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset grpo \
  --max-train-steps 100 \
  --validation-every 25 \
  --validation-prompts 128 \
  --learning-rate <selected-lr> \
  --comparison-id ftv2_grpo_baseline_s42
```

Do not test a novel update rule until this baseline demonstrates positive raw or
EMA paired held-out MQ without unacceptable VQ/TA, motion, or diversity loss.

### 5. Strict GRPO versus posterior comparison

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --paired \
  --max-train-steps 100 \
  --validation-every 25 \
  --validation-prompts 128 \
  --learning-rate <selected-lr> \
  --comparison-id ftv2_grpo_vs_posterior_s42
```

Both arms use the frozen base behavior and receive identical rollout data.
Compare matching exact sample artifacts with:

```bash
python examples/train/compare_finite_transition_paired_runs.py \
  --left <posterior-output>/paired_validation/ema_step_000100_samples.json \
  --right <grpo-output>/paired_validation/ema_step_000100_samples.json \
  --output outputs/ftv2_posterior_minus_grpo_step100.json
```

The comparison orientation is `left - right`.

### 6. Direct finite-velocity follow-up

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset velocity \
  --max-train-steps 100 \
  --validation-every 25 \
  --validation-prompts 128 \
  --comparison-id ftv2_velocity_s42
```

Run this when likelihood objectives remain tied or stochastic-policy updates do
not transfer to deterministic inference.

## W&B fields to inspect

Training:

```text
ftv2/reward_videos_per_update
ftv2/transition_records_per_update
ftv2/reward_std_current
ftv2/reward_std_ema
ftv2/posterior_ess_mean
ftv2/posterior_temperature_mean
ftv2/loss_scale_before
ftv2/loss_scale_after
ftv2/post_update_approx_kl
ftv2/post_update_logprob_delta_abs
ftv2/deterministic_map_shift_rms
ftv2/preferred_action_shift_rms
ftv2/deterministic_preference_alignment
ftv2/grad_norm
ftv2/cumulative_gpu_hours
```

Reward audit:

```text
audit/videoalign_checkpoint_numel_coverage
audit/videoalign_checkpoint_tensor_coverage
audit/videoalign_adapter_numel_coverage
audit/videoalign_reward_head_numel_coverage
audit/videoalign_repeat_delta_max
```

Validation:

```text
validation_raw/*
validation_ema/*
validation_paired_delta_raw/*
validation_paired_ci95_low_raw/*
validation_paired_delta/*
validation_paired_ci95_low/*
validation_success_raw/all_paired
validation_success/all_paired
```

## Stop rules

Stop or repair the shared substrate when:

- the luminance gate does not improve deterministically;
- reward-model coverage or repeatability audit fails;
- deterministic-preference alignment is persistently non-positive;
- post-update KL remains below `1e-6` after the controller reaches a high loss
  scale;
- training selection metrics rise while raw and EMA held-out reward stay flat;
- VQ/TA, motion, or diversity degrade materially;
- GRPO cannot demonstrate positive held-out movement under the stronger
  substrate.

Do not run 1,200 updates solely because the process is numerically stable.

## Source-of-truth code

```text
fastvideo/train/methods/rl/common/finite_transition_v2.py
fastvideo/train/methods/rl/rewards/videoalign_audit.py
fastvideo/train/methods/rl/finite_transition_v2.py
fastvideo/train/methods/rl/finite_transition_v2_exact_paired.py
fastvideo/train/methods/rl/finite_transition_v2_scientific.py
fastvideo/train/methods/rl/finite_transition_v2_final.py
examples/train/configs/rl/wan/finite_transition_*_v2_*.yaml
examples/train/compare_finite_transition_paired_runs.py
modal_train_finite_transition_v2_complete.py
```
