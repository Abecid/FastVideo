# Finite-transition v2: execution and decision plan

Read this file before launching or repairing the next AnyFlow reward-training
experiment. The original 200-step corrected-grid run was stable but produced no
held-out MQ gain for either FTPP or its matched one-update GRPO control. Running
the same recipe for 1,200 steps is not the next experiment.

## What was wrong with the shared substrate

The corrected run used only:

```text
1 prompt group / optimizer update
4 candidate videos / prompt
1 selected finite transition / video
learning rate 2e-6
group-local reward standardization
one score-function update
EMA-only validation
```

Maximum measured post-update KL was approximately `1e-7`. Both methods barely
moved. Moreover, fixed four-sample normalization gave full-strength directions
to weak/noisy local reward rankings, and the two objectives were almost the same
score estimator.

V2 changes the shared substrate before judging the update rule.

## V2 changes

### More information per optimizer update

The default GRPO/posterior presets collect:

```text
4 prompt groups / optimizer update
4 candidate videos / prompt
4 stochastic finite transitions / trajectory
16 reward-scored videos / optimizer update
64 transition likelihood records / optimizer update
```

This is still resource-aware rather than a claim of exact reproduction of the
24-candidate published image recipe. It is materially stronger than the first
four-video update.

### Stable reward normalization

GRPO v2 uses:

```text
A = (R - running_prompt_mean) / global_rollout_std
```

where the denominator is estimated from all candidate videos in the accumulated
optimizer batch and smoothed over time. It is not a four-sample group standard
deviation.

Posterior v2 uses one global/EMA reward temperature:

```text
w = softmax((R - baseline) / tau_global)
```

A group with almost equal rewards therefore produces nearly uniform weights and
a nearly zero centered update. It is no longer forcibly sharpened to ESS=2.

### Explicit update-scale calibration

Both likelihood objectives log the actual post-update policy movement and adapt
loss scale toward a configured target KL. The controller is deliberately slow:

```text
scale_next = scale * (target_kl / observed_kl)^(controller_rate / 2)
```

with bounded scale. The first target is `3e-5`, not because it is universal, but
because the old `1e-7` regime was clearly too weak to test learnability.

### Multi-transition Flow-Map GRPO

The v2 GRPO/posterior rollout uses an official AnyFlow five-segment training
schedule:

```text
4 stochastic ASFMC transitions
1 deterministic completion to data
```

All four stochastic transitions receive the same terminal reward signal. Four-
step deterministic AnyFlow remains the deployment evaluation. This is closer to
Flow-Map-GRPO's actual multi-transition formulation than selecting one local
transition per video.

### Raw and EMA evaluation

Every validation checkpoint evaluates:

```text
raw LoRA weights
EMA weights
```

The original decay-0.99 EMA could hide early raw-model movement. V2 defaults to
EMA `0.9`, updated every eight optimizer updates, and still reports raw results
separately.

### Paired statistics

For every fixed prompt index, v2 stores raw/EMA metric values under:

```text
<output_dir>/paired_validation/
```

It logs:

```text
paired mean delta
paired delta std and SEM
paired bootstrap 95% interval
```

for MQ, VQ, TA, motion, static rate, and diversity. Do not use the old
independent-SEM gate for fixed prompt/seed evaluations.

### Reward-model audit

Before training, the final method:

1. loads the prepared VideoAlign inferencer;
2. compares checkpoint tensors against runtime model keys and shapes;
3. records total, adapter, and detected reward-head coverage;
4. fails below configured coverage thresholds; and
5. scores a deterministic synthetic moving-square video twice to assert runtime
   repeatability.

The JSON audit is written to:

```text
<output_dir>/videoalign_checkpoint_audit.json
```

This does not replace a one-time numerical comparison against upstream
VideoAlign, but it prevents silent partial checkpoint loading.

### Stochastic-to-deterministic transfer diagnostic

For one probe transition each optimizer update, v2 logs:

```text
ftv2/preferred_action_shift_rms
ftv2/deterministic_map_shift_rms
ftv2/deterministic_preference_alignment
```

The last metric is the cosine between the reward-posterior preferred next-state
direction and the actual post-update shift of AnyFlow's deterministic finite
map. Positive reward selection is not useful if this alignment is zero or
negative.

## Objectives

### `grpo`

Resource-aware multi-transition Flow-Map-GRPO baseline with running prompt/global
reward statistics and target-KL control.

### `posterior`

Identical rollout and target-KL substrate, but uses centered global-temperature
Boltzmann weights. This is the fair update-rule comparison after the GRPO
baseline learns.

### `velocity`

Uses shared-state branches and a frozen-base behavior policy. Candidate actions
are converted to finite velocities:

```text
g_j = (x_t - a_j) / Delta
Delta_u = sum_j (w_j - 1/G) g_j
u_target = u_behavior + eta * Delta_u
```

`eta` is capped by a target deterministic next-state RMS change. The student is
trained by ordinary stopped velocity regression. This directly modifies the
finite map used at deterministic inference and is the meaningful follow-up when
both likelihood objectives remain tied.

### `diagnostic_motion`

Uses adjacent-frame decoded temporal L1 as the actual training reward while
still logging VideoAlign. This is a short systems-identification gate, not a
quality experiment. It asks whether the RL substrate can move a simple
measurable deterministic validation statistic at all.

## Authoritative launcher

Use only:

```text
modal_train_finite_transition_v2_complete.py
```

The other v2 launcher files are implementation stepping stones and should not be
used for new scientific jobs.

## Required execution order

### 1. Easy learnability gate

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset diagnostic_motion \
  --max-train-steps 30 \
  --validation-every 5 \
  --validation-prompts 64 \
  --comparison-id ftv2_motion_gate_s42
```

Required outcome:

- raw deterministic temporal L1 responds in the optimized direction;
- the post-update KL leaves the `1e-7` regime without NaN or collapse;
- deterministic-preference alignment is positive on average;
- MQ/VQ/TA and qualitative videos remain inspectable.

If the raw deterministic motion metric does not move, stop. Audit ASFMC,
log-probability scaling, adapter gradients, and deterministic transfer before
using VideoAlign.

### 2. Learning-rate to KL calibration

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --lr-sweep \
  --validation-prompts 64 \
  --comparison-id ftv2_grpo_lr_s42
```

This runs 20-update GRPO jobs at:

```text
2e-6
2e-5
6e-5
```

with the target-KL controller disabled. Select the largest setting that gives
stable raw deterministic behavior and post-update KL approximately in the
`1e-5` to `1e-4` diagnostic band. This band is a starting probe, not a universal
constant.

### 3. Establish a GRPO baseline that learns

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset grpo \
  --max-train-steps 100 \
  --validation-every 25 \
  --validation-prompts 128 \
  --learning-rate <selected-lr> \
  --comparison-id ftv2_grpo_baseline_s42
```

Do not test a novel objective until raw or EMA paired held-out MQ shows a
positive trend without unacceptable VQ/TA, motion, or diversity loss.

### 4. Compare posterior weighting only after step 3 succeeds

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --paired \
  --max-train-steps 100 \
  --validation-every 25 \
  --validation-prompts 128 \
  --learning-rate <selected-lr> \
  --comparison-id ftv2_grpo_vs_posterior_s42
```

The target-KL controller is enabled for both arms. Compare:

```text
paired raw and EMA MQ deltas
paired FTPP-minus-GRPO interval
reward gain per scored video
reward gain per GPU-hour
motion/diversity retention
deterministic-preference alignment
```

### 5. Test deterministic velocity regression if likelihood methods tie

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset velocity \
  --max-train-steps 100 \
  --validation-every 25 \
  --validation-prompts 128 \
  --comparison-id ftv2_velocity_s42
```

## W&B fields that must be checked

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
validation_paired_delta/*
validation_paired_sem/*
validation_paired_ci95_low/*
validation_paired_ci95_high/*
validation_success/all_paired
validation_success_raw/all_paired
```

## Stop rules

Stop a run early when any of these holds:

- reward-model audit fails;
- deterministic-preference alignment is persistently non-positive;
- post-update KL remains below `1e-6` after the controller reaches a high loss
  scale;
- training branch reward changes but raw deterministic held-out reward does not;
- VQ/TA, motion, or diversity degrades materially;
- the easy diagnostic reward cannot be learned;
- GRPO cannot demonstrate positive held-out movement under the stronger
  substrate.

Do not run 1,200 updates solely because the process is numerically stable.

## Result analysis

After outputs are available:

```bash
python examples/train/analyze_finite_transition_v2_results.py \
  --run-dir <grpo-output-dir> \
  --label grpo \
  --run-dir <posterior-output-dir> \
  --label posterior \
  --wandb-run adamlee00/finite-transition-v2-wan/<grpo-run-id> \
  --wandb-run adamlee00/finite-transition-v2-wan/<posterior-run-id> \
  --output-dir outputs/ftv2_analysis
```

## Source-of-truth code

```text
fastvideo/train/methods/rl/common/finite_transition_v2.py
fastvideo/train/methods/rl/rewards/videoalign_audit.py
fastvideo/train/methods/rl/finite_transition_v2.py
fastvideo/train/methods/rl/finite_transition_v2_paired.py
fastvideo/train/methods/rl/finite_transition_v2_final.py
examples/train/configs/rl/wan/finite_transition_*_v2_*.yaml
examples/train/analyze_finite_transition_v2_results.py
modal_train_finite_transition_v2_complete.py
```
