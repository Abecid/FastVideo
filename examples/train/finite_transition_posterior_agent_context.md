# Finite-transition experiment context for coding agents

Read these files in this order before running or modifying the experiment:

```text
examples/train/finite_transition_posterior_progress_report.md
examples/train/finite_transition_reliable_experiment.md
examples/train/finite_transition_posterior_execution_guardrails.md
```

The old centered-likelihood FTPP and four-candidate GRPO run did not improve
held-out VideoAlign MQ. The current priority is to prove that the common AnyFlow
RL substrate can learn before judging a new posterior objective.

## Current experiment sequence

```text
1. real smoke
2. easy mean-luminance sanity reward
3. target-KL calibration
4. reliable full-trajectory Flow-Map GRPO baseline
5. strict shared-behavior GRPO versus posterior likelihood pair
6. finite-velocity posterior regression
```

Do not skip directly to step 5 or 6 when the baseline has not demonstrated
learnability.

## The three active objectives

### `flowmap_grpo`

A stochastic finite-transition policy is constructed with local ASFMC. The
reliable baseline samples all four stochastic transitions in a five-segment
training trajectory, applies the terminal reward to each transition, and uses a
clipped score-function loss.

### `posterior_projection`

Uses the identical rollout and terminal rewards but replaces standardized linear
advantages with centered Boltzmann posterior coefficients. It is still a
likelihood/score-function update; do not describe it as a solved finite forward-
KL projection with a unique optimum.

### `finite_velocity_regression`

Selects one deployed transition, converts candidate next states to finite
velocities, constructs a reward-posterior mean correction, and regresses the
AnyFlow finite velocity directly toward the corrected target. This is a distinct
method, not a loss-flag ablation of GRPO.

## Non-negotiable implementation contracts

### Scheduler

Use only:

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
```

For four-step AnyFlow with shift 5:

```text
[1000.000, 937.500, 833.333, 625.000, 0.000]
```

A `24.414` target means the wrong scheduler was used.

### Model

```text
checkpoint: nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
r_embedder: true
r_embedder_fusion: gated
r_embedder_gate_value: 0.25
r_embedder_deltatime_type: r
flow_shift: 5
```

### Local stochastic policy

The RL action is the sampled next latent, not the predicted velocity. Use the
local-ASFMC implementation in:

```text
fastvideo/train/methods/rl/common/local_asfmc.py
```

Do not substitute arbitrary Gaussian perturbation.

### Rollout statistics

The reliable default uses:

```text
8 candidates per prompt group
4 prompt groups per optimizer update
4 stochastic transitions per trajectory
32 reward-scored videos per optimizer update
128 transition likelihood terms per optimizer update
```

The implementation backpropagates transition graphs sequentially. Do not retain
all four full-video transformer graphs before backward.

### Reward normalization

Default:

```text
running prompt baseline
running global standard deviation
global-scale posterior temperature
```

Do not restore four-sample group standardization or forced ESS in the main run.
Those are named ablations only.

### Update scale

The target-KL controller is part of the reliable baseline. Log and inspect:

```text
reliable/loss_scale_used
reliable/loss_scale_next
reliable/post_update_approx_kl
reliable/post_update_logprob_delta_abs
reliable/grad_norm
```

A stable run with KL near zero is not a successful run.

### Reward model

Use audited names:

```text
videoalign_mq_audited
videoalign_vq_audited
videoalign_ta_audited
```

The checkpoint must pass base/adapter/head coverage audit and deterministic clip
calibration before training.

### Validation

Every checkpoint evaluates raw and EMA weights on fixed prompt-seed pairs. Save
all per-sample values and compute paired confidence intervals.

Do not use independent baseline/current SEM for fixed paired data.

## Strict objective-only comparison

For a paired loss comparison, use:

```yaml
behavior_policy: base_adapter_disabled
```

in both arms. Candidate rollouts are then produced by the same frozen base model,
so equal seeds yield identical actions and rewards even after learner weights
diverge.

For a standalone online baseline, use:

```yaml
behavior_policy: current
```

Do not claim the online arms use identical actions after the first update.

## Tensor/data flow for one reliable prompt group

At full resolution, each rank holds two candidate latents with approximate shape:

```text
[2, 21, 16, 60, 104]
```

Across four ranks the conceptual group is:

```text
[8, 21, 16, 60, 104]
```

For each of four stochastic transitions:

1. Every rank starts from the same prompt and shared prefix state.
2. Local ASFMC computes the policy mean/std.
3. Rank-specific Gaussian noise creates candidate next states.
4. The candidate becomes the next trajectory state.
5. After four stochastic transitions, one deterministic map completes to `x_0`.
6. The VAE decodes roughly `[2, 3, 81, 480, 832]` per rank.
7. VideoAlign scores the eight videos.
8. The same terminal coefficients supervise all four stored stochastic
   transitions.
9. Each transition is recomputed with gradients and backpropagated immediately.

Four independent prompt groups repeat this process before one optimizer step.

## What counts as learning

Training-side reward selection is insufficient. A successful baseline needs:

```text
raw deterministic held-out reward increases
EMA deterministic held-out reward increases
paired bootstrap lower bound is positive
motion and diversity remain within thresholds
update KL is non-trivial and stable
qualitative videos do not reveal reward exploitation
```

`reward_selection_gain > 0` is almost automatic when weights are monotone in
reward. Do not present it as policy improvement.

## Runtime repair rules

Safe fixes that preserve the experiment:

- dependency/import corrections;
- cache race prevention;
- memory-safe sequential transition backward;
- deterministic dataset preparation;
- checkpoint coverage assertions;
- W&B logging fixes;
- resume-state serialization fixes.

Changes that create a new experiment and require a new config/run group:

- scheduler or flow shift;
- model checkpoint or CFG;
- rollout transition count;
- behavior policy;
- reward or reward preprocessing;
- reward normalization;
- target KL or controller behavior;
- LoRA rank/targets;
- optimizer hyperparameters;
- validation prompt/seed set.

Never repair only one arm of a paired run.

## Commands

Smoke:

```bash
modal run modal_train_finite_transition_reliable.py --smoke
```

Easy reward sanity:

```bash
modal run modal_train_finite_transition_reliable.py --recipe sanity
```

Target-KL calibration:

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --calibrate-kl
```

Reliable baseline:

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --objective flowmap_grpo
```

Strict shared-behavior pair:

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --paired
```

Finite-velocity method:

```bash
modal run modal_train_finite_transition_reliable.py --recipe velocity
```

Do not resume the old 200-step FTPP/GRPO checkpoints into these recipes.
