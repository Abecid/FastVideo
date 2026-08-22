# Finite-transition v2 guardrails for coding agents

Read this before launching, repairing, or modifying AnyFlow reward training on
this branch.

The first run failed scientifically twice while looking technically healthy:

1. a generic flow-matching scheduler silently produced the wrong AnyFlow
   transition grid; and
2. the corrected run used too little reward evidence and updates so small that
   neither FTPP nor GRPO moved deterministic held-out reward.

The active implementation is v2. Source of truth:

```text
modal_train_finite_transition_v2_complete.py
examples/train/finite_transition_v2_execution_plan.md
fastvideo/train/methods/rl/finite_transition_v2_final.py
```

The older original FTPP and intermediate `finite_transition_reliable*` code is
retained for history. Do not repair a new run by switching to it.

## 1. Never substitute a generic diffusion scheduler

Use:

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
```

For four-step deterministic AnyFlow with shift 5, the grid is approximately:

```text
1000.000 -> 937.500 -> 833.333 -> 625.000 -> 0.000
```

The broken generic scheduler produced a `24.414` branch target. Any run that
contains that target is invalid.

Apply flow shift exactly once.

## 2. Preserve the AnyFlow model contract

Required:

```text
checkpoint: nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
r_embedder: true
r_embedder_fusion: gated
r_embedder_gate_value: 0.25
r_embedder_deltatime_type: r
flow_shift: 5
CFG: 1 at the released operating point
```

Before a paid run, verify that changing `r_timestep` changes the model output.
Never fall back to a normal Wan checkpoint while keeping the experiment name.

## 3. Local ASFMC is the stochastic policy

The RL action is a sampled next latent, not the finite velocity itself.

For a two-time flow map:

1. AnyFlow predicts the deterministic finite target `x_r`.
2. The model is queried at `(r, r)` for the instantaneous reverse velocity.
3. Local ASFMC applies the short reverse-SDE conditional after converting the
   paper coordinate into AnyFlow's reverse coordinate.

Implementation:

```text
fastvideo/train/methods/rl/common/local_asfmc.py
```

Do not replace this with `x_r + sigma * randn` or an endpoint-anchor formula
without naming a separate ablation.

## 4. Candidate count is not transition count

The scientific GRPO/posterior presets use:

```text
8 candidate trajectories per prompt group
4 prompt groups per optimizer update
4 stochastic transitions per trajectory
32 terminal reward videos per optimizer update
128 transition likelihood records per optimizer update
```

`group_size=8` means eight alternative trajectories, not eight diffusion steps.

The implementation recomputes and backpropagates one transition at a time. Do
not retain four full 81-frame transformer graphs before backward unless full-
resolution memory has been profiled.

## 5. Five-step training and four-step evaluation are intentional

The likelihood objectives use:

```text
4 stochastic finite transitions
1 deterministic completion to x_0
```

Deterministic validation uses released four-step AnyFlow.

A two-time flow map supports arbitrary `(t, r)` pairs, so these grids need not be
identical for full-trajectory Flow-Map GRPO. The finite-velocity method is
different: it deliberately uses the four-step deployment grid because it
corrects one deployed deterministic transition directly.

Do not apply the old “all train/eval pairs must match” guardrail to the full-
trajectory baseline.

## 6. Reward normalization must not manufacture confidence

The original four-candidate run standardized every group by its own noisy
standard deviation. FTPP also forced every non-flat group to ESS 2. Tiny reward
jitter therefore caused a full-sized update.

V2 uses:

```text
running prompt baseline
global/EMA reward scale across the optimizer rollout batch
global-temperature posterior weights
```

`group_ess` is an explicit ablation only.

Monitor:

```text
ftv2/reward_std_current
ftv2/reward_std_ema
ftv2/posterior_ess_mean
ftv2/posterior_temperature_mean
ftv2/prompt_tracker_size
```

## 7. Calibrate actual policy movement

The old maximum post-update KL was around `1e-7`. A stable process that never
moves is not a successful RL run.

Required metrics:

```text
ftv2/loss_scale_before
ftv2/loss_scale_after
ftv2/target_post_update_kl
ftv2/post_update_approx_kl
ftv2/post_update_logprob_delta_abs
ftv2/grad_norm
```

Run the controller-disabled learning-rate sweep before the baseline:

```bash
modal run modal_train_finite_transition_v2_complete.py --lr-sweep
```

Choose update scale using raw deterministic held-out behavior and stability, not
training reward alone.

## 8. Optimize MQ online; hold VQ and TA out

Scientific presets use:

```text
online: videoalign_mq_audited
validation: MQ + VQ + TA audited
```

Do not add VQ and TA to every online rollout merely for logging. That triples
Qwen reward cost and makes them no longer clean held-out anti-reward-hacking
checks.

## 9. Audit the reward model, not just one score

The audit reads the actual checkpoint tensors and checks:

```text
overall tensor and numel coverage
base coverage
adapter coverage
reward-head coverage
reward-head presence
non-finite audited parameters
repeatability on a deterministic clip
```

MQ preprocessing must match upstream exactly:

```text
mean-channel grayscale
empty text prompt
```

VQ uses color with an empty prompt. TA uses color with the generation prompt.

A finite calibration score does not prove the adapter or reward head loaded.
Never disable the audit to make a run start.

## 10. Prove the common substrate can learn first

The cheapest required gate is:

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --preset diagnostic_luminance \
  --max-train-steps 30 \
  --validation-every 5
```

It uses 17 frames at 256x448 and has no VideoAlign dependency.

Continue only when raw deterministic held-out luminance increases and EMA follows
without numerical instability. Failure means the common rollout, likelihood,
LoRA, update scaling, or stochastic-to-deterministic path is still broken.

The optional `diagnostic_motion` gate is full-resolution and more expensive.

## 11. Distinguish online learning from objective isolation

Standalone GRPO uses:

```yaml
behavior_policy: on_policy
```

The strict GRPO/posterior comparison must use:

```yaml
behavior_policy: frozen_base
```

for both arms. The Modal `--paired` mode enforces this. Equal seeds then produce
identical prompts, source noise, policy means, sampled actions, final videos,
and rewards even after the learner weights diverge.

Do not claim independently evolving on-policy runs received the same actions.

Frozen behavior is an objective-isolation experiment, not necessarily the best
online policy-improvement recipe.

## 12. Use exact prompt-seed validation records

V2 evaluates raw and EMA weights with fixed prompts and fixed seeds. It writes:

```text
<output_dir>/paired_validation/
  raw_step_XXXXXX_samples.json
  ema_step_XXXXXX_samples.json
```

Every file stores exact sample keys, prompt indices, seeds, rewards, motion, and
static flags.

Confidence intervals are computed from prompt-level means across seeds, avoiding
seed-level pseudoreplication. Cross-arm comparisons must verify sample identity
before subtracting scores:

```bash
python examples/train/compare_finite_transition_paired_runs.py \
  --left <posterior-samples.json> \
  --right <grpo-samples.json>
```

Do not use independent baseline/current SEM when data are paired.

## 13. Check stochastic-to-deterministic transfer

Required fields:

```text
ftv2/preferred_action_shift_rms
ftv2/deterministic_map_shift_rms
ftv2/deterministic_preference_alignment
```

A positive reward-selection gain is nearly automatic when weights are monotone
in reward. It is not policy improvement.

If deterministic-preference alignment is consistently zero or negative, more
training of the same likelihood objective is not justified.

## 14. Raw and EMA curves answer different questions

EMA can hide early learning or early collapse. Always inspect both:

```text
validation_raw/*
validation_ema/*
validation_paired_delta_raw/*
validation_paired_delta/*
validation_paired_ci95_low_raw/*
validation_paired_ci95_low/*
```

Do not declare failure from an early EMA-only curve, and do not declare success
from a raw-only transient spike.

## 15. Runtime repairs that preserve the experiment

Safe when applied identically to all affected arms:

- dependency/import fixes;
- immutable cache/snapshot handling;
- memory-safe sequential backward;
- exact seed and prompt identity fixes;
- W&B logging and artifact fixes;
- checkpoint-state serialization;
- reward-load assertions;
- distributed deadlock fixes that do not alter samples.

Changes that create a new experiment and require a fresh W&B group/checkpoint:

- scheduler or flow shift;
- model checkpoint or CFG;
- rollout transition count;
- group size or prompt groups per update;
- behavior policy;
- reward/preprocessing;
- reward normalization;
- target KL/controller;
- LoRA rank/targets;
- optimizer hyperparameters;
- validation prompt/seed set.

Never repair only one arm of a paired run.

## 16. Stop rules

Stop or repair when:

- the luminance gate does not improve;
- VideoAlign coverage or repeatability fails;
- post-update KL stays below `1e-6` after controller escalation;
- raw and EMA held-out reward remain flat while selection metrics look good;
- deterministic-preference alignment is persistently non-positive;
- VQ/TA, motion, or diversity degrade materially;
- GRPO cannot learn under the strengthened substrate.

Do not run 1,200 updates solely because RL is stochastic. Longer training is
useful only after signal, update scale, deterministic transfer, and evaluation
are verified.

## 17. Never resume incompatible checkpoints

Do not resume the old 200-step FTPP/GRPO checkpoints into v2. V2 changes:

```text
rollout topology
candidate and prompt-group counts
reward statistics
loss-scale controller
behavior-policy options
LoRA configuration
EMA schedule
reward names and auditing
raw/EMA paired validation state
```

Start a fresh checkpoint root and W&B group.

## Source-of-truth files

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
fastvideo/train/methods/rl/common/local_asfmc.py
fastvideo/train/methods/rl/common/finite_transition_v2.py
fastvideo/train/methods/rl/rewards/videoalign_audit.py
fastvideo/train/methods/rl/finite_transition_v2.py
fastvideo/train/methods/rl/finite_transition_v2_exact_paired.py
fastvideo/train/methods/rl/finite_transition_v2_scientific.py
fastvideo/train/methods/rl/finite_transition_v2_final.py
examples/train/finite_transition_v2_execution_plan.md
modal_train_finite_transition_v2_complete.py
```
