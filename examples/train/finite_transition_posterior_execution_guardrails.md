# Finite-transition execution guardrails for coding agents

Read this before launching, repairing, or modifying any AnyFlow reward-alignment
experiment on this branch.

The first real FTPP/GRPO run was technically stable but tested the wrong
algorithm because it reused FastVideo's generic flow-matching scheduler. A later
corrected run used the proper grid but revealed a second class of failure: the
shared RL substrate used too little reward evidence, noisy four-sample
normalization, microscopic updates, EMA-only evaluation, and no explicit reward
checkpoint coverage audit.

The goal of these guardrails is to prevent code that runs, logs plausible
numbers, and silently tests the wrong or statistically powerless experiment.

## 1. Know which experiment you are running

There are now two distinct experiment families.

### Archived deployment-grid FTPP

Source config:

```text
examples/train/configs/rl/wan/
  finite_transition_posterior_anyflow_videoalign.yaml
```

This experiment selects one of the positive-target transitions from the exact
four-step deployment grid. For this experiment only, train and evaluation grids
must match:

```yaml
require_train_eval_schedule_match: true
train_map_steps: 4
eval_map_steps: 4
stochastic_steps: 3
```

### Reliable full-trajectory baseline and follow-ups

Source configs:

```text
finite_transition_reliable_anyflow_videoalign.yaml
finite_transition_velocity_anyflow_videoalign.yaml
finite_transition_reliable_sanity_luminance.yaml
```

The reliable likelihood baseline intentionally uses:

```yaml
require_train_eval_schedule_match: false
train_map_steps: 5
eval_map_steps: 4
stochastic_steps: 4
rollout_mode: full_trajectory
```

This is not a scheduler bug. A two-time flow map supports arbitrary `(t, r)`
pairs, and the full Flow-Map-GRPO-style rollout uses four stochastic transitions
plus one deterministic completion while evaluating deterministic four-step
AnyFlow.

The finite-velocity method returns to the four-step deployment grid because it
explicitly corrects one deployed deterministic transition target.

Do not copy invariants from one family into the other without changing the named
scientific question.

## 2. Always use AnyFlow's actual scheduler

Never use `FlowMatchEulerDiscreteScheduler` to build an AnyFlow finite-map
inference grid.

The released scheduler uses:

```text
base = linspace(1, 0, K + 1)[:-1]
shifted = shift * base / (1 + (shift - 1) * base)
timesteps = 1000 * shifted
append 0
```

The flow shift is applied exactly once.

At shift 5, four-step deterministic evaluation must be approximately:

```text
[1000.000, 937.500, 833.333, 625.000, 0.000]
```

The old broken grid contained:

```text
[1000.000, 909.707, 717.317, 24.414, 0.000]
```

Any run containing `24.414` as a branch target is invalid.

Source of truth:

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
```

## 3. Preserve the model contract

The experiment requires:

```text
nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
r_embedder = true
r_embedder_fusion = gated
r_embedder_gate_value = 0.25
r_embedder_deltatime_type = r
flow_shift = 5
CFG = 1 for the released AnyFlow operating point
```

Do not silently fall back to a normal Wan checkpoint, ignore `r_timestep`, or
change the flow shift to match a different Wan recipe.

Before a paid run, verify that changing `r_timestep` changes the model output.

## 4. Local ASFMC is not arbitrary latent noise

The policy action is the sampled next latent, not the predicted velocity.

For a two-time map, local ASFMC:

1. computes the deterministic target `x_r`;
2. queries the instantaneous reverse velocity at `(r, r)`;
3. applies the short reverse-SDE Gaussian after converting the paper's
   noise-to-data coordinate into AnyFlow's reverse coordinate.

The implemented policy is in:

```text
fastvideo/train/methods/rl/common/local_asfmc.py
```

Do not replace it with `x_r + sigma * randn` or reuse endpoint-anchor formulas
without naming a separate ablation.

Mandatory diagnostics:

```text
source timestep
target timestep
local anchor delta
policy standard deviation
action deviation from deterministic target
```

## 5. Candidate count and transition count are different

`group_size = 8` means eight alternative trajectories for one prompt group.
It does not mean eight denoising transitions.

The reliable baseline default is:

```text
8 candidate trajectories
4 stochastic transitions per trajectory
4 prompt groups accumulated per optimizer update
32 reward-scored videos per update
128 stochastic transition likelihood terms per update
```

Do not reduce this to one prompt and four candidates without explicitly calling
it a low-power smoke or ablation.

## 6. Avoid retaining multiple video graphs

Four video transformer graphs do not fit simply because the rollout itself was
`no_grad`.

The reliable implementation recomputes and backpropagates each stochastic
transition separately, dividing by the total accumulation denominator. Do not
rewrite it to stack four gradient-enabled log-probability forwards before one
backward call unless memory has been profiled at full 81-frame resolution.

## 7. Reward normalization must preserve confidence

The archived run normalized each four-sample group by its own standard deviation
and forced posterior ESS to two. That gives nearly the same update norm to a
clear preference and to a tiny noisy reward difference.

The reliable default uses:

```text
running per-prompt baseline
running global reward standard deviation
global-scale Boltzmann temperature
```

`fixed_ess` remains an explicit ablation only.

Monitor:

```text
reliable/reward_baseline
reliable/reward_scale
reliable/reward_baseline_source
reliable/reward_scale_source
reliable/posterior_ess
reliable/posterior_temperature
```

## 8. Calibrate actual policy movement

Equal learning rates do not imply equal updates. The video log probability is
dimension averaged, LoRA parameterizations differ, and objective coefficients
differ.

The reliable method measures post-update KL every optimizer update and adjusts a
bounded loss scale toward a configured target.

Run the target-KL calibration before a long baseline:

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe reliable \
  --calibrate-kl
```

Do not continue when KL remains effectively zero or becomes unstable. Select a
trust region from deterministic held-out behavior, not training reward alone.

## 9. Distinguish on-policy learning from objective-only comparison

Standalone reliable GRPO uses:

```yaml
behavior_policy: current
```

For a strict FTPP-versus-GRPO loss comparison, both arms use:

```yaml
behavior_policy: base_adapter_disabled
```

This disables LoRA only while collecting rollouts. With the same seeds, both
learners receive identical prompts, policy means, actions, videos, and rewards
after step one as well as at initialization.

Do not claim “same actions” for two independently evolving on-policy arms.

## 10. Audit VideoAlign before trusting RL

Use the audited reward names:

```text
videoalign_mq_audited
videoalign_vq_audited
videoalign_ta_audited
```

They verify shape-compatible checkpoint coverage and require a loaded reward
head. The launcher also runs:

```bash
python -m examples.train.audit_videoalign_checkpoint
```

Record:

```text
base coverage
adapter coverage
reward-head coverage
static clip score
smooth-motion clip score
flicker clip score
```

A stable single score is not enough. A remapped checkpoint can produce numbers
while leaving critical weights random.

## 11. Prove learnability with an easy reward

Before blaming VideoAlign or a novel posterior objective, run:

```bash
modal run modal_train_finite_transition_reliable.py \
  --recipe sanity
```

The mean-luminance sanity run must increase deterministic held-out luminance in
raw and EMA weights. Failure means the common rollout, likelihood, optimizer,
LoRA, or stochastic-to-deterministic transfer path is broken or too weak.

This is a systems test, not a publishable result.

## 12. Evaluate raw and EMA weights with paired statistics

Validation must retain exact `(prompt_index, sample_seed)` identities and save
all scalar results.

Required W&B keys include:

```text
validation_raw/reward/<name>
validation_ema/reward/<name>
paired_raw/reward/<name>/mean_delta
paired_raw/reward/<name>/sem_delta
paired_raw/reward/<name>/ci_lower
paired_raw/reward/<name>/ci_upper
paired_ema/reward/<name>/mean_delta
paired_ema/reward/<name>/ci_lower
paired_ema/reward/<name>/ci_upper
```

Do not combine baseline and current SEM as independent when prompts and seeds are
fixed. Use paired differences.

## 13. Interpret training metrics correctly

`reward_selection_gain > 0` is nearly guaranteed when weights are monotone in
reward. It proves that the current finite group was reweighted, not that the
next policy improved.

Candidate reward means from different prompts are not a learning curve.

Evidence of learning requires:

```text
fixed deterministic held-out reward improves
raw and EMA trends are coherent
paired confidence interval is positive
motion/diversity are retained
update KL is non-trivial and stable
```

## 14. Stop rules

Stop or repair the shared substrate when:

- the easy reward sanity gate fails;
- audited VideoAlign loading or calibration fails;
- post-update KL remains below the selected range after controller warmup;
- deterministic held-out reward remains flat while training selection metrics
  look positive;
- raw and EMA both decline;
- motion or diversity collapses;
- deterministic-transfer cosine is consistently non-positive for the
  single-transition probe.

Do not extend a stable but non-learning run solely because RL is stochastic.
More samples help only after the signal, update scale, and evaluation are valid.

## 15. Never resume incompatible checkpoints

Do not resume the archived 200-step FTPP/GRPO checkpoints into the reliable
method. The following changed:

```text
rollout structure
number of stochastic transitions
reward normalizer state
loss-scale controller state
optimizer hyperparameters
LoRA rank
EMA decay
raw/EMA paired validation state
```

Start a fresh W&B group and a fresh checkpoint root whenever any of those change.

## Source-of-truth files

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
fastvideo/train/methods/rl/common/local_asfmc.py
fastvideo/train/methods/rl/common/reward_statistics.py
fastvideo/train/methods/rl/finite_transition_reliable.py
fastvideo/train/methods/rl/finite_transition_paired_validation.py
fastvideo/train/methods/rl/rewards/videoalign_audit.py
examples/train/finite_transition_reliable_experiment.md
modal_train_finite_transition_reliable.py
```
