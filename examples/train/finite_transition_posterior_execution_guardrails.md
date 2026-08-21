# FTPP execution guardrails for coding agents

Read this file **before launching, repairing, or modifying** the finite-transition
posterior experiment.

The first real FTPP/GRPO run was technically stable but scientifically
misconfigured: it used FastVideo's generic flow-matching scheduler instead of
AnyFlow's released finite-map schedule. That seemingly small substitution
changed the actual transition pairs, created a pathological near-zero branch,
and made training operate off the deterministic four-step deployment grid.

The point of this document is to prevent exactly that class of mistake: code that
runs, logs plausible metrics, and quietly tests the wrong algorithm.

## 1. Scientific question

The experiment asks one narrow question:

> Given the same AnyFlow local stochastic rollout construction and terminal
> video rewards, does an ESS-controlled centered reward-posterior score update
> improve held-out reward more efficiently than a matched one-update GRPO score
> update?

The two objectives are:

```text
posterior_projection
flowmap_grpo
```

Do not add DMD, DiffusionNFT, SFT, an auxiliary consistency loss, a new reward,
or a new sampler to only one arm. Any such change is a new experiment and must
be named, configured, and evaluated separately.

## 2. The most important invariant: use AnyFlow's real deployment grid

### Never use the generic FlowMatchEulerDiscreteScheduler for this experiment

AnyFlow's released `FlowMapDiscreteScheduler` constructs the source nodes as:

```text
base = linspace(1, 0, K + 1)[:-1]
shifted = shift * base / (1 + (shift - 1) * base)
timesteps = 1000 * shifted
append 0
```

The flow shift is applied exactly once.

For four-step AnyFlow with `shift = 5`, the only accepted deployment schedule is
approximately:

```text
[1000.000, 937.500, 833.333, 625.000, 0.000]
```

The old broken run used approximately:

```text
[1000.000, 909.707, 717.317, 24.414, 0.000]
```

That grid is wrong. A run that logs `24.414` as a branch target is not a valid
AnyFlow four-step experiment.

The source of truth is:

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
```

### Train on the same transition pairs used at evaluation

AnyFlow explicitly conditions on both source and target time. Therefore training
on a five-step grid and evaluating on a four-step grid is not an innocuous
resolution change: it updates different conditional maps.

The scientific config must keep:

```yaml
require_train_eval_schedule_match: true
train_map_steps: 4
eval_map_steps: 4
stochastic_steps: 3
```

The branchable policy decisions are exactly:

```text
1000.000 -> 937.500
937.500  -> 833.333
833.333  -> 625.000
```

The final transition:

```text
625.000 -> 0.000
```

is deterministic and exists only to complete a candidate video for reward.

**Group size four means four candidate actions from one selected transition. It
does not mean four different transition indices are required.**

### Mandatory W&B schedule checks

Before trusting a run, verify:

```text
ftp/schedule_is_official_anyflow = 1
ftp/train_eval_schedule_match_required = 1
ftp/local_anchor_delta_was_clipped = 0
```

Also inspect `ftp/source_timestep` and `ftp/target_timestep`. No other transition
pairs are acceptable in the main experiment.

## 3. Do not resume checkpoints from the broken schedule

The old W&B group and checkpoints were generated from a different rollout and a
different step-zero baseline. They must not be resumed into the corrected code.

Start a new W&B comparison group and a new output directory after any change to:

- the schedule;
- flow shift;
- number of deployment steps;
- branchable transition set;
- model checkpoint;
- reward checkpoint or runtime;
- validation prompt/seed set.

A checkpoint can only be resumed when all of those remain identical.

## 4. Model-loading contract

The model must be:

```text
nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
```

Required architecture settings:

```yaml
r_embedder: true
r_embedder_fusion: gated
r_embedder_gate_value: 0.25
r_embedder_deltatime_type: r
flow_shift: 5.0
```

Do not silently fall back to base Wan. Base Wan may load and generate, but it is
not the two-time finite-map model being studied.

The model forward used for finite maps must receive both `timestep` and
`r_timestep`. The deterministic update is:

```text
x_r = x_t - ((t-r)/1000) * u_theta(x_t, t, r, prompt)
```

When local ASFMC needs the instantaneous reverse velocity at the target, the
code queries the two-time model with `(r, r)`. If this behavior is changed,
compare it numerically against the official Flow-Map-GRPO/AnyFlow convention
before launching a scientific run.

## 5. Local-anchor ASFMC guardrails

The RL action is the sampled next latent `a = x_r`, not the velocity prediction.
Local-anchor ASFMC defines a short stochastic conditional around the
deterministic finite target.

In AnyFlow reverse-time coordinates:

```text
q = r / 1000
s = max(1-q, terminal_base_sigma)
mean = x_r - delta * lambda^2 * (x_r / s + v_q(x_r, r, r))
std  = lambda * sqrt(2*q/s) * sqrt(delta)
```

Main settings:

```yaml
local_anchor_delta: 0.03
local_noise_scale: 0.7
local_terminal_base_sigma: 0.05
```

With the corrected deployment grid, every branch target is at least `q=0.625`,
so the local interval should never be clipped at the data endpoint.

Treat any of the following as a setup failure, not a harmless warning:

- non-positive or non-finite policy standard deviation;
- `local_anchor_delta_was_clipped = 1` in the main run;
- action RMS that differs by an order of magnitude between branch indices;
- a branch target close to zero;
- changing the velocity sign or time-coordinate conversion without reference
  parity tests.

Do not add arbitrary clamps to only one objective. A sampler correction must be
shared by FTPP and GRPO.

## 6. Counterfactual rollout invariants

Within one objective run, all four candidates in a group must share:

- the same prompt;
- the same initial Gaussian latent;
- the same branch index;
- the same deterministic prefix state;
- the same policy mean and standard deviation before rank-specific sampling;
- the same deterministic suffix implementation;
- the same reward model instance and checkpoint.

Each rank then receives a different action-noise seed and samples one candidate.
With four GPUs and group size four, each rank owns exactly one action.

At 81 frames and 480 x 832, the expected per-rank latent layout is approximately:

```text
[1, 21, 16, 60, 104]
```

The distributed candidate group is conceptually:

```text
[4, 21, 16, 60, 104]
```

Do not confuse seed matching across FTPP and GRPO with literal shared actions.
Once their weights diverge, equal action noise produces different actions
because the policy means differ. The current comparison is a seed-matched
on-policy comparison, not a frozen shared-rollout comparison.

## 7. Reward-model checks

The main experiment optimizes:

```text
videoalign_mq
```

and holds out:

```text
videoalign_vq
videoalign_ta
```

VideoAlign must use the same immutable cached Qwen base and reward checkpoint in
both arms.

Before a scientific run:

1. Record missing and unexpected keys after the **final** reward checkpoint load.
2. Assert high base-model, adapter, and reward-head parameter coverage.
3. Run known calibration videos through the patched runtime and compare with the
   upstream environment.
4. Do not replace VideoAlign with another scorer merely to bypass a loading
   failure.

A stable nontrivial score is not sufficient evidence that the correct reward
weights loaded.

## 8. Objective interpretation and scale

With one on-policy update, the GRPO ratio is one at the gradient evaluation and
clipping is inactive. The practical comparison is:

```text
GRPO: standardized linear reward coefficients
FTPP: centered exponential reward coefficients
```

At `G=4` and target `ESS=2`, their effective coefficient L2 norms are already
matched. Do not assume FTPP is under-updating solely because its raw coefficient
values look smaller.

The current FTPP loss is a centered score-function update, not a full finite
forward-KL minimization with a unique optimum. Describe it precisely when
interpreting results.

The fixed per-group ESS rule also normalizes away absolute reward spread. A weak
but non-degenerate ranking can receive the same update norm as a strong ranking.
Do not change this during the first corrected-grid rerun; isolate the schedule
fix. If the corrected run remains flat, a global or EMA-calibrated temperature
is the next clean ablation.

## 9. Validation must be paired

Validation uses fixed prompts and seeds, so baseline and current measurements
are paired. Do not use:

```text
1.96 * sqrt(SEM_baseline^2 + SEM_current^2)
```

as the primary significance test. That treats correlated measurements as
independent and made the old success margin roughly 0.25 for a desired effect of
0.02.

Retain one scalar row for every `(prompt_index, sample_seed)` and compute:

```text
d_i = reward_current_i - reward_baseline_i
paired_mean_delta = mean(d_i)
paired_sem = std(d_i) / sqrt(n)
paired_bootstrap_95pct_interval
```

Log the per-sample table or artifact to W&B so the statistic can be audited.

Also evaluate both:

- the raw LoRA student;
- the EMA student.

EMA decay `0.99` has an effective averaging window near 100 updates and can hide
an early raw-model trend.

## 10. What 100 training steps can and cannot establish

One update uses one prompt group and four reward-labeled videos. Therefore:

```text
100 updates = 100 prompt groups = 400 candidate videos
```

That is enough to detect:

- a sign error;
- divergence;
- static collapse;
- diversity collapse;
- a very large early improvement.

It is not enough to establish a small `0.01-0.02` held-out reward difference
with the old aggregate statistics.

For the corrected grid, run evaluations at:

```text
step 0
step 50
step 100
step 150
step 200
```

Do not immediately spend 1,200 steps. Continue only when the paired held-out
curve has a credible positive trend without quality, motion, or diversity loss.

## 11. Required W&B diagnostics

Training:

```text
ftp/source_timestep
ftp/target_timestep
ftp/branch_index
ftp/schedule_is_official_anyflow
ftp/train_eval_schedule_match_required
ftp/local_anchor_delta_was_clipped
ftp/posterior_std
ftp/action_deviation_from_deterministic
ftp/reward_std
ftp/reward_selection_gain
ftp/posterior_ess
ftp/posterior_temperature
ftp/posterior_weight_max
ftp/grad_norm
ftp/post_update_approx_kl
ftp/post_update_logprob_delta_abs
ftp/train_step_seconds
ftp/cumulative_gpu_hours
```

Post-update probe metrics are only measured on configured probe steps. A zero on
other steps is missing data, not measured zero.

Validation:

```text
held-out MQ/VQ/TA means
raw-model and EMA-model means
per-prompt paired deltas
paired confidence interval
motion ratio to base
latent diversity ratio to base
video diversity ratio to base
static-video ratio
reward gain per GPU-hour
```

Training reward alone is never a success criterion.

## 12. Stop and continue rules

After the corrected step-200 evaluation:

Continue when:

- paired held-out MQ trends positive;
- VQ and TA stay within retention tolerances;
- motion and diversity remain intact;
- raw and EMA trends are coherent;
- improvement is not isolated to one branch index.

Stop FTPP when:

- corrected GRPO improves while FTPP does not;
- FTPP remains tied within paired uncertainty;
- post-update KL grows without deterministic evaluation gain;
- reward gains occur only on training branches and not held-out four-step
  inference.

Stop both and audit the reward/generalization setup when both training rewards
rise but both held-out curves remain below the untouched model.

If the corrected likelihood objectives remain tied, the more distinct follow-up
is finite-velocity posterior regression: convert preferred actions into a
reward-corrected deterministic AnyFlow velocity target instead of continuing to
tune two nearly identical score-function estimators.

## 13. Pre-launch checklist

Run before allocating a full paired job:

```bash
pytest -q \
  fastvideo/tests/train/methods/test_anyflow_schedule.py \
  fastvideo/tests/train/methods/test_local_asfmc.py \
  fastvideo/tests/train/methods/test_finite_transition_posterior_core.py \
  fastvideo/tests/train/methods/test_finite_transition_posterior_method.py \
  fastvideo/tests/train/methods/test_finite_transition_posterior_repro.py
```

Then run:

```bash
modal run modal_train_finite_transition_posterior.py \
  --smoke \
  --paired \
  --comparison-id ftpp_gridfix_smoke_s42
```

Inspect the actual W&B timesteps before launching the 200-step run.

Fresh corrected experiment:

```bash
modal run modal_train_finite_transition_posterior.py \
  --paired \
  --max-train-steps 200 \
  --validation-every 50 \
  --comparison-id ftpp_gridfix_s42_r1
```

Do not resume the pre-grid-fix runs.

## 14. Source-of-truth files

```text
examples/train/configs/rl/wan/finite_transition_posterior_anyflow_videoalign.yaml
examples/train/finite_transition_posterior_postmortem.md
examples/train/finite_transition_posterior_progress_report.md
fastvideo/train/methods/rl/common/anyflow_schedule.py
fastvideo/train/methods/rl/common/local_asfmc.py
fastvideo/train/methods/rl/common/finite_transition.py
fastvideo/train/methods/rl/finite_transition_posterior.py
fastvideo/train/methods/rl/finite_transition_posterior_repro.py
modal_train_finite_transition_posterior.py
```

When a runtime repair conflicts with this document, do not guess. Check the
released AnyFlow and Flow-Map-GRPO implementations, write a focused parity test,
and only then change the scientific path.