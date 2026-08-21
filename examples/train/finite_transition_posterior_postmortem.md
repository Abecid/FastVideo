# FTPP 100-step postmortem and corrected rerun protocol

This note reviews the first paired FTPP/GRPO run described in
`finite_transition_posterior_progress_report.md`. It records the implementation
bug found after that run, what can and cannot be concluded from the results, and
the minimum corrected experiment needed before judging the method.

## Executive conclusion

The first run does **not** establish an FTPP win, but it also should not be used
to reject the attack vector. The run sampled and evaluated AnyFlow with the
wrong timestep grid.

The executed method reused FastVideo's generic
`FlowMatchEulerDiscreteScheduler`. That scheduler starts from a non-zero
training `sigma_min` and applies the flow shift while rebuilding the inference
grid. AnyFlow's released `FlowMapDiscreteScheduler` instead builds

```text
linspace(1, 0, K + 1)[:-1]
```

and applies the shift exactly once before appending the clean endpoint.

At shift 5, the executed and released four-step grids were:

```text
executed: [1000, 909.707, 717.317, 24.414, 0]
released: [1000, 937.500, 833.333, 625.000, 0]
```

The five-step training grids were:

```text
executed: [1000, 937.889, 834.712, 629.634, 24.414, 0]
released: [1000, 952.381, 882.353, 769.231, 555.556, 0]
```

This is not a cosmetic difference. The executed grid moved nearly the whole
late transport into a transition ending at `q ~= 0.024`, then left an almost
empty final transition. It also forced local-ASFMC delta clipping on one quarter
of updates. The released AnyFlow grid has no such near-zero branch.

The branch now overrides schedule construction in
`ReproducibleFiniteTransitionPosteriorMethod` and locks the released values in
CPU tests. Do not resume the old run into the corrected implementation: the
rollout distribution and validation baseline changed. Start a new W&B group.

## What the first run actually showed

At step 100 both methods were close to the untouched model and close to each
other:

```text
held-out MQ delta
  FTPP: -0.015262
  GRPO: -0.013678

FTPP minus GRPO MQ: -0.001584
```

Motion and diversity were retained. FTPP retained slightly more TA, while GRPO
retained more VQ. Those differences are far below the evaluation uncertainty
and must not be presented as a win for either side.

The defensible statement is only:

> The distributed local-ASFMC/reward/optimization pipeline ran stably for 100
> updates without motion or diversity collapse, but neither update rule showed
> held-out reward improvement on the mis-scheduled AnyFlow rollout.

## Why step 100 was inconclusive even without the scheduler bug

One update uses one prompt and four candidate videos. Therefore 100 updates are
only 100 prompt groups and 400 reward-labeled videos. That is enough to expose a
sign error, divergence, static collapse, or a genuinely dramatic early gain. It
is not enough by itself to establish a small effect.

The held-out MQ standard deviation in the report was approximately `0.737` over
64 prompt-level observations, giving an unpaired SEM near `0.092`. The existing
success margin incorrectly combines baseline and current SEM as though they
were independent, despite using fixed prompt/seed pairs. Validation must retain
per-prompt paired deltas and compute a paired bootstrap or paired-difference
SEM.

Even paired evaluation needs substantial correlation to resolve a `0.02`
effect with only 64 prompts. The next run should log the paired data rather than
trying to infer significance from aggregate means.

## The current FTPP and GRPO updates are closer than their names suggest

For one on-policy update, the GRPO likelihood ratio is one at the gradient
evaluation, so clipping is inactive. The comparison is primarily between two
score-function coefficient vectors:

```text
GRPO: standardized linear advantages
FTPP: centered exponential reward weights
```

With group size `G=4` and target `ESS=2`, FTPP has

```text
c_j = w_j - 1/4
sum_j c_j^2 = 1/ESS - 1/G = 1/4.
```

GRPO's population-standardized advantages satisfy `sum_j A_j^2 = G`; after DDP
averaging, `sum_j (A_j/G)^2 = 1/G = 1/4`. The two methods therefore have exactly
the same coefficient L2 norm at this operating point. Their directions are also
usually strongly aligned because both are monotone transformations of the same
four rewards.

This means the first run was not testing a radically lower-variance optimizer.
The centered FTPP loss is still a score-function update over Gaussian actions;
it is not the regression-style correction used by methods such as AWM or RAM.
A tie with matched GRPO is therefore unsurprising.

## Second design issue: fixed ESS forces a full update on weak groups

The current reward tilt standardizes each four-sample reward group and solves a
new temperature that forces `ESS=2`. Consequently, absolute reward spread is
removed. Any non-degenerate group receives the same coefficient norm even when
its reward differences are tiny.

That is useful as a hard per-state trust region, but it is not the same as a
fixed-temperature KL-regularized posterior. It can turn weak or noisy local
rankings into constant-magnitude random updates.

Do not change this in the first corrected scheduler rerun; otherwise the effect
of the scheduler repair cannot be isolated. If the corrected run is still flat,
the next ablation should replace per-group fixed ESS with a slowly adapted
global reward scale/temperature that allows nearly uniform weights and nearly
zero updates on flat groups.

## Corrected experiment sequence

### 1. Real smoke test

Run both objectives with the corrected branch and verify the W&B source/target
timesteps. Expected five-step training nodes are approximately:

```text
1000, 952.381, 882.353, 769.231, 555.556, 0
```

Expected deterministic four-step validation nodes are approximately:

```text
1000, 937.500, 833.333, 625.000, 0
```

`ftp/local_anchor_delta_was_clipped` should remain zero on every branch.

### 2. New 200-step paired run

Do not resume the old run. Launch a fresh paired W&B group for at least 200
updates. Evaluate at step 0, 50, 100, and 200. The corrected schedule changes
the untouched baseline, so all deltas must be recomputed from the new step-zero
checkpoint.

### 3. Add paired and raw-model diagnostics

For each fixed `(prompt, seed)` pair, retain the baseline and current rewards and
log:

```text
paired mean delta
paired delta standard error
paired bootstrap 95% interval
```

Also evaluate the raw student and EMA model separately at early checkpoints.
EMA decay `0.99` has an effective window around 100 updates and can obscure an
early raw-model trend.

### 4. Stop rule

After the corrected step-200 evaluation:

- continue if paired held-out MQ has a positive trend without VQ/TA, motion, or
  diversity damage;
- stop FTPP if corrected GRPO improves and FTPP does not;
- stop both and audit reward/generalization if training branch reward rises but
  deterministic held-out reward remains below baseline for both;
- do not spend 1200 steps solely because the job is stable.

## If the corrected run remains tied

Treat the centered likelihood version of FTPP as a weak/dead-end contribution.
The broader finite-transition posterior idea is still testable through a more
distinct objective.

The clean next method is posterior-mean finite-velocity regression. From the
same shared state and actions, define

```text
c_j = w_j - 1/G
g_j = (x_t - a_j) / Delta_t_to_r
Delta_u = sum_j c_j * g_j
u_target = u_old + eta * Delta_u.
```

Then regress AnyFlow's deterministic finite velocity directly toward
`u_target`, with `eta` controlled by a target finite-transition KL or RMS
change. This uses reward to correct the transition that deterministic four-step
inference actually executes, rather than optimizing the likelihood of an
auxiliary stochastic policy. It is the more meaningful PT-PDD-style follow-up.

## Source-of-truth implementation files

```text
fastvideo/train/methods/rl/common/anyflow_schedule.py
fastvideo/train/methods/rl/common/local_asfmc.py
fastvideo/train/methods/rl/finite_transition_posterior.py
fastvideo/train/methods/rl/finite_transition_posterior_repro.py
fastvideo/tests/train/methods/test_anyflow_schedule.py
fastvideo/tests/train/methods/test_finite_transition_posterior_core.py
examples/train/finite_transition_posterior_progress_report.md
```
