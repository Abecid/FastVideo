# Finite-Transition Posterior Projection for AnyFlow-Wan

This experiment tests one narrow hypothesis:

> Once ASFMC has converted a deterministic few-step flow map into a valid
> stochastic local policy, directly fitting the KL-regularized reward posterior
> should be more sample-efficient and less destructive than a clipped
> likelihood-ratio update.

The rollout is held fixed between the proposed method and the baseline. The only
scientific delta is the update rule.

## Method

A released four-step AnyFlow-Wan model defines finite transitions

```text
x_r = x_t - ((t-r)/N) u_theta(x_t, t, r, c).
```

At each optimizer step:

1. Select one of four trainable finite transitions.
2. Generate one deterministic shared prefix for a single prompt.
3. Use endpoint-anchored ASFMC to sample four valid candidate next states from
   that exact shared state.
4. Complete each branch deterministically and score the resulting video.
5. Form the local KL-regularized policy-improvement posterior

   ```text
   q_R(a|s) proportional to q_old(a|s) exp(R(a)/tau).
   ```

   The temperature is solved so that posterior ESS is half the branch group.
6. Project this posterior into the current LoRA policy with

   ```text
   L = - sum_j (w_j - 1/G) log pi_theta(a_j|s).
   ```

The `1/G` behavior-score baseline gives an exactly zero update when the reward
cannot distinguish the branches. Only one inner epoch is used, keeping the
control-variate argument on-policy.

## Matched baseline

Set:

```yaml
method:
  objective: flowmap_grpo
```

The baseline uses the same:

- AnyFlow checkpoint and four-step deterministic evaluation;
- shared prompt, initial noise and prefix;
- ASFMC candidate actions;
- branch completions and reward calls;
- optimizer, LoRA, prompts and validation seeds.

Only the loss changes to clipped Flow-Map GRPO. This directly tests whether
posterior projection is the better optimizer for the same local policy data.

## Why this is a useful open question

Flow-Map GRPO establishes that ASFMC makes few-step flow maps amenable to online
RL. AWM and RAM independently show that diffusion alignment becomes dramatically
more efficient when it is returned to matching/regression geometry rather than
high-variance trajectory PPO. The unresolved question is whether a deterministic
finite flow map should be optimized through likelihood ratios at all once its
local reward-improved posterior is available explicitly.

This experiment is intentionally not a stack of new components. ASFMC and
shared-prefix branching are the controlled rollout substrate. The proposed
single delta is:

```text
clipped policy-gradient update
        ->
direct local posterior projection.
```

## Evaluation protocol

The preparation script creates a deterministic disjoint train/validation split.
Validation always uses fixed prompts, fixed seeds, EMA weights and deterministic
four-step AnyFlow inference.

The default run optimizes VideoAlign motion quality (`videoalign_mq`) and treats
visual quality and text alignment as held-out anti-reward-hacking metrics.

Every training step logs:

- optimized and held-out branch reward means/stds;
- reward selection gain from posterior reweighting;
- posterior ESS, temperature, entropy and maximum weight;
- branch timestep and ASFMC posterior standard deviation;
- action distance from the deterministic map and posterior mean;
- branch-video temporal L1;
- gradient norm;
- periodic post-update approximate KL and log-probability displacement;
- step time and cumulative GPU-hours.

Every validation logs per-prompt means, standard deviations and standard errors
for:

- all reward heads;
- adjacent-frame temporal L1;
- static-video ratio;
- latent pairwise diversity across fixed seeds;
- decoded-video pairwise diversity;
- qualitative W&B videos with prompt and metrics.

It also logs every metric's delta from the untouched step-zero checkpoint.

## What counts as success

A smoke test only proves engineering correctness. A scientific run is considered
successful only if all of the following hold on the held-out prompt set:

1. **Primary improvement:** motion reward increases by at least the configured
   absolute margin and exceeds the combined 95% standard-error margin.
2. **No reward tradeoff:** held-out VQ and TA do not fall by more than their
   configured tolerances.
3. **No static collapse:** temporal L1 remains at least 90% of the step-zero
   baseline.
4. **No diversity collapse:** latent pairwise diversity remains at least 80% of
   baseline.
5. **Efficiency win:** posterior projection either reaches the matched GRPO
   baseline's best held-out reward in fewer updates/GPU-hours or achieves a
   higher reward-vs-GPU-hour area under the curve.

The W&B boolean `validation_success/all` encodes the first four gates. The fifth
is assessed by comparing the paired W&B curves.

## Prepare locally

```bash
python examples/train/prepare_finite_transition_posterior_assets.py \
  --check-rewards \
  --objective posterior_projection \
  --dataset world-r1-enhanced-dynamic \
  --reward videoalign_mq \
  --max-train-prompts 512 \
  --validation-prompts 64 \
  --json
```

Then run the generated config printed by the script:

```bash
bash examples/train/run.sh outputs/ftp_configs/<generated-config>.yaml
```

## Modal smoke test

```bash
modal run modal_train_finite_transition_posterior.py --smoke
```

This uses 17 frames, a smaller spatial shape, two updates and validation every
step. It must not be compared against scientific runs.

## Modal scientific run

```bash
modal run modal_train_finite_transition_posterior.py \
  --objective posterior_projection \
  --max-train-steps 1200
```

## Launch the matched pair

```bash
modal run modal_train_finite_transition_posterior.py \
  --paired \
  --max-train-steps 1200
```

The launcher first materializes the deterministic train/validation split once,
then starts posterior projection and Flow-Map-GRPO jobs against the same cached
parquets and scientific settings.
