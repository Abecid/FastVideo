# Actual GRPO v3 execution guide

Read this before launching or repairing the next AnyFlow reward-alignment run.

## VideoAlign metrics and the training objective

VideoAlign exposes three separate heads:

| Head | Meaning | Used by the GRPO v3 training loss? |
|---|---|---|
| `videoalign_mq_audited` | Motion Quality (MQ) | **Yes — the only online reward** |
| `videoalign_vq_audited` | Visual Quality (VQ) | No; held-out validation only |
| `videoalign_ta_audited` | Text Alignment (TA) | No; held-out validation only |

The v2 runs did **not** accidentally optimize VQ or TA against MQ. Their online
reward dictionary contained only audited MQ, and `optimize_reward` selected MQ.
VQ and TA were evaluated only on the fixed validation set as anti-reward-hacking
and quality-retention checks.

The v3 config preserves this contract:

```yaml
method:
  optimize_reward: videoalign_mq_audited
  reward_fn:
    rewards:
      videoalign_mq_audited: 1.0

  validation_reward_fn:
    rewards:
      videoalign_mq_audited: 1.0
      videoalign_vq_audited: 1.0
      videoalign_ta_audited: 1.0
```

Do not add VQ or TA to `reward_fn` while debugging MQ. That would change the
scientific question from "can GRPO improve MQ?" to a multi-objective tradeoff and
would make a flat MQ curve ambiguous.

## Why this run is different from v2

The v2 implementation collected one rollout buffer, computed every loss while
the learner still equaled the behavior policy, and then took one optimizer step.

Therefore, at every gradient evaluation:

```text
pi_theta = pi_old
ratio = exp(log pi_theta - log pi_old) = 1
```

PPO clipping was inert. The arm was a group-normalized policy-gradient update,
not a meaningful clipped GRPO optimization loop.

GRPO v3 freezes the rollout buffer and reuses it:

```text
4 prompt groups
x 8 candidates
= 32 terminal MQ scores per outer rollout iteration

2 policy epochs
x 4 one-group minibatches
= up to 8 optimizer steps per rollout iteration
```

After the first optimizer minibatch, later minibatches see `pi_theta != pi_old`.
The likelihood ratio and clip fraction are now real optimization quantities.

## Reward normalization

The v2 denominator pooled raw rewards across different prompts. Between-prompt
difficulty can dominate the within-prompt candidate differences that GRPO needs.

V3 uses the reference Wan-style estimator independently for each prompt group:

```text
A_gj = (R_gj - mean_j R_gj) / (std_j R_gj + epsilon)
```

A group with reward standard deviation below the configured threshold produces
an exact zero update. The main run does not use the v2 running cross-prompt
reward scale.

## The exact run being tested

Source config:

```text
examples/train/configs/rl/wan/
finite_transition_grpo_v3_anyflow_videoalign.yaml
```

Source method:

```text
fastvideo/train/methods/rl/finite_transition_grpo_v3.py
```

Authoritative launcher:

```text
modal_train_finite_transition_grpo_v3.py
```

Default scientific settings:

```text
model:                         AnyFlow-Wan 1.3B
training resolution:           81 frames, 480x832
GPUs:                           4 H100
candidate group size:           8
prompt groups / rollout:        4
terminal reward videos/rollout: 32
stochastic transitions/video:   4
policy epochs:                  2
groups/minibatch:               1
optimizer steps/rollout:        up to 8
outer rollout iterations:       20
maximum optimizer steps:        160
learning rate:                  1e-5
PPO clip range:                 0.02
old-policy KL target:           3e-5
early-stop threshold:           1.2e-4
online reward:                  audited MQ only
held-out rewards:               audited MQ, VQ, TA
validation:                     every 5 outer iterations
validation prompts:             128
samples/prompt:                 2
```

`max_train_steps` means fresh rollout buffers, not optimizer steps.

## Required execution order

### 1. Pull the branch

```bash
git fetch origin
git switch adam/finite-transition-alignment
git pull --ff-only origin adam/finite-transition-alignment
```

Never resume a v2 checkpoint into v3. The optimizer schedule, reward
normalization, rollout reuse, EMA cadence, and checkpoint state changed.

### 2. Run the real smoke

```bash
modal run modal_train_finite_transition_grpo_v3.py \
  --smoke \
  --comparison-id grpo_v3_smoke_s42
```

Smoke mode uses:

```text
2 outer rollout iterations
1 policy epoch
2 prompt groups/rollout
4 candidates/group
17 frames
256x448
8 validation prompts
```

The smoke must show:

```text
grpo_v3/optimizer_steps_this_rollout > 0
grpo_v3/ratio_abs_deviation_max > 0 after the first minibatch
grpo_v3/old_logprob_recompute_max_error <= 0.002
grpo_v3/online_reward_is_mq_only = 1
finite gradients and analytic KL
raw and EMA validation complete
VideoAlign audit coverage = 1.0
```

A ratio-deviation value of exactly zero across the whole run means we have again
failed to run actual multi-minibatch GRPO.

### 3. Run the 20-rollout scientific gate

```bash
modal run modal_train_finite_transition_grpo_v3.py \
  --max-train-steps 20 \
  --validation-every 5 \
  --validation-prompts 128 \
  --validation-samples-per-prompt 2 \
  --comparison-id grpo_v3_mq_s42_r1
```

This is a fresh run. Do not resume the least-negative v2 `2e-5` arm.

### 4. Resume only when the curve is promising

Use the original run name and the latest checkpoint:

```bash
modal run modal_train_finite_transition_grpo_v3.py \
  --run-name <existing-run-name> \
  --resume-from-checkpoint latest \
  --max-train-steps 40 \
  --skip-preprocess \
  --comparison-id grpo_v3_mq_s42_r1
```

Resume only when raw or EMA MQ is trending upward and the trust-region metrics
remain stable. `max_train_steps` is the final outer-step target, not the number
of additional steps.

## W&B metrics that decide whether this worked

### Is this actual GRPO?

```text
grpo_v3/optimizer_steps_this_rollout
grpo_v3/optimizer_steps_total
grpo_v3/policy_epochs_completed
grpo_v3/ratio_mean
grpo_v3/ratio_min
grpo_v3/ratio_max
grpo_v3/ratio_abs_deviation_max
grpo_v3/clip_fraction_mean
grpo_v3/old_policy_kl_mean
grpo_v3/old_policy_kl_max
grpo_v3/early_stopped
```

Expected qualitative behavior:

```text
first minibatch: ratio approximately 1
later minibatches: nonzero ratio deviation
clip fraction: may be zero early, but must be a meaningful measured quantity
old-policy KL: nonzero but below the early-stop region
```

### Is the reward group informative?

```text
grpo_v3/group_reward_std_mean
grpo_v3/group_reward_std_min
grpo_v3/group_reward_std_max
grpo_v3/active_group_fraction
grpo_v3/advantage_abs_mean
grpo_v3/online_mq_mean
grpo_v3/online_mq_std
```

If most groups are inactive or their standard deviation is near numerical
precision, MQ does not provide a useful candidate ranking at this group size.

### Does stochastic preference move the deterministic map?

```text
grpo_v3/deterministic_preference_alignment
grpo_v3/deterministic_map_shift_rms
grpo_v3/preferred_action_shift_rms
grpo_v3/deployment_mq_before
grpo_v3/deployment_mq_after
grpo_v3/deployment_mq_update_delta
grpo_v3/candidate_mean_minus_deployment
grpo_v3/candidate_max_minus_deployment
```

Interpretation:

```text
candidate gain positive + deployment delta positive:
    stochastic selection transfers locally to deterministic inference

candidate gain positive + deployment delta approximately zero:
    policy update is too weak or mean transfer is poor

candidate gain positive + deployment delta negative:
    stochastic reward preference is actively misaligned with the deployed map
```

### Scientific held-out outcome

Raw and EMA validation retain the v2 exact paired evaluator. Inspect:

```text
validation_paired_delta_raw/reward/videoalign_mq_audited
validation_paired_ci95_low_raw/reward/videoalign_mq_audited
validation_paired_ci95_high_raw/reward/videoalign_mq_audited

validation_paired_delta/reward/videoalign_mq_audited
validation_paired_ci95_low/reward/videoalign_mq_audited
validation_paired_ci95_high/reward/videoalign_mq_audited
```

Also inspect held-out VQ/TA, motion, and diversity.

## Success and stop rules

Continue beyond 20 outer iterations only if at least one of raw or EMA MQ:

1. has a positive paired mean delta;
2. has a visibly improving checkpoint trajectory;
3. does not trade away more than `0.02` VQ or TA;
4. retains at least 90% of baseline motion;
5. retains at least 80% of latent diversity;
6. has stable old-policy KL and finite gradients.

A publication-strength success still requires:

```text
MQ delta >= +0.02
paired 95% lower confidence bound > 0
VQ/TA retention gates pass
motion/diversity gates pass
```

Stop at or before outer step 20 when:

- raw and EMA MQ are both negative and not improving;
- candidate MQ gain is positive but deterministic deployment delta is repeatedly
  non-positive;
- deterministic-preference alignment is consistently non-positive;
- old-policy KL repeatedly trips the early-stop threshold;
- VQ, TA, motion, or diversity degrade materially.

If true GRPO v3 still fails while candidate selection looks strong, the next
experiment is the small direct finite-velocity gate. Do not jump from another
negative MQ run directly into a long posterior experiment.

## Optional reference KL

The implementation supports:

```bash
--reference-kl-beta 0.004
```

but the first run keeps it at zero. V2 showed no meaningful motion/diversity
collapse; adding a base-model KL penalty immediately would further suppress an
already weak learning signal.

Use the reference penalty only as a named follow-up when actual GRPO begins to
improve MQ but harms VQ, TA, motion, or diversity.

## Runtime repair invariants

Safe repairs:

- import/dependency fixes;
- memory-safe sequential transition recomputation;
- checkpoint/resume serialization;
- W&B naming or artifact upload;
- deterministic seed and distributed synchronization fixes.

Changes requiring a new W&B group and fresh checkpoint root:

- adding VQ or TA to the online reward;
- changing group normalization;
- changing policy epochs or minibatch structure;
- changing the AnyFlow schedule or local-ASFMC policy;
- changing clip range, learning rate, KL target, or reference-KL coefficient;
- changing the validation prompts/seeds;
- changing LoRA rank/targets.

The coding agent must not quietly "stabilize" the run by reverting it to one
optimizer step per rollout. That would recreate the v2 failure mode.
