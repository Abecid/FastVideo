# Finite-transition v2 execution report — 2026-08-23

## Executive conclusion

The repaired finite-transition v2 substrate is operational and auditable, but
the GRPO learning-rate sweep did **not** pass the scientific learning gate.
All three learning rates completed 20 updates, exact paired raw/EMA validation,
qualitative logging, and checkpointing. None produced positive held-out
VideoAlign MQ movement at step 20. The best arm, `2e-5`, was still negative:

```text
raw MQ delta = -0.008804, paired 95% CI [-0.027663, +0.010221]
EMA MQ delta = -0.003155, paired 95% CI [-0.019160, +0.014012]
```

The mandatory execution plan says not to test a novel objective until GRPO
demonstrates positive raw or EMA paired held-out MQ without unacceptable
quality, motion, or diversity loss. Therefore the 100-step GRPO baseline,
GRPO-versus-posterior comparison, and velocity follow-up were **not launched**.
This is an intentional scientific stop, not a runtime failure.

## Code and execution provenance

- Repository: `Abecid/FastVideo`
- Branch: `adam/finite-transition-alignment`
- Recovery head pulled before execution: `dc8317b9`
- Runtime/scientific repair commits produced during execution:
  - `6e6ed5a8` — preserve finite-transition likelihood precision
  - `c636bfdf` — measure finite-transition policy KL analytically
  - `3da9a356` — keep v2 reward maps contract-safe
- Source-of-truth guide:
  `examples/train/finite_transition_v2_execution_plan.md`
- Mandatory guardrails:
  `examples/train/finite_transition_posterior_execution_guardrails.md`
- Authoritative launcher: `modal_train_finite_transition_v2_complete.py`
- Model: `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- Seed: `42`

## Implementation and runtime repairs

### 1. Low-precision likelihood reduction

The bf16 diagonal-Gaussian likelihood path rounded small policy changes to
zero. That made sampled post-update log-probability deltas and the old KL proxy
look exactly zero even when LoRA weights changed.

The likelihood implementation now promotes the relevant arithmetic and
reduction to float32 while retaining gradients to the bf16 policy mean. A bf16
regression test locks the behavior, and the authoritative Modal preflight runs
that test on every launch.

### 2. Cancellation-prone KL telemetry and control

The previous KL proxy squared a dimension-averaged sampled log-probability
difference. Positive and negative coordinate contributions could cancel before
squaring, understating actual policy movement by many orders of magnitude.

The v2 path now computes the coordinate-normalized analytic KL between the old
and new diagonal Gaussian policies. The implementation uses a numerically
stable expression, including `expm1` for small log-standard-deviation changes.
The target-KL controller and W&B telemetry now use this analytic value. Unit
tests cover equality, controlled mean shifts, scale shifts, and bf16 inputs.

The corrected luminance diagnostic subsequently logged analytic KL around
`1e-6` while producing a decisive deterministic luminance improvement, proving
that the common optimization substrate can transfer an aligned reward signal
to the deployed deterministic map.

### 3. Recursive reward-map merge violated the audited contract

The first VideoAlign LR-sweep launch stopped before any optimizer update with:

```text
RuntimeError: Online VideoAlign rollouts must score MQ only
```

The launcher recursively merged a generated config with a preset. That
combined a legacy MQ/VQ/TA online reward dictionary with the audited v2 MQ-only
dictionary, even though both inputs looked valid in isolation.

The repair:

- uses `examples.train.prepare_finite_transition_v2_assets`;
- selects `videoalign_mq_audited` for online training;
- treats `reward_backend`, `reward_fn`, `optimize_reward`,
  `validation_reward_backend`, `validation_reward_fn`, and
  `videoalign_audit` as atomic during generated/preset merging;
- retains audited MQ/VQ/TA only for held-out validation;
- adds an exact regression test for the generated-plus-preset merge; and
- expands the Modal preflight to include the v2 config suite.

The repaired smoke and all sweep arms resolved to the required contract:

```text
online reward: audited MQ only
validation rewards: audited MQ, VQ, and TA
group size: 8 candidate actions
rollout groups per update: 4
reward videos per update: 32
stochastic transitions per trajectory: 4
official AnyFlow schedule: [1000, 937.5, 833.3333, 625, 0]
```

## Verification runs

### Analytic-KL smoke

- W&B: [l5me6e0k](https://wandb.ai/adamlee00/finite-transition-v2-wan/runs/l5me6e0k)
- Result: completed the H100 smoke path with nonzero analytic KL and no
  numerical/runtime failure.

### Corrected luminance systems gate

- Modal app: `ap-z3jDtJyklAT4Y6lyRhfAcw`
- W&B: [a4e9vwj8](https://wandb.ai/adamlee00/finite-transition-v2-wan/runs/a4e9vwj8)
- Duration: 39m52s; 2.487 H100 GPU-hours
- Result: 30 updates, 56 media files, `checkpoint-30`, clean exit
- Raw luminance delta: `+0.007291`, CI `[+0.006379, +0.008197]`
- EMA luminance delta: `+0.000948`, CI `[+0.000738, +0.001166]`
- Raw motion ratio: `1.00687`
- Raw latent-diversity ratio: `0.99905`
- Final analytic KL: `1.809e-6`

This gate passed decisively. It rules out a globally broken optimizer,
likelihood, LoRA-gradient, or deterministic-evaluation path.

### Audited VideoAlign reward-contract smoke

- Modal app: `ap-LHz8rhCUDhGh9Pb2xAYwcF`
- W&B: [9bpuwwwq](https://wandb.ai/adamlee00/finite-transition-v2-wan/runs/9bpuwwwq)
- Result: 46 H100 preflight tests passed; two GRPO updates, raw/EMA
  validation, 24 media files, `checkpoint-2`, clean exit
- Audit: base/adapter/checkpoint tensor/checkpoint numel/reward-head coverage
  all `1.0`; unmatched keys `0`; repeat delta max `0`

## GRPO learning-rate sweep

### Configuration

Command:

```bash
modal run modal_train_finite_transition_v2_complete.py \
  --lr-sweep \
  --validation-prompts 128 \
  --comparison-id ftv2_grpo_lr_rewardfix_s42_r1
```

- Modal app: `ap-q4TTOifWdsEczCQXNoFHqi`
- Three concurrent arms, each on 4 H100 GPUs
- Learning rates: `2e-6`, `2e-5`, `6e-5`
- 20 updates per arm; target-KL controller disabled for calibration
- 387 training prompts and 128 fixed validation prompts
- Two fixed validation samples per prompt
- Raw and EMA validation at steps 0, 10, and 20
- Shared step-zero MQ/VQ/TA:
  `0.0836081 / 0.4201892 / 0.7734056`
- 46 preflight tests passed in every arm

### Step-10 halfway gate

| LR | Raw MQ delta (95% CI) | EMA MQ delta (95% CI) | Interpretation |
|---:|---:|---:|---|
| `2e-6` | `-0.01503` `[-0.02937, -0.00018]` | `-0.01062` `[-0.02507, +0.00341]` | raw significantly worse |
| `2e-5` | `-0.01295` `[-0.03168, +0.00430]` | `-0.00795` `[-0.02150, +0.00619]` | inconclusive negative |
| `6e-5` | `-0.03032` `[-0.05636, -0.00487]` | `-0.01446` `[-0.02853, -0.00019]` | raw and EMA significantly worse |

Motion and diversity remained within their retention gates, but no LR showed
positive held-out MQ. The sweep continued to its specified 20-update endpoint
to determine whether any arm recovered.

### Final step-20 results

| LR | W&B | Raw MQ delta (95% CI) | EMA MQ delta (95% CI) | Raw VQ / TA delta | Raw motion / diversity ratio | GPU-hours |
|---:|---|---:|---:|---:|---:|---:|
| `2e-6` | [h105guh6](https://wandb.ai/adamlee00/finite-transition-v2-wan/runs/h105guh6) | `-0.01949` `[-0.03357, -0.00627]` | `-0.01524` `[-0.03114, +0.00039]` | `-0.00641 / -0.00572` | `0.99982 / 0.99999` | `9.611` |
| `2e-5` | [w2uxcm6j](https://wandb.ai/adamlee00/finite-transition-v2-wan/runs/w2uxcm6j) | `-0.00880` `[-0.02766, +0.01022]` | `-0.00316` `[-0.01916, +0.01401]` | `+0.00163 / -0.00135` | `0.99412 / 0.99987` | `9.559` |
| `6e-5` | [747f8tnv](https://wandb.ai/adamlee00/finite-transition-v2-wan/runs/747f8tnv) | `-0.02111` `[-0.05245, +0.01051]` | `-0.01091` `[-0.02536, +0.00380]` | `-0.00596 / +0.00346` | `0.98018 / 1.00542` | `9.341` |

Total reported sweep compute was approximately `28.51` H100 GPU-hours. Each
arm saved `checkpoint-20`, synced 24 media files, and exited cleanly.

Checkpoint locations on the `fastvideo-runs` Modal volume:

```text
/root/FastVideo/outputs/finite_transition_v2/anyflow_grpo_v2_lr_2e-06_s42_20260823_110920/checkpoint-20
/root/FastVideo/outputs/finite_transition_v2/anyflow_grpo_v2_lr_2e-05_s42_20260823_110920/checkpoint-20
/root/FastVideo/outputs/finite_transition_v2/anyflow_grpo_v2_lr_6e-05_s42_20260823_110920/checkpoint-20
```

All arms retained perfect VideoAlign coverage and repeatability throughout:

```text
base coverage = 1.0
adapter coverage = 1.0
checkpoint tensor coverage = 1.0
checkpoint numel coverage = 1.0
reward-head coverage = 1.0
unmatched keys = 0
repeat delta max = 0
```

Every raw and EMA `primary_paired`/`all_paired` success gate was `0`.

## Interpretation

The result isolates a scientific failure rather than a plumbing failure:

1. The luminance diagnostic proves the repaired optimizer and deterministic
   transfer path can learn a directly aligned reward.
2. VideoAlign checkpoint coverage, reward-head coverage, repeatability, and
   online/validation reward separation all pass.
3. Training MQ remains highly variable and can be large on the current rollout
   batch (for example, step-20 values around `0.61`–`0.65`) while held-out
   deterministic MQ remains negative relative to step zero.
4. Increasing LR does not fix transfer. `6e-5` causes the clearest halfway
   degradation and the largest raw motion loss by step 20.
5. `2e-5` is the least negative endpoint, but “not significantly worse” is not
   the required positive-learning result and is insufficient to justify a
   100-step baseline.

The evidence is consistent with poor generalization and/or weak alignment
between the stochastic transition action optimized by one-update GRPO and the
deployed deterministic four-step map. It does not support blaming the official
AnyFlow schedule, low-precision likelihood arithmetic, zero KL telemetry, or an
unloaded/nondeterministic reward model.

## Stop decision and recommended next investigation

Per the mandatory guardrails, do **not** launch or claim results from:

- the 100-step GRPO baseline;
- the strict GRPO-versus-posterior objective comparison; or
- the direct finite-velocity follow-up

until GRPO itself demonstrates positive paired held-out MQ on this audited
substrate.

The next investigation should be a focused, cheaper diagnosis rather than a
longer run. In priority order:

1. Measure prompt-wise correlation between online audited MQ advantage,
   deterministic-preference alignment, and subsequent held-out deterministic
   MQ change.
2. Inspect whether transition/action-level reward improvement survives the
   deterministic completion step and raw four-step map, instead of only the
   stochastic candidate rollout used for selection.
3. Compare current-prompt resampling against a genuinely disjoint prompt
   minibatch to quantify rollout overfitting before changing objectives.
4. If that diagnosis confirms stochastic-to-deterministic transfer is the
   bottleneck, run a small direct finite-velocity systems gate—not the full
   100-step experiment—with an explicit positive held-out MQ stop criterion.

## Operational caveat

One Modal GPU repeatedly emitted DCGM power-cap/hardware-slowdown/thermal-
throttling warnings. The interleaved launcher output did not identify the LR
arm unambiguously. All arms nevertheless completed, maintained finite metrics,
passed reward audits, produced exact paired artifacts, saved checkpoints, and
exited cleanly. This is a wall-time/cost caveat, not evidence of corrupted
scientific outputs.
