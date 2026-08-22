# Finite-transition v2 context for coding agents

Read these first:

```text
examples/train/finite_transition_posterior_progress_report.md
examples/train/finite_transition_v2_execution_plan.md
examples/train/finite_transition_posterior_execution_guardrails.md
```

Use this launcher:

```text
modal_train_finite_transition_v2_complete.py
```

Do not launch a new scientific job from the original FTPP or intermediate
`finite_transition_reliable*` paths.

## What we are testing

The current research sequence separates two questions.

### Question A: can the shared AnyFlow RL substrate learn at all?

Before comparing update rules, establish that:

```text
local-ASFMC exploration
+ terminal reward
+ finite-transition likelihood gradients
+ LoRA optimization
```

produce a measurable improvement under deterministic four-step inference.

This is tested first with cheap mean luminance, then with a strengthened
Flow-Map-GRPO baseline.

### Question B: how should the same reward evidence update the model?

Once the baseline learns, compare:

```text
GRPO standardized-advantage score update
versus
centered Boltzmann posterior score update
versus
finite-velocity posterior regression
```

The first two must consume identical frozen-base rollouts in the strict paired
comparison. Finite-velocity regression is a separate target-space method.

## Why the first experiment was inconclusive

The original run used:

```text
4 candidate videos
1 prompt group per optimizer update
1 selected transition
learning rate 2e-6
post-update KL around 1e-7
four-sample normalization / forced ESS
EMA-only validation
```

Both FTPP and GRPO stayed near the base checkpoint. That result does not show
that both algorithms fail after proper scaling. It shows the common substrate
had not demonstrated learnability.

## V2 rollout in practice

Default GRPO/posterior update:

```text
world size: 4 H100s
global candidates per prompt: 8
local candidates per rank: 2
prompt groups accumulated: 4
stochastic transitions per trajectory: 4
terminal reward videos per optimizer update: 32
transition likelihood records per optimizer update: 128
```

For an 81-frame 480x832 Wan video, each rank holds candidate latents with shape
approximately:

```text
[2, 21, 16, 60, 104]
```

One prompt group flows as follows:

1. All ranks receive the same prompt and initial Gaussian latent.
2. At transition 1, local ASFMC computes a policy mean/std and each rank samples
   two distinct next latents.
3. Those actions become the current states for transition 2.
4. Repeat through four stochastic finite transitions.
5. A final deterministic AnyFlow transition completes each candidate to `x_0`.
6. The VAE decodes eight final videos.
7. Audited VideoAlign MQ scores the eight videos.
8. One terminal reward coefficient supervises each of the four stochastic
   transition records from that trajectory.
9. Transition records are recomputed with gradients and backpropagated one at a
   time.
10. Four prompt groups accumulate before one AdamW step.

## What an “action” means here

The network predicts a finite velocity, but the stochastic policy action is the
sampled next latent state:

```text
state: x_t
network output: u_theta(x_t, t, r)
deterministic map: x_r = x_t - Delta u_theta
stochastic action: a_r sampled from local ASFMC around that map
```

Do not treat the velocity tensor itself as the policy action unless implementing
a separately named method.

## Local ASFMC intuition

AnyFlow's long finite map is deterministic. RL needs counterfactual next states
and their likelihoods.

Local ASFMC does not add arbitrary noise to a long jump. It uses a short reverse-
SDE conditional at the finite target. This keeps the stochastic approximation
local and compatible with the learned flow path.

Source:

```text
fastvideo/train/methods/rl/common/local_asfmc.py
```

Changing its time convention, velocity sign, or noise formula is a scientific
change, not a runtime repair.

## GRPO objective

For candidate terminal rewards `R_j`, v2 builds advantages using a running
prompt baseline and global/EMA reward scale:

```text
A_j = (R_j - baseline(prompt)) / global_std
```

Each finite transition receives the same terminal advantage for its trajectory.
The clipped likelihood-ratio loss is applied transition by transition.

For a one-update on-policy group, the initial ratio is one and clipping is
inactive. In the frozen-base paired comparison, clipping matters after the
learner diverges from the fixed behavior.

## Posterior score objective

The same rewards define weights:

```text
w_j proportional to exp((R_j - baseline) / tau_global)
```

The centered coefficient is:

```text
c_j = w_j - 1/G
```

and the score loss increases likelihood in reward-preferred action regions.

This remains a score-function update. Do not claim it is an exact finite optimum
of the full posterior distribution.

A flat group should produce weights near `1/G` and a near-zero update. If every
non-flat group is forced to the same ESS, the old noise-amplification problem has
returned.

## Finite-velocity regression

For one deployed finite transition:

```text
g_j = (x_t - a_j) / Delta
Delta_u = sum_j (w_j - 1/G) g_j
u_target = u_behavior + eta Delta_u
```

`eta` is capped by the desired RMS shift of the deterministic next state. The
student regresses directly to this stopped target.

This method attacks the stochastic-to-deterministic transfer bottleneck: the
training target is the finite velocity used by deterministic AnyFlow inference.

## Reward contract

Online scientific rollout:

```text
videoalign_mq_audited only
```

Held-out validation:

```text
MQ audited
VQ audited
TA audited
```

MQ and VQ must receive empty text prompts. TA receives the generation prompt. MQ
uses upstream mean-channel grayscale.

The checkpoint audit must verify base, adapter, and reward-head coverage. Never
replace it with “the scorer returned a finite number.”

## Strict paired comparison

The Modal `--paired` mode forces:

```text
behavior_policy = frozen_base
```

for both GRPO and posterior arms.

LoRA is disabled only during rollout collection, not during learning. Identical
seeds then create identical prompts, policy means, actions, videos, and rewards
in both arms at every update.

If one arm is repaired, apply the same data/rollout repair to the other. A
one-sided fix invalidates the comparison.

## Evaluation contract

Every validation checkpoint evaluates:

```text
raw learner
EMA learner
fixed prompts
fixed seeds
```

Prompt-level paired bootstrap confidence intervals are used for inference.
Individual prompt-seed values are saved for debugging and cross-arm subtraction.

A result is not a win unless deterministic held-out reward improves while:

```text
VQ and TA remain within tolerances
motion is retained
latent diversity is retained
qualitative videos do not expose reward hacking
update KL is non-trivial and stable
```

Training reward, posterior selection gain, or a positive gradient norm alone is
not evidence of policy improvement.

## Required run order

```text
1. smoke
2. diagnostic_luminance
3. learning-rate / KL sweep
4. standalone on-policy GRPO baseline
5. frozen-base GRPO/posterior pair
6. finite-velocity regression if likelihood objectives tie
```

Commands are maintained in:

```text
examples/train/finite_transition_v2_execution_plan.md
```

## Runtime repair checklist

Before changing code after an error, preserve:

```text
AnyFlow schedule and flow shift
model checkpoint and r-embedder contract
local ASFMC equation
candidate/group/transition counts
behavior policy
reward preprocessing and names
fixed prompt/seed evaluation set
objective parity across paired arms
```

Safe fixes include dependency, cache, checkpoint serialization, exact identity,
logging, distributed synchronization, and memory-lifetime corrections.

Changes to rollout topology, reward statistics, learning rate, target KL, LoRA,
or validation data create a new experiment and require a fresh W&B group.

## Decision rule

Do not train longer merely because RL is noisy.

Continue only when the cheaper gate and GRPO baseline show actual deterministic
movement. If the baseline learns and posterior weighting ties, the posterior
score delta is likely too weak a contribution. Move to finite-velocity
regression rather than endlessly tuning equivalent score estimators.
