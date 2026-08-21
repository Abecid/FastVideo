# FTPP experiment context for coding agents

For the latest implementation status, runtime-fix history, step-100 paired
results, and open audit questions, read
`examples/train/finite_transition_posterior_progress_report.md` first.

Read this before changing or running the finite-transition posterior experiment.
It explains the scientific question, the invariants that make the comparison
valid, the expected tensor/data flow, and which W&B signals determine whether
the method worked.

## 1. The single research question

The experiment starts from a released, competent, deterministic four-step
AnyFlow-Wan video generator. Flow-Map GRPO gives a way to turn one deterministic
finite transition into a stochastic local policy. The question is then:

> Given the exact same local stochastic actions and terminal video rewards, is
> directly fitting the reward-improved transition posterior more efficient and
> less destructive than a matched GRPO score-function update?

The proposed objective is `posterior_projection`. The matched baseline is
`flowmap_grpo`.

This is deliberately **one small delta**. Do not turn the experiment into a
mixture of DMD, DiffusionNFT, auxiliary SFT, extra KL penalties, a different
sampler, or a different reward unless that change is introduced as a separate
named ablation.

## 2. Non-negotiable scientific invariant

The proposed method and baseline must use the same:

- AnyFlow checkpoint and LoRA parameterization;
- training and held-out prompt split;
- prompt, initial Gaussian latent, and deterministic prefix per update;
- selected finite-transition index;
- local-ASFMC action sampler and candidate actions;
- deterministic suffix used to complete each action;
- decoded videos and reward calls;
- optimizer, learning rate, number of updates, and validation seeds.

Only the update weights/loss are allowed to differ.

If a runtime fix changes the rollout for only one objective, the comparison is
scientifically invalid even if the code runs.

## 3. Terminology

### State

The RL state is the current noisy video latent `x_t` at one AnyFlow timestep.

### Model output

AnyFlow predicts a finite-interval velocity

```text
u_theta(x_t, t, r, prompt)
```

and executes

```text
x_r = x_t - ((t-r)/N) * u_theta(x_t, t, r, prompt).
```

### Action

The action is **not** the velocity. The action is the sampled next latent at the
finite-transition target time:

```text
a = sampled x_r.
```

The stochastic policy is therefore a Gaussian density over next latent states.
The model's finite and instantaneous velocity predictions parameterize that
Gaussian mean.

### Reward-tilted posterior

For a shared state `s`, the current local policy is `q_old(a|s)`. Completing an
action gives a terminal reward `R(a)`. The KL-regularized improved policy is

```text
q_R(a|s) proportional to q_old(a|s) * exp(R(a) / tau).
```

Because candidate actions are already sampled from `q_old`, the finite-group
posterior weights are

```text
w_j = softmax(R_j / tau).
```

No additional `q_old(a_j)` multiplier is needed in the Monte Carlo weights.

### ESS

Effective sample size measures how concentrated the posterior weights are:

```text
ESS = 1 / sum_j w_j^2.
```

With four candidates:

- uniform weights give `ESS = 4`;
- one winner with weight one gives `ESS = 1`;
- two equal survivors give `ESS = 2`.

The default target is `ESS/G = 0.5`, so approximately two of four candidates
meaningfully survive. The code solves for `tau` by bisection on every group.
This avoids tying reward-selection strength to the arbitrary numerical scale of
a reward model.

## 4. Why local-anchor ASFMC exists

AnyFlow's finite map is deterministic, so it does not naturally define
exploration or a log probability. Adding arbitrary Gaussian noise around a
long-range deterministic output is not guaranteed to correspond to a valid
short transition of the generative process.

For a two-time map, Flow-Map GRPO recommends a **local anchor**. The code first
computes the deterministic target `x_r`, then queries AnyFlow's instantaneous
reverse velocity at the target by setting both model time arguments to `r`.
It applies the short reverse-SDE Gaussian after converting between:

- the paper coordinate `s`, which increases from noise to data; and
- AnyFlow's reverse coordinate `q = 1-s`, which decreases from noise to data.

In the implemented reverse coordinate, let

```text
q = r / N
s = max(1-q, terminal_base_sigma)
```

and let `v_q(x_r, r, r)` be AnyFlow's instantaneous reverse velocity. The local
policy is

```text
mean = x_r - delta * lambda^2 * (x_r / s + v_q)
std  = lambda * sqrt(2*q/s) * sqrt(delta).
```

If the last branchable target is closer to the data endpoint than the configured
`delta`, the implementation uses `delta_eff = min(delta, q)` so the conceptual
local anchor lands exactly at `q=0` instead of crossing the endpoint. This is
the same short-interval construction with its final interval truncated; W&B
logs both the configured and effective delta.

Defaults:

```yaml
anchor_type: local
local_anchor_delta: 0.03
local_noise_scale: 0.7
local_terminal_base_sigma: 0.05
```

`anchor_type: endpoint` is retained only as an ablation. Do not accidentally
switch the main config back to the base endpoint-anchor implementation.

The active YAML must target:

```text
ReproducibleFiniteTransitionPosteriorMethod
```

not the base `FiniteTransitionPosteriorMethod`, because the subclass supplies
both local-anchor ASFMC and resume-safe experiment state.

## 5. One full training update

The scientific default uses four H100s and group size four.

### Shared prompt and source

At outer step `n`:

1. All four ranks receive the same prompt.
2. All four ranks construct the same initial Gaussian latent from a seed based
   on the experiment seed and outer step, but not rank.
3. Rank zero chooses one of the first four branchable finite transitions and
   broadcasts the index.
4. Every rank runs the same deterministic AnyFlow prefix.

At this point all ranks hold the same state `x_t`.

### Candidate actions

The method builds the same local-ASFMC Gaussian parameters on every rank. Each
rank then uses a rank-specific action RNG, giving one different action per GPU:

```text
rank 0: a_0
rank 1: a_1
rank 2: a_2
rank 3: a_3
```

For the full 81-frame, 480x832 run, each action has approximately this layout:

```text
[batch=1, latent_time=21, channels=16, height=60, width=104].
```

The conceptual distributed action group is therefore

```text
[4, 21, 16, 60, 104].
```

### Deterministic completion and reward

Each rank completes its action with the same remaining AnyFlow transitions,
decodes approximately

```text
[1, 3, 81, 480, 832]
```

RGB video, and computes all configured VideoAlign heads:

- `videoalign_mq`: optimized motion-quality reward;
- `videoalign_vq`: held-out visual-quality diagnostic;
- `videoalign_ta`: held-out text-alignment diagnostic.

Rewards are all-gathered so every rank sees the same four-candidate reward
vector.

### Proposed posterior update

The proposal computes ESS-controlled weights `w_j` and minimizes

```text
L_FTPP = -sum_j (w_j - 1/G) * log pi_theta(a_j | s).
```

The subtraction of `1/G` is essential. If every candidate gets the same reward,
then `w_j = 1/G` and the update is exactly zero. A reward model with no local
preference must not cause random finite-sample behavior-cloning drift.

### Matched baseline

The baseline uses group-normalized advantages and the same stored action/log
probability data:

```text
A_j = normalized(R_j)
L_GRPO = clipped likelihood-ratio objective.
```

There is one on-policy update per candidate group. At the first gradient
evaluation the ratio is one, so the cleanest claim is a comparison of:

```text
linear standardized advantage weighting
versus
ESS-controlled exponential posterior weighting.
```

Do not describe the current experiment as proving that forward KL beats every
possible multi-epoch PPO implementation.

### Log probability

Gaussian log probability is averaged over latent coordinates, not summed. A
video action contains millions of coordinates; summing would make tiny
per-coordinate changes produce unusably large log-ratio magnitudes.

## 6. Source-of-truth files

Read these before modifying the experiment:

```text
modal_train_finite_transition_posterior.py
examples/train/configs/rl/wan/finite_transition_posterior_anyflow_videoalign.yaml
examples/train/prepare_finite_transition_posterior_assets.py
examples/train/check_finite_transition_posterior_environment.py
fastvideo/train/methods/rl/finite_transition_posterior.py
fastvideo/train/methods/rl/finite_transition_posterior_repro.py
fastvideo/train/methods/rl/common/finite_transition.py
fastvideo/train/methods/rl/common/local_asfmc.py
examples/train/finite_transition_posterior_wan.md
```

Relevant tests:

```text
fastvideo/tests/train/methods/test_finite_transition_posterior_core.py
fastvideo/tests/train/methods/test_finite_transition_posterior_method.py
fastvideo/tests/train/methods/test_finite_transition_posterior_repro.py
fastvideo/tests/train/methods/test_local_asfmc.py
```

## 7. W&B signals that answer the research question

### Training signal quality

Inspect:

```text
ftp/reward_std
ftp/reward_selection_gain
ftp/posterior_ess
ftp/posterior_temperature
ftp/posterior_weight_entropy
ftp/posterior_weight_max
ftp/action_deviation_from_deterministic
ftp/action_deviation_from_posterior_mean
```

Interpretation:

- `reward_std` near zero means the reward cannot distinguish local branches;
- `reward_selection_gain` near zero means posterior reweighting found no better
  local future;
- a posterior always dominated by one candidate means selection may be too
  aggressive or the reward is noisy;
- extremely large action deviation followed by quality collapse suggests the
  local policy is too stochastic.

### Optimization stability

Inspect:

```text
ftp/grad_norm
ftp/post_update_approx_kl
ftp/post_update_logprob_delta_abs
ftp/train_step_seconds
ftp/cumulative_gpu_hours
```

A reward increase accompanied by exploding post-update KL is not a robust win.

### Held-out success

The experiment must be judged on fixed held-out prompts and seeds, not training
rewards. Relevant metrics include:

```text
validation/primary_delta
validation/primary_significance_margin
validation/primary_gain_per_gpu_hour
validation/primary_gain_per_100_steps
validation/motion_ratio_to_base
validation/latent_diversity_ratio_to_base
validation_success/all
```

The default success gate requires:

```text
held-out MQ delta >= 0.02
held-out MQ delta > combined 95% standard-error margin
held-out VQ drop <= 0.02
held-out TA drop <= 0.02
motion ratio to base >= 0.90
latent diversity ratio to base >= 0.80
```

The efficiency claim additionally requires FTPP to reach a matched held-out
reward in fewer steps or GPU-hours, or to have a better reward-versus-GPU-hour
curve than `flowmap_grpo`.

Training reward alone is never sufficient evidence.

## 8. Failure modes and repair rules

### Model fails to load or lacks two-time conditioning

Verify the exact released checkpoint and the following config contract before
changing model code:

```text
nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
r_embedder: true
r_embedder_gate_value: 0.25
r_embedder_deltatime_type: r
flow_shift: 5.0
```

Do not silently fall back to base Wan. That would remove the finite flow-map
substrate and change the research problem.

### VideoAlign fails

Fix checkpoint/runtime parity. Do not replace the primary reward with a random
available scorer just to make the run proceed. The proposed and baseline runs
must use identical reward code and checkpoint revisions.

### Distributed hang or wrong group size

Check that:

```text
group_size % world_size == 0
sp_size == 1
all ranks execute every gather/broadcast in the same order
all ranks own the same number of local branches
```

With the default four GPUs and group size four, each rank owns exactly one
candidate.

### NaN or infinite likelihood

Check, in order:

1. local ASFMC `std` is positive and finite;
2. target timestep is below the initial noise endpoint;
3. terminal stabilization is active near noise;
4. log probability is dimension-averaged;
5. action and policy tensors have identical shape/dtype/device.

Do not add arbitrary clamps that affect only one objective. Add a shared sampler
fix or an explicit ablation.

### Equal rewards and zero gradient

This is expected for posterior projection. The centered objective is designed to
produce zero update when the reward contains no information. Confirm
`ftp/zero_std_group` and `ftp/reward_std` before treating it as an optimizer bug.

### Training reward rises but held-out reward does not

Treat this as overfitting or reward hacking. Do not report success. Inspect fixed
validation videos, VQ/TA, motion, and diversity.

### MQ rises while motion or diversity collapses

This is a failed run, even if MQ is nominally the optimized reward. The method
must preserve the base model's video character rather than discover a static or
low-diversity reward exploit.

### OOM

Prefer engineering-only changes first:

- reduce validation logging or validation batch size;
- run `--smoke` at the already-defined smaller frame/resolution settings;
- verify activations are under `no_grad` outside the single policy recompute;
- check that decoded videos/reward models are released promptly.

Changing group size, training resolution, sampler, or reward changes the
scientific setting and must be reflected in both objectives and the run name.

### Resume

Resume through the same run name and `--resume-from-checkpoint latest`. The
reproducible method persists the original step-zero validation baseline and
efficiency counters. Do not switch the YAML target back to the base method on a
resume.

## 9. What may be changed safely during debugging

Safe changes preserve both objectives and the scientific invariant:

- imports, dependency pins, paths, caching, and Modal volume handling;
- tensor shape/device/dtype corrections;
- distributed synchronization fixes;
- checkpoint/resume correctness;
- W&B logging and captions;
- memory cleanup that does not alter samples;
- numerical fixes applied to the shared sampler or shared likelihood code.

Algorithmic changes should be exposed as explicit config knobs and run as named
ablations. Never silently change the default after seeing one objective's result.

## 10. Recommended execution order

### Real pipeline smoke

```bash
modal run modal_train_finite_transition_posterior.py --smoke
```

### Matched two-objective smoke

```bash
modal run modal_train_finite_transition_posterior.py --smoke --paired
```

### Scientific paired run

```bash
modal run modal_train_finite_transition_posterior.py \
  --paired \
  --max-train-steps 1200
```

Both scientific runs should appear in one W&B run group with distinct job types.

## 11. How to interpret the main outcomes

### FTPP beats matched GRPO

The evidence is strongest when FTPP reaches the same held-out MQ with fewer
updates/GPU-hours while retaining VQ, TA, motion, and diversity. This supports
the hypothesis that an explicitly constructed local reward posterior uses
finite-transition rollout data more efficiently than linear advantage weighting.

### Both methods improve equally

The local-ASFMC rollout and shared-state credit assignment may be the important
part, while the two weighting rules are effectively equivalent at this group
size and reward scale.

### Both methods fail

Inspect whether the bottleneck is upstream:

- local actions do not create meaningful reward variation;
- VideoAlign is too noisy at the branch level;
- four candidates are insufficient;
- the local sampler is too weak or too destructive;
- terminal reward cannot assign useful credit at the selected transition.

Do not conclude that posterior projection is wrong until sampler and reward
signal diagnostics have been checked.

### FTPP trains faster but collapses quality or diversity

That is not success. It means the posterior weighting is more aggressive but not
better constrained. The held-out and collapse metrics are part of the method's
claim, not optional diagnostics.
