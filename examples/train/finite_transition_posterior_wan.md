# Finite-Transition Posterior Projection for AnyFlow-Wan

This branch tests one narrow hypothesis:

> Once a deterministic AnyFlow transition has been converted into a valid local
> stochastic policy, directly fitting its reward-tilted posterior should use the
> expensive video rollouts more efficiently than a matched GRPO score-function
> update.

The rollout, model, prompts, rewards, optimizer and validation are identical
between the proposed method and the baseline. The scientific delta is the
update rule.

## Why local-anchor ASFMC

AnyFlow is a two-time flow map. Its network predicts a finite average velocity
`u_theta(x_t, t, r, c)` and executes

```text
x_r = x_t - ((t-r)/N) u_theta(x_t, t, r, c).
```

That map is deterministic, so it does not directly define a stochastic policy
or action likelihood. Flow-Map GRPO introduces Anchored Stochastic Flow Map
Composition (ASFMC) to solve this without pretending that an arbitrary Gaussian
perturbation of a long-range map is path preserving.

For two-time maps, the paper recommends a **local anchor**. After the long map
`t -> r`, the model moves a short additional interval `r -> tau` toward data,
with

```text
tau = r + delta
```

in the paper's noise-to-data convention. FastVideo/AnyFlow uses the reverse
coordinate, so the implementation subtracts the same normalized interval from
the absolute reverse timestep.

At the local anchor, the method evaluates the instantaneous velocity and uses
the short reverse-SDE Euler-Maruyama Gaussian. The defaults follow the released
Flow-Map-GRPO two-time setting:

```yaml
anchor_type: local
local_anchor_delta: 0.03
local_noise_scale: 0.7
local_terminal_base_sigma: 0.05
```

The local conditional has approximation error of order
`O(delta^(3/2))`; importantly, it does not approximate an entire long-range SDE
transition as one Gaussian. `anchor_type: endpoint` remains available as an
ablation, but it injects stronger randomness and relies on an accurate long
map to the clean endpoint, which is less natural for a general two-time model.

## One training update

A released four-step AnyFlow-Wan checkpoint is evaluated deterministically. For
training, the schedule contains four stochastic branchable transitions and one
final deterministic completion.

At every optimizer step:

1. Select one branchable finite transition.
2. Sample one prompt and one initial Gaussian latent.
3. Run one deterministic shared prefix on all four GPUs.
4. Use local-anchor ASFMC to sample one candidate next latent per GPU from the
   same current state.
5. Complete each candidate with the same deterministic AnyFlow suffix.
6. Decode the four 81-frame videos and score all VideoAlign heads.
7. Use MQ as the optimized reward; retain VQ and TA as held-out diagnostics.
8. Convert the four MQ rewards into an ESS-controlled Boltzmann posterior.
9. Update the LoRA policy with either posterior projection or the matched GRPO
   loss.

With four GPUs and group size four, the global conceptual action tensor is

```text
[4, 21, 16, 60, 104]
```

for 81-frame, 480x832 Wan video latents. Each GPU physically owns one
`[1, 21, 16, 60, 104]` branch.

## Proposed update

The current stochastic transition policy is `q_old(a|s)`. Completing action
`a` gives a terminal reward `R(a)`. The KL-regularized locally improved policy
is

```text
q_R(a|s) proportional to q_old(a|s) exp(R(a)/tau).
```

Because the candidates were sampled from `q_old`, the finite-group posterior
weights are simply

```text
w_j = softmax(R_j / tau).
```

`tau` is solved per group so that the effective sample size

```text
ESS = 1 / sum_j w_j^2
```

is half of the four-candidate group by default. This makes selection strength
comparable across reward scales.

The proposed centered forward-KL projection is

```text
L = -sum_j (w_j - 1/G) log pi_theta(a_j|s).
```

Subtracting `1/G` is a behavior-score control variate. If every branch receives
the same reward, `w_j = 1/G` and the update is exactly zero rather than a random
finite-sample behavior-cloning drift.

## Matched baseline

Set

```yaml
method:
  objective: flowmap_grpo
```

The baseline uses the same prompt, source noise, prefix, local-ASFMC actions,
completed videos, rewards, LoRA, optimizer and held-out seeds. Only the loss
changes to clipped likelihood-ratio GRPO.

The current experiment performs one on-policy update per candidate group. At
that first gradient evaluation the policy ratio is one, so the sharpest
interpretation is:

```text
group-normalized linear advantage weighting
versus
ESS-controlled Boltzmann posterior weighting.
```

It is not a blanket claim that forward KL dominates every multi-epoch PPO
variant.

## Evaluation

The preparation script creates a deterministic disjoint train/validation prompt
split. Validation uses fixed prompts, fixed seeds, EMA weights and deterministic
four-step AnyFlow inference.

Every training step logs to W&B:

- optimized and held-out reward means and standard deviations;
- reward selection gain;
- posterior ESS, temperature, entropy and maximum weight;
- selected transition and local anchor timestep;
- local ASFMC noise scale and action displacement;
- temporal L1 motion;
- gradient norm;
- periodic post-update approximate KL and log-probability shift;
- wall time and cumulative GPU-hours.

Every validation logs:

- mean, standard deviation and standard error for all reward heads;
- delta from the untouched step-zero model;
- adjacent-frame temporal L1 and static-video rate;
- pairwise latent and decoded-video diversity across fixed seeds;
- fixed qualitative videos with prompt and reward captions;
- primary reward gain per update and per GPU-hour.

A scientific run counts as successful only when held-out MQ improves by the
configured absolute and 95%-standard-error margins while VQ/TA, motion and
latent diversity stay within their retention thresholds. W&B logs this gate as
`validation_success/all`. The efficiency claim requires comparing the paired
reward-versus-step and reward-versus-GPU-hour curves.

## Reproducibility and fail-fast behavior

The branch includes:

- released AnyFlow checkpoint contract validation and full snapshot caching;
- deterministic World-R1 train/held-out splitting;
- text-only parquet preprocessing;
- VideoAlign checkpoint download and runtime preflight;
- LoRA, HSDP, checkpoint and EMA configuration;
- resume-safe persistence of the original validation baseline, best reward
  delta, target-reaching step and cumulative training time;
- periodic Modal volume commits while `torchrun` is active;
- CPU math tests and fake-model end-to-end tests executed during Modal image
  construction before H100 allocation.

## One-time setup on macOS

The local Conda environment contains only the Modal client. CUDA, PyTorch,
FlashAttention, FastVideo and reward dependencies are built inside Modal.

```bash
git switch adam/finite-transition-alignment
git pull
conda env create -f environment.ftpp-modal.yml
conda activate fastvideo-ftpp-modal
modal setup
```

Required Modal resources:

```text
Secrets: wandb-adamlee00, hf-adamlee00
Volumes: fastvideo-data, fastvideo-runs, fastvideo-cache
```

`wandb-adamlee00` must expose `WANDB_API_KEY`.

## Smoke test

```bash
modal run modal_train_finite_transition_posterior.py --smoke
```

This uses 17 frames, reduced spatial resolution, two updates and validation at
every update. It verifies the real distributed model, reward and W&B path but
is not a scientific result.

To smoke-test both update rules on the same cached split:

```bash
modal run modal_train_finite_transition_posterior.py --smoke --paired
```

## Recommended experiment

```bash
modal run modal_train_finite_transition_posterior.py \
  --paired \
  --max-train-steps 1200
```

The launcher prepares and commits the deterministic split first, then starts the
posterior-projection and matched-GRPO jobs in one W&B run group.

## Resume an interrupted single run

```bash
modal run modal_train_finite_transition_posterior.py \
  --objective posterior_projection \
  --run-name-override <previous-run-name> \
  --resume-from-checkpoint latest
```

The launcher commits run/cache volumes every ten minutes by default and again
on normal or failed process exit.
