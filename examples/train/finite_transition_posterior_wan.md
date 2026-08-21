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

Only the loss changes to clipped Flow-Map GRPO. With one on-policy update per
candidate group, the initial likelihood ratio is one, so the sharpest
interpretation is a comparison between group-normalized linear advantage
weighting and ESS-controlled Boltzmann posterior weighting. It is not a blanket
claim that forward KL dominates every possible multi-epoch PPO implementation.

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

## Reproducibility and fail-fast checks

The branch includes all experiment plumbing:

- released `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers` model loading;
- deterministic train/held-out prompt splitting and text-only preprocessing;
- VideoAlign checkpoint download and runtime preflight;
- FastVideo config/model-contract validation;
- LoRA, HSDP, checkpoint and EMA configuration;
- fixed-seed held-out evaluation and W&B video logging;
- periodic Modal volume commits while `torchrun` is active;
- resume-safe persistence of the step-zero validation baseline, best reward
  delta, target-reaching step and cumulative training time.

The Modal image compiles the new modules and runs the CPU math and fake-model
regression tests before an H100 container can start. A syntax/import/unit-test
failure therefore stops during image construction rather than after expensive
GPU allocation.

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
Paired Modal launches share a `WANDB_RUN_GROUP`, while `WANDB_JOB_TYPE`
distinguishes `posterior_projection` from `flowmap_grpo`.

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

## One-time launcher setup on macOS

The local Conda environment contains only the Modal client. CUDA, PyTorch,
FlashAttention, FastVideo, reward libraries and all training dependencies are
built inside the reproducible Modal image.

```bash
git switch adam/finite-transition-alignment
git pull

conda env create -f environment.ftpp-modal.yml
conda activate fastvideo-ftpp-modal
modal setup
```

The launcher expects these existing Modal resources:

```text
Secrets:  wandb-adamlee00, hf-adamlee00
Volumes:  fastvideo-data, fastvideo-runs, fastvideo-cache
```

The W&B secret must expose `WANDB_API_KEY`. The Hugging Face checkpoint and
World-R1 dataset are public, but the HF secret is mounted for authenticated,
rate-limit-safe downloads.

## Modal smoke test

Run this first after pulling a new revision:

```bash
modal run modal_train_finite_transition_posterior.py --smoke
```

This uses 17 frames, a smaller spatial shape, two updates and validation every
step. It verifies the real distributed model/reward/tracker path but must not be
compared with scientific runs.

To smoke-test both update rules against the exact same cached split:

```bash
modal run modal_train_finite_transition_posterior.py --smoke --paired
```

## Recommended scientific experiment

```bash
modal run modal_train_finite_transition_posterior.py \
  --paired \
  --max-train-steps 1200
```

The launcher first materializes and commits the deterministic split once, then
starts posterior-projection and Flow-Map-GRPO jobs against the same cached
parquets and scientific settings. Both runs appear in one W&B group.

A single proposed-method run is also supported:

```bash
modal run modal_train_finite_transition_posterior.py \
  --objective posterior_projection \
  --max-train-steps 1200
```

## Resume an interrupted single run

Use the same run name so the output directory is reused, and ask FastVideo to
resolve the latest checkpoint:

```bash
modal run modal_train_finite_transition_posterior.py \
  --objective posterior_projection \
  --run-name-override <previous-run-name> \
  --resume-from-checkpoint latest
```

The launcher commits the run and cache volumes every ten minutes by default, as
well as on normal or failed subprocess exit.

## Manual local preparation

For debugging outside Modal, run the module form so repository-relative imports
are deterministic:

```bash
python -m examples.train.check_finite_transition_posterior_environment \
  --require-wandb \
  --json

python -m examples.train.prepare_finite_transition_posterior_assets \
  --check-rewards \
  --objective posterior_projection \
  --dataset world-r1-enhanced-dynamic \
  --reward videoalign_mq \
  --max-train-prompts 512 \
  --validation-prompts 64 \
  --json
```

Then run the generated config printed by the preparation script:

```bash
bash examples/train/run.sh outputs/ftp_configs/<generated-config>.yaml
```
