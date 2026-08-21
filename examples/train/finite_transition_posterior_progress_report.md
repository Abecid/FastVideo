# Finite-Transition Posterior Alignment: implementation and experiment handoff

Last updated: 2026-08-20 22:17 PDT

This report is a technical handoff for reviewing the finite-transition
posterior-projection experiment on AnyFlow-Wan. It records what was implemented,
what was changed while executing the real Modal jobs, what the experiment has
shown so far, and which scientific or implementation questions remain open.

## Executive summary

- Repository: `Abecid/FastVideo`
- Branch: `adam/finite-transition-alignment`
- Tested implementation commit: `9f14c58b3f17f37fb1ba6aeb6049494fc7a686f0`
- Main experiment group: `ftpp_pair_s42_20260820_r3`
- Proposed run: [posterior projection `d7t6o9or`](https://wandb.ai/adamlee00/finite-transition-posterior-wan/runs/d7t6o9or)
- Control run: [matched Flow-Map GRPO `mylt6v99`](https://wandb.ai/adamlee00/finite-transition-posterior-wan/runs/mylt6v99)
- Modal app: [active paired run](https://modal.com/apps/hao-ai-lab/main/ap-0WOCFHjEjs4INhqig5ixrK)
- State at the time of this report: running, posterior past step 115 and GRPO
  past step 137 of 1,200.

The implementation is now runtime-stable through model and reward loading,
full-resolution baseline evaluation, the final shortened local transition,
100 optimizer updates, checkpoint saving, and a second full held-out evaluation.
The paired smoke tests also completed end to end.

The first scientific checkpoint does **not** establish that posterior projection
is better. At step 100, both methods slightly reduced held-out VideoAlign MQ.
Posterior projection trailed GRPO by `0.00158` MQ, which is negligible relative
to the current uncertainty estimate. Motion and diversity were retained by both.
The full 1,200-step run remains active.

Two review issues deserve particular attention:

1. The success gate combines baseline and current SEM as if the fixed-prompt,
   fixed-seed measurements were independent. With the observed MQ standard
   deviation and only 64 prompt-level observations, this produces a significance
   margin near `0.25`, despite the configured minimum useful delta being `0.02`.
   Per-prompt paired differences should be logged and evaluated instead.
2. The two jobs use identical prompt, latent, branch, and action-noise seeds.
   Once their LoRA weights diverge, however, their policy means and therefore
   their realized actions/videos also diverge. The documentation's literal
   claim that both objectives use the “same candidate actions” is only true at
   initialization. A stricter update-rule comparison needs a shared rollout
   producer or an explicitly frozen behavior policy.

## Research question and intended comparison

The starting policy is the released deterministic four-step
`nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers` generator. The experiment asks:

> Given local stochastic finite-transition rollouts and their terminal video
> rewards, is projecting the reward-tilted local posterior more efficient and
> less destructive than a matched score-function/GRPO update?

The action is the sampled next latent `x_r`, not a velocity. For a shared state,
the local policy is a diagonal Gaussian over this next latent. Candidate actions
are completed with the deterministic AnyFlow suffix, decoded, and scored with
VideoAlign.

For rewards `R_j`, the posterior weights are

```text
w_j = softmax(R_j / tau)
```

where `tau` is solved per group so that `ESS/G = 0.5`. With `G=4`, the target
effective sample size is two candidates.

The proposed objective is

```text
L_FTPP = -sum_j (w_j - 1/G) * log pi_theta(a_j | s).
```

The `1/G` behavior-score baseline gives exactly zero update for an
uninformative/equal-reward group. The control uses standardized group advantages
and the clipped likelihood-ratio form:

```text
L_GRPO = -mean(min(ratio * A, clipped_ratio * A)).
```

There is one optimizer update per on-policy candidate group. Consequently,
`ratio=1` during the pre-update loss evaluation and clipping is normally inactive.
In the present implementation this control is best understood as a matched
standardized score-function update with GRPO/PPO bookkeeping, not a multi-epoch
clipped-policy optimization procedure.

## Local-anchor ASFMC policy

AnyFlow predicts a finite map, so a local stochastic policy is constructed at
the deterministic target. The code queries the instantaneous reverse velocity
at `(r, r)` and performs the paper-to-AnyFlow time/sign conversion.

Using AnyFlow reverse time `q=r/N` and `s=max(1-q, terminal_base_sigma)`, the
implemented Gaussian is

```text
mean = x_r - delta * lambda^2 * (x_r / s + v_q(x_r, r, r))
std  = lambda * sqrt(2*q/s) * sqrt(delta).
```

Defaults are:

```yaml
anchor_type: local
local_anchor_delta: 0.03
local_noise_scale: 0.7
local_terminal_base_sigma: 0.05
```

When `q < delta` on the final branchable transition, the executed interval is
now `delta_eff=min(delta, q)`. This places the conceptual anchor exactly at the
data endpoint instead of crossing it. Both configured and effective deltas,
the clipping indicator, and the resulting anchor time are logged.

The primary files for auditing this derivation are:

- `fastvideo/train/methods/rl/common/local_asfmc.py`
- `fastvideo/train/methods/rl/common/finite_transition.py`
- `fastvideo/train/methods/rl/finite_transition_posterior.py`
- `fastvideo/train/methods/rl/finite_transition_posterior_repro.py`
- `examples/train/finite_transition_posterior_agent_context.md`

## Reproducibility and data flow

The scientific configuration is
`examples/train/configs/rl/wan/finite_transition_posterior_anyflow_videoalign.yaml`.
It uses:

| Setting | Value |
|---|---:|
| Model | AnyFlow-Wan 1.3B |
| GPUs | 4 H100 per objective, 8 total |
| Frames | 81 |
| Resolution | 480 × 832 |
| Train prompts | 451 |
| Validation prompts | 64 |
| Validation samples per prompt | 2 |
| Candidate group | 4 |
| LoRA rank/alpha | 256/256 |
| Learning rate | `2e-6` |
| EMA decay | `0.99` from step 1 |
| Training updates | 1,200 |
| Validation cadence | every 100 updates |
| Validation videos | 8 per checkpoint |
| Seed | 42 |

Within a given objective run, each update uses:

1. A deterministically selected prompt.
2. Shared initial Gaussian noise across all four ranks.
3. A branch index selected on rank zero and broadcast.
4. A shared deterministic prefix.
5. One rank-specific Gaussian action-noise seed per candidate.
6. A deterministic suffix after the candidate action.
7. All three VideoAlign heads: MQ for optimization, VQ and TA as held-out
   anti-reward-hacking diagnostics.

The two objective runs use the same seed formulas and initial checkpoint. Their
step-zero validation values match exactly, which confirms the initial model,
validation split, and evaluation seeds are aligned. After the first different
gradient update, the runs are seed-matched but no longer action-matched because
their policy parameters differ.

Resume-safe method state includes the original validation baseline, best MQ
delta, first step reaching the target, cumulative train time, EMA, optimizer,
and RNG/checkpoint state. The validation baseline is serialized to a fixed-shape
64 KiB `uint8` tensor so distributed-checkpoint load planning remains stable.

## Implementations already present on the branch

Before the real run, the branch contained:

- Reusable finite-transition math primitives and tests.
- Local-anchor ASFMC construction for two-time maps.
- The `FiniteTransitionPosteriorMethod` and reproducible AnyFlow subclass.
- The posterior-projection and Flow-Map-GRPO objectives.
- AnyFlow prompt preparation and deterministic train/validation splitting.
- VideoAlign MQ/VQ/TA reward wrappers and media conversion.
- LoRA, EMA, DCP checkpointing, fixed validation, and success gates.
- A paired Modal launcher and persistent data/run/cache volumes.
- W&B scalar, efficiency, success-gate, caption, and video logging.

The branch also contains earlier DiffusionNFT work inherited by this feature
branch. Reviewers should scope the finite-transition audit to the files listed
in this report and the finite-transition commits unless they intentionally want
to review the entire branch relative to upstream.

## Runtime fixes made during execution

All fixes below are in implementation commit `9f14c58b`.

### 1. GPU-dependent imports during Modal image build

**Failure:** focused tests and reward import checks ran in the image builder,
where Triton-backed FastVideo kernels could not find a GPU driver.

**Fix:** compilation remains in the image build, while import-dependent pytest,
CLI help, reward-runtime, and FlashAttention gates run after H100 allocation.

**Impact on science:** none; this only changes where fail-fast checks execute.

### 2. AnyFlow registry and architecture overrides

**Failure:** the released AnyFlow HF ID and cached `AnyFlowPipeline` snapshot did
not reliably resolve to the Wan T2V pipeline configuration. Training config
overrides for the `r` embedder were not in the allowed architecture override
set.

**Fix:** register the exact HF ID, recognize both `WanPipeline` and
`AnyFlowPipeline`, and allow `r_embedder`, fusion, gate value, and delta-time
type through the training config resolver.

**Impact on science:** required to instantiate the released two-time AnyFlow
architecture specified by the experiment.

### 3. Wan non-persistent gate buffer under meta/FSDP initialization

**Failure:** the gated `r` embedder stores its scalar gate as a non-persistent
buffer. FSDP constructs on `meta`, while the checkpoint correctly omits this
buffer, leaving a meta tensor at runtime.

**Fix:** retain the configured gate value and implement
`materialize_non_persistent_buffers()` on the Wan embedding and transformer.
The existing FSDP loader calls this hook after materialization.

**Impact on science:** restores the configured AnyFlow gate (`0.25`) without
adding it to the checkpoint state dict.

### 4. Transformers/Accelerate compatibility

**Failure:** the installed Transformers 5.x path requires Accelerate at least
1.1.0, while the project pinned 1.0.1.

**Fix:** pin `accelerate==1.1.0` with the compatibility reason documented in
`pyproject.toml`.

### 5. Concurrent VideoAlign/Qwen cache race

**Failure:** the first paired smoke run launched both jobs against a mutable
shared Hugging Face cache. One arm observed the Qwen2-VL base snapshot before
all indexed shards were visible and failed during VideoAlign initialization.

**Fix:** the one-time preparation job completes and commits the snapshot. Each
paired arm resolves a snapshot whose index and every referenced shard exist,
then sets `VIDEOALIGN_BASE_MODEL_PATH` to that immutable directory. The reward
wrapper patches VideoAlign to load that path rather than mutating Hub metadata.

**Impact on science:** both reward models now load the same cached Qwen base.

### 6. Full-resolution validation OOM

**Failure:** both first scientific arms completed all 128 held-out samples at
81×480×832, then retained every decoded float video and called
`all_gather_object`. This requested an additional 46.28 GiB per device.

**Fix:** apply the configured global eight-video qualitative cap before object
gathering and convert only selected videos to CPU `uint8`. Aggregate rewards,
motion/diversity metrics, prompt coverage, and video selection order are
unchanged.

**Impact on science:** none on scalar validation; only the representation and
early filtering of qualitative payloads changed.

### 7. Final local interval crossed the data endpoint

**Failure:** at optimizer step 5, deterministic branch selection chose target
time `24.414` on the 1,000-step scale. The configured 30-timestep local interval
would cross `q=0`, so both objectives raised the geometry guard.

**Fix:** use `delta_eff=min(0.03, target/N)`. For this branch,
`delta_eff=0.0244140663`, making the anchor exactly zero. Ordinary branches
still use `0.03`.

**Impact on science:** this is a truncated final short interval, applied
identically to both objectives. It should nevertheless be checked against the
official Flow-Map-GRPO reference because exploration variance becomes small
near the endpoint.

### 8. Paired launcher lifetime and error propagation

**Failure:** fire-and-forget spawning could let the local Modal app exit without
waiting for both jobs, and exceptions were not reliably surfaced together.

**Fix:** spawn both calls, wait on both handles, retain completed-arm results,
and raise a combined error if either arm fails. Single-objective mode now uses
a blocking remote call. Run and cache volumes are periodically committed and
again at subprocess exit.

## Regression and execution evidence

### Local/static checks

- `python -m compileall -q` passed for the modified training and launcher code.
- `git diff --check` passed.
- Pre-commit could not be executed because it was unavailable in both the base
  shell and the existing `fastvideo` Conda environment. Project instructions
  prohibit bypassing the configured pre-commit chain with direct formatter or
  linter invocations.

### Remote focused gate

Each scientific arm passed the same focused runtime gate before training:

```text
28 passed, 15 warnings
```

Coverage includes finite-transition math/method behavior, reproducible state,
local ASFMC, prepared local VideoAlign base loading, AnyFlow registry/config
resolution, and Wan gate materialization after meta initialization.

### Successful smoke runs

Single proposed-method smoke:

- [W&B `pi3iz3b2`](https://wandb.ai/adamlee00/finite-transition-posterior-wan/runs/pi3iz3b2)
- Two optimizer steps, real AnyFlow, real VideoAlign, four H100s.
- Checkpoint: `/root/FastVideo/outputs/finite_transition_posterior/ftpp_anyflow_videoalign_mq_smoke_20260820/checkpoint-2`.

Successful paired smoke after the cache fix:

- [Posterior `lyqa57yw`](https://wandb.ai/adamlee00/finite-transition-posterior-wan/runs/lyqa57yw)
- [GRPO `o0ooktoj`](https://wandb.ai/adamlee00/finite-transition-posterior-wan/runs/o0ooktoj)
- Both completed two updates, checkpointing, 116 scalar fields, and 12 videos.
- Posterior final grad norm `0.00186`, advantage absolute mean `0.92925`,
  cumulative GPU hours `0.06747`.
- GRPO final grad norm `0.00154`, advantage absolute mean `0.79949`,
  cumulative GPU hours `0.04615`.
- This was a plumbing test, not evidence of quality superiority.

## Full-run failure and fix timeline

| Attempt | Runs | Outcome | Resolution |
|---|---|---|---|
| Initial paired smoke | `f59wcssc`, `b40j5v9n` | Posterior saw a partial shared Qwen cache; GRPO finished | Resolve and validate one immutable snapshot before launching arms |
| Paired smoke r4 | `lyqa57yw`, `o0ooktoj` | Both completed | Established end-to-end plumbing |
| Scientific r1 | `2fw3djj1`, `iz2p1mwy` | Both failed after step-zero scoring with 46.28 GiB gather OOM | Cap and compact qualitative media before gather |
| Scientific r2 | `voi2t54q`, `1qu6xcmc` | Both passed baseline and four updates, then failed on target `24.414` | Truncate the final local interval at `q=0` |
| Scientific r3 | `d7t6o9or`, `mylt6v99` | Active; both passed step 100 validation | Current main comparison |

The shared failures in r1 and r2 are useful evidence that both objectives
followed the same validation and branch-selection paths.

## Experiment results through step 100

Both arms have the exact same step-zero validation baseline:

| Metric | Shared baseline |
|---|---:|
| VideoAlign MQ | -0.069407 |
| VideoAlign VQ | 0.273385 |
| VideoAlign TA | 0.801972 |
| Temporal L1 | 0.0322293 |
| Latent diversity RMS | 0.971575 |
| Video diversity RMS | 0.321547 |
| MQ prompt-level standard deviation | 0.737353 |
| MQ SEM | 0.092169 |

Step-100 results:

| Metric | Posterior projection | Flow-Map GRPO | Posterior − GRPO |
|---|---:|---:|---:|
| MQ | -0.084669 | -0.083085 | -0.001584 |
| MQ delta from baseline | -0.015262 | -0.013678 | -0.001584 |
| VQ | 0.267112 | 0.276047 | -0.008934 |
| VQ delta | -0.006273 | +0.002661 | -0.008934 |
| TA | 0.797672 | 0.791889 | +0.005783 |
| TA delta | -0.004300 | -0.010083 | +0.005783 |
| Temporal L1 | 0.0322236 | 0.0322301 | -0.0000064 |
| Motion/base | 0.999824 | 1.000024 | -0.000200 |
| Latent diversity/base | 1.000227 | 0.999992 | +0.000236 |
| Video diversity RMS | 0.321592 | 0.321526 | +0.000066 |
| MQ SEM | 0.090690 | 0.091924 | — |
| Current independent-SEM margin | 0.253438 | 0.255140 | — |
| Cumulative GPU hours at validation | 4.7941 | 3.9660 | +0.8281 |
| Primary MQ gate | 0 | 0 | — |
| VQ retention gate | 1 | 1 | — |
| TA retention gate | 1 | 1 | — |
| Motion retention gate | 1 | 1 | — |
| Diversity retention gate | 1 | 1 | — |
| All-objective gate | 0 | 0 | — |

Interpretation:

- Neither method improved held-out MQ by step 100.
- Posterior is `0.00158` worse than GRPO on MQ, far too small to distinguish
  with the currently logged aggregate statistics.
- Posterior is worse on VQ but better on TA at this checkpoint. Both remain
  within the configured `-0.02` held-out-drop tolerances.
- Motion and diversity are essentially unchanged, so there is no evidence of
  early collapse.
- GRPO is currently faster per update. At validation step 100 it used about
  17% fewer GPU-hours than posterior (`3.97` versus `4.79`).
- The full trajectory, later checkpoints, and paired per-prompt statistics are
  required before making an efficiency or quality claim.

An asynchronous live snapshot (posterior step 115, GRPO step 137) confirms that
the mechanics remain active: posterior ESS ratio is exactly `0.5`, posterior
maximum weight is `0.680`, reward-selection gain is positive, and both jobs are
still checkpointing. These rows are from different optimizer steps and prompts,
so their raw rewards or gradient norms must not be compared as a paired result.

## W&B coverage

Training logs include:

- MQ/VQ/TA candidate reward means and standard deviations.
- Reward selection gain and normalized-advantage magnitude.
- ESS, ESS ratio, solved temperature, weight maximum, and weight entropy.
- Source/target/anchor times, configured/effective local delta, and endpoint
  clipping indicator.
- Action deviation from deterministic target and posterior mean.
- Temporal L1, gradient norm, post-update log-probability change, and KL probe.
- Step time, cumulative GPU hours, and EMA update count.
- Objective-specific posterior coefficient or GRPO ratio/clip diagnostics.

Held-out validation logs include:

- MQ/VQ/TA means, standard deviations, and SEM.
- Deltas from the untouched step-zero baseline.
- Motion, static-sample ratio, latent diversity, and video diversity.
- Gain per 100 steps and per GPU-hour.
- Primary, held-out, motion, diversity, and aggregate success gates.
- Eight fixed qualitative videos with prompts and reward captions per validation.

## Concerns and hypotheses for the next reviewer

### P0: replace independent validation significance with paired statistics

Validation uses fixed prompts and seeds, but the code computes

```text
1.96 * sqrt(SEM_current^2 + SEM_baseline^2)
```

as if the measurements were independent. At the observed standard deviation,
the resulting margin is approximately `0.25`. The configured useful MQ target
is only `0.02`, so the gate cannot recognize a modest real gain with 64 prompts.

Recommended change:

1. Preserve a stable key `(prompt_index, sample_seed)`.
2. Log per-prompt current and baseline rewards as a W&B table/artifact or JSON.
3. Compute the mean and SEM of paired differences directly.
4. Compare posterior and GRPO with paired differences or a paired bootstrap.
5. Keep the absolute MQ delta threshold as a separate practical-significance
   criterion.

### P0: clarify or enforce the “same actions” invariant

Both objectives use the same exogenous randomness, but actions are sampled as

```text
a = mean_theta(s) + std_theta(s) * epsilon.
```

After the parameter updates diverge, equal `epsilon` does not produce equal
actions. This is a valid seed-matched on-policy comparison, but it is not a
literal shared-action comparison.

Possible stricter designs:

- Generate each iteration's rollout group from a shared frozen behavior policy,
  then feed identical actions/rewards/log probabilities to both learners.
- Alternate data collection and synchronized objective-only updates from the
  same behavior checkpoint.
- Retain the current on-policy comparison but correct the documentation and
  treat rollout-distribution divergence as part of the method effect.

### P0: audit objective scale and actual update size

Posterior coefficients `w-1/G` and standardized GRPO advantages have different
scales. Both objectives currently share one learning rate, but that does not
ensure equal KL, equal gradient norm, or equal parameter-space step size.
`gaussian_log_prob_mean` also averages over a very large latent action, which
keeps gradients numerically manageable but makes the absolute scale a design
choice.

Recommended analysis:

- Plot synchronized per-step gradient norm, post-update KL/log-probability
  delta, coefficient RMS, and reward gain.
- Confirm that post-update probes are populated on their configured cadence;
  zeros on non-probe steps should be treated as missing, not measured zero.
- Add a trust-region-matched ablation, e.g. tune learning rates so both methods
  reach similar early KL or log-probability change.
- Consider coefficient normalization or an explicit target-KL controller before
  concluding that one objective is intrinsically less sample efficient.

### P1: verify local ASFMC numerically against the official reference

The sign and time-coordinate derivation has unit tests and passed real runtime,
but it has not been reported here as a layer/output parity comparison against
the official Flow-Map-GRPO implementation. Check:

- Whether AnyFlow's `(r, r)` output is exactly the instantaneous reverse
  velocity assumed by the derivation.
- The sign of `v_q` after converting from the paper coordinate.
- Timestep normalization and `flow_shift=5.0` interaction.
- Whether the released reference excludes the last branch rather than truncating
  its local interval.
- Whether `terminal_base_sigma=0.05` is applied at the intended endpoint.

### P1: analyze the clipped final branch separately

For target `q=0.024414`, setting `delta_eff=q` makes the local standard deviation
shrink roughly with `q`. The latest GRPO snapshot on that branch had action
deviation `0.0257`, versus approximately `0.23` on an ordinary posterior branch
snapshot. This is expected mathematically, but it means branch-dependent
exploration and reward signal differ sharply.

Log and compare metrics stratified by branch index. Consider an ablation that
excludes the final branch or uses an official endpoint treatment.

### P1: validate VideoAlign checkpoint coverage and calibration

Transformers 5.x changed Qwen2-VL module names. The wrapper remaps state-dict
keys and the deterministic preflight produced stable nontrivial MQ/VQ/TA scores,
but model loading emitted a large base-model missing/unexpected-key report before
the patched reward checkpoint was applied.

Recommended checks:

- Record exact missing/unexpected keys after the final reward checkpoint load.
- Assert a high loaded-parameter coverage ratio for base, LoRA, and reward head.
- Score known VideoAlign calibration videos and compare with the upstream
  implementation/environment.
- Prefer the upstream-supported Transformers/PEFT versions if practical instead
  of relying only on runtime remapping.

### P1: investigate training/held-out reward mismatch

Recent candidate-group MQ values are often strongly positive, while the fixed
held-out mean remains slightly negative and declined at step 100. This could be
ordinary prompt/action variance, rapid overfitting to only 451 prompts, or local
reward exploitation that does not survive fixed four-step evaluation.

Useful diagnostics:

- Train versus held-out reward distributions on the same prompt subset.
- Reward by branch index and prompt frequency.
- Fixed behavior-policy evaluation at more frequent early checkpoints.
- Human inspection of the eight W&B videos and a larger downloadable sample.
- A second seed before interpreting a small single-seed difference.

### P2: improve experiment power and evaluation breadth

- Save all per-prompt scalar validation results, not only aggregates and eight
  qualitative videos.
- Run at least three seeds for any final claim.
- Add a human preference or stronger video-quality evaluation for shortlisted
  checkpoints; VideoAlign alone should not establish superiority.
- Evaluate the raw student as well as EMA to determine whether EMA lag obscures
  early changes.
- Report reward versus wall time and GPU-hours, since posterior currently takes
  longer per step.

## Suggested next actions

1. Let r3 reach at least the step-200 validation unless instability appears.
2. Export and inspect synchronized W&B curves for coefficient scale, gradient
   norm, post-update KL, reward selection gain, and branch index.
3. Implement per-prompt paired validation logging and paired confidence
   intervals before relying on `validation_success/all`.
4. Numerically compare `local_anchor_gaussian_parameters()` with the official
   Flow-Map-GRPO reference on fixed tensors and an AnyFlow checkpoint.
5. Audit final VideoAlign key coverage and calibration scores.
6. Decide whether the intended claim is a seed-matched on-policy comparison or
   a strict shared-rollout update-rule comparison, then align code and docs.
7. If the posterior objective is consistently under-updating, run a
   trust-region-matched learning-rate/normalization ablation rather than changing
   several scientific variables at once.
8. Repeat the best corrected configuration with multiple seeds.

## Reproduction and monitoring

Do not launch another full pair while r3 is active. The original launch command
was equivalent to:

```bash
conda run --no-capture-output -n fastvideo \
  modal run modal_train_finite_transition_posterior.py \
  --paired \
  --max-train-steps 1200 \
  --comparison-id ftpp_pair_s42_20260820_r3
```

Current checkpoint roots on Modal volume `fastvideo-runs`:

```text
/root/FastVideo/outputs/finite_transition_posterior/
  anyflow_ftp_posterior_projection_videoalign_mq_f81_s42_20260821_033617/
  anyflow_ftp_flowmap_grpo_videoalign_mq_f81_s42_20260821_033621/
```

Both have persisted `checkpoint-100`; checkpoints are configured every 50
updates with a retention limit of four.

If one arm fails, resume it with the same run name and `latest` checkpoint rather
than preparing a new prompt split or W&B identity. See
`examples/train/finite_transition_posterior_wan.md` for the exact launcher flags.

## Relevant code map

| Area | File |
|---|---|
| Full method and evaluation | `fastvideo/train/methods/rl/finite_transition_posterior.py` |
| AnyFlow local policy and resume state | `fastvideo/train/methods/rl/finite_transition_posterior_repro.py` |
| Posterior/GRPO math | `fastvideo/train/methods/rl/common/finite_transition.py` |
| Local ASFMC math | `fastvideo/train/methods/rl/common/local_asfmc.py` |
| VideoAlign wrapper | `fastvideo/train/methods/rl/rewards/videoalign.py` |
| Scientific YAML | `examples/train/configs/rl/wan/finite_transition_posterior_anyflow_videoalign.yaml` |
| Asset preparation | `examples/train/prepare_finite_transition_posterior_assets.py` |
| Environment preflight | `examples/train/check_finite_transition_posterior_environment.py` |
| Modal launcher | `modal_train_finite_transition_posterior.py` |
| Method explanation | `examples/train/finite_transition_posterior_wan.md` |
| Agent context/invariants | `examples/train/finite_transition_posterior_agent_context.md` |
| Core/method tests | `fastvideo/tests/train/methods/test_finite_transition_posterior_*.py` |
| Local ASFMC tests | `fastvideo/tests/train/methods/test_local_asfmc.py` |
| AnyFlow integration test | `fastvideo/tests/training/distill/test_anyflow_pretrain.py` |

## Bottom line

The run is no longer blocked by infrastructure or known geometry errors, and
the implementation now produces rich quantitative and qualitative evidence.
The available data do not show an FTPP advantage at step 100. The small negative
MQ difference is inconclusive, while the current evaluation statistics and
literal shared-action claim need correction before a strong scientific
conclusion can be supported.
