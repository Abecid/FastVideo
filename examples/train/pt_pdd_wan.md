# Posterior-Tilted Parallel Decoding Distillation for video

This branch implements the first runnable test of **Posterior-Tilted Parallel
Decoding Distillation (PT-PDD)** for Wan video generation.

The core research question is:

> Can reward alignment change the finite teacher transition being distilled,
> instead of being appended as a second policy-gradient objective after the
> student has already been compressed?

The implementation deliberately starts from NVIDIA's released
`nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers` checkpoint. It does **not** evaluate
raw Wan at four unsupported steps.

FastGen-PDD's public project page still lists its training code as forthcoming.
Consequently, the checked-in experiment uses AnyFlow as the proven finite-map
substrate and tests the exact PT-PDD contribution: a centered, reward-posterior
correction to a finite teacher velocity target. Once official PDD code is
available, the same correction replaces PDD's ordinary Runge--Kutta interval
target.

## 1. Problem

A conventional staged pipeline first learns a few-step student

\[
q_K = \Pi_K(p_T),
\]

then aligns that already-compressed generator:

\[
q_{K,R}=\mathcal A_R(q_K).
\]

The concern is that \(q_K\) may already have discarded teacher modes and motion
trajectories that are difficult to represent with \(K\) finite transitions.
Later RL can only steer inside the support and transition geometry retained by
that student.

A naive joint objective

\[
\mathcal L_{\rm distill}+\lambda_R\mathcal L_{\rm reward}
\]

has its own problems:

- dense teacher regression and sparse terminal reward use different gradient
  geometries;
- early few-step outputs may be outside the reward model's training manifold;
- one terminal scalar provides weak credit to individual denoising transitions;
- independent video comparisons often favor low-motion, low-artifact outputs;
- reward weighting a fixed deterministic teacher pair does not change its
  pointwise regression optimum.

PT-PDD instead changes the **teacher transition target itself**.

## 2. Prior-work components

The runnable method uses only mechanisms supported by released prior work.

| Component | Source |
|---|---|
| Competent finite video map | AnyFlow |
| Four stochastic transition positions | Flow-Map GRPO |
| Path-preserving endpoint posterior | ASFMC in Flow-Map GRPO |
| Several futures from one shared state | GLASS, Diamond Maps, BranchGRPO |
| Feynman--Kac reward tilt | Diamond Maps / diffusion SMC |
| Reward-corrected score or velocity target | RSM, RAM, AWM |
| Within-group centering | GRPO baselines, CRD, AWM |
| Single supervised regression geometry | PDD |
| Motion-first isolated reward experiment | DenseDPO and video-RL failure analyses |

No free stochastic-noise multiplier or ad hoc transition schedule is introduced.

## 3. Runnable AnyFlow proxy

Let

\[
\psi_\theta^{t\rightarrow r}(x_t,c)
\]

be AnyFlow's learned finite map from source time \(t\) to target time \(r<t\).
Its reverse-time update is

\[
x_r=x_t-\frac{t-r}{N}u_\theta(x_t,t,r,c),
\]

where \(N=1000\) is the training-time scale and \(u_\theta\) is an interval
mean velocity.

For every optimizer update, PT-PDD performs the following steps.

### 3.1 Select one on-policy transition

The released four-step AnyFlow generator is rolled out to one selected
intermediate state \(x_t\). All candidates share this exact prefix.

### 3.2 Build a path-preserving stochastic posterior

The behavior model first predicts a deterministic next state

\[
\widehat x_r=\psi_{\theta_{\rm old}}^{t\rightarrow r}(x_t,c)
\]

and its clean endpoint

\[
\widehat x_0=\psi_{\theta_{\rm old}}^{r\rightarrow0}(\widehat x_r,c).
\]

For the affine AnyFlow path

\[
x_r=(1-r)x_0+r\epsilon,
\]

conditioning on \(\widehat x_0\) gives the exact endpoint-anchor posterior

\[
q_{\rm old}(a\mid x_t,c)
=
\mathcal N\!\left((1-r)\widehat x_0,r^2I\right).
\]

One candidate action is sampled on each of four GPUs, giving a global group of
four same-prefix futures.

### 3.3 Complete and reward every future

Each sampled action \(a_j\) is completed deterministically to a clean video and
scored:

\[
R_j=R(Y_j,c).
\]

The first experiment optimizes only VideoAlign motion quality. Visual quality
and text alignment remain held-out diagnostics. This avoids the uncontrolled
metric seesaw observed when raw VQ, MQ, and TA scores are simply summed.

### 3.4 Fit a reward posterior by effective sample size

The empirical Feynman--Kac posterior is

\[
w_j
=
\frac{\exp((R_j-\overline R)/\tau)}
{\sum_\ell\exp((R_\ell-\overline R)/\tau)}.
\]

The temperature is solved by bisection so the posterior has a target effective
sample size

\[
\operatorname{ESS}(w)=\frac{1}{\sum_jw_j^2}=G/2.
\]

Thus reward scale does not require a manually transferred temperature.

### 3.5 Convert posterior futures into finite velocities

Every realized transition \((x_t,a_j)\) defines

\[
g_j
=
\frac{x_t-a_j}{(t-r)/N}.
\]

These are candidate finite-transition velocities available from the same state.

### 3.6 Center the reward correction

The reward-induced velocity shift is

\[
\boxed{
\Delta u_R
=
\sum_{j=1}^{G}
\left(w_j-\frac1G\right)g_j
}
\]

The uniform term is a control variate. If reward has no variation, then
\(w_j=1/G\) and

\[
\Delta u_R=0
\]

exactly.

This is different from multiplying an unchanged teacher MSE by reward weights.
Here reward changes the target velocity.

### 3.7 Perform one regression update

The frozen released AnyFlow map supplies the reference velocity

\[
u_{\rm ref}(x_t,t,r,c).
\]

The trainable LoRA student minimizes

\[
\boxed{
\mathcal L_{\rm PT-PDD}
=
\left\|
 u_\theta(x_t,t,r,c)
-
\operatorname{sg}
\left[u_{\rm ref}(x_t,t,r,c)+\Delta u_R\right]
\right\|_2^2.
}
\]

There is one loss and one optimization geometry. Reward does not backpropagate
through the decoder, reward model, rollout, or teacher.

## 4. Exact matched-compute controls

The command-line objective can be one of:

### `posterior_tilted_regression`

The proposed method. It regresses to

\[
u_{\rm ref}+\Delta u_R.
\]

### `reference_regression`

The strongest zero-correction control. It still performs the same:

- shared-prefix rollout;
- stochastic posterior sampling;
- video completions;
- VideoAlign calls;
- optimizer invocation;

but sets

\[
\Delta u_R=0.
\]

At the untouched LoRA initialization, this objective has exactly zero gradient.
Any difference from it is attributable to the reward correction, not extra
sampling or reward compute.

### `posterior_distillation`

The previous forward-KL likelihood projection of the same reward posterior.
This tests whether target regression is better than fitting a transition
likelihood.

### `flowmap_grpo`

A clipped likelihood-ratio baseline using the same states, actions, completions,
and rewards.

## 5. Scientific configuration

The non-smoke configuration is locked to released AnyFlow or Flow-Map-GRPO
settings.

| Setting | Value | Provenance |
|---|---:|---|
| Base checkpoint | AnyFlow-Wan2.1 T2V 1.3B | AnyFlow |
| Video shape | 81 frames, 480 x 832 | AnyFlow |
| Playback | 16 fps | AnyFlow |
| Flow shift | 5 | AnyFlow |
| Evaluation | deterministic four-step, CFG 1 | AnyFlow |
| Trainable parameters | LoRA only | AnyFlow on-policy |
| LoRA rank / alpha | 256 / 256 | AnyFlow on-policy |
| AdamW learning rate | 2e-6 | AnyFlow on-policy |
| Adam betas | 0, 0.999 | AnyFlow on-policy |
| Weight decay | 0 | AnyFlow on-policy |
| Gradient clipping | 1 | AnyFlow / Flow-Map GRPO |
| EMA | 0.99 after 200 updates | AnyFlow on-policy |
| Full target | 1,200 updates | AnyFlow on-policy |
| Candidate group | 4 | released video Flow-GRPO |
| Stochastic positions | 4 | Flow-Map GRPO |
| Posterior ESS target | half the group | SMC trust-region convention |

The `--smoke` flag intentionally relaxes only resolution, frame count,
validation coverage, and total updates. Smoke outputs are not scientific
comparisons.

## 6. Prompt datasets and evaluation subsets

The model is trained from generated videos and therefore needs prompt data, not
paired ground-truth clips. The branch exposes reproducible named profiles.

| Profile | Intended use |
|---|---|
| `world_r1_enhanced_dynamic` | Default motion- and camera-heavy pilot |
| `world_r1_enhanced_train` | Broader post-training prompt distribution |
| `world_r1_enhanced_test` | Default held-out validation prompts |
| `world_r1_final_dynamic` | Independent dynamic-prompt replication |
| `world_r1_final_train` | Alternate broad World-R1 training set |
| `world_r1_final_test` | Alternate held-out World-R1 evaluation |
| `vimix_public_sample` | Small public ViMix distribution-shift probe |
| `vidprom_unique` | Large real-user prompt pool, streamed and capped |

The default pilot uses:

```text
training:   world_r1_enhanced_dynamic, capped at 515 prompts
validation: world_r1_enhanced_test, fixed 16 prompts
qualitative: first 8 held-out prompts at every evaluation
```

The exact prompt lists are written to `prompt_split.json` in the output
directory. Resuming with a different dataset profile or prompt count fails
instead of silently changing the experiment.

Before using VidProM or another external prompt set in a publication, verify its
current dataset-card license and filtering requirements.

## 7. Recommended initial experiment matrix

### Gate 0: sampler and reward preflight

```bash
modal run examples/train/modal_pt_pdd.py --preflight-only
```

This must verify:

1. the custom sampler matches the pinned official AnyFlow sampler in latent
   space;
2. the untouched four-step model generates coherent moving videos;
3. VideoAlign loads and scores those videos;
4. eight qualitative videos are visible on W&B before training.

Do not proceed if this gate fails.

### Gate 1: engineering smoke

```bash
modal run examples/train/modal_pt_pdd.py --smoke
```

This is a two-update reduced-resolution test for distributed backward,
checkpointing, W&B, and resume behavior.

### Gate 2: exact zero-correction control

```bash
modal run examples/train/modal_pt_pdd.py \
  --max-steps 50 \
  --objective reference_regression \
  --run-name wan_anyflow_reference_regression_mq_50
```

### Gate 3: PT-PDD pilot

```bash
modal run examples/train/modal_pt_pdd.py \
  --max-steps 50 \
  --objective posterior_tilted_regression \
  --run-name wan_anyflow_pt_pdd_mq_50
```

### Gate 4: matched policy baselines

```bash
modal run examples/train/modal_pt_pdd.py \
  --max-steps 50 \
  --objective posterior_distillation \
  --run-name wan_anyflow_posterior_distillation_mq_50

modal run examples/train/modal_pt_pdd.py \
  --max-steps 50 \
  --objective flowmap_grpo \
  --run-name wan_anyflow_flowmap_grpo_mq_50
```

### Gate 5: prompt-distribution replication

Run the winning objective on an independent dynamic prompt profile:

```bash
modal run examples/train/modal_pt_pdd.py \
  --max-steps 100 \
  --dataset-profile world_r1_final_dynamic \
  --validation-profile world_r1_final_test \
  --max-train-prompts 500 \
  --run-name wan_anyflow_pt_pdd_mq_final_dynamic_100
```

Then test broader prompts:

```bash
modal run examples/train/modal_pt_pdd.py \
  --max-steps 100 \
  --dataset-profile world_r1_enhanced_train \
  --max-train-prompts 1000 \
  --run-name wan_anyflow_pt_pdd_mq_enhanced_100
```

## 8. W&B diagnostics

Every optimization update logs:

- training loss and gradient norm;
- reward mean, standard deviation, minimum, and maximum;
- zero-variance group rate;
- selected transition index and source/target times;
- posterior ESS, entropy, maximum weight, and fitted temperature;
- posterior action deviation from the deterministic map;
- candidate finite-velocity RMS;
- reference-velocity RMS;
- reward-correction RMS;
- correction/reference norm ratio;
- correction/reference cosine;
- student/reference MSE;
- temporal frame-difference diagnostic;
- objective indicator flags;
- GRPO ratio, clipping, and approximate KL for that baseline.

Every evaluation logs:

- fixed held-out MQ, VQ, and TA means;
- temporal frame difference;
- number of validation prompts;
- up to eight videos at 16 fps;
- prompt and reward values in each video caption.

## 9. What failures mean

### Base model is bad at step zero

This is an inference or checkpoint-parity failure. Stop immediately. It is not a
training-duration problem.

### `reference_regression` changes the base model

At initialized LoRA weights its gradient should be zero up to numerical noise.
A visible change indicates that the frozen-reference path or adapter disabling
is wrong.

### Posterior reward variance is almost always zero

The reward cannot distinguish candidate futures at that transition. Possible
responses, in order:

1. verify the reward model on recognizable held-out videos;
2. move the branch transition to a time where futures differ semantically;
3. increase the candidate group in a dedicated ablation;
4. train a time-aware latent reward model on few-step video states.

Do not increase reward strength when there is no reward information.

### Correction/reference ratio is extremely large

The reward posterior is selecting a direction much larger than the competent
base transition. Inspect:

- candidate videos;
- posterior ESS;
- correction cosine;
- reward calibration;
- transition index.

The first principled stabilizer is a KL/ESS trust-region sweep, not an arbitrary
loss coefficient.

### MQ improves while VQ or TA collapses

This is a true objective trade-off. Keep single-objective training for the
method paper and report a Pareto curve. Multi-objective optimization should be a
separate constrained or specialist-teacher extension.

### Motion quality rises but diversity falls

Measure cross-seed flow and video-feature diversity. This would indicate that a
reward-posterior correction is becoming mode-seeking. Relevant follow-ups are
JS/forward-divergence regularization, DFD repair, or a native stochastic
transition head—not a larger reward coefficient.

## 10. Full PDD extension

The current executable target is

\[
u_{\rm AnyFlow,ref}+\Delta u_R.
\]

When FastGen-PDD code is released, preserve its exact:

- interval grid;
- block length;
- parallel output heads;
- Euler or Midpoint teacher estimator;
- optimizer and data recipe;
- CFG and skip-layer configuration.

Then change only the target:

\[
u_{k}^{T,\mathrm{RK}}
\quad\longrightarrow\quad
u_{k}^{T,\mathrm{RK}}+\Delta u_{R,k}.
\]

That is the full Posterior-Tilted PDD method. The AnyFlow experiment is the
lowest-risk falsification test of its unique contribution before undertaking a
full PDD reproduction.
