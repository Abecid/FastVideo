# Video Posterior Transition Distillation

**Reward-tilted, on-manifold alignment of a released four-step AnyFlow video model**

## Abstract

Few-step video generators are attractive deployment policies but awkward objects
for reward alignment. Their learned transitions are deterministic long-range
flow maps, while standard diffusion RL assumes many local stochastic transitions
with tractable likelihoods. A naïve workaround—score complete videos and regress
reward-weighted noise/endpoints—either destroys the pretrained transport coupling
or leaves the optimal deterministic map unchanged. Both failures appeared in the
previous RTFD/RTRFD experiments.

Video Posterior Transition Distillation (VPTD) begins from NVIDIA's released,
already functional **AnyFlow-Wan2.1-1.3B** four-step generator. At one finite
transition, it keeps a shared on-manifold prefix, uses Flow-Map GRPO's exact
endpoint-anchor stochasticization to draw valid conditional next states, completes
each branch with the pretrained flow map, and evaluates terminal video reward.
Those candidate actions define a Feynman--Kac reward-tilted transition posterior.
VPTD then projects that posterior back into the AnyFlow LoRA policy with one
advantage-weighted maximum-likelihood update. Deterministic four-step inference is
unchanged.

The experiment tests one precise hypothesis:

> A few-step video model can be aligned more safely by changing a valid finite
> transition distribution at a shared on-manifold state than by optimizing
> unrelated completed videos or by stochasticizing an infinitesimal velocity
> sampler.

The method uses a released few-step checkpoint, released video-model settings,
released LoRA/optimizer settings, an exact path conditional, a video-domain group
size, and a mandatory inference-parity gate. No raw Wan checkpoint is ever treated
as a four-step generator.

---

## 1. The bottleneck

Let

\[
\psi_\theta^{t\to r}(x_t,c)
\]

be a deterministic AnyFlow transition from source time \(t\) to target time
\(r\), conditioned on prompt \(c\). A standard four-step rollout is

\[
x_{r_j}=\psi_\theta^{r_{j-1}\to r_j}(x_{r_{j-1}},c),
\qquad j=1,\ldots,4.
\]

This is fast, but it has no stochastic action distribution
\(\pi_\theta(x_{r_j}\mid x_{r_{j-1}},c)\). Consequently:

1. ordinary PPO/GRPO likelihood ratios are undefined;
2. one intermediate state has only one deterministic future;
3. terminal reward gives poor credit to individual finite transitions;
4. arbitrary Gaussian perturbation can leave the pretrained probability path;
5. reward models become unreliable when exploration produces off-manifold video.

The previous endpoint-weighting experiment did not solve this. Independent
noise/video pairing changed the transport problem and averaged incompatible video
motions. Reusing the teacher's exact noise/video pair preserved the transport map,
but reward weighting did not change the pointwise optimum of that deterministic
map. VPTD instead changes the **conditional next-state distribution at a fixed
valid state**.

---

## 2. Prior work used directly

### AnyFlow: the finite-transition substrate

AnyFlow learns a two-time flow map rather than evaluating an instantaneous Wan
velocity field at an unsupported four-step schedule. The released Wan-1.3B model
supports arbitrary finite transitions and has a verified four-step operating
point. VPTD loads this checkpoint unchanged and adds LoRA only.

### Flow-Map GRPO: path-preserving stochastic actions

Flow-Map GRPO introduces Anchored Stochastic Flow Map Composition (ASFMC). For a
finite transition \(t\to r\):

1. compute the deterministic state
   \(\widehat x_r=\psi_{\theta_{\rm old}}^{t\to r}(x_t,c)\);
2. map it to the clean endpoint
   \(\widehat x_0=\psi_{\theta_{\rm old}}^{r\to 0}(\widehat x_r,c)\);
3. sample back onto the affine probability path.

AnyFlow uses the reverse convention

\[
x_r=(1-r)x_0+r\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\]

where \(r=1\) is Gaussian noise and \(r=0\) is clean video. Therefore the exact
endpoint-anchor policy is

\[
\boxed{
\pi_{\theta_{\rm old}}(a\mid s)
=
\mathcal N\!\left(
(1-r)\,\widehat x_0,
\;r^2 I
\right)
}
\]

with state

\[
s=(c,t,r,x_t)
\]

and action \(a=x_r\). There is no hand-selected noise multiplier or variance
floor.

### Diamond Maps and Feynman--Kac steering: tilt the posterior

Diamond Maps frames reward alignment around stochastic futures from a fixed
intermediate state. Given candidate actions from the behavior posterior and
terminal rewards \(R_j\), the desired local posterior is

\[
q_R(a\mid s)
\propto
q_{\rm old}(a\mid s)
\exp\!\left(\frac{R(a)}{\tau}\right).
\]

VPTD constructs this posterior empirically over candidate finite transitions.
Unlike Diamond Maps, it does not train a new posterior network from scratch; it
uses ASFMC on a released AnyFlow map and amortizes the reward-improved transition
into the existing LoRA policy.

### Advantage-Weighted Matching: retain supervised geometry

AWM shows that diffusion/flow alignment can retain the same score- or
flow-matching geometry as pretraining instead of relying only on a high-variance
trajectory policy gradient. VPTD follows this principle by performing a weighted
likelihood projection of the transition posterior.

### BranchGRPO / Flow-GRPO-Fast: shared-prefix local credit

All candidates share the same prompt, initial noise, deterministic prefix, and
selected transition state. They differ only in the stochastic action at that
transition. Reward differences therefore provide local credit for one finite
video transition instead of being copied indiscriminately over an entire
trajectory.

---

## 3. Method

### 3.1 Start from a valid four-step generator

VPTD initializes from

```text
nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
```

and uses the released AnyFlow code at the pinned commit recorded in the config.
Before optimization, the trainer compares its deterministic rollout against the
official `WanAnyFlowPipeline.training_rollout` implementation on identical
latents and embeddings. Training aborts if the latent outputs do not match under
PyTorch's dtype-specific numerical parity check.

The untouched base model is then evaluated on the fixed validation prompts and
logged to W&B at step zero. This makes bad inference impossible to misdiagnose as
slow training.

### 3.2 Choose one finite transition

The training schedule has five transitions:

\[
r_0\to r_1\to r_2\to r_3\to r_4\to 0.
\]

The first \(K=4\) transitions are stochastic policy steps, matching Flow-Map
GRPO. The last transition is deterministic and exists only to produce the reward
video.

At update \(n\), choose one index

\[
k\sim\operatorname{Uniform}\{0,1,2,3\}.
\]

Generate one shared prefix

\[
x_{r_k}
=
\psi_{\theta_{\rm old}}^{r_0\to r_k}(x_{r_0},c).
\]

Every reward candidate now answers the same counterfactual question:

> Which valid next state from this exact video state leads to a better final
> video?

### 3.3 Sample a conditional posterior group

For candidate \(j=1,\ldots,G\), draw

\[
a_j
\sim
q_{\rm old}(a\mid s_k)
=
\mathcal N(\mu_{\rm old},\sigma_k^2 I),
\]

where

\[
\mu_{\rm old}=(1-r_{k+1})\widehat x_0,
\qquad
\sigma_k=r_{k+1}.
\]

Complete the remaining transitions deterministically:

\[
x_0^{(j)}
=
\psi_{\theta_{\rm old}}^{r_{k+1}\to0}(a_j,c),
\]

then score the decoded video:

\[
R_j=R(x_0^{(j)},c).
\]

The default run optimizes **VideoAlign motion quality only**. Visual quality and
text alignment remain held-out diagnostics. Separate reward runs avoid silently
trading motion against a differently scaled reward head.

### 3.4 Feynman--Kac posterior tilt

Compute normalized weights

\[
w_j
=
\frac{\exp((R_j-\bar R)/\tau)}
{\sum_\ell\exp((R_\ell-\bar R)/\tau)}.
\]

Rather than transferring a reward-specific temperature from an image model,
VPTD solves \(\tau\) so that the effective sample size is half the candidate
population:

\[
\operatorname{ESS}(w)
=
\frac{1}{\sum_j w_j^2}
=
\frac{G}{2}.
\]

This uses the half-particle ESS threshold that is the default in Pyro
`SMCFilter`, expressed here as a scale-free temperature rule. If rewards are degenerate, the weights are exactly
uniform.

### 3.5 Posterior transition distillation

The ideal local target is

\[
q_R(a\mid s_k)
\propto
q_{\rm old}(a\mid s_k)e^{R(a)/\tau}.
\]

VPTD projects this target into the updated AnyFlow policy by descending

\[
D_{\rm KL}(q_R\,\|\,\pi_\theta).
\]

At the behavior parameters, the score identity

\[
\mathbb E_{q_{\rm old}}
[\nabla_\theta\log\pi_{\theta_{\rm old}}(a\mid s)]=0
\]

allows an exact control variate. The sampled loss is

\[
\boxed{
\mathcal L_{\rm VPTD}
=
-\sum_{j=1}^{G}
\left(w_j-\frac1G\right)
\log\pi_\theta(a_j\mid s_k)
}
\]

for one inner epoch.

The subtraction does not change the first-order posterior-projection gradient at
\(\theta_{\rm old}\), but it prevents a finite-sample random walk: if all rewards
are equal, \(w_j=1/G\) and the update is **exactly zero**.

Only the AnyFlow LoRA adapter is updated. The base flow map remains frozen, and
EMA is used for deterministic validation.

### 3.6 Inference

Training-time stochasticity is discarded. Evaluation uses the standard released
deterministic AnyFlow sampler:

\[
x_{r_{j+1}}
=
\psi_\theta^{r_j\to r_{j+1}}(x_{r_j},c),
\qquad j=0,\ldots,3.
\]

No reward model, branch search, SMC, or extra denoising call is needed at
deployment.

---

## 4. Why this is different

### Versus the failed RTFD/RTRFD endpoint regression

RTFD weighted unrelated completed teacher videos. VPTD compares valid next states
from one fixed intermediate state. The target is a changed conditional transition
distribution, not the same deterministic map under nonuniform regression density.

### Versus ordinary Flow-Map GRPO

Flow-Map GRPO applies a clipped trajectory policy-gradient objective. VPTD uses
the same path-preserving ASFMC policy but performs a Feynman--Kac posterior
projection at one shared-prefix transition. This is intended to reduce temporal
credit variance and preserve the regression-like optimization geometry that
works well in diffusion pretraining and AWM.

The repository includes `objective: flowmap_grpo` as a matched implementation
ablation. The same model, branches, rewards, prompts, and evaluation are used;
only the update rule changes.

### Versus Diamond Maps

Diamond Maps redesigns the model to efficiently produce stochastic posterior
samples for arbitrary inference-time rewards. VPTD tackles a narrower video
post-training problem: use exact ASFMC posterior samples from an existing
AnyFlow map, then amortize one chosen reward into its LoRA. It is cheaper to test
and leaves deterministic inference unchanged.

### Versus sequential few-step distillation then generic video RL

VPTD does begin from a released distilled map; this is deliberate. The previous
experiment showed that simultaneously inventing a four-step transport and
optimizing an off-manifold reward is underconstrained. The new contribution is
joint **finite-transition posterior distillation and reward alignment** on a
proven step-distilled substrate. It directly tests the posterior-transition
bottleneck without conflating it with whether raw Wan can be distilled at all.

---

## 5. Locked scientific configuration

The trainer rejects changes to the core scientific setup unless `--smoke` is
explicitly enabled.

| Setting | Value | Source |
|---|---:|---|
| Base model | AnyFlow-Wan2.1-T2V-1.3B | released AnyFlow checkpoint |
| Video | 81 frames, 480×832, 16 fps | released AnyFlow Wan config |
| Flow shift | 5 | released AnyFlow Wan config |
| Deterministic evaluation | 4 steps, CFG 1 | released AnyFlow Wan config |
| Stochastic policy transitions | 4 + final deterministic transition | Flow-Map GRPO |
| Posterior group | 4 videos/prompt | released Flow-GRPO Wan config |
| LoRA | rank 256, alpha 256, dropout 0 | released AnyFlow Wan config |
| LoRA targets | Q/K/V/out and FFN projections | released AnyFlow on-policy config |
| AdamW | LR 2e-6, betas (0, 0.999), WD 0 | released AnyFlow on-policy config |
| Gradient clipping | 1.0 | AnyFlow / Flow-Map GRPO |
| EMA | 0.99 after 200 updates | released AnyFlow on-policy config |
| Full target | 1,200 updates | released AnyFlow on-policy config |
| Checkpoint/qualitative diagnostic | 50 / 200 updates | released AnyFlow on-policy config |
| GRPO ablation clip | 1e-4 | Flow-Map GRPO |
| GRPO advantage clip/epsilon | 5 / 1e-4 | Flow-Map GRPO |
| Validation | 16 fixed prompts; 8 W&B videos | FastVideo RL default + requested qualitative coverage |

The target-ESS rule is part of the proposed method, not a transferred model
hyperparameter. It replaces a reward-scale-dependent temperature with a
scale-free SMC criterion and must be ablated against direct Flow-Map GRPO.

---

## 6. W&B diagnostics

Every update logs:

- posterior-distillation or GRPO loss;
- gradient norm;
- reward mean, standard deviation, minimum, and maximum;
- selected transition and source/target times;
- posterior standard deviation;
- action deviation from the deterministic map;
- effective sample size and fitted temperature;
- maximum posterior weight and posterior entropy;
- adjacent-frame temporal L1;
- policy-ratio/clip diagnostics for the GRPO ablation;
- exact zero-reward-group indicator.

Every evaluation logs:

- means and standard deviations for VideoAlign MQ, VQ, and TA;
- temporal L1;
- 16 fixed validation prompts;
- 8 deterministic four-step videos at 16 fps with prompt and reward captions.

At step zero, the untouched AnyFlow model is logged before any optimization.

---

## 7. Modal commands

Create the named secret once:

```bash
modal secret create fastvideo-training \
  HF_TOKEN="$HF_TOKEN" \
  WANDB_API_KEY="$WANDB_API_KEY"
```

### Inference and reward preflight only

```bash
modal run examples/train/modal_vptd.py --preflight-only
```

This is the mandatory first gate. It performs:

1. CPU unit tests;
2. exact AnyFlow source checkout verification;
3. custom-vs-official sampler latent parity;
4. untouched base-model validation and eight W&B videos;
5. no optimizer step.

### Engineering smoke test

```bash
modal run examples/train/modal_vptd.py --smoke
```

This intentionally reduces video shape and validation coverage for two updates.
It proves plumbing only and must not be plotted against the scientific run.

### First scientific pilot

```bash
modal run examples/train/modal_vptd.py \
  --max-steps 50 \
  --run-name wan_anyflow_vptd_videoalign_mq
```

### Full target

```bash
modal run examples/train/modal_vptd.py \
  --max-steps 1200 \
  --run-name wan_anyflow_vptd_videoalign_mq
```

The target is absolute and `resume=auto` is the default. If the 24-hour Modal
allocation is insufficient, rerun with monotonically increasing targets:

```text
200 → 400 → 600 → 800 → 1000 → 1200
```

### Matched Flow-Map GRPO ablation

```bash
modal run examples/train/modal_vptd.py \
  --max-steps 50 \
  --objective flowmap_grpo \
  --run-name wan_anyflow_flowmap_grpo_videoalign_mq
```

---

## 8. Required decision gates

1. **Base parity:** official and custom AnyFlow latents must match; base videos
   must be recognizable and moving.
2. **No-reward sanity:** zero reward variance must produce zero VPTD gradient.
3. **Pilot stability:** no non-finite values, quality collapse, or temporal-L1
   collapse through 50 updates.
4. **Reward validity:** MQ improvement must not be explained by flicker; inspect
   eight videos and held-out VQ/TA.
5. **Method value:** VPTD must beat the matched Flow-Map GRPO update at comparable
   reward calls, or show materially lower variance/stability cost.
6. **Publication gate:** evaluate on a real held-out prompt benchmark with VBench,
   independent rewards, and human pairwise judgments. The built-in 16-prompt
   panel is a development diagnostic, not paper evidence.

---

## 9. Limitations and next step

VPTD uses endpoint-anchor ASFMC rather than a learned GLASS/Diamond posterior
map. This is the simplest path-preserving experiment with released code, but the
endpoint anchor depends on the quality of the long-range \(r\to0\) AnyFlow map
and injects strong posterior variance at early times.

If the experiment improves reward but remains sample-inefficient, the principled
next step is to replace ASFMC with a learned video posterior map distilled from
GLASS-style conditional transitions. If it fails while direct Flow-Map GRPO
works, the posterior-projection update—not the stochastic transition—is the
failed hypothesis. If both fail despite a strong base, the likely bottleneck is
VideoAlign credit/reward robustness rather than step distillation.
