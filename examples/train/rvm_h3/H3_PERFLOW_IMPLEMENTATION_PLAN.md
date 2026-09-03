# H3 → FastH3 reward-filtered PeRFlow implementation plan

## Status and source branch

- Working branch: `adam/h3-teacher-perflow-distill`
- Branched from: `adam/h3-rvm-posttraining`
- Base commit: `74907dd347805e12fb12b8f093afc41d6184d312`
- Downstream stage retained unchanged: paper-faithful on-policy FastH3 RVM
- PeRFlow source audit: `magic-research/piecewise-rectified-flow@87bac762f049d069497e83a76f528f007e9adabd`
- Reference training path: `scripts/perflow_accelerate_sd.py`

This document is the durable design contract for the implementation. The living execution record is [`H3_PERFLOW_PROGRESS.md`](H3_PERFLOW_PROGRESS.md).

## Project objective

Distill high-quality trajectories sampled by the full MiniMax H3 teacher into the deployed four-forward FastH3 model, then optionally continue with the existing on-policy RVM implementation.

The teacher stage should transfer **where good H3 trajectories go and how they move between FastH3’s four deployed boundaries**. The RVM stage should then improve FastH3 using rewards on samples from FastH3’s own current policy.

The intended pipeline is:

```text
full H3, dense teacher
    ↓ K candidates per prompt
frozen existing reward profile
    ↓ deterministic top-q selection
five FastH3-aligned trajectory anchors
    ↓ piecewise rectified-flow regression
FastH3 quality LoRA
    ↓ optional initialization
existing paper-faithful on-policy RVM
```

## Core methodological decision

Use **reward-filtered Piecewise Rectified Flow (PeRFlow)** for H3 teacher distillation.

The official PeRFlow velocity-matching recipe samples a query time inside a teacher window, linearly interpolates the window endpoints, and regresses the endpoint secant velocity. H3 requires one careful adaptation: video and audio share one base timestep but apply different scheduler shifts, so the two modalities are straightened in their own sigma coordinates.

Let adjacent cached anchors be

\[
(z_{v,m},z_{a,m},\tau_m),
\qquad
(z_{v,m+1},z_{a,m+1},\tau_{m+1}),
\]

where:

- \(v\) denotes video;
- \(a\) denotes audio;
- \(\tau\) is the repository’s shared base timestep, descending from `1000` to `0`;
- \(\sigma_v(\tau)\) and \(\sigma_a(\tau)\) are the existing H3 video and audio noise-amount maps, with native shifts `12` and `3`.

Draw one shared fraction

\[
s\sim\mathcal U(0,1)
\]

and query the shared base schedule at

\[
\tau_q=(1-s)\tau_m+s\tau_{m+1}.
\]

For each modality \(j\in\{v,a\}\), define

\[
\alpha_j
=
\frac{
\sigma_j(\tau_q)-\sigma_j(\tau_m)
}{
\sigma_j(\tau_{m+1})-\sigma_j(\tau_m)
},
\]

\[
z_{j,q}
=
z_{j,m}
+
\alpha_j
\left(z_{j,m+1}-z_{j,m}\right),
\]

and the piecewise-constant target field

\[
u_{j,m}
=
\frac{
z_{j,m+1}-z_{j,m}
}{
\sigma_j(\tau_{m+1})-\sigma_j(\tau_m)
}.
\]

FastH3’s packed transformer predicts `noise - clean`, which is the derivative of

\[
x_{\sigma}=(1-\sigma)x_0+\sigma\epsilon
\]

with respect to \(\sigma\). Therefore the sigma-domain denominator is required. Using the raw base-timestep delta for both modalities would be wrong, and taking the denominator’s absolute value would reverse the field because cached trajectories run from high sigma to low sigma.

Train with separate modality reductions:

\[
\mathcal L_{\mathrm{PeRFlow}}
=
\mathcal L_v
+
\lambda_a\mathcal L_a
+
\beta\mathcal L_{\mathrm{anchor}},
\]

\[
\mathcal L_j
=
\mathbb E\left[
w_i\,
\rho\!\left(
v_{\theta,j}(z_q,\tau_q,c)
-
\operatorname{sg}(u_{j,m})
\right)
\right].
\]

Here:

- \(v_{\theta,j}\) is FastH3’s existing packed velocity predictor restricted to modality \(j\);
- \(c\) is the original H3 prompt conditioning;
- \(\operatorname{sg}\) stops gradients through cached teacher states and targets;
- \(w_i=1/q\) for every retained candidate, normalized as a weighted mean;
- \(\rho\) is **MSE by default**, matching the official PeRFlow velocity-matching implementation;
- Huber is available only as an explicit robustness ablation;
- \(\mathcal L_{\mathrm{anchor}}\) is an optional LoRA-off function-space anchor and is disabled by default.

The same sampled base timestep is sent through the existing H3 model adapter, preserving its packed row timesteps, VSA metadata, and distinct video/audio schedule shifts.

## Reward filtering contract

The existing H3 REST cache already contains all `K` trajectories per prompt, every raw reward component, the mixed reward advantage, hashes, prompt identity, candidate index, seed, and five FastH3-aligned anchors. PeRFlow therefore filters the immutable cache at dataset construction time rather than generating a second physical cache.

Default selection:

```text
K = 8 cached H3 candidates per prompt
q = 2 selected candidates per prompt
ranking scalar = mixed_advantage
ordering = descending score, then ascending candidate_index
selected sample weight = 1/q
```

Important design rules:

1. Reward determines **which teacher trajectories are retained**, not the magnitude or sign of the supervised PeRFlow gradient.
2. Every retained candidate receives equal weight by default.
3. Ties are deterministic.
4. Every prompt must contain exactly `K` candidates before selection.
5. Selection configuration and selected trajectory IDs must be fingerprinted and logged.
6. Held-out evaluation prompts must not enter teacher-bank generation or reward calibration.

## Why not put H3 rollouts directly into RVM?

The current RVM objective is on-policy: its group-relative advantages and velocity update are defined using FastH3 behavior rollouts. Replacing those rollouts with samples from full H3 would create an off-policy objective without an importance correction or a supporting derivation. It would no longer be the published RVM algorithm.

Therefore:

- H3 samples are used only by the supervised teacher-distillation stage.
- RVM continues to collect endpoints from the current FastH3 behavior snapshot.
- The only connection is checkpoint initialization: `PeRFlow → RVM`.

## Why PeRFlow instead of the alternatives?

| Method | Uses full H3 trajectories | Directly matches FastH3 velocity | Extra learned critic | Decision |
|---|---:|---:|---:|---|
| Reward-filtered PeRFlow | Yes | Yes | No | Primary method |
| T2V-Turbo-style consistency distillation | Yes | Indirectly | No | Strong fallback, larger parameterization change |
| DMD/DMD2 | Distribution-level | No | Yes | Existing baseline; too invasive for this teacher-transfer stage |
| Diffusion-DPO | Mostly final pairs | No | No | Discards intermediate H3 trajectory supervision |
| H3 samples inside RVM | Yes | Superficially | No | Rejected: invalid off-policy substitution |

PeRFlow is the narrowest established method that matches the object already trained by FastH3: a few-step velocity field over explicit temporal windows.

## Reuse map: existing repository components

The implementation must reuse, rather than duplicate:

### Full-H3 trajectory generation

- `fastvideo.train.methods.knowledge_distillation.h3_rest_sampler`
- `fastvideo.train.models.minimax_h3.MiniMaxH3RESTTeacherModel`
- `examples/train/rvm_h3/build_h3_rest_cache.py`

These already perform dense H3 sampling, record the exact five FastH3 boundaries, score `K` candidates, and write a provenance-sealed cache.

### Cached trajectory loading

- `fastvideo.dataset.h3_rest_cache`
- `fastvideo.dataset.h3_perflow_cache`
- `fastvideo.train.models.minimax_h3.MiniMaxH3RESTModel`

PeRFlow adds an opt-in deterministic top-q view over this cache. Existing H3 REST behavior remains backward compatible.

### FastH3 training path

- `MiniMaxH3RVMModel` / `MiniMaxH3RESTModel`
- packed video/audio modality slices
- native video/audio scheduler shifts
- VSA metadata refresh
- LoRA-only training with FP32 trainable adapter masters
- repository optimizer, trainer, checkpoint, and resume machinery

### Reward profile

- existing original RVM reward profile
- existing calibrated Physion/MJ profile
- the same immutable reward configuration and calibration artifact used when the H3 cache was built

### Stage-two RVM

- `RVMFaithfulMethod` / `RVMRewardProfileMethod`
- released four-forward FastH3 VSA rollout
- existing evaluation, export, and inference tooling

No RVM rollout or loss code is copied into PeRFlow.

## Task subsets

### T0 — Freeze design and provenance contract

- [x] Create remote feature branch from current RVM/H3-REST head.
- [x] Audit H3 REST cache, sampler, model, and RVM interfaces.
- [x] Write this plan and the living progress report.
- [x] Record exact source references and implementation non-goals in tests/docs.

Acceptance: the design explicitly distinguishes supervised teacher transfer from on-policy RVM.

### T1 — Deterministic reward-filtered cache view

- [x] Add pure top-q selection utilities.
- [x] Add an opt-in selected cache view while preserving existing REST defaults.
- [x] Validate complete prompt groups before ranking.
- [x] Fingerprint the selection policy and expose selected trajectory metadata.
- [x] Add tests for ties, malformed prompt groups, non-finite scores, equal weights, and deterministic ordering.

Acceptance: for every prompt, exactly `q` of `K` candidates are exposed, with weight `1/q`, reproducibly across process restarts and distributed ranks.

### T2 — Piecewise interpolation and velocity-loss primitives

- [x] Add continuous base-timestep sampling and modality-specific sigma interpolation.
- [x] Add MSE and Huber packed-modality losses.
- [x] Keep video and audio reductions separate before weighting.
- [x] Detach cached targets and fail on zero/non-finite sigma intervals.
- [x] Add exact linear-field, sign, weighting, and gradient tests.

Acceptance: an exact linear velocity predictor has zero loss and teacher tensors receive no gradients.

### T3 — `H3PeRFlowMethod`

- [ ] Implement a YAML-selectable method alongside `H3RESTMethod`.
- [ ] Reuse FastH3 LoRA, VSA, optimizer, checkpoint, and resume code.
- [ ] Train at continuous points inside each cached segment.
- [ ] Default to equal selected-candidate weighting with no signed reward coefficient.
- [ ] Add optional LoRA-off function-space anchor as an isolated ablation, disabled by default.
- [ ] Log interpolation, segment, target-norm, selection, and cache-fingerprint diagnostics.

Acceptance: one tiny CPU end-to-end step produces finite loss and non-zero student gradients while leaving the frozen base untouched.

### T4 — Configs and execution scripts

- [ ] Add compact one-GPU correctness config.
- [ ] Add four-/eight-GPU topology and pilot configs.
- [ ] Add cache-build, cache-verify, PeRFlow-train, resume, export, and `PeRFlow → RVM` scripts.
- [ ] Reuse `examples/train/rvm_h3/common.sh` environment and reward paths.
- [ ] Make every run fail closed on missing cache, model, reward calibration, or source mismatch.

Acceptance: scripts pass shell parsing and YAMLs pass configuration tests.

### T5 — Evaluation and reporting

Run matched controls with fixed prompts, seeds, initialization, topology, optimizer budget, and reward profile:

```text
C0  released FastH3
C1  original published-profile RVM
C2  Physion/MJ RVM
T1  unfiltered H3 PeRFlow
T2  reward-filtered H3 PeRFlow
T3  reward-filtered H3 PeRFlow → on-policy RVM
```

Primary gate:

- at least 2% improvement in the held-out aggregate reward over the relevant FastH3 control;
- positive lower bound under paired bootstrap confidence intervals;
- less than 1% degradation on any temporal-integrity, motion, or diversity guard metric;
- no NaN, Inf, collapse, duplicate output, or checkpoint/resume mismatch.

Acceptance: quality claims are made only after the real H3 checkpoint, real reward models, and matched held-out evaluation have run.

## Non-goals

- Inventing a new RL algorithm.
- Backpropagating through MJ-VIDEO, VideoAlign, RAFT, or the H3 sampling rollout.
- Replacing FastH3 behavior rollouts with H3 rollouts inside RVM.
- Training a second 35B reference model during the PeRFlow stage.
- Re-scoring or rewriting the immutable teacher cache during student training.
- Mixing held-out evaluation prompts into cache generation or calibration.
- Claiming empirical improvement from CPU/static tests alone.

## Expected resource profile

Teacher-cache generation remains the expensive full-H3 stage and uses the existing dense H3 sequence-parallel sampler. Student PeRFlow training loads only FastH3 plus its trainable LoRA and cached packed latents; it does not keep full H3 or reward models resident.

This separation is intentional: teacher inference and reward scoring happen once, while the much cheaper student loss can reuse the immutable bank for many ablations.

## References

1. Liu, Gong, and Liu, **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**, ICLR 2023. https://arxiv.org/abs/2209.03003
2. Yan et al., **PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator**, NeurIPS 2024. https://arxiv.org/abs/2405.07510
3. Official PeRFlow implementation, source pin `87bac762f049d069497e83a76f528f007e9adabd`. https://github.com/magic-research/piecewise-rectified-flow
4. Liu et al., **InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation**. https://arxiv.org/abs/2309.06380
5. Li et al., **T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback**. https://arxiv.org/abs/2405.18750
6. Yin et al., **One-step Diffusion with Distribution Matching Distillation**. https://arxiv.org/abs/2311.18828
7. Yin et al., **Improved Distribution Matching Distillation for Fast Image Synthesis**. https://arxiv.org/abs/2405.14867
8. Wallace et al., **Diffusion Model Alignment Using Direct Preference Optimization**. https://arxiv.org/abs/2311.12908
9. Dong et al., **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**. https://arxiv.org/abs/2003.12039

## Implementation rule

Every completed task subset must update `H3_PERFLOW_PROGRESS.md`, be committed to this branch, and be verified against the remote branch head before the next subset begins.
