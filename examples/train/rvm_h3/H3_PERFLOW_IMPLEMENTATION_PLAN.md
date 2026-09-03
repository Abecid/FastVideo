# H3 → FastH3 reward-filtered PeRFlow implementation plan

## Status and source branch

- Working branch: `adam/h3-teacher-perflow-distill`
- Branched from: `adam/h3-rvm-posttraining`
- Base commit: `74907dd347805e12fb12b8f093afc41d6184d312`
- Downstream stage retained unchanged: paper-faithful on-policy FastH3 RVM

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

For one selected teacher trajectory, let the adjacent cached anchors be

\[
(z_m,t_m),\qquad (z_{m+1},t_{m+1}),
\]

where `t` is FastH3 model time, increasing from noise to clean data. Draw

\[
s\sim\mathcal U(0,1),
\]

and construct

\[
t=(1-s)t_m+s t_{m+1},
\qquad
z_t=(1-s)z_m+s z_{m+1}.
\]

The exact velocity of this straight segment is

\[
u_m=\frac{z_{m+1}-z_m}{t_{m+1}-t_m}.
\]

Train FastH3 with

\[
\mathcal L_{\mathrm{PeRFlow}}
=
\mathbb E\left[
\rho\!\left(v_\theta(z_t,t,c)-\operatorname{sg}(u_m)\right)
\right],
\]

where:

- `v_θ` is the existing FastH3 packed video/audio velocity model;
- `c` is the original H3 prompt conditioning;
- `sg` stops gradients through cached teacher targets;
- `ρ` is Huber loss by default, with MSE as an explicit ablation.

FastH3’s repository timestep convention is decreasing `[1000, 750, 500, 250, 0]`, while the corresponding model-time convention is increasing `[0, 0.25, 0.5, 0.75, 1]`. The implementation must convert through the existing model-time helper and must never silently remove the sign of the interval denominator.

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
- `fastvideo.train.models.minimax_h3.MiniMaxH3RESTModel`

PeRFlow adds an opt-in deterministic top-q view over this cache. Existing H3 REST behavior must remain backward compatible.

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

No RVM rollout or loss code should be copied into PeRFlow.

## Task subsets

### T0 — Freeze design and provenance contract

- [x] Create remote feature branch from current RVM/H3-REST head.
- [x] Audit H3 REST cache, sampler, model, and RVM interfaces.
- [x] Write this plan and the living progress report.
- [ ] Record exact source references and implementation non-goals in tests/docs.

Acceptance: the design explicitly distinguishes supervised teacher transfer from on-policy RVM.

### T1 — Deterministic reward-filtered cache view

- [ ] Add pure top-q selection utilities.
- [ ] Extend `H3RESTCacheDataset` with an opt-in selection configuration while preserving existing REST defaults.
- [ ] Validate complete prompt groups before ranking.
- [ ] Fingerprint the selection policy and expose selected trajectory metadata.
- [ ] Add tests for ties, malformed prompt groups, non-finite scores, equal weights, and deterministic ordering.

Acceptance: for every prompt, exactly `q` of `K` candidates are exposed, with weight `1/q`, reproducibly across process restarts and distributed ranks.

### T2 — Piecewise interpolation and velocity-loss primitives

- [ ] Add model-time conversion, continuous segment interpolation, and secant-target helpers.
- [ ] Add Huber and MSE packed-modality losses.
- [ ] Keep video and audio reductions separate before weighting.
- [ ] Detach cached targets and fail on zero/non-finite time intervals.
- [ ] Add exact linear-field and gradient tests.

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
3. Official PeRFlow implementation. https://github.com/magic-research/piecewise-rectified-flow
4. Liu et al., **InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation**. https://arxiv.org/abs/2309.06380
5. Li et al., **T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback**. https://arxiv.org/abs/2405.18750
6. Yin et al., **One-step Diffusion with Distribution Matching Distillation**. https://arxiv.org/abs/2311.18828
7. Yin et al., **Improved Distribution Matching Distillation for Fast Image Synthesis**. https://arxiv.org/abs/2405.14867
8. Wallace et al., **Diffusion Model Alignment Using Direct Preference Optimization**. https://arxiv.org/abs/2311.12908
9. Dong et al., **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**. https://arxiv.org/abs/2003.12039

## Implementation rule

Every completed task subset must update `H3_PERFLOW_PROGRESS.md`, be committed to this branch, and be verified against the remote branch head before the next subset begins.
