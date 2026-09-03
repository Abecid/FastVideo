# H3 → FastH3 PeRFlow progress report

## Branch

- Branch: `adam/h3-teacher-perflow-distill`
- Parent branch: `adam/h3-rvm-posttraining`
- Initial parent SHA: `74907dd347805e12fb12b8f093afc41d6184d312`
- Design document: [`H3_PERFLOW_IMPLEMENTATION_PLAN.md`](H3_PERFLOW_IMPLEMENTATION_PLAN.md)

This file is updated after every completed task subset. A task is not marked complete until its implementation is committed to the remote branch and the new remote head is verified.

## Current status

| Task | Status | Remote commit | Notes |
|---|---|---|---|
| T0 design and provenance contract | Complete | `cd11e0b56e73a7166fcd055c4fb147ab7a5663ec`, `e4ce6a73cfac852ca2221aba06a1e6b0d99434d2`, `15c2bb9e5326cb2f7c2f4e0310a3d0d1197c4d3e` | Plan, exact source pin, corrected multimodal math, references, and task decomposition committed |
| T1 reward-filtered cache view | Code complete; full suite pending | `171ded3722c1f600290123ea2751ca49a2147b41`, `9f5c5e93fd0bebdecc1c2e094a9076e2e1c22264` | Deterministic top-q view and focused tests committed |
| T2 PeRFlow interpolation/loss primitives | Complete; full suite pending | `6455833d3cf2f00bfc0547c8d33e23480c9ccb30`, `f35f3fe230bb906b09d24d10f70fc76390b43aeb` | 12/12 isolated focused tests passed |
| T3 `H3PeRFlowMethod` | In progress | — | Model-integrated continuous segment trainer next |
| T4 configs and execution scripts | Not started | — | — |
| T5 validation and evaluation | Not started | — | — |

## Execution log

### 2026-09-02 — Branch creation

Created `adam/h3-teacher-perflow-distill` from the latest `adam/h3-rvm-posttraining` head at `74907dd347805e12fb12b8f093afc41d6184d312`.

The base already contains:

- paper-faithful on-policy FastH3 RVM;
- the calibrated Physion/MJ reward profile;
- full-H3 dense trajectory sampling;
- immutable scored H3 trajectory caches;
- FastH3-aligned five-anchor extraction;
- a FastH3 LoRA REST/AMD trainer;
- cache validation, hashes, configs, and CPU tests.

This changed the implementation strategy substantially. Instead of introducing generic runtime factories or a second cache format, PeRFlow reuses the existing H3 REST cache and adds a deterministic selected view plus a sibling training method.

### 2026-09-02 — Method audit

Audited the following source paths before modifying code:

- `examples/train/rvm_h3/build_h3_rest_cache.py`
- `fastvideo/dataset/h3_rest_cache.py`
- `fastvideo/train/methods/knowledge_distillation/h3_rest.py`
- `fastvideo/train/methods/knowledge_distillation/h3_rest_sampler.py`
- `fastvideo/train/methods/knowledge_distillation/h3_rest_utils.py`
- `fastvideo/train/models/minimax_h3/minimax_h3_rest.py`
- `fastvideo/train/models/minimax_h3/minimax_h3_rvm.py`
- `fastvideo/train/methods/rl/rvm.py`
- `fastvideo/train/methods/rl/rvm_faithful.py`
- `fastvideo/train/methods/rl/rvm_reward_profile.py`

Key findings:

1. The H3 cache already stores all `K=8` candidates per prompt and every field needed for deterministic top-q filtering.
2. Each trajectory already contains the exact FastH3 deployment boundary anchors `[1000, 750, 500, 250, 0]`.
3. Cache validation already enforces complete prompt groups, finite reward data, path confinement, byte counts, hashes, and provenance fingerprints.
4. H3 REST currently applies a signed reward coefficient directly to segment MSE. PeRFlow must not inherit that behavior: reward is used only for candidate selection, then all retained candidates receive equal positive supervised weight.
5. The existing FastH3 REST model already provides the correct packed video/audio layout, independent scheduler shifts, VSA metadata, FP32 LoRA masters, dataloader integration, checkpointing, and resume plumbing.
6. The existing RVM stage must remain on-policy. H3 samples will never be inserted into RVM’s rollout buffer.

### 2026-09-02 — Design and source contract

Committed the durable implementation plan at:

- `cd11e0b56e73a7166fcd055c4fb147ab7a5663ec` — `docs: plan reward-filtered H3 PeRFlow distillation`
- `e4ce6a73cfac852ca2221aba06a1e6b0d99434d2` — `docs: start H3 PeRFlow execution log`
- `15c2bb9e5326cb2f7c2f4e0310a3d0d1197c4d3e` — `docs: correct PeRFlow math for H3 modality shifts`

The official PeRFlow source was inspected at:

```text
magic-research/piecewise-rectified-flow
commit 87bac762f049d069497e83a76f528f007e9adabd
scripts/perflow_accelerate_sd.py
```

The reference implementation linearly interpolates each teacher window and uses the window secant as the velocity-matching target with MSE. Two corrections were made to the initial H3 design before the model method was written:

1. MSE is the paper-faithful default. Huber remains an opt-in robustness ablation.
2. H3 video and audio use different shifted noise coordinates. They share one sampled base timestep, but each modality must be interpolated and differentiated in its own sigma coordinate because the FastH3 output is `noise - clean = dx/dsigma`.

This correction prevents an attractive but wrong implementation that divides both modality state deltas by the same raw base-time interval.

### 2026-09-02 — T1 deterministic reward-filtered cache view

Implemented `fastvideo/dataset/h3_perflow_cache.py` as a non-destructive view over the existing immutable H3 REST cache.

Implementation details:

- Groups manifest entries by `prompt_id` and verifies every group has the same complete candidate set `0..K-1`.
- Supports ranking by `mixed_advantage`, `reward_scores.<name>`, or `reward_advantages.<name>`.
- Rejects missing, boolean, non-numeric, NaN, and infinite ranking values.
- Sorts by descending score, then ascending `candidate_index`, then `trajectory_id` for deterministic ties.
- Retains exactly `q` trajectories per prompt.
- Assigns every retained trajectory equal positive weight `1/q`; no reward sign or magnitude enters the later supervised gradient.
- Fingerprints the selection schema, ranking key, `K`, `q`, segment count, prompt IDs, selected trajectory IDs, ranks, and scores.
- Rebuilds the distributed sampler over only the selected examples while preserving the original cache files.
- Exposes selection rank, score, equal weight, and fingerprint in each training batch.
- Leaves the existing `H3RESTCacheDataset` and all REST configs unchanged.

Remote commits:

- `171ded3722c1f600290123ea2751ca49a2147b41` — `feat: add deterministic top-q H3 teacher cache view`
- `9f5c5e93fd0bebdecc1c2e094a9076e2e1c22264` — `test: cover deterministic H3 PeRFlow cache selection`

Focused tests cover deterministic results under manifest reordering, stable tie-breaking, equal selected weights, raw reward-component ranking, incomplete groups, non-finite scores, selected dataset length, selected trajectory identities, and batch-level selection metadata.

### 2026-09-02 — T2 multimodal PeRFlow primitives

Implemented `fastvideo/train/methods/knowledge_distillation/h3_perflow_utils.py`.

Implementation details:

- Samples one continuous query fraction inside each cached base-timestep segment.
- Maps the same base query through the existing video and audio noise schedules.
- Interpolates each modality using its own sigma-domain fraction.
- Computes the signed piecewise velocity target `(next_state - current_state) / (sigma_next - sigma_current)`.
- Detaches both cached states before constructing query states and targets.
- Fails closed on mismatched shapes, NaN/Inf states, NaN/Inf scalars, zero sigma intervals, out-of-window query sigmas, and invalid sample weights.
- Provides normalized per-sample weighting, so equal `1/q` top-q weights express the intended average without silently shrinking the effective learning rate.
- Uses MSE by default, matching official PeRFlow velocity matching.
- Provides Huber as an explicit ablation.
- Reduces video and audio independently before applying `audio_loss_weight`, preventing the packed video stream from numerically drowning audio.
- Provides an optional detached LoRA-off function-space MSE anchor, disabled when its coefficient is zero.

Remote commits:

- `6455833d3cf2f00bfc0547c8d33e23480c9ccb30` — `feat: add multimodal PeRFlow interpolation and losses`
- `f35f3fe230bb906b09d24d10f70fc76390b43aeb` — `test: validate multimodal PeRFlow interpolation math`

Validation completed in an isolated package containing the exact committed utility and test source:

```text
12 passed in 1.65s
```

The tests cover signed high-sigma-to-low-sigma velocity, endpoint-orientation invariance, exact interpolation, zero loss for an exact field, detached teacher states, query-range and degenerate-interval failures, seeded base-time sampling, normalized weights, Huber gradients, independent video/audio reductions, anchoring, and invalid weights.

The environment could not clone GitHub directly because outbound DNS is disabled, so this was an exact-source isolated test rather than the complete repository suite. Full-repository tests remain a required later gate.

## Validation boundary

No real H3/FastH3 GPU run has been executed on this new branch yet. T0–T2 are committed remotely and their pure contracts have focused tests, but end-to-end model integration, full repository CI, checkpoint/resume, distributed topology, and quality evaluation remain open. No empirical quality claim is being made.
