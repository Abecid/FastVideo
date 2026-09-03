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
| T0 design and provenance contract | Complete | `cd11e0b56e73a7166fcd055c4fb147ab7a5663ec`, `e4ce6a73cfac852ca2221aba06a1e6b0d99434d2` | Plan, references, source audit, task decomposition, and progress log committed |
| T1 reward-filtered cache view | Code complete; full suite pending | `171ded3722c1f600290123ea2751ca49a2147b41`, `9f5c5e93fd0bebdecc1c2e094a9076e2e1c22264` | Deterministic top-q view and focused tests committed; remote head verified |
| T2 PeRFlow interpolation/loss primitives | In progress | — | Continuous segment math and packed Huber loss next |
| T3 `H3PeRFlowMethod` | Not started | — | — |
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

This changed the implementation strategy substantially. Instead of introducing generic runtime factories or a second cache format, PeRFlow will reuse the existing H3 REST cache and add a deterministic selected view plus a sibling training method.

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

### 2026-09-02 — Design document committed

Committed the durable implementation plan at:

- `cd11e0b56e73a7166fcd055c4fb147ab7a5663ec` — `docs: plan reward-filtered H3 PeRFlow distillation`
- `e4ce6a73cfac852ca2221aba06a1e6b0d99434d2` — `docs: start H3 PeRFlow execution log`

The plan records the mathematical objective, time convention, top-q contract, reuse map, alternatives rejected, non-goals, task subsets, evaluation controls, and primary references.

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

Static Python compilation of both new files succeeded before upload. The complete repository test suite has not yet run on this branch; that remains an explicit validation gate rather than an assumed result.

Remote verification:

- Verified branch head after T1: `9f5c5e93fd0bebdecc1c2e094a9076e2e1c22264`.

## Validation boundary

No real H3/FastH3 GPU run has been executed on this new branch yet. Current completed work is branch creation, repository audit, design/provenance documentation, and the committed top-q cache view with focused tests. GPU and quality claims remain explicitly gated in this report.
