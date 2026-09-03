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
| T0 design and provenance contract | In progress | `cd11e0b56e73a7166fcd055c4fb147ab7a5663ec` | Plan committed; reference and source audit in progress |
| T1 reward-filtered cache view | Not started | — | — |
| T2 PeRFlow interpolation/loss primitives | Not started | — | — |
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

The plan records the mathematical objective, time convention, top-q contract, reuse map, alternatives rejected, non-goals, task subsets, evaluation controls, and primary references.

## Validation boundary

No real H3/FastH3 GPU run has been executed on this new branch yet. Current completed work is branch creation, repository audit, and design/provenance documentation. GPU claims will remain explicitly gated in this report.
