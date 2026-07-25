# Reward-Tilted Flow Distillation for Wan

`RewardTiltedFlowDistillationMethod` is a single-stage experiment for jointly
compressing a multi-step Wan teacher and steering the resulting few-step model
toward higher reward.

## Bottleneck

DMD followed by RL solves two different projections in sequence: first project
the teacher into a restricted few-step family, then move that already-restricted
model toward reward. RTFD instead defines the reward-aligned distribution before
the few-step projection.

For prompt `c`, frozen-teacher endpoint `x0`, and terminal reward `R`,

```text
q_beta(x0 | c) ∝ q_teacher(x0 | c) exp(R(x0, c) / tau).
```

RTFD draws several teacher endpoints per prompt, chooses `tau` to maintain a
target effective sample size, and mixes the tilted weights with a uniform
teacher floor. It then samples fresh independent Gaussian noise `eps` and trains
on the exact four-step deployment grid:

```text
x_sigma = (1 - sigma) x0 + sigma eps
u_target = eps - x0
```

The single loss is reward-weighted conditional flow matching:

```text
L = sum_i sum_k w_i |delta_sigma_k|
        ||v_student(x_sigma_k, sigma_k, c_i) - (eps_i - x0_i)||^2.
```

Independent `eps` matters: reward changes only the target endpoint law while the
source remains the standard Gaussian used at inference.

## What is deliberately absent

- no pretrained distilled checkpoint;
- no cold-start stage;
- no old policy or PPO/GRPO ratio;
- no separate frozen reference copy;
- no fake-score critic;
- no additive reward loss.

The only method-specific stabilizers are:

- `reward_ess_ratio`: controls how concentrated reward weighting may become;
- `uniform_mix`: preserves teacher coverage. Set it to `1.0` for the exact
  matched-compute no-reward baseline.

## Local launch

Prepare the prompt parquet and VideoAlign checkpoint with the existing reusable
DMDR asset script:

```bash
python examples/train/prepare_dmdr_assets.py \
  --config examples/train/configs/reward_tilted_flow/wan/rtfd_videoalign.yaml \
  --data-root data/rtfd \
  --cache-root .cache/rtfd \
  --output-dir outputs/wan2.1_rtfd_videoalign \
  --run-config-dir outputs/rtfd_run_configs \
  --num-gpus 4 \
  --num-frames 49 \
  --max-train-steps 100 \
  --sample-num-steps 16 \
  --validation-num-steps 4 \
  --check-rewards
```

The helper currently writes `dmdr_wan_run.yaml`; it is method-agnostic despite
the historical filename. Launch it directly:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
  outputs/rtfd_run_configs/dmdr_wan_run.yaml
```

## Modal launch

Load the Hugging Face and W&B credentials directly from the checkout's `.env`
file (or set `RTFD_DOTENV_PATH` to start the search from another path):

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

The launcher passes these values to the Modal function as an ephemeral secret;
no named Modal secret is required.

Run a cheap pipeline smoke test:

```bash
modal run examples/train/modal_rtfd.py \
  --max-steps 5 \
  --num-frames 17 \
  --height 256 \
  --width 448 \
  --max-prompts 32
```

Run the first meaningful reward-tilted experiment:

```bash
modal run examples/train/modal_rtfd.py \
  --max-steps 100 \
  --num-frames 49 \
  --height 448 \
  --width 832 \
  --trajectories-per-prompt 4 \
  --ess-ratio 0.60 \
  --uniform-mix 0.25
```

Run the matched-compute plain four-step distillation baseline:

```bash
modal run examples/train/modal_rtfd.py \
  --max-steps 100 \
  --num-frames 49 \
  --height 448 \
  --width 832 \
  --trajectories-per-prompt 4 \
  --uniform-mix 1.0 \
  --run-name wan2.1_rtfd_uniform_baseline
```

## First decision criterion

Compare reward tilt versus `uniform_mix=1.0` at equal:

- teacher calls;
- prompts and random seeds;
- optimizer updates;
- four-step validation NFE.

The method clears the first gate only if held-out four-step reward improves
without a clear collapse in prompt-level diversity or visual quality. Log
`rtfd/reward_ess_ratio_final`, `rtfd/max_trajectory_weight`, each transition MSE,
and teacher/student validation rewards to diagnose the mechanism rather than
only the final score.
