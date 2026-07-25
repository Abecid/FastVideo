# Experiment Journal

Living log of all experiments. Each entry captures what was tried, the result,
and any insights. Newest entries go at the top.

## [2026-07-24] Experiment: wan2.1_rtfd_tilted
- **Hypothesis**: ESS-controlled VideoAlign reward tilting with a 25% uniform teacher mixture improves the four-step student's held-out reward over 100 updates without collapse or unstable transition losses.
- **Config**: model=Wan2.1-T2V-1.3B, lr=2e-6, sp_size=1, gpus=4xH100, script=examples/train/modal_rtfd.py, frames=49, resolution=448x832, prompts=64, teacher_steps=16, student_steps=4, trajectories_per_prompt=4, ess_ratio=0.60, uniform_mix=0.25, max_steps=100
- **W&B run**: [attentionx2023/rtfd_wan/ul4oo1wt](https://wandb.ai/attentionx2023/rtfd_wan/runs/ul4oo1wt) (retroactively synced from `/runs/outputs/wan2.1_rtfd_tilted/tracker/wandb/offline-run-20260725_021800-ul4oo1wt`)
- **Duration**: ~3h16m end-to-end; 3h11m33s for 100 training steps
- **Key metrics**: final_loss=0.06729, mean_loss=0.09420, loss_range=[0.06417, 0.13796], avg_step_time=111.67s, teacher_reward_avg=-0.05971, reward_ess_raw=0.60000, reward_ess_final=0.72727, max_trajectory_weight=0.47616, validation_reward=-1.64390 (step 0) → -1.26609 (best, step 5) → -1.58785 (step 100)
- **Checkpoint**: `/runs/outputs/wan2.1_rtfd_tilted/checkpoint-100`
- **Insight**: The 100-step plumbing and optimization run completed without NaN, OOM, loss spikes, or distributed failures. Fixed-prompt held-out reward improved slightly from -1.64390 to -1.58785, but oscillated substantially and peaked early, so this run does not establish a convincing reward-quality gain. A matched uniform-teacher baseline is still required to evaluate the reward tilt itself.
- **Status**: completed
- **Related lessons**: `.agents/lessons/2026-07-24_modal-rtfd-launcher-compatibility.md`

## [2026-07-24] Experiment: wan2.1_rtfd_smoke
- **Hypothesis**: The Modal launcher completes cloning, dependency installation, preprocessing, VideoAlign loading, distributed startup, one RTFD forward/backward path, DMDR-style fixed-prompt validation logging, and checkpointing in five outer steps.
- **Config**: model=Wan2.1-T2V-1.3B, lr=2e-6, sp_size=1, gpus=4xH100, script=examples/train/modal_rtfd.py, frames=17, resolution=256x448, prompts=32, max_steps=5
- **W&B run**: [attentionx2023/rtfd_wan/q3uxn51y](https://wandb.ai/attentionx2023/rtfd_wan/runs/q3uxn51y) (retroactively synced from `/runs/outputs/wan2.1_rtfd_smoke/tracker/wandb/offline-run-20260725_021015-q3uxn51y`)
- **Duration**: ~6m end-to-end; 1m29s for five training steps
- **Key metrics**: final_loss=0.16510, teacher_reward_avg=-0.99602, reward_ess_raw=0.60000, reward_ess_final=0.72727, max_trajectory_weight=0.48304, avg_train_step_time≈17.9s
- **Checkpoint**: `/runs/outputs/wan2.1_rtfd_smoke`
- **Insight**: The full Modal plumbing passed after launcher compatibility fixes. Four fixed-prompt validation videos were logged at steps 0 and 5; no NaN, OOM, loss-spike, or NCCL failure occurred.
- **Status**: completed
- **Related lessons**: `.agents/lessons/2026-07-24_modal-rtfd-launcher-compatibility.md`

<!-- TEMPLATE — copy and fill for each new experiment:

## [YYYY-MM-DD] Experiment: <name>
- **Hypothesis**: <what you expected to learn>
- **Config**: model=..., lr=..., sp_size=..., gpus=..., script=...
- **W&B run**: <run_id or URL>
- **Duration**: <total wall time>
- **Key metrics**: loss=..., step_time=..., grad_norm=...
- **Checkpoint**: <path>
- **Insight**: <what was learned>
- **Status**: running | completed | failed | abandoned
- **Related lessons**: `.agents/lessons/<filename>.md`

-->
