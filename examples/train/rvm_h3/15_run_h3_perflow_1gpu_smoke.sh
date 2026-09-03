#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

require_path "${FASTH3_MODEL_DIR}"
require_path "${H3_REST_COMPACT_CACHE}/COMPLETE"
PERFLOW_EXPECT_K=2 \
PERFLOW_SELECTED_PER_PROMPT=2 \
PERFLOW_METADATA_ONLY=0 \
    bash examples/train/rvm_h3/14_verify_h3_perflow_cache.sh \
        "${H3_REST_COMPACT_CACHE}"

export NUM_GPUS=1
export RVM_SP_SIZE=1
CONFIG="examples/train/configs/knowledge_distillation/minimax_h3/h3_perflow_1gpu_smoke.yaml"
OUTPUT="${PERFLOW_SMOKE_OUTPUT:-outputs/h3_perflow/1gpu_smoke}"
RUN_NAME="${PERFLOW_SMOKE_RUN_NAME:-h3-perflow-1gpu-smoke}"

run_rvm_training \
    "${CONFIG}" \
    --models.student.init_from "${FASTH3_MODEL_DIR}" \
    --training.model_path "${FASTH3_MODEL_DIR}" \
    --training.data.data_path "${H3_REST_COMPACT_CACHE}" \
    --training.loop.max_train_steps "${PERFLOW_SMOKE_MAX_STEPS:-1}" \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.checkpoint.training_state_checkpointing_steps 1 \
    --training.checkpoint.checkpoints_total_limit 2 \
    --training.tracker.run_name "${RUN_NAME}"

cat <<EOF
H3 PeRFlow one-GPU correctness smoke completed.
  cache: ${H3_REST_COMPACT_CACHE}
  output: ${OUTPUT}
This is a runtime/gradient/checkpoint gate, not a quality result.
EOF
