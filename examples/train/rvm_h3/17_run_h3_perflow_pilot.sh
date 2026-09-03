#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

require_path "${FASTH3_MODEL_DIR}"
require_path "${H3_REST_FULL_CACHE}/COMPLETE"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
if [[ "${NUM_GPUS}" != "8" && "${NUM_GPUS}" != "16" ]]; then
    echo "PeRFlow pilot supports NUM_GPUS=8 or 16." >&2
    exit 2
fi
if [[ "${RVM_SP_SIZE}" != "4" ]] || (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "Use RVM_SP_SIZE=4 for the PeRFlow pilot." >&2
    exit 2
fi

PERFLOW_EXPECT_K="${PERFLOW_EXPECT_K:-8}" \
PERFLOW_EXPECT_PROMPTS="${PERFLOW_EXPECT_PROMPTS:-100}" \
PERFLOW_SELECTED_PER_PROMPT="${PERFLOW_SELECTED_PER_PROMPT:-2}" \
PERFLOW_RANKING_KEY="${PERFLOW_RANKING_KEY:-mixed_advantage}" \
PERFLOW_REQUIRE_REWARDS="${PERFLOW_REQUIRE_REWARDS:-videoalign_ta,mj_video_coherence_consistency,mj_video_fineness,dynamic_tracking}" \
PERFLOW_METADATA_ONLY="${PERFLOW_VERIFY_FULL_HASHES:-0}" \
    bash examples/train/rvm_h3/14_verify_h3_perflow_cache.sh \
        "${H3_REST_FULL_CACHE}"

CONFIG="examples/train/configs/knowledge_distillation/minimax_h3/h3_perflow_8gpu_pilot.yaml"
OUTPUT="${PERFLOW_PILOT_OUTPUT:-outputs/h3_perflow/${NUM_GPUS}gpu_pilot}"
RUN_NAME="${PERFLOW_PILOT_RUN_NAME:-h3-perflow-${NUM_GPUS}gpu-pilot}"

run_rvm_training \
    "${CONFIG}" \
    --models.student.init_from "${FASTH3_MODEL_DIR}" \
    --models.student.selected_per_prompt "${PERFLOW_SELECTED_PER_PROMPT:-2}" \
    --models.student.ranking_key "${PERFLOW_RANKING_KEY:-mixed_advantage}" \
    --training.model_path "${FASTH3_MODEL_DIR}" \
    --training.data.data_path "${H3_REST_FULL_CACHE}" \
    --training.optimizer.learning_rate "${PERFLOW_LEARNING_RATE:-1e-5}" \
    --training.loop.max_train_steps "${PERFLOW_PILOT_STEPS:-100}" \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.tracker.run_name "${RUN_NAME}"

cat <<EOF
H3 PeRFlow pilot completed:
  GPUs: ${NUM_GPUS}
  cache: ${H3_REST_FULL_CACHE}
  top-q: ${PERFLOW_SELECTED_PER_PROMPT:-2}
  output: ${OUTPUT}
Do not select a checkpoint from training loss alone. Export candidate adapters
and compare fixed-seed held-out reward, temporal integrity, motion, and diversity.
EOF
