#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

require_path "${FASTH3_MODEL_DIR}"
require_path "${H3_REST_COMPACT_CACHE}/COMPLETE"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
if [[ "${NUM_GPUS}" != "8" && "${NUM_GPUS}" != "16" ]]; then
    echo "PeRFlow topology smoke supports NUM_GPUS=8 or 16." >&2
    exit 2
fi
if [[ "${RVM_SP_SIZE}" != "4" ]] || (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "Use RVM_SP_SIZE=4 for the 8/16-GPU PeRFlow topology gate." >&2
    exit 2
fi
DP_REPLICAS=$((NUM_GPUS / RVM_SP_SIZE))

PERFLOW_EXPECT_K=2 \
PERFLOW_SELECTED_PER_PROMPT=2 \
PERFLOW_METADATA_ONLY=1 \
    bash examples/train/rvm_h3/14_verify_h3_perflow_cache.sh \
        "${H3_REST_COMPACT_CACHE}"

CONFIG="examples/train/configs/knowledge_distillation/minimax_h3/h3_perflow_1gpu_smoke.yaml"
OUTPUT="${PERFLOW_TOPOLOGY_OUTPUT:-outputs/h3_perflow/${NUM_GPUS}gpu_topology_smoke}"
RUN_NAME="${PERFLOW_TOPOLOGY_RUN_NAME:-h3-perflow-${NUM_GPUS}gpu-topology-smoke}"
COMMON_OVERRIDES=(
    --models.student.init_from "${FASTH3_MODEL_DIR}"
    --models.student.verify_cache_hashes false
    --training.model_path "${FASTH3_MODEL_DIR}"
    --training.data.data_path "${H3_REST_COMPACT_CACHE}"
    --training.checkpoint.output_dir "${OUTPUT}"
    --training.checkpoint.training_state_checkpointing_steps 1
    --training.checkpoint.checkpoints_total_limit 3
)

# Phase 1: distributed forward/backward/update plus a complete checkpoint.
run_rvm_training \
    "${CONFIG}" \
    "${COMMON_OVERRIDES[@]}" \
    --training.loop.max_train_steps 1 \
    --training.tracker.run_name "${RUN_NAME}-phase1"

# Phase 2: restore model, optimizer, scheduler, dataloader, and RNG state.
run_rvm_training \
    "${CONFIG}" \
    "${COMMON_OVERRIDES[@]}" \
    --training.loop.max_train_steps 2 \
    --training.checkpoint.resume_from_checkpoint latest \
    --training.tracker.run_name "${RUN_NAME}-resume"

ADAPTER="${OUTPUT}/fasth3_perflow_topology_smoke.safetensors"
bash examples/train/rvm_h3/09_export_lora.sh \
    "${CONFIG}" \
    "${OUTPUT}/checkpoint-2" \
    "${ADAPTER}"
require_path "${ADAPTER}"

cat <<EOF
H3 PeRFlow topology gate completed:
  GPUs: ${NUM_GPUS}
  topology: SP4 x DP${DP_REPLICAS}
  output: ${OUTPUT}
  exported adapter: ${ADAPTER}
Verify finite modality losses, nonzero LoRA gradients, checkpoint-2, exact
resume advance, and the export manifest before a quality-bearing run.
EOF
