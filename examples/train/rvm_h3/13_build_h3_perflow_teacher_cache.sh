#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

MODE="${1:-compact}"
case "${MODE}" in
    compact)
        CONFIG="examples/train/configs/knowledge_distillation/minimax_h3/h3_rest_cache_4gpu_compact.yaml"
        CACHE_DIR="${H3_REST_COMPACT_CACHE}"
        EXPECT_K=2
        EXPECT_PROMPTS="${PERFLOW_EXPECT_PROMPTS:-2}"
        ;;
    full)
        CONFIG="examples/train/configs/knowledge_distillation/minimax_h3/h3_rest_cache_4gpu_full.yaml"
        CACHE_DIR="${H3_REST_FULL_CACHE}"
        EXPECT_K=8
        EXPECT_PROMPTS="${PERFLOW_EXPECT_PROMPTS:-100}"
        ;;
    *)
        echo "Usage: $0 {compact|full}" >&2
        exit 2
        ;;
esac

require_path "${H3_TEACHER_MODEL_DIR}"
require_path "${RVM_TRAIN_DATA}"
require_path "${VIDEOALIGN_RUNTIME_PATH}"
require_path "${VIDEOALIGN_CHECKPOINT_PATH}"
require_path "${MJ_VIDEO_RUNTIME_PATH}"
require_path "${MJ_VIDEO_MODEL_PATH}"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}"
python examples/train/rvm_h3/verify_clean_source.py

if [[ -e "${CACHE_DIR}" && "${PERFLOW_CACHE_OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to replace existing cache ${CACHE_DIR}." >&2
    echo "Set PERFLOW_CACHE_OVERWRITE=1 only after preserving any reported artifact." >&2
    exit 2
fi

OVERWRITE_ARGS=()
if [[ "${PERFLOW_CACHE_OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
python -m torch.distributed.run \
    --nproc_per_node 4 \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29531}" \
    examples/train/rvm_h3/build_h3_rest_cache.py \
    --config "${CONFIG}" \
    --output-dir "${CACHE_DIR}" \
    "${OVERWRITE_ARGS[@]}" \
    --models.student.init_from "${H3_TEACHER_MODEL_DIR}" \
    --training.model_path "${H3_TEACHER_MODEL_DIR}" \
    --training.data.data_path "${RVM_TRAIN_DATA}"

PERFLOW_EXPECT_K="${EXPECT_K}" \
PERFLOW_EXPECT_PROMPTS="${EXPECT_PROMPTS}" \
PERFLOW_SELECTED_PER_PROMPT="${PERFLOW_SELECTED_PER_PROMPT:-2}" \
PERFLOW_REQUIRE_REWARDS="videoalign_ta,mj_video_coherence_consistency,mj_video_fineness,dynamic_tracking" \
PERFLOW_METADATA_ONLY=0 \
    bash examples/train/rvm_h3/14_verify_h3_perflow_cache.sh "${CACHE_DIR}"

cat <<EOF
Completed immutable H3 teacher cache for PeRFlow:
  mode: ${MODE}
  cache: ${CACHE_DIR}
  K: ${EXPECT_K}
  prompts: ${EXPECT_PROMPTS}
The cache contains all K candidates. Top-q filtering remains a deterministic
read-only dataset view and never rewrites this artifact.
EOF
