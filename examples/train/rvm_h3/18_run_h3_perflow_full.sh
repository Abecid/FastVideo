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
    echo "Full PeRFlow pass supports NUM_GPUS=8 or 16." >&2
    exit 2
fi
if [[ "${RVM_SP_SIZE}" != "4" ]] || (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "Use RVM_SP_SIZE=4 for the full PeRFlow pass." >&2
    exit 2
fi
DP_REPLICAS=$((NUM_GPUS / RVM_SP_SIZE))
TOP_Q="${PERFLOW_SELECTED_PER_PROMPT:-2}"

METADATA_ONLY=1
if [[ "${PERFLOW_VERIFY_FULL_HASHES:-0}" == "1" ]]; then
    METADATA_ONLY=0
fi
PERFLOW_EXPECT_K="${PERFLOW_EXPECT_K:-8}" \
PERFLOW_EXPECT_PROMPTS="${PERFLOW_EXPECT_PROMPTS:-100}" \
PERFLOW_SELECTED_PER_PROMPT="${TOP_Q}" \
PERFLOW_RANKING_KEY="${PERFLOW_RANKING_KEY:-mixed_advantage}" \
PERFLOW_REQUIRE_REWARDS="${PERFLOW_REQUIRE_REWARDS:-videoalign_ta,mj_video_coherence_consistency,mj_video_fineness,dynamic_tracking}" \
PERFLOW_METADATA_ONLY="${METADATA_ONLY}" \
    bash examples/train/rvm_h3/14_verify_h3_perflow_cache.sh \
        "${H3_REST_FULL_CACHE}"

EXPECTED_STEPS="$(python - "${H3_REST_FULL_CACHE}" "${TOP_Q}" "${DP_REPLICAS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
q = int(sys.argv[2])
dp = int(sys.argv[3])
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
examples = int(metadata["num_prompts"]) * q * int(metadata["num_segments"])
if examples % dp:
    raise SystemExit(
        f"selected examples {examples} are not divisible by DP replicas {dp}"
    )
print(examples // dp)
PY
)"
MAX_STEPS="${PERFLOW_FULL_STEPS:-${EXPECTED_STEPS}}"
if [[ "${PERFLOW_ALLOW_PARTIAL_PASS:-0}" != "1" && "${MAX_STEPS}" != "${EXPECTED_STEPS}" ]]; then
    echo "A deterministic one-pass run requires ${EXPECTED_STEPS} steps, got ${MAX_STEPS}." >&2
    echo "Set PERFLOW_ALLOW_PARTIAL_PASS=1 only for an intentional ablation." >&2
    exit 2
fi

CONFIG="examples/train/configs/knowledge_distillation/minimax_h3/h3_perflow_8gpu_full.yaml"
OUTPUT="${PERFLOW_FULL_OUTPUT:-outputs/h3_perflow/${NUM_GPUS}gpu_full}"
RUN_NAME="${PERFLOW_FULL_RUN_NAME:-h3-perflow-${NUM_GPUS}gpu-full}"

run_rvm_training \
    "${CONFIG}" \
    --models.student.init_from "${FASTH3_MODEL_DIR}" \
    --models.student.selected_per_prompt "${TOP_Q}" \
    --models.student.ranking_key "${PERFLOW_RANKING_KEY:-mixed_advantage}" \
    --training.model_path "${FASTH3_MODEL_DIR}" \
    --training.data.data_path "${H3_REST_FULL_CACHE}" \
    --training.optimizer.learning_rate "${PERFLOW_LEARNING_RATE:-1e-5}" \
    --training.loop.max_train_steps "${MAX_STEPS}" \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.checkpoint.training_state_checkpointing_steps "${PERFLOW_CHECKPOINT_EVERY:-20}" \
    --training.tracker.run_name "${RUN_NAME}"

cat <<EOF
H3 PeRFlow full pass completed:
  GPUs: ${NUM_GPUS}
  topology: SP${RVM_SP_SIZE} x DP${DP_REPLICAS}
  selected examples: $((EXPECTED_STEPS * DP_REPLICAS))
  optimizer steps: ${MAX_STEPS}
  output: ${OUTPUT}
Export and evaluate fixed checkpoints; do not use training loss as model selection.
EOF
