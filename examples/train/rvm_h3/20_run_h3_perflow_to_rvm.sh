#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
if [[ "${NUM_GPUS}" != "8" && "${NUM_GPUS}" != "16" ]]; then
    echo "PeRFlow -> RVM continuation supports NUM_GPUS=8 or 16." >&2
    exit 2
fi
if [[ "${RVM_SP_SIZE}" != "4" ]] || (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "Use RVM_SP_SIZE=4 for PeRFlow -> RVM continuation." >&2
    exit 2
fi

PROFILE="${RVM_STAGE_PROFILE:-physion_mj}"
case "${PROFILE}" in
    physion_mj)
        CONFIG="examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_physion_mj.yaml"
        require_path "${MJ_VIDEO_CALIBRATION_PATH}"
        require_path "${MJ_VIDEO_RUNTIME_PATH}"
        require_path "${MJ_VIDEO_MODEL_PATH}"
        require_path "${MJ_VIDEO_BASE_MODEL_PATH}"
        ;;
    published)
        CONFIG="examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml"
        require_path "${VIDEOALIGN_RUNTIME_PATH}"
        require_path "${VIDEOALIGN_CHECKPOINT_PATH}"
        ;;
    *)
        echo "RVM_STAGE_PROFILE must be 'physion_mj' or 'published'." >&2
        exit 2
        ;;
esac

ADAPTER="${1:-${PERFLOW_LORA_PATH:-}}"
if [[ -z "${ADAPTER}" ]]; then
    SEARCH_ROOT="${PERFLOW_EXPORT_RUN_DIR:-outputs/h3_perflow/${NUM_GPUS}gpu_full}"
    ADAPTER="$(find "${SEARCH_ROOT}" -type f -name 'fasth3_perflow_lora.safetensors' 2>/dev/null | sort -V | tail -n 1)"
fi
if [[ -z "${ADAPTER}" ]]; then
    echo "No exported PeRFlow adapter found. Pass it as argument 1 or set PERFLOW_LORA_PATH." >&2
    exit 2
fi

require_path "${FASTH3_MODEL_DIR}"
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"
require_path "${ADAPTER}"
require_path "${CONFIG}"

# Fail before loading 35B weights if the adapter cannot preserve the configured
# RVM LoRA function exactly. The model performs the same checks again on every
# distributed rank during construction.
python - "${CONFIG}" "${ADAPTER}" <<'PY'
import sys
from pathlib import Path

import yaml
from safetensors import safe_open

config_path = Path(sys.argv[1])
adapter_path = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
method_target = str(config["method"]["_target_"])
if "RVM" not in method_target or "PeRFlow" in method_target:
    raise SystemExit(
        f"stage two must remain an on-policy RVM method, got {method_target}"
    )
lora = config["models"]["student"]["lora"]
expected_rank = int(lora["rank"])
expected_alpha = int(lora["alpha"])
with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    alpha_keys = [key for key in keys if key.endswith(".lora_alpha")]
    a_keys = [key for key in keys if key.endswith(".lora_A")]
    if not alpha_keys or len(alpha_keys) != len(a_keys):
        raise SystemExit("adapter has an incomplete FastVideo LoRA key set")
    observed_alphas = {int(handle.get_tensor(key).item()) for key in alpha_keys}
    observed_ranks = {int(handle.get_tensor(key).shape[0]) for key in a_keys}
if observed_alphas != {expected_alpha}:
    raise SystemExit(
        f"LoRA alpha mismatch: adapter={observed_alphas}, config={expected_alpha}"
    )
if observed_ranks != {expected_rank}:
    raise SystemExit(
        f"LoRA rank mismatch: adapter={observed_ranks}, config={expected_rank}"
    )
print(
    f"Validated PeRFlow -> RVM handoff: method={method_target}, "
    f"rank={expected_rank}, alpha={expected_alpha}, adapter={adapter_path}"
)
PY

OUTPUT="${RVM_STAGE_OUTPUT:-outputs/h3_perflow/rvm_${PROFILE}_${NUM_GPUS}gpu}"
RUN_NAME="${RVM_STAGE_RUN_NAME:-h3-perflow-to-rvm-${PROFILE}-${NUM_GPUS}gpu}"

run_rvm_training \
    "${CONFIG}" \
    --models.student.init_from "${FASTH3_MODEL_DIR}" \
    --models.student.lora_init_from "${ADAPTER}" \
    --training.model_path "${FASTH3_MODEL_DIR}" \
    --training.data.data_path "${RVM_TRAIN_DATA}" \
    --method.validation.data_path "${RVM_EVAL_DATA}" \
    --training.optimizer.learning_rate "${RVM_STAGE_LEARNING_RATE:-1e-5}" \
    --training.loop.max_train_steps "${RVM_STAGE_STEPS:-20}" \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.tracker.run_name "${RUN_NAME}"

cat <<EOF
PeRFlow -> on-policy RVM continuation completed:
  profile: ${PROFILE}
  initialization adapter: ${ADAPTER}
  output: ${OUTPUT}
The H3 teacher cache was not inserted into RVM. Stage two generated fresh
FastH3 behavior rollouts and retained the existing paper-faithful RVM objective.
EOF
