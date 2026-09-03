#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

CONFIG="${1:-examples/train/configs/knowledge_distillation/minimax_h3/h3_perflow_8gpu_full.yaml}"
RUN_DIR="${PERFLOW_EXPORT_RUN_DIR:-outputs/h3_perflow/${NUM_GPUS:-8}gpu_full}"
CHECKPOINT="${2:-}"
if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" ]]; then
    echo "No PeRFlow checkpoint found under ${RUN_DIR}." >&2
    exit 2
fi
OUTPUT="${3:-${CHECKPOINT%/}/fasth3_perflow_lora.safetensors}"

require_path "${CONFIG}"
require_path "${CHECKPOINT}"
bash examples/train/rvm_h3/09_export_lora.sh \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${OUTPUT}"
require_path "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import math
import sys
from pathlib import Path

from safetensors import safe_open

path = Path(sys.argv[1]).resolve()
with safe_open(str(path), framework="pt", device="cpu") as handle:
    keys = sorted(handle.keys())
    if not keys or len(keys) % 3:
        raise SystemExit(f"unexpected LoRA tensor count: {len(keys)}")
    suffixes = {key.rsplit(".", 1)[-1] for key in keys}
    if suffixes != {"lora_A", "lora_B", "lora_alpha"}:
        raise SystemExit(f"unexpected LoRA key suffixes: {sorted(suffixes)}")
    alphas = []
    for key in keys:
        tensor = handle.get_tensor(key)
        if not tensor.isfinite().all():
            raise SystemExit(f"non-finite tensor in exported adapter: {key}")
        if key.endswith(".lora_alpha"):
            if tensor.numel() != 1:
                raise SystemExit(f"non-scalar alpha tensor: {key}")
            alphas.append(int(tensor.item()))
    if len(set(alphas)) != 1:
        raise SystemExit(f"inconsistent LoRA alpha values: {sorted(set(alphas))}")
    print(
        f"Verified {path}: layers={len(keys) // 3}, tensors={len(keys)}, "
        f"alpha={alphas[0]}, bytes={path.stat().st_size}"
    )
PY

echo "Exported trainable PeRFlow adapter: ${OUTPUT}"
