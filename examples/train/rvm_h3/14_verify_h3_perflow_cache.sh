#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

CACHE_DIR="${1:-${H3_REST_FULL_CACHE}}"
require_path "${CACHE_DIR}"
require_path "${CACHE_DIR}/COMPLETE"

ARGS=(
    "${CACHE_DIR}"
    --selected-per-prompt "${PERFLOW_SELECTED_PER_PROMPT:-2}"
    --ranking-key "${PERFLOW_RANKING_KEY:-mixed_advantage}"
)
if [[ "${PERFLOW_METADATA_ONLY:-0}" == "1" ]]; then
    ARGS+=(--metadata-only)
fi
if [[ -n "${PERFLOW_EXPECT_K:-}" ]]; then
    ARGS+=(--expect-samples-per-prompt "${PERFLOW_EXPECT_K}")
fi
if [[ -n "${PERFLOW_EXPECT_PROMPTS:-}" ]]; then
    ARGS+=(--expect-prompts "${PERFLOW_EXPECT_PROMPTS}")
fi
if [[ -n "${PERFLOW_EXPECT_CACHE_FINGERPRINT:-}" ]]; then
    ARGS+=(--expect-cache-fingerprint "${PERFLOW_EXPECT_CACHE_FINGERPRINT}")
fi
if [[ -n "${PERFLOW_REQUIRE_REWARDS:-}" ]]; then
    IFS=',' read -r -a REWARDS <<<"${PERFLOW_REQUIRE_REWARDS}"
    for reward in "${REWARDS[@]}"; do
        [[ -n "${reward}" ]] && ARGS+=(--require-reward-name "${reward}")
    done
fi

python examples/train/rvm_h3/verify_h3_perflow_cache.py "${ARGS[@]}"
