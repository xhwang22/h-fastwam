#!/usr/bin/env bash
# InternData V-JEPA predictor experiment using fixed offline global statistics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INTERN_A1_ROOT="${INTERN_A1_ROOT:-/fsx/pretrain_data/InternData-A1}"
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v3}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${INTERN_A1_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_global_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[vjepa21-global-norm] ERROR: stats file not found: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Run scripts/precompute_interndata_vjepa21_global_stats_single8.sh first." >&2
  exit 1
fi

export STANDARDISE_OUTPUT=true
export RUN_NAME="${RUN_NAME:-interndata_a1_vjepa21_predictor_global_norm_single8_b48_acc2}"

exec bash \
  "${SCRIPT_DIR}/run_interndata_a1_vjepa21_predictor_pretrain_single8_b48_acc2_cudnn.sh" \
  "$@"
