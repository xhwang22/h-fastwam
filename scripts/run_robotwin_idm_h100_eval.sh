#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export MODEL_KIND=idm
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/robotwin_hfastwam/robotwin_vjepa21_predictor_idm_full_condition_causal_tubelet_32gpu_b48_cudnn_overlap_efa}"
exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
