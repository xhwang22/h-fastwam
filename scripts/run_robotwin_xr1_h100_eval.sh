#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export MODEL_KIND=xr1
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/robotwin_hfastwam/robotwin_xr1_vision_causal_tubelet_32gpu_b48_cudnn_overlap_efa}"
export XR1_CHECKPOINT="${XR1_CHECKPOINT:-${REPO_ROOT}/checkpoints/XiaomiRobotics/Xiaomi-Robotics-1-5B}"
exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
