#!/usr/bin/env bash
# Non-IDM H-FastWAM with the Xiaomi Robotics-1-tuned vision tower.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

export XR1_CHECKPOINT="${XR1_CHECKPOINT:-${REPO_ROOT}/checkpoints/XiaomiRobotics/Xiaomi-Robotics-1-5B/model_states.pt}"
if [[ ! -e "${XR1_CHECKPOINT}" ]]; then
  echo "[xr1-vision] ERROR: missing XR-1 checkpoint: ${XR1_CHECKPOINT}" >&2
  echo "[xr1-vision] Set XR1_CHECKPOINT to the XR-1 HF model directory or converted model_states.pt." >&2
  exit 1
fi

export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_xr1_vision}"
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_xr1_vision_causal_tubelet_ds}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-robotwin-small-xr1-vision-causal-tubelet}"
export VISUAL_ENCODER_DESCRIPTION="Xiaomi Robotics-1 vision tower, raw 1024-d features + Wan DiT"
export CAUSAL_TUBELET_ENCODING=true
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
