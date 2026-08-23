#!/usr/bin/env bash
# Continue the same trained model for one epoch with the online encoder on/off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"

MODE="${1:-}"
if [[ "${MODE}" != "on" && "${MODE}" != "off" ]]; then
  echo "Usage: FASTWAM_CHECKPOINT=/path/to/weights.pt NODE_IP_LIST=ip1,ip2,ip3,ip4 bash $0 {on|off}" >&2
  exit 2
fi
shift
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: this A/B launcher does not accept positional Hydra overrides; use environment variables so every node receives identical settings." >&2
  exit 2
fi

if [[ -z "${FASTWAM_CHECKPOINT:-}" ]]; then
  echo "ERROR: FASTWAM_CHECKPOINT is required for continuation training." >&2
  exit 2
fi
if [[ ! -f "${FASTWAM_CHECKPOINT}" ]]; then
  echo "ERROR: FASTWAM_CHECKPOINT not found: ${FASTWAM_CHECKPOINT}" >&2
  exit 2
fi
if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "ERROR: set NODE_IP_LIST to exactly four AWS nodes." >&2
  exit 2
fi
IFS=',' read -ra ENCODER_AB_NODES <<< "${NODE_IP_LIST}"
if [[ "${#ENCODER_AB_NODES[@]}" -ne 4 ]]; then
  echo "ERROR: expected 4 nodes, got ${#ENCODER_AB_NODES[@]}: ${NODE_IP_LIST}" >&2
  exit 2
fi

export PER_GPU_BATCH_SIZE=48
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=1536
export NUM_EPOCHS=1
export MAX_STEPS=null
export VIDEO_LATENT_CACHE_ENABLED=0
export NO_CKPT="${NO_CKPT:-1}"
export FASTWAM_ADAM_FUSED=1
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-0}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
export FIXED_TARGET_ENCODER=true

if [[ "${MODE}" == "on" ]]; then
  export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_continue_1epoch_encoder_on_4x8_b48_efa}"
  export VISUAL_ENCODER_FREEZE_BACKBONE=false
  export VISUAL_ENCODER_ACTIVATION_CHECKPOINTING=true
  export TRAINABLE_COMPONENTS='[dit,visual_encoder]'
  export VISUAL_ENCODER_LR_MULTIPLIER="${VISUAL_ENCODER_LR_MULTIPLIER:-0.1}"
else
  export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_continue_1epoch_encoder_off_4x8_b48_efa}"
  export VISUAL_ENCODER_FREEZE_BACKBONE=true
  export VISUAL_ENCODER_ACTIVATION_CHECKPOINTING=false
  export TRAINABLE_COMPONENTS='[dit]'
  unset VISUAL_ENCODER_LR_MULTIPLIER
fi

exec bash \
  "${SCRIPT_DIR}/run_robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa.sh"
