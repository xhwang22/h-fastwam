#!/usr/bin/env bash
# 32-GPU DINOv3 ViT-L/16 300M + Flow-DiT on AWS HyperPod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

export FASTWAM_EXPECTED_WORLD_SIZE=32
export FASTWAM_USE_EFA=1

# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${REPO_ROOT}/checkpoints/dinov3-vitl16-pretrain-lvd1689m}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
mkdir -p "$(dirname "${DINOV3_MODEL_PATH}")"
(
  flock -x 9
  if [[ ! -f "${DINOV3_MODEL_PATH}/config.json" || ! -f "${DINOV3_MODEL_PATH}/model.safetensors" ]]; then
    if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
      echo "[aws-dinov3] ERROR: DINOv3 is gated; set HF_TOKEN or pre-download ${DINOV3_MODEL_PATH}." >&2
      exit 2
    fi
    echo "[aws-dinov3] downloading facebook/dinov3-vitl16-pretrain-lvd1689m to ${DINOV3_MODEL_PATH}"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - "${DINOV3_MODEL_PATH}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
    local_dir=sys.argv[1],
    allow_patterns=["config.json", "model.safetensors"],
)
PY
  fi
) 9>"${DINOV3_MODEL_PATH}.download.lock"

export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=1
export SAVE_EVERY=2000
export LOG_EVERY=10
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export MODEL_CONFIG=hfastwam_small_dinov3_vitl
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true
export VIDEO_LATENT_CACHE_ENABLED=0
export RUN_NAME="${RUN_NAME:-robotwin_dinov3_vitl_300m_32gpu_b48_cudnn_overlap_efa}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-dinov3-vitl-flow-dit}"
export LAUNCH_LABEL=aws-dinov3-vitl-300m
export VISUAL_ENCODER_DESCRIPTION="DINOv3 ViT-L/16 300M, raw 1024-d features + Wan DiT"

unset CAUSAL_TUBELET_ENCODING
exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
