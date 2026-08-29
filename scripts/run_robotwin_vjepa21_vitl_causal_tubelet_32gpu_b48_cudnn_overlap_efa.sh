#!/usr/bin/env bash
# 32-GPU V-JEPA 2.1 ViT-L/16 300M + Flow-DiT on AWS HyperPod.
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

export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
export VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}"
export VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
mkdir -p "$(dirname "${VJEPA21_CHECKPOINT}")" "$(dirname "${VJEPA21_REPO}")"

(
  flock -x 9
  if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
    echo "[aws-vjepa21-vitl] downloading V-JEPA 2.1 ViT-L checkpoint"
    curl -fL --retry 5 \
      https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
      -o "${VJEPA21_CHECKPOINT}.part"
    mv "${VJEPA21_CHECKPOINT}.part" "${VJEPA21_CHECKPOINT}"
  fi
) 9>"${VJEPA21_CHECKPOINT}.download.lock"

(
  flock -x 9
  if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
    if [[ -e "${VJEPA21_REPO}" ]]; then
      echo "[aws-vjepa21-vitl] ERROR: incomplete source tree at ${VJEPA21_REPO}; remove or repair it before retrying." >&2
      exit 2
    fi
    git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "${VJEPA21_REPO}"
  fi
) 9>"${VJEPA21_REPO}.download.lock"

export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=1
export SAVE_EVERY=2000
export LOG_EVERY=10
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export MODEL_CONFIG=hfastwam_small_vjepa21_vitl
export CAUSAL_TUBELET_ENCODING=true
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true
export VIDEO_LATENT_CACHE_ENABLED=0
export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "[aws-vjepa21-vitl] ERROR: completed WebDataset not found: ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
  echo "Set ROBOTWIN_WEBDATASET_ROOT to the completed indexed RoboTwin dataset." >&2
  exit 2
fi
export LOG_ROOT="${LOG_ROOT:-/efs/shaunxhwang}"
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_vitl_300m_causal_tubelet_32gpu_b48_cudnn_overlap_efa}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-vjepa21-vitl-flow-dit}"
export VISUAL_ENCODER_DESCRIPTION="V-JEPA 2.1 ViT-L/16 300M, raw 1024-d features + Wan DiT"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21.sh" "$@"
