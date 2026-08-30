#!/usr/bin/env bash
# 64-GPU V-JEPA 2.1 ViT-L image mode + semantic-wm S-VAE-96 + Flow-DiT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FASTWAM_EXPECTED_WORLD_SIZE=64
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
    echo "[aws-vjepa21-svae96] downloading V-JEPA 2.1 ViT-L checkpoint"
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
      echo "[aws-vjepa21-svae96] ERROR: incomplete source tree at ${VJEPA21_REPO}." >&2
      exit 2
    fi
    git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "${VJEPA21_REPO}"
  fi
) 9>"${VJEPA21_REPO}.download.lock"

SEMANTIC_SVAE_ROOT="${SEMANTIC_SVAE_ROOT:-/efs/shaunxhwang/checkpoints/semantic-wm/vjepa}"
SEMANTIC_SVAE_SOURCE="${SEMANTIC_SVAE_ROOT}/adapter_vjepa_image_96_multi.full.pt"
export SEMANTIC_SVAE_CHECKPOINT="${SEMANTIC_SVAE_CHECKPOINT:-${SEMANTIC_SVAE_ROOT}/adapter_vjepa_image_96_multi.encoder_bf16.pt}"
SEMANTIC_SVAE_READY="${SEMANTIC_SVAE_CHECKPOINT}.ready"
SEMANTIC_SVAE_URL="https://huggingface.co/Nilaksh404/semantic-wm/resolve/ba06ced314d61e313ff670b0f932cfecad5126a6/vjepa/adapter_vjepa_image_96_multi.pt?download=true"
SEMANTIC_SVAE_SHA256="179dc0809262bca042e2a05d834df40f3e5c953271223727c94eb9e4484f6ec8"
mkdir -p "${SEMANTIC_SVAE_ROOT}"

(
  flock -x 9
  if [[ ! -s "${SEMANTIC_SVAE_CHECKPOINT}" || ! -f "${SEMANTIC_SVAE_READY}" ]]; then
    echo "[aws-vjepa21-svae96] preparing semantic-wm multi-view S-VAE-96 weights"
    curl -fL --retry 5 --retry-delay 5 \
      "${SEMANTIC_SVAE_URL}" \
      -o "${SEMANTIC_SVAE_SOURCE}.part"
    mv "${SEMANTIC_SVAE_SOURCE}.part" "${SEMANTIC_SVAE_SOURCE}"
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      python scripts/prepare_semantic_svae_checkpoint.py \
        --source "${SEMANTIC_SVAE_SOURCE}" \
        --output "${SEMANTIC_SVAE_CHECKPOINT}" \
        --expected-source-sha256 "${SEMANTIC_SVAE_SHA256}" \
        --remove-source
    touch "${SEMANTIC_SVAE_READY}"
  fi
) 9>"${SEMANTIC_SVAE_CHECKPOINT}.prepare.lock"

export GLOBAL_BATCH_SIZE=$(( 24 * FASTWAM_EXPECTED_WORLD_SIZE ))
export GRADIENT_ACCUMULATION_STEPS=1
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export MODEL_CONFIG=hfastwam_small_vjepa21_vitl_image_svae96
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=false
export VIDEO_LATENT_CACHE_ENABLED=0
export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "[aws-vjepa21-svae96] ERROR: completed WebDataset not found: ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
  exit 2
fi
export LOG_ROOT="${LOG_ROOT:-/efs/shaunxhwang}"
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_vitl_image_svae96_64gpu_b24_cudnn_overlap_efa}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-vjepa21-vitl-image-svae96-flow-dit}"
export LAUNCH_LABEL=aws-vjepa21-vitl-image-svae96
export VISUAL_ENCODER_DESCRIPTION="V-JEPA 2.1 ViT-L image mode + semantic-wm multi-view S-VAE-96 + Wan DiT"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
