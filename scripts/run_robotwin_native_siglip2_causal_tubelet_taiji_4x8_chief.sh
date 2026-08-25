#!/usr/bin/env bash
# Chief-only Taiji 4x8 native SigLIP2 So400M causal-state Flow-DiT training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err() {
  echo "[native-siglip2-taiji] ERROR: $*" >&2
  exit 1
}

HOSTFILE="${HOSTFILE:-/etc/taiji/hostfile}"
[[ -f "${HOSTFILE}" ]] || err "hostfile not found: ${HOSTFILE}"
mapfile -t HOSTS < <(
  awk 'NF && $1 !~ /^#/ && !seen[$1]++ {print $1}' "${HOSTFILE}"
)
if [[ "${#HOSTS[@]}" -ne 4 ]]; then
  err "expected exactly four unique hosts in ${HOSTFILE}, got ${#HOSTS[@]}: ${HOSTS[*]}"
fi
printf -v NODE_IP_LIST '%s,' "${HOSTS[@]}"
export NODE_IP_LIST="${NODE_IP_LIST%,}"

export SIGLIP2_MODEL_PATH="${SIGLIP2_MODEL_PATH:-${REPO_ROOT}/checkpoints/siglip2-so400m-patch16-384}"
if [[ ! -f "${SIGLIP2_MODEL_PATH}/config.json" || ! -f "${SIGLIP2_MODEL_PATH}/model.safetensors" ]]; then
  if [[ "${SIGLIP2_AUTO_DOWNLOAD:-1}" != "1" ]]; then
    err "SigLIP2 checkpoint missing at ${SIGLIP2_MODEL_PATH}; set SIGLIP2_AUTO_DOWNLOAD=1 or provide SIGLIP2_MODEL_PATH"
  fi
  CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
  if [[ -f "${CONDA_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_ACTIVATE}" fastwam
  fi
  echo "[native-siglip2-taiji] downloading google/siglip2-so400m-patch16-384 to ${SIGLIP2_MODEL_PATH}"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - "${SIGLIP2_MODEL_PATH}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-so400m-patch16-384",
    local_dir=sys.argv[1],
    allow_patterns=["config.json", "model.safetensors"],
)
PY
fi

export SSH_PORT="${SSH_PORT:-36000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=32
export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=1

export FASTWAM_USE_EFA=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"

export MODEL_CONFIG=hfastwam_small_native_siglip2
export CAUSAL_TUBELET_ENCODING=true
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true
export VIDEO_LATENT_CACHE_ENABLED=0
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export RUN_NAME="${RUN_NAME:-robotwin_native_siglip2_so400m_causal_tubelet_t3_taiji_4x8_b48}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-native-siglip2-so400m-flow-dit}"
export LAUNCH_LABEL=native-siglip2-so400m-taiji-4x8
export VISUAL_ENCODER_DESCRIPTION="native SigLIP2 So400M patch16, raw 1152-d features + Wan DiT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[native-siglip2-taiji] hosts=${NODE_IP_LIST} model=${MODEL_CONFIG} global_batch=${GLOBAL_BATCH_SIZE}"
  exit 0
fi

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
