#!/usr/bin/env bash
# Chief-only Taiji 4x8 direct RoboTwin training with VAE latents + predictor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err() {
  echo "[vae-predictor-taiji] ERROR: $*" >&2
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

export SSH_PORT="${SSH_PORT:-36000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=32
export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=2

export FASTWAM_USE_EFA=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"

export MODEL_CONFIG=hfastwam_small_vae_predictor
export USE_VJEPA21_VISUAL_ENCODER=0
export VIDEO_LATENT_CACHE_ENABLED=0
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export RUN_NAME="${RUN_NAME:-robotwin_vae_predictor_taiji_4x8_b24_acc2_gb1536}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-vae-predictor}"

unset VJEPA21_NORMALISE_STATS_PATH STANDARDISE_OUTPUT TEMPORAL_DOWNSAMPLE
unset CAUSAL_TUBELET_ENCODING FRAME_GAP FIXED_TARGET_ENCODER
unset VISUAL_ENCODER_FREEZE_BACKBONE VISUAL_ENCODER_ACTIVATION_CHECKPOINTING
unset TRAINABLE_COMPONENTS VISUAL_ENCODER_LR_MULTIPLIER

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[vae-predictor-taiji] hosts=${NODE_IP_LIST} model=${MODEL_CONFIG} global_batch=${GLOBAL_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
  exit 0
fi

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" "$@"
