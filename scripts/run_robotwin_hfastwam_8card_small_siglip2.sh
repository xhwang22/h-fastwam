#!/usr/bin/env bash
# RoboTwin 2.0 8-GPU H-FastWAM SMALL with the frozen Qwen3-VL-2B
# visual tower feeding the existing flow-matching video DiT.
set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -n "${NODE_IP_LIST:-}" ]]; then
  IFS=',' read -ra _NODES <<< "${NODE_IP_LIST}"
  NNODES="${#_NODES[@]}"
  MASTER_ADDR="${MASTER_ADDR:-${_NODES[0]%%:*}}"
elif [[ -n "${NNODES:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
else
  NNODES=1
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
NODE_RANK="${NODE_RANK:-0}"
# shellcheck source=_multinode_ssh_dispatch.sh
source "${SCRIPT_DIR}/_multinode_ssh_dispatch.sh"
# shellcheck source=_auto_resume.sh
source "${SCRIPT_DIR}/_auto_resume.sh"
# shellcheck source=_video_latent_cache_overrides.sh
source "${SCRIPT_DIR}/_video_latent_cache_overrides.sh"
# Resolve WandB key before dispatch so it gets forwarded to remote nodes
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_key")"
fi
_multinode_dispatch "${BASH_SOURCE[0]}"

unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset DS_CONFIG
unset FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER
ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"

if [[ "${FASTWAM_DISABLE_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy ftp_proxy all_proxy
  unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
  unset no_proxy NO_PROXY
else

export no_proxy="${no_proxy:-.woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com,.myqcloud.com,.tencentcos.cn}"
export NO_PROXY="${no_proxy}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export ftp_proxy="${ftp_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
fi

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_robotwin_8card_small_siglip2_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
LAUNCH_LABEL="${LAUNCH_LABEL:-robotwin-small-siglip2}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
_BATCH_DENOM=$(( WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))
if (( GLOBAL_BATCH_SIZE % _BATCH_DENOM != 0 )); then
  echo "[${LAUNCH_LABEL}] ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by world_size*grad_accum=${_BATCH_DENOM} (nproc=${NPROC_PER_NODE} nnodes=${NNODES} grad_accum=${GRADIENT_ACCUMULATION_STEPS})." >&2
  exit 1
fi
BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / _BATCH_DENOM ))

RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_siglip2_ds}"
MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_siglip2}"
LOG_DIR="${REPO_ROOT}/runs/robotwin_hfastwam/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log.rank${NODE_RANK}"

export FASTWAM_KEEP_LAST_CKPT="${FASTWAM_KEEP_LAST_CKPT:-3}"
SAVE_EVERY="${SAVE_EVERY:-200}"
_compute_resume_override "${LOG_DIR}"

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_key")"
fi
WANDB_OVERRIDES=("wandb.enabled=false")
if [[ "${WANDB:-1}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[${LAUNCH_LABEL}] ERROR: set WANDB_API_KEY/~/.wandb_key or use WANDB=0." >&2
    exit 1
  fi
  export WANDB_API_KEY
  WANDB_OVERRIDES=(
    "wandb.enabled=true"
    "wandb.project=${WANDB_PROJECT:-fast-wam}"
    "wandb.name=${RUN_NAME}"
    "wandb.mode=${WANDB_MODE:-online}"
  )
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_OVERRIDES+=("wandb.workspace=${WANDB_ENTITY}")
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_OVERRIDES+=("wandb.group=${WANDB_GROUP}")
fi

CKPT_OVERRIDES=()
if [[ "${NO_CKPT:-0}" == "1" ]]; then
  CKPT_OVERRIDES=(
    "model.video_dit_config.use_gradient_checkpointing=false"
    "model.action_dit_config.use_gradient_checkpointing=false"
  )
fi

STANDARDISE_OVERRIDES=()
if [[ -n "${STANDARDISE_OUTPUT:-}" ]]; then
  STANDARDISE_OVERRIDES=(
    "model.visual_encoder_config.standardise_output=${STANDARDISE_OUTPUT}"
  )
fi

TEMPORAL_OVERRIDES=()
if [[ -n "${TEMPORAL_DOWNSAMPLE:-}" ]]; then
  TEMPORAL_OVERRIDES=(
    "model.visual_encoder_config.temporal_downsample=${TEMPORAL_DOWNSAMPLE}"
  )
fi

CAUSAL_TUBELET_OVERRIDES=()
if [[ -n "${CAUSAL_TUBELET_ENCODING:-}" ]]; then
  CAUSAL_TUBELET_OVERRIDES=(
    "++model.visual_encoder_config.causal_tubelet_encoding=${CAUSAL_TUBELET_ENCODING}"
  )
fi

CMD=(
  accelerate launch
    --config_file "${ACCEL_CONFIG}"
    --num_machines "${NNODES}"
    --machine_rank "${NODE_RANK}"
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
    --num_processes "$(( NPROC_PER_NODE * NNODES ))"
    --deepspeed_multinode_launcher standard
    scripts/train.py
      task=robotwin_uncond_3cam_384_1e-4
      data=robotwin_interleaved
      model="${MODEL_CONFIG}"
      output_dir="${LOG_DIR}"
      "${WANDB_OVERRIDES[@]}"
      batch_size="${BATCH_SIZE}"
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every=1
      num_workers="${NUM_WORKERS:-8}"
      dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR:-4}"
      dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS:-true}"
      dataloader_timeout=0
      data.train.num_segments=1
      data.val.num_segments=1
      "data.train.dataset_dirs=[${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0]"
      "data.train.pretrained_norm_stats=${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json"
      "data.val.dataset_dirs=[${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0]"
      "data.val.pretrained_norm_stats=${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json"
      num_epochs=5
      max_steps=null
      save_every="${SAVE_EVERY}"
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
      "model.action_loss_detach_video_expert=${DETACH_VIDEO:-false}"
      "${CKPT_OVERRIDES[@]}"
      "${STANDARDISE_OVERRIDES[@]}"
      "${TEMPORAL_OVERRIDES[@]}"
      "${CAUSAL_TUBELET_OVERRIDES[@]}"
      "${VIDEO_LATENT_CACHE_OVERRIDES[@]}"
      "${RESUME_OVERRIDES[@]}"
)

echo "[${LAUNCH_LABEL}] model=${MODEL_CONFIG} (Qwen3-VL-2B visual tower, raw 1024-d features + Wan DiT)"
echo "[${LAUNCH_LABEL}] global_batch=${GLOBAL_BATCH_SIZE} world_size=${WORLD_SIZE} batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[${LAUNCH_LABEL}] master=${MASTER_ADDR}:${MASTER_PORT} log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
