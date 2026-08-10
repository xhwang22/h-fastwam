#!/usr/bin/env bash
# RoboTwin 2.0 8-GPU H-FastWAM SMALL with V-JEPA 2.1 ViT-G/16 (2B)
# and the deterministic JEPA predictor replacing the video DiT.
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
# shellcheck source=_robotwin_data_source.sh
source "${SCRIPT_DIR}/_robotwin_data_source.sh"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
fastwam_select_robotwin_data_source
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
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_robotwin_8card_small_vjepa21_predictor_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
  echo "[robotwin-small-vjepa21-predictor] ERROR: missing ${VJEPA21_CHECKPOINT}" >&2
  echo "Download: curl -L https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -o ${VJEPA21_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
  echo "[robotwin-small-vjepa21-predictor] ERROR: V-JEPA 2.1 source tree missing at ${VJEPA21_REPO}." >&2
  exit 1
fi

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

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TASK_CONFIG="${TASK_CONFIG:-robotwin_uncond_3cam_384_1e-4}"
DATA_CONFIG="${DATA_CONFIG:-${ROBOTWIN_DATA_CONFIG}}"
MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_vjepa21_predictor}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-null}"
WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
_BATCH_DENOM=$(( WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))
if (( GLOBAL_BATCH_SIZE % _BATCH_DENOM != 0 )); then
  echo "[8card-small-vjepa-predictor] ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by world_size*grad_accum=${_BATCH_DENOM} (nproc=${NPROC_PER_NODE} nnodes=${NNODES} grad_accum=${GRADIENT_ACCUMULATION_STEPS})." >&2
  exit 1
fi
BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / _BATCH_DENOM ))

RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_vjepa21_predictor_ds}"
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
    echo "[robotwin-small-vjepa21-predictor] ERROR: set WANDB_API_KEY/~/.wandb_key or use WANDB=0." >&2
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

PRETRAIN_OVERRIDES=()
if [[ -n "${FASTWAM_CHECKPOINT:-}" ]]; then
  if [[ ! -f "${FASTWAM_CHECKPOINT}" ]]; then
    echo "[8card-small-vjepa-predictor] ERROR: FASTWAM_CHECKPOINT not found: ${FASTWAM_CHECKPOINT}" >&2
    exit 1
  fi
  PRETRAIN_OVERRIDES=(
    "model.fastwam_checkpoint=${FASTWAM_CHECKPOINT}"
  )
fi

STANDARDISE_OVERRIDES=()
if [[ -n "${STANDARDISE_OUTPUT:-}" ]]; then
  STANDARDISE_OVERRIDES=(
    "model.visual_encoder_config.standardise_output=${STANDARDISE_OUTPUT}"
  )
fi

GAP_OVERRIDES=()
if [[ "${FRAME_GAP:-}" == "3" ]]; then
  GAP_OVERRIDES=(
    "data.train.action_video_freq_ratio=1"
    "data.val.action_video_freq_ratio=1"
    "model.visual_encoder_config.temporal_downsample=3"
  )
fi

TEMPORAL_OVERRIDES=()
if [[ -n "${TEMPORAL_DOWNSAMPLE:-}" ]]; then
  TEMPORAL_OVERRIDES=(
    "model.visual_encoder_config.temporal_downsample=${TEMPORAL_DOWNSAMPLE}"
  )
fi

DATA_SOURCE_OVERRIDES=()
if [[ "${USE_ROBOTWIN_DATA_OVERRIDES:-1}" == "1" ]]; then
  DATA_SOURCE_OVERRIDES=("${ROBOTWIN_DATA_OVERRIDES[@]}")
fi

SEGMENT_OVERRIDES=()
if [[ "${SET_NUM_SEGMENTS:-1}" == "1" ]]; then
  SEGMENT_OVERRIDES=(
    "data.train.num_segments=1"
    "data.val.num_segments=1"
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
      task="${TASK_CONFIG}"
    data="${DATA_CONFIG}"
      model="${MODEL_CONFIG}"
      output_dir="${LOG_DIR}"
      "${WANDB_OVERRIDES[@]}"
      batch_size="${BATCH_SIZE}"
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every="${LOG_EVERY:-1}"
      num_workers="${NUM_WORKERS:-8}"
      dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR:-4}"
      dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS:-true}"
      dataloader_timeout=0
      "${SEGMENT_OVERRIDES[@]}"
      "${DATA_SOURCE_OVERRIDES[@]}"
      num_epochs="${NUM_EPOCHS}"
      max_steps="${MAX_STEPS}"
      save_every="${SAVE_EVERY}"
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
      "model.action_loss_detach_video_expert=${DETACH_VIDEO:-false}"
      model.visual_encoder_config.checkpoint_path="${VJEPA21_CHECKPOINT}"
      model.visual_encoder_config.repo_path="${VJEPA21_REPO}"
      "${PRETRAIN_OVERRIDES[@]}"
      "${CKPT_OVERRIDES[@]}"
      "${STANDARDISE_OVERRIDES[@]}"
      "${GAP_OVERRIDES[@]}"
      "${TEMPORAL_OVERRIDES[@]}"
      "${VIDEO_LATENT_CACHE_OVERRIDES[@]}"
      "${RESUME_OVERRIDES[@]}"
)

echo "[robotwin-small-vjepa21-predictor] task=${TASK_CONFIG} model=${MODEL_CONFIG}"
echo "[robotwin-small-vjepa21-predictor] global_batch=${GLOBAL_BATCH_SIZE} world_size=${WORLD_SIZE} batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[robotwin-small-vjepa21-predictor] checkpoint=${VJEPA21_CHECKPOINT}"
echo "[robotwin-small-vjepa21-predictor] master=${MASTER_ADDR}:${MASTER_PORT} log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
