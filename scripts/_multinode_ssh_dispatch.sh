#!/usr/bin/env bash
# _multinode_ssh_dispatch.sh — one-click multi-node SSH launcher.
#
# Source this file AFTER the topology block (_NODES / NNODES / MASTER_ADDR /
# NODE_RANK / MASTER_PORT are already set). Then call:
#
#   _multinode_dispatch "${BASH_SOURCE[0]}"
#
# What it does (only when NNODES > 1 and _MULTINODE_LAUNCHED is unset):
#   1. SSH into each non-0 node and run the same script with NODE_RANK=i and
#      all relevant env vars forwarded.
#   2. Sets NODE_RANK=0 and _MULTINODE_LAUNCHED=1 locally so the calling
#      script continues as rank-0.
#   3. Registers a trap so rank-0 waits for all SSH sessions before exiting.
#
# Prerequisites:
#   * Passwordless SSH from rank-0 to all other nodes (ssh-copy-id).
#   * The script path must be accessible on all nodes (shared FS).
#
# Usage:
#   NODE_IP_LIST="10.0.0.1,10.0.0.2" bash scripts/run_libero_hfastwam_8card_small_dino.sh
#   # NODE_RANK defaults to 0; the script SSHes into 10.0.0.2 as rank 1 automatically.

_DISPATCH_PIDS=()

_multinode_wait() {
  [[ "${#_DISPATCH_PIDS[@]}" -eq 0 ]] && return
  echo "[multinode] waiting for ${#_DISPATCH_PIDS[@]} remote rank(s) ..."
  local rc=0
  for pid in "${_DISPATCH_PIDS[@]}"; do
    wait "${pid}" || rc=$?
  done
  [[ "${rc}" -ne 0 ]] && echo "[multinode] WARNING: one or more remote ranks exited non-zero (rc=${rc})." >&2
}

_multinode_dispatch() {
  # No-op for single-node or if already dispatched by a remote invocation.
  [[ "${NNODES:-1}" -le 1 || -n "${_MULTINODE_LAUNCHED:-}" || "${FASTWAM_MANAGED_DISTRIBUTED:-0}" == "1" ]] && return

  local caller="$1"

  # Collect env vars to forward to remote nodes.
  local kvs=(
    "NODE_IP_LIST=${NODE_IP_LIST}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "MASTER_PORT=${MASTER_PORT}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "_MULTINODE_LAUNCHED=1"
  )
  for v in RUN_NAME TASK_CONFIG MODEL_CONFIG LAUNCH_LABEL GLOBAL_BATCH_SIZE PER_DEVICE_BATCH_SIZE \
            GRADIENT_ACCUMULATION_STEPS SAVE_EVERY LOG_EVERY LIBERO_DATA_ROOT ROBOTWIN_DATA_ROOT \
            ROBOTWIN_WEBDATASET_ROOT ROBOTWIN_DATA_CONFIG \
            NO_CKPT FASTWAM_PROFILE_STEPS CUDA_VISIBLE_DEVICES \
            FASTWAM_EXPECTED_WORLD_SIZE \
            FASTWAM_TORCH_PROFILE FASTWAM_TORCH_PROFILE_TRACE \
            WANDB WANDB_API_KEY WANDB_API_KEY_FILE WANDB_PROJECT WANDB_GROUP \
            WANDB_MODE WANDB_ENTITY \
            NCCL_SOCKET_IFNAME NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL \
            GLOO_SOCKET_IFNAME NCCL_SOCKET_FAMILY TP_SOCKET_IFNAME \
            FASTWAM_USE_EFA FI_PROVIDER FI_EFA_USE_DEVICE_RDMA FI_EFA_USE_HUGE_PAGE NCCL_BUFFSIZE \
            NCCL_DEBUG NCCL_DEBUG_SUBSYS \
            HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HOME TORCH_HOME HF_DATASETS_CACHE \
            FASTWAM_DISABLE_PROXY \
            http_proxy https_proxy ftp_proxy no_proxy \
            HTTP_PROXY HTTPS_PROXY NO_PROXY \
            STANDARDISE_OUTPUT TEMPORAL_DOWNSAMPLE CAUSAL_TUBELET_ENCODING FRAME_GAP DETACH_VIDEO \
            FASTWAM_ROPE_IMPL \
            FASTWAM_SDPA_BACKEND \
            VIDEO_LATENT_CACHE_ENABLED VIDEO_LATENT_CACHE_DIR LATENT_CACHE_SHARD_SIZE LATENT_CACHE_BATCH_SIZE \
            LATENT_CACHE_NUM_WORKERS LATENT_CACHE_DTYPE LATENT_CACHE_DROP_TRAIN_VIDEO \
            LATENT_CACHE_INCLUDE_VAL \
            FFMPEG_PREFIX LD_LIBRARY_PATH \
            VJEPA21_CHECKPOINT VJEPA21_REPO \
            XR1_CHECKPOINT XR1_CONFIG_MODEL VISUAL_ENCODER_DESCRIPTION \
            ACCEL_CONFIG NUM_WORKERS DATALOADER_PREFETCH_FACTOR \
            DATALOADER_PERSISTENT_WORKERS FASTWAM_KEEP_LAST_CKPT \
            FRESH RESUME ACTION_DIT_PRETRAINED_PATH \
            FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER; do
    [[ -n "${!v+x}" ]] && kvs+=("${v}=${!v}")
  done

  # Build a shell-safe env prefix string.
  local env_prefix=""
  for kv in "${kvs[@]}"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    env_prefix+="${key}=$(printf '%q' "${val}") "
  done

  # Where the repo lives + conda env to use on every (local & remote) node.
  # Override via env if your layout differs.
  local conda_activate="${CONDA_ACTIVATE_PATH:-/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate}"
  local conda_env="${CONDA_ENV_NAME:-fastwam}"
  local repo_dir="${REPO_ROOT:-/apdcephfs_csgl/share_306089109/shaunxhwang/h-fastwam}"

  for (( i=1; i<NNODES; i++ )); do
    local node="${_NODES[$i]}"
    local ip="${node%%:*}"
    echo "[multinode] SSH dispatch → rank ${i} @ ${ip}:${SSH_PORT:-36000}"
    # Remote runs a non-login shell: explicitly cd into the repo and activate
    # the conda env before exec'ing the training script.
    local remote_cmd
    remote_cmd="cd $(printf '%q' "${repo_dir}") && source $(printf '%q' "${conda_activate}") $(printf '%q' "${conda_env}") && NODE_RANK=${i} ${env_prefix}bash $(printf '%q' "${caller}")"
    # shellcheck disable=SC2029
    ssh -p "${SSH_PORT:-36000}" -o StrictHostKeyChecking=no -o BatchMode=yes "${ip}" \
      "${remote_cmd}" &
    _DISPATCH_PIDS+=($!)
  done

  # Wait for all SSH sessions when this process exits.
  trap '_multinode_wait' EXIT

  export NODE_RANK=0
  export _MULTINODE_LAUNCHED=1
}
