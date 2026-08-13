#!/usr/bin/env bash

_fastwam_require_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${name} must be a non-negative integer, got '${value}'." >&2
    return 2
  fi
}

_fastwam_require_positive_uint() {
  local name="$1"
  local value="$2"
  _fastwam_require_uint "${name}" "${value}" || return
  if (( value < 1 )); then
    echo "ERROR: ${name} must be positive, got ${value}." >&2
    return 2
  fi
}

_fastwam_configure_proxy() {
  export FASTWAM_DISABLE_PROXY="${FASTWAM_DISABLE_PROXY:-1}"
  case "${FASTWAM_DISABLE_PROXY}" in
    1|true|yes|on)
      unset http_proxy https_proxy ftp_proxy all_proxy
      unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
      unset no_proxy NO_PROXY
      echo "[aws-setup] outbound proxy disabled."
      ;;
    0|false|no|off)
      echo "[aws-setup] preserving caller-provided proxy settings."
      ;;
    *)
      echo "ERROR: FASTWAM_DISABLE_PROXY must be 0/1, true/false, yes/no, or on/off." >&2
      return 2
      ;;
  esac
}

_fastwam_resolve_nproc_per_node() {
  if [[ -n "${NPROC_PER_NODE:-}" && "${NPROC_PER_NODE}" != "auto" ]]; then
    return
  fi
  if [[ -n "${PET_NPROC_PER_NODE:-}" && "${PET_NPROC_PER_NODE}" != "auto" ]]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local devices="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -n "${devices}" && "${devices}" != "NoDevFiles" ]]; then
      local -a _fastwam_visible_devices
      IFS=',' read -ra _fastwam_visible_devices <<< "${devices}"
      NPROC_PER_NODE="${#_fastwam_visible_devices[@]}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    NPROC_PER_NODE="${NPROC_PER_NODE//[[:space:]]/}"
    if [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
      return
    fi
  fi
  NPROC_PER_NODE=8
}

_fastwam_configure_hyperpod_topology() {
  local topology_source="single-node"

  if [[ -n "${PET_NNODES:-}" ]]; then
    NNODES="${PET_NNODES}"
    _fastwam_resolve_nproc_per_node
    NODE_RANK="${PET_NODE_RANK:?PET_NODE_RANK is required when PET_NNODES is set}"
    MASTER_ADDR="${PET_MASTER_ADDR:?PET_MASTER_ADDR is required when PET_NNODES is set}"
    MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
    topology_source="HyperPod PET"
  elif [[ -n "${NNODES:-}" && -n "${NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]]; then
    _fastwam_resolve_nproc_per_node
    MASTER_PORT="${MASTER_PORT:-29500}"
    topology_source="explicit managed"
  elif [[ -n "${WORLD_SIZE:-}" && -n "${RANK:-}" && -n "${MASTER_ADDR:-}" && -z "${LOCAL_RANK:-}${LOCAL_WORLD_SIZE:-}" ]]; then
    NNODES="${WORLD_SIZE}"
    _fastwam_resolve_nproc_per_node
    NODE_RANK="${RANK}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    topology_source="PyTorchJob node-level"
  elif [[ -n "${LOCAL_RANK:-}${LOCAL_WORLD_SIZE:-}" ]]; then
    echo "ERROR: this AWS wrapper must run once per node, not once per GPU process." >&2
    echo "Use the HyperPod/PyTorchJob container command directly without an outer torchrun." >&2
    return 2
  else
    NNODES=1
    _fastwam_resolve_nproc_per_node
    NODE_RANK=0
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-29500}"
  fi

  _fastwam_require_positive_uint NNODES "${NNODES}" || return
  _fastwam_require_positive_uint NPROC_PER_NODE "${NPROC_PER_NODE}" || return
  _fastwam_require_uint NODE_RANK "${NODE_RANK}" || return
  _fastwam_require_positive_uint MASTER_PORT "${MASTER_PORT}" || return
  if (( NODE_RANK >= NNODES )); then
    echo "ERROR: NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}." >&2
    return 2
  fi

  export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
  if [[ "${topology_source}" != "single-node" ]]; then
    unset NODE_IP_LIST
    export FASTWAM_MANAGED_DISTRIBUTED=1
    export _MULTINODE_LAUNCHED=1
  fi

  echo "[aws-setup] topology=${topology_source} node_rank=${NODE_RANK}/${NNODES} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
}

_fastwam_configure_aws_network() {
  local network_iface
  local use_efa=0
  network_iface="${NETWORK_IFACE:-$(awk '$2 == "00000000" && $1 != "lo" { print $1; exit }' /proc/net/route)}"
  if [[ -z "${network_iface}" || ! -d "/sys/class/net/${network_iface}" ]]; then
    echo "ERROR: unable to detect a usable network interface; set NETWORK_IFACE explicitly." >&2
    return 2
  fi

  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${network_iface}}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${network_iface}}"
  export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${network_iface}}"

  if [[ "${FASTWAM_MANAGED_DISTRIBUTED:-0}" == "1" && "${NNODES}" -gt 1 ]]; then
    case "${FASTWAM_USE_EFA:-auto}" in
      1|true|yes|on)
        use_efa=1
        ;;
      0|false|no|off)
        use_efa=0
        ;;
      auto)
        if [[ "${NCCL_NET:-}" != "Socket" ]] && compgen -G '/sys/class/infiniband/*' >/dev/null; then
          if ! command -v fi_info >/dev/null 2>&1 || fi_info -p efa >/dev/null 2>&1; then
            use_efa=1
          fi
        fi
        ;;
      *)
        echo "ERROR: FASTWAM_USE_EFA must be auto, 0/1, true/false, yes/no, or on/off." >&2
        return 2
        ;;
    esac
  fi

  if (( use_efa )); then
    [[ "${NCCL_NET:-}" == "Socket" ]] && unset NCCL_NET
    [[ "${NCCL_NET_PLUGIN:-}" == "none" ]] && unset NCCL_NET_PLUGIN
    export FI_PROVIDER="${FI_PROVIDER:-efa}"
    export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
    export FI_EFA_USE_HUGE_PAGE="${FI_EFA_USE_HUGE_PAGE:-0}"
    export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
    echo "[aws-setup] network=${network_iface} transport=EFA"
  else
    export NCCL_NET=Socket
    export NCCL_NET_PLUGIN=none
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    [[ "${FI_PROVIDER:-}" == "efa" ]] && unset FI_PROVIDER
    unset FI_EFA_USE_DEVICE_RDMA
    echo "[aws-setup] network=${network_iface} transport=Socket"
  fi
}

_fastwam_install_shared_ffmpeg() {
  export FFMPEG_PREFIX="${FFMPEG_PREFIX:-/fsx/ffmpeg-shared}"
  if [[ "${FASTWAM_SKIP_FFMPEG_SETUP:-0}" != "1" && ! -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]]; then
    local install_lock="${FFMPEG_PREFIX}.install.lock"
    if mkdir "${install_lock}" 2>/dev/null; then
      local mamba_tmp
      mamba_tmp="$(mktemp -d /tmp/fastwam-micromamba.XXXXXX)"
      trap 'rm -rf "${mamba_tmp}"; rmdir "${install_lock}" 2>/dev/null || true' EXIT
      if [[ ! -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]]; then
        (
          cd "${mamba_tmp}"
          curl --fail --location --silent --show-error \
            https://micro.mamba.pm/api/micromamba/linux-64/latest |
            tar -xj bin/micromamba
          ./bin/micromamba create -y \
            -p "${FFMPEG_PREFIX}" \
            -c conda-forge \
            "ffmpeg=7.*"
        )
      fi
      rm -rf "${mamba_tmp}"
      rmdir "${install_lock}"
      trap - EXIT
    else
      echo "[aws-setup] waiting for shared FFmpeg installation at ${FFMPEG_PREFIX}..."
      local attempt
      for (( attempt=0; attempt<900; attempt++ )); do
        [[ -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]] && break
        sleep 2
      done
      if [[ ! -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]]; then
        echo "ERROR: timed out waiting for FFmpeg installation; check ${install_lock}." >&2
        return 1
      fi
    fi
  fi

  export PATH="${FFMPEG_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

_fastwam_configure_wandb() {
  local key_file="${WANDB_API_KEY_FILE:-/fsx/.secrets/wandb_api_key}"
  if [[ -z "${WANDB_API_KEY:-}" && -f "${key_file}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${key_file}")"
  elif [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_key")"
  fi

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    export WANDB="${WANDB:-1}"
    echo "[aws-setup] W&B enabled using a protected key source."
  else
    export WANDB="${WANDB:-0}"
  fi
}

fastwam_prepare_aws_hyperpod_runtime() {
  _fastwam_configure_proxy
  _fastwam_configure_hyperpod_topology
  if [[ -n "${FASTWAM_EXPECTED_WORLD_SIZE:-}" ]]; then
    _fastwam_require_positive_uint \
      FASTWAM_EXPECTED_WORLD_SIZE \
      "${FASTWAM_EXPECTED_WORLD_SIZE}"
    local actual_world_size=$(( NNODES * NPROC_PER_NODE ))
    if (( actual_world_size != FASTWAM_EXPECTED_WORLD_SIZE )); then
      echo "ERROR: expected world_size=${FASTWAM_EXPECTED_WORLD_SIZE}, got ${actual_world_size} (${NNODES} nodes x ${NPROC_PER_NODE} processes)." >&2
      return 2
    fi
  fi
  _fastwam_configure_aws_network
  _fastwam_install_shared_ffmpeg
}

fastwam_prepare_aws_hyperpod() {
  fastwam_prepare_aws_hyperpod_runtime
  _fastwam_configure_wandb

  if ! command -v accelerate >/dev/null 2>&1; then
    echo "ERROR: accelerate is not installed in the current Python environment." >&2
    echo "Run on every worker node:" >&2
    echo "  bash scripts/install_aws_python_dependencies.sh" >&2
    return 1
  fi

  export VIDEO_LATENT_CACHE_ENABLED=0
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
  export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
}
