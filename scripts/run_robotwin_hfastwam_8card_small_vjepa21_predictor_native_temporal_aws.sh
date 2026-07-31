#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NETWORK_IFACE="${NETWORK_IFACE:-$(awk '$2 == "00000000" && $1 != "lo" { print $1; exit }' /proc/net/route)}"
if [[ -z "${NETWORK_IFACE}" || ! -d "/sys/class/net/${NETWORK_IFACE}" ]]; then
  echo "ERROR: unable to detect a usable network interface; set NETWORK_IFACE explicitly." >&2
  exit 1
fi

export NCCL_SOCKET_IFNAME="${NETWORK_IFACE}"
export GLOO_SOCKET_IFNAME="${NETWORK_IFACE}"
export TP_SOCKET_IFNAME="${NETWORK_IFACE}"
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export VIDEO_LATENT_CACHE_ENABLED=0

export FFMPEG_PREFIX="${FFMPEG_PREFIX:-/fsx/ffmpeg-shared}"
if [[ ! -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]]; then
  MAMBA_TMP="$(mktemp -d /tmp/fastwam-micromamba.XXXXXX)"
  trap 'rm -rf "${MAMBA_TMP}"' EXIT
  (
    cd "${MAMBA_TMP}"
    curl --fail --location --silent --show-error \
      https://micro.mamba.pm/api/micromamba/linux-64/latest |
      tar -xj bin/micromamba
    ./bin/micromamba create -y \
      -p "${FFMPEG_PREFIX}" \
      -c conda-forge \
      "ffmpeg=7.*"
  )
  rm -rf "${MAMBA_TMP}"
  trap - EXIT
fi

export PATH="${FFMPEG_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_native_temporal.sh" "$@"
