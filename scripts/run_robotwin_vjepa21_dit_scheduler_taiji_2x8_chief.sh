#!/usr/bin/env bash
# Chief-only Taiji launcher for the 2x8 V-JEPA 2.1 video-DiT scheduler study.
# By default, runs baseline, noise, middle, and data sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err() {
  echo "[vjepa21-dit-scheduler-taiji] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[vjepa21-dit-scheduler-taiji] $*"
}

EXPERIMENT="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "${EXPERIMENT}" in
  all)
    PRESETS=(baseline noise middle data)
    ;;
  baseline|noise|middle|data)
    PRESETS=("${EXPERIMENT}")
    ;;
  *)
    err "expected one of: all, baseline, noise, middle, data; got ${EXPERIMENT}"
    ;;
esac

HOSTFILE="${HOSTFILE:-/etc/taiji/hostfile}"
[[ -f "${HOSTFILE}" ]] || err "hostfile not found: ${HOSTFILE}"
mapfile -t HOSTS < <(
  awk 'NF && $1 !~ /^#/ && !seen[$1]++ {print $1}' "${HOSTFILE}"
)
if [[ "${#HOSTS[@]}" -ne 2 ]]; then
  err "expected exactly two unique hosts in ${HOSTFILE}, got ${#HOSTS[@]}: ${HOSTS[*]}"
fi

printf -v NODE_IP_LIST '%s,' "${HOSTS[@]}"
export NODE_IP_LIST="${NODE_IP_LIST%,}"
export SSH_PORT="${SSH_PORT:-36000}"

export FASTWAM_USE_EFA=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"
export ROBOTWIN_DATA_ROOT
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
[[ -f "${VJEPA21_NORMALISE_STATS_PATH}" ]] || \
  err "normalization stats not found: ${VJEPA21_NORMALISE_STATS_PATH}"

RUN_SET="${RUN_SET:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_PREFIX="${RUN_PREFIX:-robotwin_vjepa21_dit_scheduler}"
INNER_SCRIPT="${SCRIPT_DIR}/run_robotwin_vjepa21_dit_timestep_oldcluster_2x8.sh"
[[ -f "${INNER_SCRIPT}" ]] || err "inner launcher not found: ${INNER_SCRIPT}"

# shellcheck source=_timestep_sampling_preset.sh
source "${SCRIPT_DIR}/_timestep_sampling_preset.sh"

info "hosts=${NODE_IP_LIST} ssh_port=${SSH_PORT}"
info "presets=${PRESETS[*]} run_set=${RUN_SET}"
info "stats=${VJEPA21_NORMALISE_STATS_PATH}"
info "execution is sequential; the next preset starts only after the current one exits"

for preset in "${PRESETS[@]}"; do
  fastwam_video_timestep_sampling_preset "${preset}"
  run_name="${RUN_PREFIX}_${TIMESTEP_PRESET_SUFFIX}_globalnorm_taiji_2x8_b48_acc2_gb1536_${RUN_SET}"

  info "starting preset=${preset} run_name=${run_name}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    info "dry-run: TIMESTEP_SAMPLING_PRESET=${preset} RUN_NAME=${run_name} bash ${INNER_SCRIPT} $*"
    continue
  fi

  TIMESTEP_SAMPLING_PRESET="${preset}" \
  RUN_NAME="${run_name}" \
    bash "${INNER_SCRIPT}" "$@"
  info "completed preset=${preset}"
done

info "requested scheduler experiments completed"
