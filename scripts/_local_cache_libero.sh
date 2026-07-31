#!/usr/bin/env bash
# Per-node helper: copy LIBERO LeRobot datasets from shared storage to local disk.
# The destination keeps the same root layout expected by LIBERO_DATA_ROOT:
#   ${DST_ROOT}/libero_mujoco3.3.2/<subset>/{meta,data,videos}

set -euo pipefail

SRC_ROOT="${LIBERO_SOURCE_ROOT:-data}"
DST_ROOT="${LOCAL_LIBERO_DATA_ROOT:-/tmp/fastwam_data/libero}"
PARALLEL="${LOCAL_CACHE_PARALLEL:-4}"
LOG_DIR="${LOCAL_CACHE_LOG_DIR:-/tmp/local_cache_libero}"

SUBSETS=(
  libero_spatial_no_noops_lerobot
  libero_object_no_noops_lerobot
  libero_goal_no_noops_lerobot
  libero_10_no_noops_lerobot
)

mkdir -p "${LOG_DIR}" "${DST_ROOT}/libero_mujoco3.3.2"
LOG_FILE="${LOG_DIR}/copy.log"
DONE_FILE="${LOG_DIR}/done.flag"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

copy_subset() {
  local subset="$1"
  local src="${SRC_ROOT}/libero_mujoco3.3.2/${subset}"
  local dst="${DST_ROOT}/libero_mujoco3.3.2/${subset}"
  local done="${LOG_DIR}/${subset}.done"

  if [[ -f "${done}" && -d "${dst}/meta" && -d "${dst}/data" && -d "${dst}/videos" ]]; then
    log "skip ${subset}: already cached"
    return 0
  fi
  [[ -d "${src}" ]] || { log "ERROR: missing source dataset dir: ${src}"; return 1; }

  log "copy ${subset}: ${src} -> ${dst}"
  mkdir -p "${dst}"

  if [[ -d "${src}/meta" ]]; then
    rsync -a --delete "${src}/meta/" "${dst}/meta/"
  fi
  if [[ -d "${src}/data" ]]; then
    rsync -a --delete "${src}/data/" "${dst}/data/"
  fi
  if [[ -d "${src}/videos" ]]; then
    rsync -a --delete "${src}/videos/" "${dst}/videos/"
  fi

  touch "${done}"
  log "done ${subset}"
}

: > "${LOG_FILE}"
log "host=$(hostname) SRC_ROOT=${SRC_ROOT} DST_ROOT=${DST_ROOT} PARALLEL=${PARALLEL}"
df -h "${DST_ROOT}" 2>/dev/null | tee -a "${LOG_FILE}" || true

running=0
for subset in "${SUBSETS[@]}"; do
  (
    copy_subset "${subset}"
  ) &
  running=$((running + 1))
  if (( running >= PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done
wait

for subset in "${SUBSETS[@]}"; do
  dst="${DST_ROOT}/libero_mujoco3.3.2/${subset}"
  [[ -d "${dst}/meta" && -d "${dst}/data" && -d "${dst}/videos" ]] || {
    log "ERROR: incomplete cached dataset: ${dst}"
    exit 1
  }
done

du -sh "${DST_ROOT}/libero_mujoco3.3.2" 2>/dev/null | tee -a "${LOG_FILE}" || true
echo "OK $(date +%s)" > "${DONE_FILE}"
log "ALL DONE"
