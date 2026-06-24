#!/usr/bin/env bash
# Per-node helper: cache robotwin2.0 dataset to LOCAL DISK /tmp (not /dev/shm,
# so it does NOT consume RAM). Streams the tar.gz parts sequentially so the
# CephFS source side does big sequential reads (fast) instead of per-file
# random metadata lookups (slow). Shows a live progress bar.
#
# Caches (VAE run — NO vjepa checkpoint needed):
#   /tmp/fwam/data/robotwin2.0/dataset_stats.json    (small)
#   /tmp/fwam/data/robotwin2.0/robotwin2.0/...        (~75GB, from tar.gz parts)
#
# Progress: uses `pv` if available (exact %, based on the ~20GB compressed
# input). Otherwise falls back to a background `du` that prints extracted size
# every 15s (target ~75GB).
#
# Usage (per node): bash scripts/_local_cache_robotwin_tmp.sh

set -e

SRC_TAR_PARTS_GLOB="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/robotwin2.0.tar.gz.part-*"
SRC_STATS="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/dataset_stats.json"

DST_BASE="${LOCAL_CACHE_BASE:-/tmp/fwam}"
DST_DATA_PARENT="${DST_BASE}/data/robotwin2.0"
DST_DATA="${DST_DATA_PARENT}/robotwin2.0"
DST_STATS="${DST_DATA_PARENT}/dataset_stats.json"

LOG="/tmp/local_cache_tmp.log"
DONE="/tmp/local_cache_tmp.done"

mkdir -p "${DST_DATA_PARENT}"
: > "${LOG}"
START=$(date +%s)
# log() prints to BOTH stdout (so the chief's per-rank cache.log captures it)
# and the local logfile.
log() { echo "$(date '+%H:%M:%S') [$(hostname)] $* (t+$(( $(date +%s) - START ))s)" | tee -a "${LOG}"; }

log "starting /tmp cache  DST_BASE=${DST_BASE}"
df -h "${DST_BASE}" 2>/dev/null | tee -a "${LOG}" || true

# 1) tiny: dataset_stats.json
if [[ ! -s "${DST_STATS}" ]]; then
  log "copy dataset_stats.json"
  cp "${SRC_STATS}" "${DST_STATS}"
fi

# 2) dataset via tar.gz parts
NEED_DATA=1
if [[ -d "${DST_DATA}/videos/chunk-027" && -d "${DST_DATA}/data/chunk-027" ]]; then
  cur_mb=$(du -sm "${DST_DATA}" 2>/dev/null | awk '{print $1}')
  if [[ "${cur_mb:-0}" -gt 70000 ]]; then
    NEED_DATA=0
    log "dataset already cached (${cur_mb} MB), skipping"
  fi
fi

if [[ "${NEED_DATA}" == "1" ]]; then
  rm -rf "${DST_DATA}"

  # Total compressed size (for pv progress).
  total_bytes=$(cat ${SRC_TAR_PARTS_GLOB} 2>/dev/null | wc -c 2>/dev/null || echo 0)
  # The above re-reads everything just to size it, which is wasteful; instead
  # sum file sizes via stat.
  total_bytes=0
  for f in ${SRC_TAR_PARTS_GLOB}; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    total_bytes=$(( total_bytes + sz ))
  done
  log "extract dataset from tar.gz parts (${total_bytes} compressed bytes)"

  if command -v pv >/dev/null 2>&1; then
    # pv shows a live progress bar on stderr (captured into the per-rank log).
    cat ${SRC_TAR_PARTS_GLOB} \
      | pv -s "${total_bytes}" -name "robotwin tar" \
      | tar -xzf - -C "${DST_DATA_PARENT}/" 2>>"${LOG}"
  else
    log "pv not found — falling back to periodic du progress (target ~75000 MB)"
    # Background progress printer.
    (
      while :; do
        sleep 15
        m=$(du -sm "${DST_DATA}" 2>/dev/null | awk '{print $1}')
        [[ -n "${m}" ]] && echo "$(date '+%H:%M:%S') [$(hostname)] extracted ${m} MB / ~75000 MB (t+$(( $(date +%s) - START ))s)" | tee -a "${LOG}"
      done
    ) &
    PROG_PID=$!
    cat ${SRC_TAR_PARTS_GLOB} | tar -xzf - -C "${DST_DATA_PARENT}/" 2>>"${LOG}"
    kill "${PROG_PID}" 2>/dev/null || true
  fi
  log "extract dataset done"
fi

data_mb=$(du -sm "${DST_DATA}" 2>/dev/null | awk '{print $1}')
file_count=$(find "${DST_DATA}" -type f 2>/dev/null | wc -l)
log "final: ${data_mb} MB, ${file_count} files at ${DST_DATA}"
log "ALL DONE total=$(( $(date +%s) - START ))s"

echo "OK $(date +%s)" > "${DONE}"
