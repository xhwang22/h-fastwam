#!/usr/bin/env bash
# Per-node helper: cache the robotwin2.0 dataset + stats to /dev/shm so the
# DINO+RoboTwin training run does not hit FUSE/CephFS during data loading.
#
# Mirrors _local_cache_robotwin.sh but skips the vjepa2 checkpoint (DINO
# DiT-projection does not use V-JEPA).
#
# Caches:
#   /dev/shm/fwam/data/robotwin2.0/dataset_stats.json    (small)
#   /dev/shm/fwam/data/robotwin2.0/robotwin2.0/...       (~75GB, from tar.gz parts)

set -e

SRC_TAR_PARTS_GLOB="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/robotwin2.0.tar.gz.part-*"
SRC_STATS="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/dataset_stats.json"

DST_BASE="/dev/shm/fwam"
DST_DATA_PARENT="${DST_BASE}/data/robotwin2.0"
DST_DATA="${DST_DATA_PARENT}/robotwin2.0"
DST_STATS="${DST_DATA_PARENT}/dataset_stats.json"

LOG="/tmp/local_cache_dino.log"

mkdir -p "${DST_DATA_PARENT}"

: > "${LOG}"
START=$(date +%s)
log() { echo "$(date '+%H:%M:%S') $* (t+$(( $(date +%s) - START ))s)" | tee -a "${LOG}"; }
log "host=$(hostname) starting tar-based cache (DINO; no vjepa2 ckpt)"

# 1) tiny: dataset_stats.json
if [[ ! -s "${DST_STATS}" ]]; then
  log "copy dataset_stats.json"
  cp "${SRC_STATS}" "${DST_STATS}"
fi

# 2) dataset via tar.gz parts — extracts to DST_DATA_PARENT/robotwin2.0/
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
  log "extract dataset from tar.gz parts (sequential stream)"
  cat ${SRC_TAR_PARTS_GLOB} | tar -xzf - -C "${DST_DATA_PARENT}/" 2>>"${LOG}"
  log "extract dataset done"
fi

data_mb=$(du -sm "${DST_DATA}" 2>/dev/null | awk '{print $1}')
file_count=$(find "${DST_DATA}" -type f 2>/dev/null | wc -l)
log "final: ${data_mb} MB, ${file_count} files at ${DST_DATA}"
log "ALL DONE total=$(( $(date +%s) - START ))s"
