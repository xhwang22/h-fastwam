#!/usr/bin/env bash
# Per-node helper v2: fetch dataset via tar.gz parts (avoids per-file FUSE metadata cost).
# Streams `cat parts | tar -xzf -` so the source-side reads sequentially.
#
# Caches:
#   /dev/shm/fwam/ckpt/vjepa2/vjepa2-ac-vitg.pt          (11GB)
#   /dev/shm/fwam/data/robotwin2.0/dataset_stats.json    (small)
#   /dev/shm/fwam/data/robotwin2.0/robotwin2.0/...       (75GB, from tar.gz parts)

set -e

SRC_TAR_PARTS_GLOB="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/robotwin2.0.tar.gz.part-*"
SRC_STATS="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/dataset_stats.json"
SRC_CKPT="${CKPT_BASE}/vjepa2/vjepa2-ac-vitg.pt"  # set CKPT_BASE or place file here

DST_BASE="/dev/shm/fwam"
DST_DATA_PARENT="${DST_BASE}/data/robotwin2.0"
DST_DATA="${DST_DATA_PARENT}/robotwin2.0"
DST_STATS="${DST_DATA_PARENT}/dataset_stats.json"
DST_CKPT_DIR="${DST_BASE}/ckpt/vjepa2"
DST_CKPT="${DST_CKPT_DIR}/vjepa2-ac-vitg.pt"

LOG="/tmp/local_cache.log"
DONE="/tmp/local_cache.done"

mkdir -p "${DST_DATA_PARENT}" "${DST_CKPT_DIR}"

: > "${LOG}"
START=$(date +%s)
log() { echo "$(date '+%H:%M:%S') $* (t+$(( $(date +%s) - START ))s)" >> "${LOG}"; }
log "host=$(hostname) starting tar-based cache"

# 1) tiny: dataset_stats.json
if [[ ! -s "${DST_STATS}" ]]; then
  log "copy dataset_stats.json"
  cp "${SRC_STATS}" "${DST_STATS}"
fi

# 2) ckpt (11GB) — skip if size matches
NEED_CKPT=1
if [[ -s "${DST_CKPT}" ]]; then
  src_size=$(stat -c %s "${SRC_CKPT}" 2>/dev/null || echo 0)
  dst_size=$(stat -c %s "${DST_CKPT}" 2>/dev/null || echo 0)
  if [[ "${src_size}" == "${dst_size}" && "${src_size}" -gt 0 ]]; then
    NEED_CKPT=0
    log "ckpt already cached (${dst_size} bytes), skipping"
  fi
fi
if [[ "${NEED_CKPT}" == "1" ]]; then
  log "copy vjepa2-ac-vitg.pt start"
  cp "${SRC_CKPT}" "${DST_CKPT}.tmp"
  mv "${DST_CKPT}.tmp" "${DST_CKPT}"
  log "copy vjepa2-ac-vitg.pt done"
fi

# 3) dataset via tar.gz parts — extracts to DST_DATA_PARENT/robotwin2.0/
NEED_DATA=1
if [[ -d "${DST_DATA}/videos/chunk-027" && -d "${DST_DATA}/data/chunk-027" ]]; then
  # Final chunk present, assume complete
  cur_mb=$(du -sm "${DST_DATA}" 2>/dev/null | awk '{print $1}')
  if [[ "${cur_mb:-0}" -gt 70000 ]]; then
    NEED_DATA=0
    log "dataset already cached (${cur_mb} MB), skipping"
  fi
fi

if [[ "${NEED_DATA}" == "1" ]]; then
  # Wipe partial dst
  rm -rf "${DST_DATA}"
  log "extract dataset from tar.gz parts (sequential stream)"
  # Order matters; sort lexically
  cat ${SRC_TAR_PARTS_GLOB} | tar -xzf - -C "${DST_DATA_PARENT}/" 2>>"${LOG}"
  log "extract dataset done"
fi

data_mb=$(du -sm "${DST_DATA}" 2>/dev/null | awk '{print $1}')
file_count=$(find "${DST_DATA}" -type f 2>/dev/null | wc -l)
log "final: ${data_mb} MB, ${file_count} files at ${DST_DATA}"
log "ALL DONE total=$(( $(date +%s) - START ))s"

echo "OK $(date +%s)" > "${DONE}"
