#!/usr/bin/env bash
# Parallel copy RobotWin 2.0 dataset from CephFS to /tmp on local node.
# Uses per-chunk parallelism (one rsync per chunk) to bypass FUSE single-thread
# stat bottleneck. Idempotent via per-chunk done flags.
#
# Run via launch_all_nodes.sh to execute on all 4 nodes simultaneously.

set -euo pipefail

SRC="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/robotwin2.0"
DST="/tmp/fastwam_data/robotwin2.0/robotwin2.0"
PARALLEL="${PARALLEL:-16}"      # concurrent chunks to copy

LOG_DIR="/tmp/copy_robotwin"
mkdir -p "${LOG_DIR}" "${DST}"
LOG_FILE="${LOG_DIR}/copy_parallel.log"
DONE_FILE="${LOG_DIR}/done.flag"

if [[ -f "${DONE_FILE}" ]]; then
  echo "[copy_parallel] already done"
  exit 0
fi

nohup bash -c '
  set -e
  SRC="'"${SRC}"'"
  DST="'"${DST}"'"
  LOG_DIR="'"${LOG_DIR}"'"
  PARALLEL='"${PARALLEL}"'

  echo "[copy_parallel] start $(date)"
  mkdir -p "${DST}"

  # 1. Copy top-level small files (README, stats json) + meta/ first (small, blocking)
  echo "[copy_parallel] phase 1: top-level + meta/"
  mkdir -p "${DST}/meta"
  cp -n "'"${SRC}"'"/README.md "${DST}/" 2>/dev/null || true
  # dataset_stats.json lives one level above (in /apdcephfs.../robotwin2.0/)
  cp -n /apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0/dataset_stats.json \
        "${DST}/../dataset_stats.json" 2>/dev/null || true
  rsync -a "${SRC}/meta/" "${DST}/meta/"

  # 2. Launch per-chunk parallel rsyncs for data/ and videos/
  echo "[copy_parallel] phase 2: parallel chunks ($PARALLEL concurrency)"
  mkdir -p "${DST}/data" "${DST}/videos"

  chunks=$(ls "${SRC}/videos" | sort)
  running=0
  for chunk in $chunks; do
    (
      if [[ ! -f "${LOG_DIR}/chunk_${chunk}.done" ]]; then
        mkdir -p "${DST}/data/${chunk}" "${DST}/videos/${chunk}"
        rsync -a "${SRC}/data/${chunk}/" "${DST}/data/${chunk}/" 2>&1
        rsync -a "${SRC}/videos/${chunk}/" "${DST}/videos/${chunk}/" 2>&1
        touch "${LOG_DIR}/chunk_${chunk}.done"
        echo "[copy_parallel] chunk ${chunk} done at $(date +%H:%M:%S)"
      fi
    ) &
    running=$((running + 1))
    if (( running >= PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
  wait

  echo "[copy_parallel] all chunks done"
  touch "'"${DONE_FILE}"'"
  echo "[copy_parallel] DONE $(date)"
' > "${LOG_FILE}" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || disown || true
echo "${PID}" > "${LOG_DIR}/pid_parallel"
echo "[copy_parallel] background PID=${PID} log=${LOG_FILE}"
