#!/usr/bin/env bash
# Copy RobotWin 2.0 dataset from CephFS to /tmp on the local node, in background.
# Run via launch_all_nodes.sh so it executes on all 4 nodes simultaneously.
# Idempotent: skips files that already exist with matching size.

set -euo pipefail

SRC="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/robotwin2.0"
DST="/tmp/fastwam_data/robotwin2.0"

LOG_DIR="/tmp/copy_robotwin"
mkdir -p "${LOG_DIR}" "${DST}"
LOG_FILE="${LOG_DIR}/copy.log"
DONE_FILE="${LOG_DIR}/done.flag"

# Skip if already finished
if [[ -f "${DONE_FILE}" ]]; then
  echo "[copy_robotwin] Already done. Flag: ${DONE_FILE}"
  exit 0
fi

# Use rsync, transfer the contents of robotwin2.0/ (NOT the .tar.gz parts)
# We only need: dataset_stats.json, README.md, robotwin2.0/ subdir
nohup bash -c "
  set -e
  echo \"[copy_robotwin] start at \$(date)\"
  rsync -a --info=progress2 --no-i-r \\
    --exclude='*.tar.gz.part-*' \\
    \"${SRC}/\" \"${DST}/\" \\
    && touch '${DONE_FILE}' \\
    && echo \"[copy_robotwin] DONE at \$(date)\" \\
    || echo \"[copy_robotwin] FAILED at \$(date)\"
" > "${LOG_FILE}" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || disown || true
echo "${PID}" > "${LOG_DIR}/pid"
echo "[copy_robotwin] background PID=${PID} log=${LOG_FILE}"
