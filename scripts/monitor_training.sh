#!/usr/bin/env bash
# =============================================================================
# Training Monitor Script
# Checks all nodes, detects crashes, reports progress.
# Usage: bash scripts/monitor_training.sh [pretrain_run_id] [finetune_run_id]
# =============================================================================

REPO_ROOT="/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM"
PRETRAIN_LOG_ROOT="${REPO_ROOT}/runs/pretrain_dino_ditproj"
FINETUNE_LOG_ROOT="${REPO_ROOT}/runs/robotwin_dino_ditproj"

NODES=(28.216.19.198 28.216.19.213 28.216.19.159 28.216.18.220)
NODE_NAMES=(rank0 rank1 rank2 rank3)

CHECK_INTERVAL=60   # seconds between checks

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*" >&2; }
err()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

# Find the most recent pretrain run log on each node
find_latest_log() {
  local node_idx="$1"
  local log_root="$2"
  local rank="$3"
  ls -t "${log_root}"/*/train.log.rank${rank} 2>/dev/null | head -1
}

# Count training processes on a node (local or remote)
count_procs() {
  local host="$1"
  if [[ "${host}" == "28.216.19.198" ]] || [[ "${host}" == "$(hostname -I | awk '{print $1}')" ]]; then
    ps aux | grep -E "train\.py|accelerate launch" | grep -v grep | wc -l
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${host}" \
      "ps aux | grep -E 'train\.py|accelerate launch' | grep -v grep | wc -l" 2>/dev/null || echo "0"
  fi
}

# Get last training step from log
get_last_step() {
  local logfile="$1"
  grep -oE "step[_ ]+[0-9]+" "${logfile}" 2>/dev/null | tail -1 || \
  grep -oE "\[ *[0-9]+/[0-9]+ *\]" "${logfile}" 2>/dev/null | tail -1 || \
  echo "N/A"
}

# Get last loss from log
get_last_loss() {
  local logfile="$1"
  grep -oE "loss[= :]+[0-9]+\.[0-9]+" "${logfile}" 2>/dev/null | tail -1 || \
  grep -oE "train_loss[= :]+[0-9]+\.[0-9]+" "${logfile}" 2>/dev/null | tail -1 || \
  echo "N/A"
}

# Check if log has been updated recently (within N seconds)
log_is_active() {
  local logfile="$1"
  local max_age="${2:-300}"   # 5 min default
  if [[ ! -f "${logfile}" ]]; then return 1; fi
  local mtime
  mtime=$(stat -c %Y "${logfile}" 2>/dev/null || echo 0)
  local now
  now=$(date +%s)
  (( now - mtime < max_age ))
}

# ─── Main monitor loop ────────────────────────────────────────────────────────
log "Starting training monitor. Checking every ${CHECK_INTERVAL}s."
log "Nodes: ${NODES[*]}"

PREV_LOG_LINES=0
STALL_COUNT=0
MAX_STALL=5    # alert after 5 consecutive no-progress checks

while true; do
  echo ""
  log "======== Health Check ========"

  # ── 1. Find current phase (pretrain vs finetune) ──────────────────────────
  CURRENT_PHASE="none"
  CURRENT_RUN_DIR=""

  # Check for active pretrain
  for dir in $(ls -t "${PRETRAIN_LOG_ROOT}/" 2>/dev/null); do
    logfile="${PRETRAIN_LOG_ROOT}/${dir}/train.log.rank0"
    if [[ -f "${logfile}" ]] && log_is_active "${logfile}" 600; then
      CURRENT_PHASE="pretrain"
      CURRENT_RUN_DIR="${PRETRAIN_LOG_ROOT}/${dir}"
      break
    fi
  done

  # Check for active finetune
  if [[ "${CURRENT_PHASE}" == "none" ]]; then
    for dir in $(ls -t "${FINETUNE_LOG_ROOT}/" 2>/dev/null); do
      logfile="${FINETUNE_LOG_ROOT}/${dir}/train.log.rank0"
      if [[ -f "${logfile}" ]] && log_is_active "${logfile}" 600; then
        CURRENT_PHASE="finetune"
        CURRENT_RUN_DIR="${FINETUNE_LOG_ROOT}/${dir}"
        break
      fi
    done
  fi

  log "Phase: ${CURRENT_PHASE}  |  Run dir: ${CURRENT_RUN_DIR:-N/A}"

  if [[ "${CURRENT_PHASE}" == "none" ]]; then
    warn "No active training found. Checking if training completed or crashed..."
    # Check if checkpoint exists (completed)
    latest_ckpt=$(ls -t "${PRETRAIN_LOG_ROOT}"/*/checkpoints/weights/*.pt 2>/dev/null | head -1)
    if [[ -n "${latest_ckpt}" ]]; then
      log "Found pretrain checkpoint: ${latest_ckpt}"
      # Check if finetune started/completed
      latest_ft=$(ls -t "${FINETUNE_LOG_ROOT}"/*/train.log.rank0 2>/dev/null | head -1)
      if [[ -n "${latest_ft}" ]]; then
        log "Finetune log found: ${latest_ft}"
      fi
    fi
    log "Sleeping ${CHECK_INTERVAL}s before next check..."
    sleep "${CHECK_INTERVAL}"
    continue
  fi

  # ── 2. Check rank0 log progress ───────────────────────────────────────────
  RANK0_LOG="${CURRENT_RUN_DIR}/train.log.rank0"
  CURRENT_LINES=$(wc -l < "${RANK0_LOG}" 2>/dev/null || echo 0)
  LAST_STEP=$(get_last_step "${RANK0_LOG}")
  LAST_LOSS=$(get_last_loss "${RANK0_LOG}")
  LOG_AGE=$(( $(date +%s) - $(stat -c %Y "${RANK0_LOG}" 2>/dev/null || echo 0) ))

  log "rank0 log: ${CURRENT_LINES} lines | age: ${LOG_AGE}s | step: ${LAST_STEP} | loss: ${LAST_LOSS}"
  log "rank0 tail:"
  tail -3 "${RANK0_LOG}" 2>/dev/null | sed 's/^/  /'

  # Check for stall (no new log lines)
  if (( CURRENT_LINES <= PREV_LOG_LINES )); then
    STALL_COUNT=$(( STALL_COUNT + 1 ))
    warn "No log progress for ${STALL_COUNT} consecutive checks (${STALL_COUNT}×${CHECK_INTERVAL}s = $((STALL_COUNT*CHECK_INTERVAL))s stall)"
    if (( STALL_COUNT >= MAX_STALL )); then
      err "Log stalled for $(( STALL_COUNT * CHECK_INTERVAL ))s! Training may be hung."
    fi
  else
    if (( STALL_COUNT > 0 )); then
      log "Log progressing again after ${STALL_COUNT} stall checks."
    fi
    STALL_COUNT=0
  fi
  PREV_LOG_LINES="${CURRENT_LINES}"

  # ── 3. Check errors in log ────────────────────────────────────────────────
  ERRORS=$(grep -i "error\|traceback\|exception\|killed\|oom\|out of memory\|nccl\|timeout" \
    "${RANK0_LOG}" 2>/dev/null | grep -vi "no error\|info.*error\|WARNING" | tail -3)
  if [[ -n "${ERRORS}" ]]; then
    warn "Potential errors in rank0 log:"
    echo "${ERRORS}" | sed 's/^/  /'
  fi

  # ── 4. Check process count per node ───────────────────────────────────────
  log "Process counts per node:"
  for i in 0 1 2 3; do
    host="${NODES[$i]}"
    procs=$(count_procs "${host}")
    if (( procs == 0 )); then
      warn "  ${host} (rank${i}): NO TRAINING PROCESSES FOUND!"
    else
      log "  ${host} (rank${i}): ${procs} processes"
    fi
  done

  # ── 5. Show GPU utilization on rank0 ──────────────────────────────────────
  GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | awk '{sum_util+=$1; sum_mem+=$2; total_mem+=$3} END {printf "avg_util=%d%% mem=%dMiB/%dMiB", sum_util/NR, sum_mem, total_mem}')
  log "rank0 GPU: ${GPU_INFO}"

  # ── 6. Check checkpoints ──────────────────────────────────────────────────
  CKPT_COUNT=$(ls "${CURRENT_RUN_DIR}/checkpoints/weights/"*.pt 2>/dev/null | wc -l)
  if (( CKPT_COUNT > 0 )); then
    LATEST_CKPT=$(ls -t "${CURRENT_RUN_DIR}/checkpoints/weights/"*.pt 2>/dev/null | head -1)
    log "Checkpoints saved: ${CKPT_COUNT}  |  Latest: $(basename ${LATEST_CKPT})"
  else
    log "No checkpoints saved yet"
  fi

  log "Next check in ${CHECK_INTERVAL}s..."
  sleep "${CHECK_INTERVAL}"
done
