#!/usr/bin/env bash
# Multi-node training: V-JEPA2-AC predictor on RoboTwin.
#
# Reads NODE_IP_LIST (format: "ip1:ngpu,ip2:ngpu,...") to resolve topology.
# Run this script on EVERY node.
#
# Usage:
#   NODE_IP_LIST="1.2.3.4:8,1.2.3.5:8" bash scripts/run_robotwin_vjepa2ac_predictor_multinode.sh
#   FOREGROUND=1 NODE_IP_LIST=... bash scripts/run_robotwin_vjepa2ac_predictor_multinode.sh
#   EXTRA="max_steps=50000" NODE_IP_LIST=... bash scripts/run_robotwin_vjepa2ac_predictor_multinode.sh

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[robotwin-vjepa2-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[robotwin-vjepa2-mn] $*"; }

# ---------------------------------------------------------------------------
# Resolve topology from NODE_IP_LIST env var (format: ip1:ngpu,ip2:ngpu,...)
# ---------------------------------------------------------------------------
: "${NODE_IP_LIST:?NODE_IP_LIST must be set, e.g. 1.2.3.4:8,1.2.3.5:8}"

IFS=',' read -ra NODE_LIST <<< "${NODE_IP_LIST}"
NNODES="${#NODE_LIST[@]}"

declare -a NODE_IPS=()
declare -a NODE_GPUS=()
for entry in "${NODE_LIST[@]}"; do
  NODE_IPS+=("${entry%%:*}")
  NODE_GPUS+=("${entry##*:}")
done

MASTER_ADDR="${MASTER_ADDR:-${NODE_IPS[0]}}"
MASTER_PORT="${MASTER_PORT:-29501}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NODE_GPUS[0]}}"

# Determine current node rank by matching local IP
LOCAL_IPS=$(hostname -I 2>/dev/null || true)
NODE_RANK="${NODE_RANK:-}"
if [[ -z "${NODE_RANK}" ]]; then
  for i in "${!NODE_IPS[@]}"; do
    for lip in ${LOCAL_IPS}; do
      if [[ "${lip}" == "${NODE_IPS[$i]}" ]]; then
        NODE_RANK="${i}"
        break 2
      fi
    done
  done
fi
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"

info "Detected topology:"
info "  NNODES         = ${NNODES}"
info "  NODE_RANK      = ${NODE_RANK}"
info "  MASTER_ADDR    = ${MASTER_ADDR}"
info "  MASTER_PORT    = ${MASTER_PORT}"
info "  NPROC_PER_NODE = ${NPROC_PER_NODE}"
info "  TOTAL_GPUS     = $(( NNODES * NPROC_PER_NODE ))"

CKPT_BASE="${REPO_ROOT}/checkpoints"
VJEPA2_CKPT="${CKPT_BASE}/vjepa2/vjepa2-ac-vitg.pt"

# Rank-0-only sanity checks
if [[ "${NODE_RANK}" == "0" ]]; then
  python -c "import timm" >/dev/null 2>&1 \
    || err "timm not installed. Run: pip install timm"

  [[ -f "${VJEPA2_CKPT}" ]] \
    || err "Missing V-JEPA2-AC checkpoint at ${VJEPA2_CKPT}"

  HUBCONF="${HOME}/.cache/torch/hub/facebookresearch_vjepa2_main/src/hub/backbones.py"
  if [[ -f "${HUBCONF}" ]] && grep -q '^VJEPA_BASE_URL = "http://localhost:8300"' "${HUBCONF}"; then
    info "Patching vjepa2 hubconf URL (localhost:8300 -> fbaipublicfiles)"
    python - <<PY
import pathlib
p = pathlib.Path("${HUBCONF}")
src = p.read_text()
src = src.replace(
    '# VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"\n\n# for testing\nVJEPA_BASE_URL = "http://localhost:8300"',
    'VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"\n\n# for testing (disabled)\n# VJEPA_BASE_URL = "http://localhost:8300"',
)
p.write_text(src)
PY
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
    if (( used > 1024 )); then
      info "WARNING: rank-0 GPUs not idle (${used} MiB in use). Continuing anyway."
    fi
  fi
fi

export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_BASE}/"
export MODEL="${MODEL:-fastwam_vjepa2ac_predictor}"
export TASK="${TASK:-robotwin_uncond_3cam_384_1e-4}"
export DATA="${DATA:-robotwin}"
export RUN_PREFIX="${RUN_PREFIX:-robotwin_vjepa2_mn}"
export WANDB_NAME="${WANDB_NAME:-robotwin_vjepa2ac_predictor_mn}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/robotwin_vjepa2ac_predictor}"
export FOREGROUND="${FOREGROUND:-0}"
export EXTRA="${EXTRA:-}"

export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"

exec bash "${SCRIPT_DIR}/launch_torchrun_multinode.sh"
