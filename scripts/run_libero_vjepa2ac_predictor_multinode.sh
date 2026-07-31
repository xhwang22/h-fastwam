#!/usr/bin/env bash
# Thin wrapper: V-JEPA 2-AC predictor (W-1) on LIBERO, multi-node.
#
# This wrapper does V-JEPA-specific sanity checks (ckpt presence, hubconf
# URL patch, idle-GPU warning) and then `exec`s the generic multi-node
# launcher at scripts/run_multinode.sh, which handles topology resolution
# (NNODES/NODE_RANK/...) and the actual accelerate launch.
#
# Run this on every node of the Taiji job. Each node already has its own
# INDEX env var injected by the platform.
#
# Usage:
#   bash scripts/run_libero_vjepa2ac_predictor_multinode.sh
#   FOREGROUND=1 bash scripts/run_libero_vjepa2ac_predictor_multinode.sh
#   RUN_NAME=myrun bash scripts/run_libero_vjepa2ac_predictor_multinode.sh
#   EXTRA="model.loss.lambda_action=0.5" \
#       bash scripts/run_libero_vjepa2ac_predictor_multinode.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[w1-multinode] ERROR: $*" >&2; exit 1; }
info() { echo "[w1-multinode] $*"; }

CKPT_BASE="${REPO_ROOT}/checkpoints"
VJEPA2_CKPT="${CKPT_BASE}/vjepa2/vjepa2-ac-vitg.pt"

# Run rank-0-only checks (the launcher itself runs on every node).
NODE_RANK_RESOLVED="${NODE_RANK:-${INDEX:-0}}"
if [[ "${NODE_RANK_RESOLVED}" == "0" ]]; then
  python -c "import timm" >/dev/null 2>&1 \
    || err "timm not installed. Run: pip install timm"

  [[ -f "${VJEPA2_CKPT}" ]] \
    || err "Missing V-JEPA2-AC checkpoint at ${VJEPA2_CKPT}.
       Download: curl -fL --retry 10 -C - -o ${VJEPA2_CKPT} \\
         https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"

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
      info "WARNING: rank-0 GPUs are not idle (${used} MiB total in use). Continuing anyway."
    fi
  fi
fi

# Resolve multi-node topology from Taiji platform env vars.
export NNODES="${NNODES:-${HOST_NUM:-1}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${HOST_GPU_NUM:-8}}"
export NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Derive a shared RUN_NAME if not already set externally.
export RUN_PREFIX="${RUN_PREFIX:-w1mn}"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"

# Hand off to the torchrun-based launcher with W-1-specific env set.
export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_BASE}/"
export MODEL="${MODEL:-fastwam_vjepa2ac_predictor}"
export TASK="${TASK:-libero_uncond_2cam224_1e-4}"
export DATA="${DATA:-libero_2cam}"
export WANDB_NAME="${WANDB_NAME:-libero_vjepa2ac_predictor_w1_mn}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/libero_vjepa2ac_predictor}"

exec bash "${SCRIPT_DIR}/launch_torchrun_multinode.sh"
