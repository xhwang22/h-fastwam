#!/usr/bin/env bash
# Launch FastWAM V-JEPA 2-AC predictor training on LIBERO (8-GPU, ZeRO-1).
#
# This is the W-1 design: V-JEPA-AC predictor 24L/16H/64dim wrapped in
# Wan-style MoT blocks, deterministic video loss, ActionDiT 24L/16H/64dim
# from scratch.
#
# Usage:
#   bash scripts/run_libero_vjepa2ac_predictor.sh                    # default 8 GPU, sane defaults
#   NUM_GPUS=4 bash scripts/run_libero_vjepa2ac_predictor.sh         # override GPU count
#   FOREGROUND=1 bash scripts/run_libero_vjepa2ac_predictor.sh       # run in foreground
#   RUN_NAME=myrun bash scripts/run_libero_vjepa2ac_predictor.sh     # custom run dir name
#   EXTRA="model.loss.lambda_action=0.5" bash scripts/run_libero_vjepa2ac_predictor.sh
#                                                                     # forward extra hydra overrides

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-w1_$(date +%Y-%m-%d_%H-%M-%S)}"
FOREGROUND="${FOREGROUND:-0}"
EXTRA="${EXTRA:-}"

TJ5_BASE="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints"
VJEPA2_CKPT="${TJ5_BASE}/vjepa2/vjepa2-ac-vitg.pt"

LOG_DIR="${REPO_ROOT}/runs/libero_vjepa2ac_predictor/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"

err()  { echo "[run_libero_vjepa2ac_predictor] ERROR: $*" >&2; exit 1; }
info() { echo "[run_libero_vjepa2ac_predictor] $*"; }

# 1. timm available
python -c "import timm" >/dev/null 2>&1 \
  || err "timm not installed. Run: pip install timm"

# 2. V-JEPA-AC ckpt present
[[ -f "${VJEPA2_CKPT}" ]] \
  || err "Missing V-JEPA2-AC checkpoint at ${VJEPA2_CKPT}.
       Download: curl -fL --retry 10 -C - -o ${VJEPA2_CKPT} \\
         https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"

# 3. patch vjepa2 hubconf URL if upstream cache still has the localhost stub
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

# 4. GPUs idle warning
if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if (( used > 1024 )); then
    info "WARNING: GPUs are not idle (${used} MiB total in use). Continuing anyway."
  fi
fi

mkdir -p "${LOG_DIR}"
info "REPO_ROOT     = ${REPO_ROOT}"
info "NUM_GPUS      = ${NUM_GPUS}"
info "RUN_NAME      = ${RUN_NAME}"
info "LOG_FILE      = ${LOG_FILE}"
info "FOREGROUND    = ${FOREGROUND}"
[[ -n "${EXTRA}" ]] && info "EXTRA hydra   = ${EXTRA}"

HYDRA_OVERRIDES=(
  "task=libero_uncond_2cam224_1e-4"
  "data=libero_2cam"
  "model=fastwam_vjepa2ac_predictor"
  "output_dir=${LOG_DIR}"
  "wandb.name=libero_vjepa2ac_predictor_w1"
)
if [[ -n "${EXTRA}" ]]; then
  # shellcheck disable=SC2206
  HYDRA_OVERRIDES+=( ${EXTRA} )
fi

COMMON_ENV=(
  "DIFFSYNTH_MODEL_BASE_PATH=${TJ5_BASE}/"
  "NUM_GPUS=${NUM_GPUS}"
  "RUN_ID=${RUN_NAME}"
)

if [[ "${FOREGROUND}" == "1" ]]; then
  info "Running in foreground."
  env "${COMMON_ENV[@]}" bash scripts/train_zero1.sh "${NUM_GPUS}" "${HYDRA_OVERRIDES[@]}" \
    2>&1 | tee "${LOG_FILE}"
else
  nohup setsid env "${COMMON_ENV[@]}" \
    bash scripts/train_zero1.sh "${NUM_GPUS}" "${HYDRA_OVERRIDES[@]}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
  PID=$!
  disown || true
  echo "${PID}" > "${LOG_DIR}/.launcher.pid"
  info "Launched in background, PID=${PID}"
  info "Tail the log:  tail -f ${LOG_FILE}"
  info "Stop the run:  pkill -TERM -f scripts/train.py && pkill -TERM -f train_zero1"
fi
