#!/usr/bin/env bash
# Launch FastWAM V-JEPA2-AC training on LIBERO (8-GPU, ZeRO-1).
#
# Usage:
#   bash scripts/run_libero_vjepa2ac.sh                 # default 8 GPU, sane defaults
#   NUM_GPUS=4 bash scripts/run_libero_vjepa2ac.sh      # override GPU count
#   FOREGROUND=1 bash scripts/run_libero_vjepa2ac.sh    # run in foreground (don't detach)
#   RUN_NAME=myrun bash scripts/run_libero_vjepa2ac.sh  # custom run dir name
#   EXTRA="model.loss.lambda_action=0.5" bash scripts/run_libero_vjepa2ac.sh
#                                                       # forward extra hydra overrides
#
# What this does (encapsulates everything we set up on 2026-05-10):
#   1. Verifies prereqs: timm installed, vjepa2 hubconf URL patched,
#      vjepa2-ac-vitg.pt present on tj5, GPUs idle.
#   2. Exports DIFFSYNTH_MODEL_BASE_PATH so HF model loads stay on tj5
#      (cross-cluster reads cripple multi-rank training).
#   3. Launches scripts/train_zero1.sh with model=fastwam_vjepa2ac_ditproj.
#   4. By default detaches via nohup+setsid so the job survives shell exit;
#      set FOREGROUND=1 to attach.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (this script lives in scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-wan_init_$(date +%Y-%m-%d_%H-%M-%S)}"
FOREGROUND="${FOREGROUND:-0}"
EXTRA="${EXTRA:-}"

# Cluster paths — keep big artifacts under /apdcephfs_tj5 because the compute
# nodes live on the tj cluster (cross-cluster cephfs reads make 8-rank
# torch.load stall in D-state for hours).
TJ5_BASE="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints"
ACTION_DIT_CKPT="${TJ5_BASE}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
VJEPA2_CKPT="${TJ5_BASE}/vjepa2/vjepa2-ac-vitg.pt"

LOG_DIR="${REPO_ROOT}/runs/libero_vjepa2ac_ditproj/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"

# ---------------------------------------------------------------------------
# Pre-flight checks (fail fast, with clear remediation hints)
# ---------------------------------------------------------------------------
err() { echo "[run_libero_vjepa2ac] ERROR: $*" >&2; exit 1; }
info() { echo "[run_libero_vjepa2ac] $*"; }

# 1. timm available in the fastwam env (vjepa2 hub entry depends on it).
python -c "import timm" >/dev/null 2>&1 \
  || err "timm not installed. Run: pip install timm"

# 2. vjepa2-ac-vitg.pt present on tj5 (model config points here).
[[ -f "${VJEPA2_CKPT}" ]] \
  || err "Missing V-JEPA2-AC checkpoint at ${VJEPA2_CKPT}.
       Download with: curl -fL --retry 10 -C - -o ${VJEPA2_CKPT} \\
         https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
       (~11GB; place on tj5, not gy2.)"

# 3. ActionDiT pretrain ckpt exists.
[[ -f "${ACTION_DIT_CKPT}" ]] \
  || err "Missing ActionDiT pretrain at ${ACTION_DIT_CKPT}"

# 4. vjepa2 hubconf URL patched (upstream main branch ships localhost:8300).
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

# 5. GPUs idle (warn, don't block — user may know what they're doing).
if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if (( used > 1024 )); then
    info "WARNING: GPUs are not idle (${used} MiB total in use). Continuing anyway."
  fi
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
info "REPO_ROOT     = ${REPO_ROOT}"
info "NUM_GPUS      = ${NUM_GPUS}"
info "RUN_NAME      = ${RUN_NAME}"
info "LOG_FILE      = ${LOG_FILE}"
info "FOREGROUND    = ${FOREGROUND}"
[[ -n "${EXTRA}" ]] && info "EXTRA hydra   = ${EXTRA}"

# Hydra overrides — match the working command from 2026-05-10.
HYDRA_OVERRIDES=(
  "task=libero_uncond_2cam224_1e-4"
  "data=libero_2cam"
  "model=fastwam_vjepa2ac_ditproj"
  "output_dir=${LOG_DIR}"
  "wandb.name=libero_vjepa2ac_ditproj_waninit"
  "model.action_dit_pretrained_path=${ACTION_DIT_CKPT}"
)
# Append user-provided EXTRA (space-separated key=value tokens).
if [[ -n "${EXTRA}" ]]; then
  # shellcheck disable=SC2206
  HYDRA_OVERRIDES+=( ${EXTRA} )
fi

# Env vars consumed by train_zero1.sh / hydra / training stack.
COMMON_ENV=(
  "DIFFSYNTH_MODEL_BASE_PATH=${TJ5_BASE}/"
  "NUM_GPUS=${NUM_GPUS}"
  "RUN_ID=${RUN_NAME}"
)

if [[ "${FOREGROUND}" == "1" ]]; then
  info "Running in foreground (Ctrl+C to abort)."
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
  info "Find workers:  ps -ef | grep scripts/train.py | grep -v grep"
  info "Stop the run:  pkill -TERM -f scripts/train.py && pkill -TERM -f train_zero1"
fi
