#!/usr/bin/env bash
# Multi-node launcher: H-FastWAM on LIBERO 2-camera, **V-JEPA 2 ViT-L encoder**
# variant. Companion to scripts/run_libero_hfastwam_multinode.sh (DINO baseline).
#
# Defaults to the smallest registered V-JEPA encoder:
#   facebook/vjepa2-vitl-fpc64-256  (ViT-L/16, hidden_dim=1024)
# loaded via HuggingFace AutoModel (no torch.hub plumbing needed).
#
# Compared to the DINO script, this wrapper:
#   1. Prepends V-JEPA encoder overrides to EXTRA so they win over the
#      `model.visual_encoder*=null` defaults in run_libero_hfastwam_multinode.sh,
#      while still letting the user's EXTRA (passed in via env) override them.
#   2. Execs run_libero_hfastwam_multinode.sh — all topology resolution,
#      data caching, action-DiT ckpt resolution etc. is reused as-is.
#
# Encoder defaults (override via env if needed):
#   JEPA_MODEL_NAME=facebook/vjepa2-vitl-fpc64-256   # registered: vitl (1024) / vith (1280)
#   JEPA_HIDDEN_DIM=1024                              # must match the encoder's hidden_dim
#
# Resulting hydra overrides:
#   model.visual_encoder_config={
#     encoder_type: vjepa2,
#     model_name: ${JEPA_MODEL_NAME},
#     skip_projection: true,
#     freeze_backbone: true,
#     spatial_downsample: 16,
#     temporal_downsample: 4,
#     standardise_output: false,
#   }
#   model.video_dit_config.in_dim=${JEPA_HIDDEN_DIM}
#   model.video_dit_config.out_dim=${JEPA_HIDDEN_DIM}
#
# Usage (mirrors the DINO script):
#   bash scripts/run_libero_hfastwam_vjepa2_multinode.sh
#   FOREGROUND=1 bash scripts/run_libero_hfastwam_vjepa2_multinode.sh
#   RUN_NAME=myrun bash scripts/run_libero_hfastwam_vjepa2_multinode.sh
#   EXTRA="num_epochs=3 ..." bash scripts/run_libero_hfastwam_vjepa2_multinode.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[hfastwam-vjepa2-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[hfastwam-vjepa2-mn] $*"; }

JEPA_MODEL_NAME="${JEPA_MODEL_NAME:-facebook/vjepa2-vitl-fpc64-256}"
JEPA_HIDDEN_DIM="${JEPA_HIDDEN_DIM:-1024}"

case "${JEPA_MODEL_NAME}" in
  facebook/vjepa2-vitl-fpc64-256) expected_dim=1024 ;;
  facebook/vjepa2-vith-fpc64-256) expected_dim=1280 ;;
  *) expected_dim="" ;;
esac
if [[ -n "${expected_dim}" && "${JEPA_HIDDEN_DIM}" != "${expected_dim}" ]]; then
  err "JEPA_HIDDEN_DIM=${JEPA_HIDDEN_DIM} does not match expected ${expected_dim} for ${JEPA_MODEL_NAME}."
fi

# Build the encoder-swap EXTRA. Hydra applies overrides in order, so anything
# in the user's EXTRA (appended after) still wins.
VJEPA_EXTRA=(
  "model.visual_encoder_config={encoder_type:vjepa2,model_name:${JEPA_MODEL_NAME},skip_projection:true,freeze_backbone:true,spatial_downsample:16,temporal_downsample:4,standardise_output:false}"
  "model.video_dit_config.in_dim=${JEPA_HIDDEN_DIM}"
  "model.video_dit_config.out_dim=${JEPA_HIDDEN_DIM}"
)

USER_EXTRA="${EXTRA:-}"
printf -v VJEPA_EXTRA_JOINED '%s ' "${VJEPA_EXTRA[@]}"
export EXTRA="${VJEPA_EXTRA_JOINED}${USER_EXTRA}"

# Default RUN_NAME mirrors the DINO convention (e.g. "vjepa2l" for ViT-L).
short=""
case "${JEPA_MODEL_NAME}" in
  *vitl*) short="vjepa2l" ;;
  *vith*) short="vjepa2h" ;;
  *)       short="vjepa2"  ;;
esac
export RUN_PREFIX="${RUN_PREFIX:-libero_hfastwam_${short}_mn}"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
export WANDB_NAME="${WANDB_NAME:-libero_hfastwam_${short}_mn}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/libero_hfastwam_${short}}"

info "JEPA_MODEL_NAME = ${JEPA_MODEL_NAME}  (hidden_dim=${JEPA_HIDDEN_DIM})"
info "EXTRA = ${EXTRA}"

exec bash "${SCRIPT_DIR}/run_libero_hfastwam_multinode.sh"
