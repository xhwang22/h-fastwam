#!/usr/bin/env bash
# LIBERO V-JEPA 2.1 Predictor with 5 independent causal tubelet states.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset FRAME_GAP
export RUN_NAME="${RUN_NAME:-libero_hfastwam_8card_small_vjepa21_predictor_native_t5_ds}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export TEMPORAL_DOWNSAMPLE=2
export VIDEO_LATENT_CACHE_DIR="${VIDEO_LATENT_CACHE_DIR:-${VIDEO_LATENT_CACHE_ROOT:-${REPO_ROOT}/data/video_latent_cache}/libero/vjepa21_predictor_native_t5_causal}"
export LATENT_CACHE_BATCH_SIZE="${LATENT_CACHE_BATCH_SIZE:-1}"
exec bash "${SCRIPT_DIR}/run_libero_hfastwam_8card_small_vjepa21_predictor.sh" "$@"
