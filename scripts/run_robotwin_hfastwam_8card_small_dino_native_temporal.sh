#!/usr/bin/env bash
# RoboTwin DINO native temporal resolution: retain all 9 sampled image states.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_dino_native_t9_ds}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export TEMPORAL_DOWNSAMPLE=1
export STANDARDISE_OUTPUT="${STANDARDISE_OUTPUT:-false}"
export VIDEO_LATENT_CACHE_DIR="${VIDEO_LATENT_CACHE_DIR:-${VIDEO_LATENT_CACHE_ROOT:-${REPO_ROOT}/data/video_latent_cache}/robotwin/dino_native_t9_raw}"
export LATENT_CACHE_BATCH_SIZE="${LATENT_CACHE_BATCH_SIZE:-2}"
exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_dino.sh" "$@"
