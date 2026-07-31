#!/usr/bin/env bash
# RoboTwin Qwen3-VL native temporal resolution: retain all 5 video tubelets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_siglip2}"
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_qwen3vl_native_t5_ds}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-robotwin-small-qwen3vl-native-t5}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export TEMPORAL_DOWNSAMPLE=2
export STANDARDISE_OUTPUT="${STANDARDISE_OUTPUT:-false}"
export VIDEO_LATENT_CACHE_DIR="${VIDEO_LATENT_CACHE_DIR:-${VIDEO_LATENT_CACHE_ROOT:-${REPO_ROOT}/data/video_latent_cache}/robotwin/qwen3vl_native_t5_raw}"
export LATENT_CACHE_BATCH_SIZE="${LATENT_CACHE_BATCH_SIZE:-1}"
exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
