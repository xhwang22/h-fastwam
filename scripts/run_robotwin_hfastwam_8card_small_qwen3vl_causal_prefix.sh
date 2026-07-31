#!/usr/bin/env bash
# RoboTwin Qwen3-VL with causal prefix targets anchored at s0, s16, and s32.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_qwen3vl_causal_prefix}"
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_qwen3vl_causal_prefix_ds}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-robotwin-small-qwen3vl-causal-prefix}"
exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
