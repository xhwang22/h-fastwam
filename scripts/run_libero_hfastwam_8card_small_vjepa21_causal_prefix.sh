#!/usr/bin/env bash
# LIBERO V-JEPA 2.1 with causal prefix targets anchored at s0, s16, and s32.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_vjepa21_causal_prefix}"
export RUN_NAME="${RUN_NAME:-libero_hfastwam_8card_small_vjepa21_causal_prefix_ds}"
exec bash "${SCRIPT_DIR}/run_libero_hfastwam_8card_small_vjepa21.sh" "$@"
