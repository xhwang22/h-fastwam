#!/usr/bin/env bash
# Qwen3-VL Flow DiT with three independent causal tubelet states.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_siglip2}"
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_qwen3vl_causal_tubelet_t3_ds}"
export LAUNCH_LABEL="${LAUNCH_LABEL:-robotwin-small-qwen3vl-causal-tubelet-t3}"
export CAUSAL_TUBELET_ENCODING=true
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_siglip2.sh" "$@"
