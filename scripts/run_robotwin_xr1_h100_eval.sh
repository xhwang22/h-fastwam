#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_KIND=xr1
exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
