#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TIMESTEP_SAMPLING_PRESET=baseline
exec bash "${SCRIPT_DIR}/run_robotwin_vjepa21_dit_timestep_oldcluster_2x8.sh" "$@"
