#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TIMESTEP_SAMPLING_PRESET=middle
exec bash "${SCRIPT_DIR}/run_robotwin_vjepa21_dit_timestep_oldcluster_4x8.sh" "$@"
