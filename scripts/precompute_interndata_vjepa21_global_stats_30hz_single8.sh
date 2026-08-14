#!/usr/bin/env bash
# Compute fixed V-JEPA statistics for the historical InternData 30Hz contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INTERN_A1_TARGET_CONTROL_HZ=30
exec bash "${SCRIPT_DIR}/precompute_interndata_vjepa21_global_stats_single8.sh" "$@"
