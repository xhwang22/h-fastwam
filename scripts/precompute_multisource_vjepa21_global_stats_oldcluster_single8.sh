#!/usr/bin/env bash
# Old non-AWS cluster: fixed V-JEPA statistics for the multisource mixture.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INTERN_A1_ROOT="${INTERN_A1_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/InternData-A1_LeRobotv3}"
export AGIBOT_WORLD_ROOT="${AGIBOT_WORLD_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/AgiBot-World-Beta_lerobotv3}"
export DROID_ROOT="${DROID_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/DROID_LeRobot-v3.0}"
export OPEN_X_ROOT="${OPEN_X_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/Open-X-Embodiment}"
export ROBOCOIN_ROOT="${ROBOCOIN_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/RoboCOIN_v3.0_official}"
export GALAXEA_ROOT="${GALAXEA_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain/Galaxea-Open-World-Dataset-LeRobot-v3.0}"

MANIFEST_ROOT="${MULTISOURCE_MANIFEST_ROOT:-/apdcephfs_csgl/share_306089109/shaunxhwang/fastwam_manifests}"
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${MANIFEST_ROOT}/interndata_manifest_v5_10hz}"
export MULTISOURCE_VIDEO_MANIFEST_DIR="${MULTISOURCE_VIDEO_MANIFEST_DIR:-${MANIFEST_ROOT}/multisource_canonical_v5}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${MULTISOURCE_VIDEO_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_declared_weights_10hz_stats.pt}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
export MULTISOURCE_SAMPLES_PER_EPOCH="${MULTISOURCE_SAMPLES_PER_EPOCH:-768000}"
export STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-16}"
export STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-8}"
export STATS_PREFETCH_FACTOR="${STATS_PREFETCH_FACTOR:-2}"

exec bash "${SCRIPT_DIR}/precompute_multisource_vjepa21_global_stats_single8.sh" "$@"
