#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROBOTWIN_DATA_ROOT}/robotwin2.0_webdataset}"
STATS_PATH="${STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json}"
WDS_WORKERS="${WDS_WORKERS:-32}"
WDS_EPISODES_PER_SHARD="${WDS_EPISODES_PER_SHARD:-32}"
WDS_PNG_COMPRESS_LEVEL="${WDS_PNG_COMPRESS_LEVEL:-1}"
WDS_DECODE_CHUNK_FRAMES="${WDS_DECODE_CHUNK_FRAMES:-64}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_ROOT}/meta/info.json" ]]; then
  echo "ERROR: RoboTwin source dataset not found: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "ERROR: normalization statistics not found: ${STATS_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[robotwin-webdataset] source=${SOURCE_ROOT}"
echo "[robotwin-webdataset] output=${OUTPUT_ROOT}"
echo "[robotwin-webdataset] workers=${WDS_WORKERS} episodes_per_shard=${WDS_EPISODES_PER_SHARD}"
echo "[robotwin-webdataset] png_compress_level=${WDS_PNG_COMPRESS_LEVEL} decode_chunk_frames=${WDS_DECODE_CHUNK_FRAMES}"
echo "[robotwin-webdataset] estimated final size is about 3.2-3.6 TB; existing completed shards will be reused."
df -h "${OUTPUT_ROOT}" || true

ARGS=(
  --source-root "${SOURCE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --stats-path "${STATS_PATH}"
  --workers "${WDS_WORKERS}"
  --episodes-per-shard "${WDS_EPISODES_PER_SHARD}"
  --png-compress-level "${WDS_PNG_COMPRESS_LEVEL}"
  --decode-chunk-frames "${WDS_DECODE_CHUNK_FRAMES}"
)
if [[ -n "${WDS_MAX_EPISODES:-}" ]]; then
  ARGS+=(--max-episodes "${WDS_MAX_EPISODES}")
fi
if [[ "${WDS_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

exec "${PYTHON_BIN}" scripts/preprocess_robotwin_webdataset.py "${ARGS[@]}"
