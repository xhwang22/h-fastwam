#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/fsx/venv312/bin/python}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/robotwin2.0_webdataset.stage}"
STATS_PATH="${STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json}"
S3_OUTPUT_ROOT="${S3_OUTPUT_ROOT:-s3://ams-gpu-p5-training-ohio-roboticsx/data/pretrain_data/RoboTwin/robotwin2.0_webdataset}"
AWS_PROFILE="${AWS_PROFILE:-roboticsx}"
AWS_REGION="${AWS_REGION:-us-east-2}"
AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-/fsx/.aws/credentials}"
WDS_WORKERS="${WDS_WORKERS:-4}"
WDS_EPISODES_PER_SHARD="${WDS_EPISODES_PER_SHARD:-32}"
WDS_PNG_COMPRESS_LEVEL="${WDS_PNG_COMPRESS_LEVEL:-1}"
WDS_DECODE_CHUNK_FRAMES="${WDS_DECODE_CHUNK_FRAMES:-64}"

# TorchCodec needs shared FFmpeg libraries. Reuse the same installation used
# by the AWS training launchers before starting worker processes.
# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
_fastwam_install_shared_ffmpeg

case "${OUTPUT_ROOT}" in
  s3://*|/s3/*)
    echo "ERROR: OUTPUT_ROOT must be local POSIX staging, not ${OUTPUT_ROOT}" >&2
    exit 1
    ;;
esac
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import pyarrow; from PIL import Image; from torchcodec.decoders import VideoDecoder" >/dev/null 2>&1; then
  echo "ERROR: PyArrow/Pillow/TorchCodec cannot load after FFmpeg setup." >&2
  echo "FFMPEG_PREFIX=${FFMPEG_PREFIX}" >&2
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: AWS CLI is required." >&2
  exit 1
fi
if [[ ! -f "${AWS_SHARED_CREDENTIALS_FILE}" ]]; then
  echo "ERROR: AWS credentials file not found: ${AWS_SHARED_CREDENTIALS_FILE}" >&2
  exit 1
fi
export AWS_PROFILE AWS_REGION AWS_DEFAULT_REGION="${AWS_REGION}"
export AWS_SHARED_CREDENTIALS_FILE
export AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"
export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"
AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-/tmp/robotwin_webdataset_aws_config}"
cat > "${AWS_CONFIG_FILE}" <<EOF
[profile ${AWS_PROFILE}]
region = ${AWS_REGION}
s3 =
    max_concurrent_requests = 32
    multipart_threshold = 128MB
    multipart_chunksize = 64MB
EOF
export AWS_CONFIG_FILE
aws sts get-caller-identity --output json >/dev/null
if [[ ! -f "${SOURCE_ROOT}/meta/info.json" ]]; then
  echo "ERROR: RoboTwin source dataset not found: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "ERROR: normalization statistics not found: ${STATS_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required." >&2; exit 1; }
exec 9>"${OUTPUT_ROOT}/.preprocess.lock"
flock -n 9 || { echo "ERROR: another preprocessing job is using ${OUTPUT_ROOT}." >&2; exit 1; }
# Shared FSx lock prevents another host/staging directory from writing the same
# production S3 prefix concurrently.
s3_lock_id=$(printf '%s' "${S3_OUTPUT_ROOT}" | sha256sum | cut -d' ' -f1)
S3_LOCK_ROOT="${S3_LOCK_ROOT:-/fsx/h-fastwam/.s3_upload_locks}"
mkdir -p "${S3_LOCK_ROOT}"
exec 8>"${S3_LOCK_ROOT}/${s3_lock_id}.lock"
flock -n 8 || { echo "ERROR: another job is publishing to ${S3_OUTPUT_ROOT}." >&2; exit 1; }
available_bytes=$(df -P -B1 "${OUTPUT_ROOT}" | tail -1 | tr -s ' ' | cut -d' ' -f4)
if (( available_bytes < 68719476736 )); then
  echo "ERROR: local staging requires at least 64 GiB free: ${OUTPUT_ROOT}" >&2
  exit 1
fi
echo "[robotwin-webdataset] source=${SOURCE_ROOT}"
echo "[robotwin-webdataset] staging=${OUTPUT_ROOT}"
echo "[robotwin-webdataset] s3=${S3_OUTPUT_ROOT}"
echo "[robotwin-webdataset] aws_profile=${AWS_PROFILE} region=${AWS_REGION}"
echo "[robotwin-webdataset] workers=${WDS_WORKERS} episodes_per_shard=${WDS_EPISODES_PER_SHARD}"
echo "[robotwin-webdataset] png_compress_level=${WDS_PNG_COMPRESS_LEVEL} decode_chunk_frames=${WDS_DECODE_CHUNK_FRAMES}"
echo "[robotwin-webdataset] estimated final size is about 3.2-3.6 TB; existing completed shards will be reused."
df -h "${OUTPUT_ROOT}" || true

ARGS=(
  --source-root "${SOURCE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --s3-output-root "${S3_OUTPUT_ROOT}"
  --aws-profile "${AWS_PROFILE}"
  --aws-region "${AWS_REGION}"
  --aws-credentials-file "${AWS_SHARED_CREDENTIALS_FILE}"
  --stats-path "${STATS_PATH}"
  --workers "${WDS_WORKERS}"
  --episodes-per-shard "${WDS_EPISODES_PER_SHARD}"
  --png-compress-level "${WDS_PNG_COMPRESS_LEVEL}"
  --decode-chunk-frames "${WDS_DECODE_CHUNK_FRAMES}"
)
if [[ -n "${WDS_MAX_EPISODES:-}" ]]; then
  if [[ "${WDS_ALLOW_PARTIAL_S3:-0}" != "1" ]]; then
    echo "ERROR: partial S3 conversion requires WDS_ALLOW_PARTIAL_S3=1 and a dedicated smoke prefix." >&2
    exit 1
  fi
  ARGS+=(--max-episodes "${WDS_MAX_EPISODES}" --allow-partial-s3)
fi
if [[ "${WDS_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

exec "${PYTHON_BIN}" scripts/preprocess_robotwin_webdataset.py "${ARGS[@]}"
