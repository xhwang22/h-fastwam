#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data/robotwin2.0}"
ROBOTWIN_DATA_REPO="${ROBOTWIN_DATA_REPO:-yuanty/robotwin2.0-fastwam}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
VJEPA21_URL="${VJEPA21_URL:-https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt}"
VJEPA21_EXPECTED_SIZE="${VJEPA21_EXPECTED_SIZE:-30238058912}"

usage() {
  cat <<EOF
Usage:
  bash scripts/setup_robotwin_vjepa21_predictor_training.sh

Environment overrides:
  ROBOTWIN_DATA_ROOT=/path/to/data/robotwin2.0
  HF_HOME=/path/to/huggingface/cache
  TORCH_HOME=/path/to/torch/cache
  SKIP_DATA=1
  SKIP_MODELS=1
  START_TRAINING=1
  VIDEO_LATENT_CACHE_ROOT=/large/cache/path
  KEEP_DATA_ARCHIVES=1
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local output="$2"
  local expected_size="$3"
  local part="${output}.download"
  local size=0

  if [[ -f "${output}" ]]; then
    size="$(stat -c '%s' "${output}")"
    if [[ "${size}" == "${expected_size}" ]]; then
      return
    fi
    echo "ERROR: ${output} has size ${size}; expected ${expected_size}." >&2
    exit 1
  fi

  mkdir -p "$(dirname "${output}")"
  echo "Downloading $(basename "${output}")..."
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    "${url}" \
    --output "${part}"

  size="$(stat -c '%s' "${part}")"
  if [[ "${size}" != "${expected_size}" ]]; then
    echo "ERROR: ${part} has size ${size}; expected ${expected_size}." >&2
    exit 1
  fi
  mv "${part}" "${output}"
}

download_data() {
  require_command curl
  require_command tar
  mkdir -p "${ROBOTWIN_DATA_ROOT}"

  local base_url="https://huggingface.co/datasets/${ROBOTWIN_DATA_REPO}/resolve/main"
  local names=(
    robotwin2.0.tar.gz.part-00
    robotwin2.0.tar.gz.part-01
    robotwin2.0.tar.gz.part-02
    robotwin2.0.tar.gz.part-03
    robotwin2.0.tar.gz.part-04
    robotwin2.0.tar.gz.part-05
    robotwin2.0.tar.gz.part-06
    robotwin2.0.tar.gz.part-07
  )
  local sizes=(
    10737418240
    10737418240
    10737418240
    10737418240
    10737418240
    10737418240
    10737418240
    3924503300
  )
  local index

  download_file \
    "${base_url}/dataset_stats.json?download=true" \
    "${ROBOTWIN_DATA_ROOT}/dataset_stats.json" \
    88715

  if [[ ! -f "${ROBOTWIN_DATA_ROOT}/robotwin2.0/meta/info.json" ]]; then
    for index in "${!names[@]}"; do
      download_file \
        "${base_url}/${names[$index]}?download=true" \
        "${ROBOTWIN_DATA_ROOT}/${names[$index]}" \
        "${sizes[$index]}"
    done

    echo "Extracting RoboTwin 2.0 (requires about 80 GB additional free space)..."
    (
      cd "${ROBOTWIN_DATA_ROOT}"
      cat "${names[@]}" | tar -xzf -
    )
  fi

  if [[ ! -f "${ROBOTWIN_DATA_ROOT}/robotwin2.0/meta/info.json" ]]; then
    echo "ERROR: RoboTwin extraction did not create robotwin2.0/meta/info.json." >&2
    exit 1
  fi

  if [[ "${KEEP_DATA_ARCHIVES:-0}" != "1" ]]; then
    for index in "${!names[@]}"; do
      rm -f "${ROBOTWIN_DATA_ROOT}/${names[$index]}"
    done
  fi
}

download_vjepa21() {
  require_command curl
  require_command git
  mkdir -p "$(dirname "${VJEPA21_CHECKPOINT}")" "$(dirname "${VJEPA21_REPO}")"

  download_file "${VJEPA21_URL}" "${VJEPA21_CHECKPOINT}" "${VJEPA21_EXPECTED_SIZE}"

  if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
    if [[ -e "${VJEPA21_REPO}" ]]; then
      echo "ERROR: incomplete V-JEPA directory exists: ${VJEPA21_REPO}" >&2
      exit 1
    fi
    echo "Cloning V-JEPA 2.1 source..."
    git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "${VJEPA21_REPO}"
  fi
}

download_qwen() {
  require_command python3
  if ! python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    echo "ERROR: huggingface_hub is not installed. Install the project first:" >&2
    echo "  python3 -m pip install -e ." >&2
    exit 1
  fi
  echo "Downloading Qwen3-VL-2B-Instruct into ${HF_HOME}/hub..."
  HF_HOME="${HF_HOME}" python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    cache_dir=None,
    resume_download=True,
)
PY
}

validate_assets() {
  local required=(
    "${ROBOTWIN_DATA_ROOT}/robotwin2.0/meta/info.json"
    "${ROBOTWIN_DATA_ROOT}/dataset_stats.json"
    "${VJEPA21_CHECKPOINT}"
    "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py"
    "${HF_HOME}/hub/models--Qwen--Qwen3-VL-2B-Instruct"
  )
  local path
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "ERROR: required asset is missing: ${path}" >&2
      exit 1
    fi
  done
}

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  download_data
fi
if [[ "${SKIP_MODELS:-0}" != "1" ]]; then
  download_vjepa21
  download_qwen
fi

validate_assets

cat <<EOF
Assets are ready.

RoboTwin data: ${ROBOTWIN_DATA_ROOT}
V-JEPA checkpoint: ${VJEPA21_CHECKPOINT}
V-JEPA source: ${VJEPA21_REPO}
Qwen cache: ${HF_HOME}/hub/models--Qwen--Qwen3-VL-2B-Instruct
EOF

if [[ "${START_TRAINING:-0}" == "1" ]]; then
  export ROBOTWIN_DATA_ROOT="$(dirname "${ROBOTWIN_DATA_ROOT}")"
  export HF_HOME TORCH_HOME VJEPA21_CHECKPOINT VJEPA21_REPO
  export VIDEO_LATENT_CACHE_ROOT="${VIDEO_LATENT_CACHE_ROOT:-${REPO_ROOT}/data/video_latent_cache}"
  export WANDB="${WANDB:-0}"
  exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_native_temporal.sh"
fi

cat <<EOF

Start training with:
  WANDB=0 \\
  VIDEO_LATENT_CACHE_ROOT=/path/with/tens-of-terabytes/free \\
  bash scripts/run_robotwin_hfastwam_8card_small_vjepa21_predictor_native_temporal.sh
EOF
