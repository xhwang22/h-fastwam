#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FASTWAM_DISABLE_PROXY="${FASTWAM_DISABLE_PROXY:-1}"
if [[ "${FASTWAM_DISABLE_PROXY}" == "1" ]]; then
  unset http_proxy https_proxy ftp_proxy all_proxy
  unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
  unset no_proxy NO_PROXY
fi

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data/robotwin2.0}"
ROBOTWIN_DATA_REPO="${ROBOTWIN_DATA_REPO:-yuanty/robotwin2.0-fastwam}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
VJEPA21_URL="${VJEPA21_URL:-https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt}"
VJEPA21_EXPECTED_SIZE="${VJEPA21_EXPECTED_SIZE:-30238058912}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_PYTHON_DEPS="${INSTALL_PYTHON_DEPS:-1}"
SETUP_WAIT_TIMEOUT_SECONDS="${SETUP_WAIT_TIMEOUT_SECONDS:-43200}"
SETUP_MARKER="${SETUP_MARKER:-${REPO_ROOT}/.robotwin_vjepa21_setup_complete}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-run_robotwin_hfastwam_8card_small_vjepa21_predictor_native_temporal_aws.sh}"

usage() {
  cat <<EOF
Usage:
  bash scripts/setup_robotwin_vjepa21_predictor_training.sh

Environment overrides:
  ROBOTWIN_DATA_ROOT=/path/to/data/robotwin2.0
  HF_HOME=/path/to/huggingface/cache
  TORCH_HOME=/path/to/torch/cache
  PYTHON_BIN=/path/to/python
  SKIP_DATA=1
  SKIP_MODELS=1
  INSTALL_PYTHON_DEPS=0
  START_TRAINING=1
  SETUP_WAIT_TIMEOUT_SECONDS=43200
  SETUP_MARKER=/shared/path/setup-complete
  TRAINING_SCRIPT=run_robotwin_hfastwam_8card_small_vjepa21_predictor_native_temporal_aws.sh
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

ensure_system_tools() {
  local missing=()
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v tmux >/dev/null 2>&1 || missing+=(tmux)
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return
  fi

  echo "Installing missing system tools: ${missing[*]}"
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y "${missing[@]}"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y "${missing[@]}"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y "${missing[@]}"
  else
    echo "ERROR: missing system tools (${missing[*]}), but no supported package manager was found." >&2
    exit 1
  fi

  require_command git
  require_command tmux
}

install_python_dependencies() {
  if [[ "${INSTALL_PYTHON_DEPS}" == "0" ]]; then
    return
  fi
  echo "Installing the FastWAM Python training environment..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_aws_python_dependencies.sh"
}

setup_node_rank() {
  if [[ -n "${PET_NNODES:-}" ]]; then
    printf '%s\n' "${PET_NODE_RANK:?PET_NODE_RANK is required when PET_NNODES is set}"
  elif [[ -n "${NNODES:-}" && "${NNODES}" -gt 1 ]]; then
    printf '%s\n' "${NODE_RANK:?NODE_RANK is required when NNODES is greater than 1}"
  elif [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" -gt 1 && -z "${LOCAL_RANK:-}" ]]; then
    printf '%s\n' "${RANK:?RANK is required when WORLD_SIZE is greater than 1}"
  else
    printf '0\n'
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
  ensure_system_tools
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
  require_command "${PYTHON_BIN}"
  if ! "${PYTHON_BIN}" -c 'import huggingface_hub, safetensors' >/dev/null 2>&1; then
    if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
      echo "ERROR: pip is unavailable for ${PYTHON_BIN}." >&2
      exit 1
    fi
    echo "Installing Hugging Face download dependencies..."
    "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
      "huggingface-hub>=0.34.0" \
      "safetensors>=0.5.3"
  fi
  echo "Downloading Qwen3-VL-2B-Instruct into ${HF_HOME}/hub..."
  HF_HOME="${HF_HOME}" "${PYTHON_BIN}" - <<'PY'
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

assets_ready() {
  [[ -f "${SETUP_MARKER}" ]] &&
    [[ -f "${ROBOTWIN_DATA_ROOT}/robotwin2.0/meta/info.json" ]] &&
    [[ -f "${ROBOTWIN_DATA_ROOT}/dataset_stats.json" ]] &&
    [[ -f "${VJEPA21_CHECKPOINT}" ]] &&
    [[ -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]] &&
    [[ -d "${HF_HOME}/hub/models--Qwen--Qwen3-VL-2B-Instruct" ]]
}

wait_for_shared_assets() {
  local deadline=$(( SECONDS + SETUP_WAIT_TIMEOUT_SECONDS ))
  echo "Waiting for node rank 0 to prepare shared data and models..."
  until assets_ready; do
    if (( SECONDS >= deadline )); then
      echo "ERROR: timed out waiting for shared setup marker ${SETUP_MARKER}." >&2
      exit 1
    fi
    sleep 30
  done
}

install_python_dependencies
ensure_system_tools

SETUP_NODE_RANK="$(setup_node_rank)"
if [[ "${SETUP_NODE_RANK}" == "0" ]]; then
  rm -f "${SETUP_MARKER}"
  if [[ "${SKIP_DATA:-0}" != "1" ]]; then
    download_data
  fi
  if [[ "${SKIP_MODELS:-0}" != "1" ]]; then
    download_vjepa21
    download_qwen
  fi
  validate_assets
  touch "${SETUP_MARKER}"
else
  wait_for_shared_assets
fi

validate_assets

cat <<EOF
Assets are ready.

RoboTwin data: ${ROBOTWIN_DATA_ROOT}
V-JEPA checkpoint: ${VJEPA21_CHECKPOINT}
V-JEPA source: ${VJEPA21_REPO}
Qwen cache: ${HF_HOME}/hub/models--Qwen--Qwen3-VL-2B-Instruct
Setup node rank: ${SETUP_NODE_RANK}
EOF

if [[ "${START_TRAINING:-0}" == "1" ]]; then
  export ROBOTWIN_DATA_ROOT="$(dirname "${ROBOTWIN_DATA_ROOT}")"
  export HF_HOME TORCH_HOME VJEPA21_CHECKPOINT VJEPA21_REPO
  export VIDEO_LATENT_CACHE_ROOT="${VIDEO_LATENT_CACHE_ROOT:-${REPO_ROOT}/data/video_latent_cache}"
  export WANDB="${WANDB:-0}"
  exec bash "${SCRIPT_DIR}/${TRAINING_SCRIPT}"
fi

cat <<EOF

Run setup and training as one command on every worker node:
  START_TRAINING=1 bash scripts/setup_robotwin_vjepa21_predictor_training.sh
EOF
