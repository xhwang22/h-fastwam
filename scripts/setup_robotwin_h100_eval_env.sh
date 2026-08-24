#!/usr/bin/env bash
# Create the pinned Python, simulator, model, and GPU-rendering environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CURRENT_USER="${USER:-$(id -un)}"
readonly ROBOTWIN_REVISION="bf44be51cf5717a5595ce59447f2cf5263d2aa95"
readonly CUROBO_REVISION="v0.7.8"
readonly VJEPA21_EXPECTED_SIZE="30238058912"
readonly VJEPA21_SOURCE_REVISION="204698b45b3712590f06245fbfba32d3be539812"
readonly QWEN_REVISION="89644892e4d85e24eaac8bacfd4f463576704203"

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/fsx/miniforge3}"
if [[ ! -d "$(dirname "${MINIFORGE_ROOT}")" ]]; then
  MINIFORGE_ROOT="/fsx/${CURRENT_USER}/miniforge3"
fi
FASTWAM_EVAL_ENV="${FASTWAM_EVAL_ENV:-/fsx/conda-envs/fastwam-eval}"
if [[ ! -d "$(dirname "${FASTWAM_EVAL_ENV}")" ]]; then
  FASTWAM_EVAL_ENV="/fsx/${CURRENT_USER}/conda-envs/fastwam-eval"
fi
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/checkpoints/RoboTwin}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
QWEN_DIR="${QWEN_DIR:-${REPO_ROOT}/checkpoints/Qwen/Qwen3-VL-2B-Instruct}"
CUROBO_ROOT="${CUROBO_ROOT:-${REPO_ROOT}/external/curobo-v0.7.8}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
DOWNLOAD_ROBOTWIN_ASSETS="${DOWNLOAD_ROBOTWIN_ASSETS:-1}"
FORCE_EVAL_ENV_INSTALL="${FORCE_EVAL_ENV_INSTALL:-0}"

run_as_root() {
  if [[ "$(id -u)" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[h100-setup] ERROR: root privileges are required for: $*" >&2
    exit 1
  fi
}

install_system_dependencies() {
  local required_commands=(git curl bzip2 tar unzip vulkaninfo)
  local missing_commands=()
  local command_name
  for command_name in "${required_commands[@]}"; do
    command -v "${command_name}" >/dev/null 2>&1 \
      || missing_commands+=("${command_name}")
  done
  if (( ${#missing_commands[@]} == 0 )); then
    return
  fi
  if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
    echo "[h100-setup] ERROR: missing system commands: ${missing_commands[*]}" >&2
    exit 1
  fi
  if command -v dnf >/dev/null 2>&1; then
    run_as_root dnf install -y \
      git curl bzip2 tar unzip vulkan-loader vulkan-tools libglvnd-egl
  elif command -v yum >/dev/null 2>&1; then
    run_as_root yum install -y \
      git curl bzip2 tar unzip vulkan-loader vulkan-tools libglvnd-egl
  elif command -v apt-get >/dev/null 2>&1; then
    run_as_root apt-get update
    run_as_root apt-get install -y \
      git curl bzip2 tar unzip libvulkan1 vulkan-tools libegl1
  else
    echo "[h100-setup] ERROR: no supported system package manager found." >&2
    exit 1
  fi
  for command_name in "${required_commands[@]}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo "[h100-setup] ERROR: command is still missing: ${command_name}" >&2
      exit 1
    fi
  done
}

install_miniforge() {
  if [[ -x "${MINIFORGE_ROOT}/bin/conda" ]]; then
    return
  fi
  local installer="${MINIFORGE_ROOT}.installer.sh"
  mkdir -p "$(dirname "${MINIFORGE_ROOT}")"
  curl --fail --location --retry 5 \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    --output "${installer}"
  bash "${installer}" -b -p "${MINIFORGE_ROOT}"
  rm -f "${installer}"
}

activate_eval_environment() {
  local conda_sh="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_sh}" ]]; then
    echo "[h100-setup] ERROR: conda activation script missing: ${conda_sh}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set +u
  source "${conda_sh}"
  if [[ ! -x "${FASTWAM_EVAL_ENV}/bin/python" ]]; then
    conda create -y -p "${FASTWAM_EVAL_ENV}" python=3.10 pip
  fi
  conda activate "${FASTWAM_EVAL_ENV}"
  set -u
  hash -r
}

install_python_dependencies() {
  local marker="${FASTWAM_EVAL_ENV}/.fastwam_robotwin_h100_eval_v2"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    conda install -y -p "${FASTWAM_EVAL_ENV}" -c conda-forge "ffmpeg=7.1"
  fi
  if [[ "${FORCE_EVAL_ENV_INSTALL}" != "1" && -f "${marker}" ]]; then
    echo "[h100-setup] Reusing Python environment: ${FASTWAM_EVAL_ENV}"
    return
  fi

  python -m pip install --upgrade pip setuptools wheel ninja
  python -m pip install \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
  python -m pip install -e "${REPO_ROOT}"
  python -m pip install --no-cache-dir --upgrade \
    "transformers==5.12.1" \
    "huggingface-hub>=0.34.0" \
    "safetensors>=0.5.3" \
    "timm>=1.0.19"
  python -m pip uninstall -y torchaudio || true

  python -m pip install --no-cache-dir \
    "transforms3d==0.4.2" \
    "sapien==3.0.0b1" \
    "scipy==1.10.1" \
    "mplib==0.2.1" \
    "gymnasium==0.29.1" \
    "trimesh==4.4.3" \
    "open3d==0.18.0" \
    "imageio==2.34.2" \
    "moviepy==1.0.3" \
    "warp-lang==1.12.1" \
    "zarr<3" \
    "h5py" \
    "pyglet<2" \
    "opencv-python-headless==4.10.0.84" \
    "ffmpeg-python"

  if [[ ! -d "${CUROBO_ROOT}/.git" ]]; then
    if [[ -e "${CUROBO_ROOT}" ]]; then
      echo "[h100-setup] ERROR: non-git cuRobo path exists: ${CUROBO_ROOT}" >&2
      exit 1
    fi
    git clone --branch "${CUROBO_REVISION}" --depth 1 \
      https://github.com/NVlabs/curobo.git "${CUROBO_ROOT}"
  fi
  local curobo_head
  local curobo_expected
  curobo_head="$(git -C "${CUROBO_ROOT}" rev-parse HEAD)"
  curobo_expected="$(git -C "${CUROBO_ROOT}" rev-parse "${CUROBO_REVISION}^{commit}")"
  if [[ "${curobo_head}" != "${curobo_expected}" ]]; then
    echo "[h100-setup] ERROR: cuRobo checkout is not ${CUROBO_REVISION}." >&2
    exit 1
  fi
  python -m pip install -e "${CUROBO_ROOT}" --no-build-isolation

  python - <<'PY'
from pathlib import Path

import mplib
import sapien

planner = Path(mplib.__file__).resolve().parent / "planner.py"
text = planner.read_text(encoding="utf-8")
old = (
    "if np.linalg.norm(delta_twist) < 1e-4 or collide "
    "or not within_joint_limit:"
)
new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
if old in text:
    planner.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError(f"Unsupported mplib planner layout: {planner}")

urdf_loader = Path(sapien.__file__).resolve().parent / "wrapper" / "urdf_loader.py"
text = urdf_loader.read_text(encoding="utf-8")
text = text.replace(
    'with open(urdf_file, "r") as f:',
    'with open(urdf_file, "r", encoding="utf-8") as f:',
)
text = text.replace(
    'with open(srdf_file, "r") as f:',
    'with open(srdf_file, "r", encoding="utf-8") as f:',
)
text = text.replace('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"')
urdf_loader.write_text(text, encoding="utf-8")
PY
  touch "${marker}"
}

setup_nvidia_graphics() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[h100-setup] ERROR: nvidia-smi is unavailable." >&2
    exit 1
  fi
  local host_version
  host_version="$(
    nvidia-smi --query-gpu=driver_version --format=csv,noheader \
      | sort -u
  )"
  if [[ "$(printf '%s\n' "${host_version}" | wc -l)" != "1" ]]; then
    echo "[h100-setup] ERROR: GPUs report different driver versions:" >&2
    printf '%s\n' "${host_version}" >&2
    exit 1
  fi
  local version="${NVIDIA_DRIVER_VERSION:-${host_version}}"
  if [[ "${version}" != "${host_version}" ]]; then
    echo "[h100-setup] ERROR: userspace ${version} does not match host driver ${host_version}." >&2
    exit 1
  fi

  NVIDIA_GRAPHICS_ROOT="${NVIDIA_GRAPHICS_ROOT:-/fsx/nvidia-userspace/${version}}"
  local runfile="${NVIDIA_GRAPHICS_ROOT}/NVIDIA-Linux-x86_64-${version}.run"
  local lib_dir="${NVIDIA_GRAPHICS_ROOT}/extracted"
  local env_file="${NVIDIA_GRAPHICS_ROOT}/activate.sh"
  if [[ ! -f "${lib_dir}/libGLX_nvidia.so.${version}" ]]; then
    mkdir -p "${NVIDIA_GRAPHICS_ROOT}"
    curl --fail --location --retry 5 \
      "https://us.download.nvidia.com/tesla/${version}/NVIDIA-Linux-x86_64-${version}.run" \
      --output "${runfile}"
    chmod +x "${runfile}"
    "${runfile}" --extract-only --target "${lib_dir}"
  fi

  for target in \
    "libGLX_nvidia.so.${version}" \
    "libEGL_nvidia.so.${version}" \
    "libGL.so.1.7.0" \
    "libEGL.so.1.1.0" \
    "libGLESv2.so.2.1.0" \
    "libGLESv1_CM.so.1.2.0"; do
    if [[ ! -f "${lib_dir}/${target}" ]]; then
      echo "[h100-setup] ERROR: extracted NVIDIA library missing: ${target}" >&2
      exit 1
    fi
  done
  ln -sfn "libGLX_nvidia.so.${version}" "${lib_dir}/libGLX_nvidia.so.0"
  ln -sfn "libEGL_nvidia.so.${version}" "${lib_dir}/libEGL_nvidia.so.0"
  ln -sfn "libGL.so.1.7.0" "${lib_dir}/libGL.so.1"
  ln -sfn "libEGL.so.1.1.0" "${lib_dir}/libEGL.so.1"
  ln -sfn "libGLESv2.so.2.1.0" "${lib_dir}/libGLESv2.so.2"
  ln -sfn "libGLESv1_CM.so.1.2.0" "${lib_dir}/libGLESv1_CM.so.1"

  mkdir -p "${NVIDIA_GRAPHICS_ROOT}/icd"
  cat > "${NVIDIA_GRAPHICS_ROOT}/icd/nvidia_icd.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${lib_dir}/libGLX_nvidia.so.0",
    "api_version": "1.3.239"
  }
}
EOF
  cat > "${NVIDIA_GRAPHICS_ROOT}/icd/10_nvidia.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${lib_dir}/libEGL_nvidia.so.0"
  }
}
EOF
  cat > "${env_file}" <<EOF
export NVIDIA_GRAPHICS_ROOT="${NVIDIA_GRAPHICS_ROOT}"
_fastwam_python_env_prefix="\${FASTWAM_PYTHON_ENV_PREFIX:-\${CONDA_PREFIX:-}}"
export LD_LIBRARY_PATH="${lib_dir}:\${_fastwam_python_env_prefix}/lib:\${_fastwam_python_env_prefix}/targets/x86_64-linux/lib:/usr/lib64:/lib64:\${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_GRAPHICS_ROOT}/icd/nvidia_icd.json"
export VK_DRIVER_FILES="${NVIDIA_GRAPHICS_ROOT}/icd/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_GRAPHICS_ROOT}/icd/10_nvidia.json"
export XDG_RUNTIME_DIR=/tmp
unset FASTWAM_ROBOTWIN_CPU_RENDER
unset _fastwam_python_env_prefix
EOF
  # shellcheck disable=SC1090
  source "${env_file}"
  export NVIDIA_GRAPHICS_ENV="${env_file}"

  vulkaninfo --summary > "${NVIDIA_GRAPHICS_ROOT}/vulkaninfo.log"
  if ! grep -q "NVIDIA" "${NVIDIA_GRAPHICS_ROOT}/vulkaninfo.log"; then
    echo "[h100-setup] ERROR: Vulkan did not enumerate an NVIDIA device." >&2
    exit 1
  fi
}

setup_robotwin() {
  if [[ ! -d "${ROBOTWIN_ROOT}/.git" ]]; then
    if [[ -e "${ROBOTWIN_ROOT}" ]]; then
      echo "[h100-setup] ERROR: non-git RoboTwin path exists: ${ROBOTWIN_ROOT}" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${ROBOTWIN_ROOT}")"
    git clone --no-checkout \
      https://github.com/RoboTwin-Platform/RoboTwin.git "${ROBOTWIN_ROOT}"
    git -C "${ROBOTWIN_ROOT}" fetch --depth 1 origin "${ROBOTWIN_REVISION}"
    git -C "${ROBOTWIN_ROOT}" checkout --detach "${ROBOTWIN_REVISION}"
  fi
  local revision
  revision="$(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD)"
  if [[ "${revision}" != "${ROBOTWIN_REVISION}" ]]; then
    echo "[h100-setup] ERROR: RoboTwin is ${revision}; expected ${ROBOTWIN_REVISION}." >&2
    exit 1
  fi

  local missing_assets=0
  for relative in assets/objects assets/embodiments assets/background_texture; do
    [[ -e "${ROBOTWIN_ROOT}/${relative}" ]] || missing_assets=1
  done
  if (( missing_assets )); then
    if [[ "${DOWNLOAD_ROBOTWIN_ASSETS}" != "1" ]]; then
      echo "[h100-setup] ERROR: RoboTwin assets are missing under ${ROBOTWIN_ROOT}." >&2
      exit 1
    fi
    (
      cd "${ROBOTWIN_ROOT}"
      bash script/_download_assets.sh
    )
  fi
}

download_model_assets() {
  mkdir -p "$(dirname "${VJEPA21_CHECKPOINT}")"
  if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
    curl --fail --location --retry 5 --continue-at - \
      "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt" \
      --output "${VJEPA21_CHECKPOINT}.download"
    mv "${VJEPA21_CHECKPOINT}.download" "${VJEPA21_CHECKPOINT}"
  fi
  local vjepa_size
  vjepa_size="$(stat -c '%s' "${VJEPA21_CHECKPOINT}")"
  if [[ "${vjepa_size}" != "${VJEPA21_EXPECTED_SIZE}" ]]; then
    echo "[h100-setup] ERROR: invalid V-JEPA checkpoint size ${vjepa_size}." >&2
    exit 1
  fi
  if [[ ! -d "${VJEPA21_REPO}/.git" ]]; then
    if [[ -e "${VJEPA21_REPO}" ]]; then
      echo "[h100-setup] ERROR: non-git V-JEPA source: ${VJEPA21_REPO}" >&2
      exit 1
    fi
    git clone --no-checkout \
      https://github.com/facebookresearch/vjepa2.git "${VJEPA21_REPO}"
    git -C "${VJEPA21_REPO}" fetch --depth 1 \
      origin "${VJEPA21_SOURCE_REVISION}"
    git -C "${VJEPA21_REPO}" checkout --detach \
      "${VJEPA21_SOURCE_REVISION}"
  fi
  local vjepa_revision
  vjepa_revision="$(git -C "${VJEPA21_REPO}" rev-parse HEAD)"
  if [[ "${vjepa_revision}" != "${VJEPA21_SOURCE_REVISION}" ]]; then
    echo "[h100-setup] ERROR: V-JEPA is ${vjepa_revision}; expected ${VJEPA21_SOURCE_REVISION}." >&2
    exit 1
  fi
  if [[ -n "$(git -C "${VJEPA21_REPO}" status --porcelain --untracked-files=no)" ]]; then
    echo "[h100-setup] ERROR: V-JEPA has modified tracked files: ${VJEPA21_REPO}" >&2
    exit 1
  fi
  if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
    echo "[h100-setup] ERROR: incomplete V-JEPA source: ${VJEPA21_REPO}" >&2
    exit 1
  fi

  if ! QWEN_DIR="${QWEN_DIR}" QWEN_REVISION="${QWEN_REVISION}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["QWEN_DIR"])
revision = os.environ["QWEN_REVISION"]
marker = root / ".fastwam_hf_revision"
if marker.is_file():
    actual_revision = marker.read_text(encoding="utf-8").strip()
elif root.parent.name == "snapshots":
    actual_revision = root.name
else:
    actual_revision = None
if actual_revision != revision:
    raise SystemExit(1)
if not (root / "config.json").is_file():
    raise SystemExit(1)
single = root / "model.safetensors"
if single.is_file():
    raise SystemExit(0)
index = root / "model.safetensors.index.json"
if not index.is_file():
    raise SystemExit(1)
with index.open("r", encoding="utf-8") as handle:
    weight_map = json.load(handle).get("weight_map", {})
if not weight_map:
    raise SystemExit(1)
if any(not (root / filename).is_file() for filename in set(weight_map.values())):
    raise SystemExit(1)
PY
  then
    QWEN_DIR="${QWEN_DIR}" QWEN_REVISION="${QWEN_REVISION}" python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    revision=os.environ["QWEN_REVISION"],
    local_dir=os.environ["QWEN_DIR"],
    resume_download=True,
)
marker = os.path.join(os.environ["QWEN_DIR"], ".fastwam_hf_revision")
with open(marker, "w", encoding="utf-8") as handle:
    handle.write(f"{os.environ['QWEN_REVISION']}\n")
PY
  fi
}

install_system_dependencies
install_miniforge
activate_eval_environment
install_python_dependencies
setup_nvidia_graphics
setup_robotwin
download_model_assets

export HF_HOME TORCH_HOME ROBOTWIN_ROOT VJEPA21_CHECKPOINT VJEPA21_REPO QWEN_DIR
python scripts/check_robotwin_h100_eval_env.py \
  --robotwin-root "${ROBOTWIN_ROOT}" \
  --render-backend gpu

cat <<EOF
[h100-setup] Environment is ready.
FASTWAM_EVAL_ENV=${FASTWAM_EVAL_ENV}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
VJEPA21_CHECKPOINT=${VJEPA21_CHECKPOINT}
VJEPA21_REPO=${VJEPA21_REPO}
VJEPA21_SOURCE_REVISION=${VJEPA21_SOURCE_REVISION}
QWEN_DIR=${QWEN_DIR}
QWEN_REVISION=${QWEN_REVISION}
NVIDIA_GRAPHICS_ENV=${NVIDIA_GRAPHICS_ENV}
EOF
