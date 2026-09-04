#!/usr/bin/env bash
# Create the pinned Python, simulator, model, and GPU-rendering environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CURRENT_USER="${USER:-$(id -un)}"
readonly ROBOTWIN_REVISION="bf44be51cf5717a5595ce59447f2cf5263d2aa95"
readonly CUROBO_REVISION="v0.7.8"
readonly VJEPA21_VITG_EXPECTED_SIZE="30238058912"
readonly VJEPA21_VITL_EXPECTED_SIZE="5151198524"
readonly VJEPA21_SOURCE_REVISION="204698b45b3712590f06245fbfba32d3be539812"
readonly QWEN_REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
readonly DINOV3_REVISION="c807c9eeea853df70aec4069e6f56b28ddc82acc"
readonly SIGLIP2_LARGE_REVISION="1b426889ea62b5a72bf9839009a1b184bfc9c178"
readonly SETUPTOOLS_REQUIREMENT="setuptools<82"
readonly CUDA_TOOLKIT_VERSION="12.8"
readonly CUDA_TOOLKIT_RUNFILE="cuda_12.8.0_570.86.10_linux.run"
readonly CUDA_TOOLKIT_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/${CUDA_TOOLKIT_RUNFILE}"

FASTWAM_EVAL_USE_CURRENT_ENV="${FASTWAM_EVAL_USE_CURRENT_ENV:-0}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/fsx/miniforge3}"
if [[ ! -d "$(dirname "${MINIFORGE_ROOT}")" ]]; then
  MINIFORGE_ROOT="/fsx/${CURRENT_USER}/miniforge3"
fi
if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV}" == "1" ]]; then
  if [[ -z "${PYTHON_BIN:-}" && -x "/opt/venv/bin/python" ]]; then
    PYTHON_BIN=/opt/venv/bin/python
  fi
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
  if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "[h100-setup] ERROR: current Python not found; set PYTHON_BIN." >&2
    exit 1
  fi
  PYTHON_ENV_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
  FASTWAM_EVAL_ENV="${PYTHON_ENV_PREFIX}"
else
  FASTWAM_EVAL_ENV="${FASTWAM_EVAL_ENV:-/fsx/conda-envs/fastwam-eval}"
  if [[ ! -d "$(dirname "${FASTWAM_EVAL_ENV}")" ]]; then
    FASTWAM_EVAL_ENV="/fsx/${CURRENT_USER}/conda-envs/fastwam-eval"
  fi
fi
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/checkpoints/RoboTwin}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_VITL_CHECKPOINT="${VJEPA21_VITL_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${REPO_ROOT}/checkpoints/dinov3-vith16plus-pretrain-lvd1689m}"
SIGLIP2_MODEL_PATH="${SIGLIP2_MODEL_PATH:-${REPO_ROOT}/checkpoints/siglip2-so400m-patch16-384}"
SIGLIP2_LARGE_MODEL_PATH="${SIGLIP2_LARGE_MODEL_PATH:-${REPO_ROOT}/checkpoints/siglip2-large-patch16-384}"
ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_WEBDATASET_ROOT}/vjepa21_vitG_causal_tubelet_global_stats.pt}"
if [[ -z "${QWEN_DIR:-}" ]]; then
  CACHED_QWEN_DIR="${HF_HOME}/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/${QWEN_REVISION}"
  if [[ -f "${CACHED_QWEN_DIR}/config.json" ]]; then
    QWEN_DIR="${CACHED_QWEN_DIR}"
  else
    QWEN_DIR="${REPO_ROOT}/checkpoints/Qwen/Qwen3-VL-2B-Instruct"
  fi
fi
CUROBO_ROOT="${CUROBO_ROOT:-${REPO_ROOT}/external/curobo-v0.7.8}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
DOWNLOAD_ROBOTWIN_ASSETS="${DOWNLOAD_ROBOTWIN_ASSETS:-1}"
FORCE_EVAL_ENV_INSTALL="${FORCE_EVAL_ENV_INSTALL:-0}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-1}"
USE_SYSTEM_NVIDIA_GRAPHICS="${USE_SYSTEM_NVIDIA_GRAPHICS:-${FASTWAM_EVAL_USE_CURRENT_ENV}}"
NVIDIA_GRAPHICS_ENV="${NVIDIA_GRAPHICS_ENV:-}"
PIP_REINSTALL_ARGS=()
if [[ "${FORCE_EVAL_ENV_INSTALL}" == "1" ]]; then
  PIP_REINSTALL_ARGS=(--force-reinstall)
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

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
  if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV}" == "1" ]]; then
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
    export FASTWAM_PYTHON_ENV_PREFIX="${PYTHON_ENV_PREFIX}"
    return
  fi

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
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  PYTHON_ENV_PREFIX="${CONDA_PREFIX}"
  export FASTWAM_PYTHON_ENV_PREFIX="${PYTHON_ENV_PREFIX}"
}

python_environment_ready() {
  CUROBO_ROOT="${CUROBO_ROOT}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.metadata
import os
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

import accelerate
import av
import boto3
import cv2
import curobo
import datasets
import einops
import ffmpeg
import git
import gymnasium
import h5py
import huggingface_hub
import hydra
import imageio
import imageio_ffmpeg
import jsonlines
import mplib
import moviepy.editor
import numpy
import omegaconf
import open3d
import PIL
import pkg_resources
import pyglet
import pyarrow
import rich
import safetensors
import sapien
import scipy
import termcolor
import timm
import torch
import torchvision
import transformers
import transforms3d
import trimesh
import tqdm
import warp
import yaml
import zarr
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from experiments.robotwin.fastwam_policy.deploy_policy import (
    WorldActionRobotWinPolicy,
)

assert torch.__version__.startswith("2.7.1+cu128")
assert torchvision.__version__.startswith("0.22.1+cu128")
assert Version(transformers.__version__) >= Version("4.57.0")
assert str(warp.__version__) == "1.12.1"
requirements = [
    "accelerate==1.12.0",
    "av==16.0.1",
    "boto3==1.35.99",
    "datasets==3.6.0",
    "einops==0.8.1",
    "gitpython==3.1.45",
    "transforms3d==0.4.2",
    "sapien==3.0.0b1",
    "mplib==0.2.1",
    "gymnasium==0.29.1",
    "hydra-core==1.3.2",
    "huggingface-hub>=0.34.0",
    "jsonlines==4.0.0",
    "numpy==1.26.4",
    "pillow==12.0.0",
    "pyarrow==23.0.0",
    "rich==14.2.0",
    "safetensors>=0.5.3",
    "termcolor==2.5.0",
    "trimesh==4.4.3",
    "imageio==2.34.2",
    "moviepy==1.0.3",
    "omegaconf==2.3.0",
    "tqdm==4.66.5",
    "warp-lang==1.12.1",
    "zarr<3",
    "pyglet<2",
    "opencv-python-headless==4.10.0.84",
]
requirements.extend(
    ["scipy>=1.11.4,<1.14", "open3d>=0.19,<0.20"]
    if sys.version_info >= (3, 12)
    else ["scipy==1.10.1", "open3d==0.18.0"]
)
for requirement_text in requirements:
    requirement = Requirement(requirement_text)
    installed = importlib.metadata.version(requirement.name)
    assert installed in requirement.specifier
curobo_path = Path(curobo.__file__).resolve()
curobo_root = Path(os.environ["CUROBO_ROOT"]).resolve()
assert curobo_path.is_relative_to(curobo_root)
PY
}

ensure_python_requirement() {
  local module="$1"
  local requirement="$2"
  shift
  shift
  if [[ "${FORCE_EVAL_ENV_INSTALL}" != "1" ]] && \
      MODULE="${module}" REQUIREMENT="${requirement}" \
      "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib
import importlib.metadata
import os

from packaging.requirements import Requirement

module = os.environ["MODULE"]
requirement = Requirement(os.environ["REQUIREMENT"])
importlib.import_module(module)
version = importlib.metadata.version(requirement.name)
assert version in requirement.specifier
PY
  then
    return
  fi
  "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
    "${PIP_REINSTALL_ARGS[@]}" "${requirement}" "$@"
}

curobo_checkout_ready() {
  [[ -d "${CUROBO_ROOT}/.git" ]] || return 1
  local curobo_head
  local curobo_expected
  curobo_head="$(git -C "${CUROBO_ROOT}" rev-parse HEAD 2>/dev/null)" || return 1
  curobo_expected="$(
    git -C "${CUROBO_ROOT}" rev-parse "${CUROBO_REVISION}^{commit}" 2>/dev/null
  )" || return 1
  [[ "${curobo_head}" == "${curobo_expected}" ]] || return 1
  [[ -z "$(git -C "${CUROBO_ROOT}" status --porcelain --untracked-files=no)" ]]
}

curobo_python_ready() {
  local output="/dev/null"
  if [[ "${1:-}" == "--verbose" ]]; then
    output="/dev/stderr"
  fi
  CUROBO_ROOT="${CUROBO_ROOT}" "${PYTHON_BIN}" - <<'PY' >"${output}" 2>&1
import os
from pathlib import Path

import torch
import curobo
from curobo.curobolib import (
    geom_cu,
    kinematics_fused_cu,
    lbfgs_step_cu,
    line_search_cu,
    tensor_step_cu,
)
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen

root = Path(os.environ["CUROBO_ROOT"]).resolve()
assert Path(curobo.__file__).resolve().is_relative_to(root)
PY
}

cuda_toolkit_matches_torch() {
  local cuda_root="$1"
  [[ -x "${cuda_root}/bin/nvcc" ]] || return 1
  local torch_cuda
  local nvcc_cuda
  torch_cuda="$(
    "${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "")'
  )"
  nvcc_cuda="$(
    "${cuda_root}/bin/nvcc" --version \
      | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
      | head -1
  )"
  [[ -n "${torch_cuda}" && "${nvcc_cuda}" == "${torch_cuda}" ]]
}

install_cuda_toolkit_for_torch() {
  local torch_cuda
  torch_cuda="$(
    "${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "")'
  )"
  if [[ "${torch_cuda}" != "${CUDA_TOOLKIT_VERSION}" ]]; then
    echo "[h100-setup] ERROR: expected torch CUDA ${CUDA_TOOLKIT_VERSION}, found ${torch_cuda}." >&2
    exit 1
  fi

  local explicit_cuda_root="${CUDA_TOOLKIT_ROOT:-}"
  local nvcc_root=""
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_root="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  local candidates=()
  if [[ -n "${explicit_cuda_root}" ]]; then
    candidates+=("${explicit_cuda_root}")
  else
    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("${CUDA_HOME}")
    candidates+=(
      "/usr/local/cuda-${torch_cuda}"
      "/usr/local/cuda-${torch_cuda}.0"
      "/fsx/cuda-toolkit/${torch_cuda}"
      "/fsx/cuda-toolkit/${torch_cuda}.0"
      "${REPO_ROOT}/checkpoints/cuda-toolkit/${torch_cuda}.0"
    )
    [[ -n "${nvcc_root}" ]] && candidates+=("${nvcc_root}")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if cuda_toolkit_matches_torch "${candidate}"; then
      CUDA_TOOLKIT_ROOT="${candidate}"
      break
    fi
  done

  if [[ -z "${CUDA_TOOLKIT_ROOT:-}" ]] || \
      ! cuda_toolkit_matches_torch "${CUDA_TOOLKIT_ROOT}"; then
    if [[ -n "${explicit_cuda_root}" ]]; then
      echo "[h100-setup] ERROR: CUDA_TOOLKIT_ROOT does not provide nvcc ${torch_cuda}: ${explicit_cuda_root}" >&2
      exit 1
    fi
    if [[ "${INSTALL_CUDA_TOOLKIT}" != "1" ]]; then
      echo "[h100-setup] ERROR: CUDA toolkit ${torch_cuda} is required to build cuRobo." >&2
      echo "Set CUDA_TOOLKIT_ROOT to a matching toolkit or INSTALL_CUDA_TOOLKIT=1." >&2
      exit 1
    fi

    CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_INSTALL_ROOT:-${REPO_ROOT}/checkpoints/cuda-toolkit/${torch_cuda}.0}"
    local installer_dir
    installer_dir="$(dirname "${CUDA_TOOLKIT_ROOT}")/installers"
    local installer="${installer_dir}/${CUDA_TOOLKIT_RUNFILE}"
    mkdir -p "${installer_dir}" "${CUDA_TOOLKIT_ROOT}"
    if [[ ! -f "${installer}" ]]; then
      echo "[h100-setup] Downloading CUDA toolkit ${torch_cuda} (toolkit only; no driver)."
      curl --fail --location --retry 5 --continue-at - \
        "${CUDA_TOOLKIT_URL}" \
        --output "${installer}"
    fi
    echo "[h100-setup] Installing CUDA toolkit ${torch_cuda} into ${CUDA_TOOLKIT_ROOT}."
    bash "${installer}" \
      --silent \
      --toolkit \
      --toolkitpath="${CUDA_TOOLKIT_ROOT}" \
      --defaultroot="${CUDA_TOOLKIT_ROOT}" \
      --no-opengl-libs \
      --no-man-page \
      --override
    if ! cuda_toolkit_matches_torch "${CUDA_TOOLKIT_ROOT}"; then
      echo "[h100-setup] ERROR: CUDA toolkit ${torch_cuda} installation failed." >&2
      exit 1
    fi
  fi

  export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
  export CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
  export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_TOOLKIT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  export MAX_JOBS="${MAX_JOBS:-8}"
  printf '%s\n' "${CUDA_TOOLKIT_ROOT}" \
    > "${PYTHON_ENV_PREFIX}/.fastwam_cuda_toolkit_root"
  echo "[h100-setup] cuRobo build toolkit: $("${CUDA_HOME}/bin/nvcc" --version | grep release | tail -1)"
}

install_current_python_dependencies() {
  "${PYTHON_BIN}" -m pip --version >/dev/null
  if ! "${PYTHON_BIN}" -c 'import packaging, pkg_resources' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install \
      "packaging==25.0" \
      "${SETUPTOOLS_REQUIREMENT}"
  fi
  "${PYTHON_BIN}" -m pip install --upgrade \
    "${SETUPTOOLS_REQUIREMENT}" wheel ninja
  "${PYTHON_BIN}" -m pip install "${PIP_REINSTALL_ARGS[@]}" \
    -e "${REPO_ROOT}" \
    --extra-index-url https://download.pytorch.org/whl/cu128

  if [[ "${FORCE_EVAL_ENV_INSTALL}" == "1" ]] || \
      ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
import torchvision

assert torch.__version__.startswith("2.7.1+cu128")
assert torchvision.__version__.startswith("0.22.1+cu128")
PY
  then
    "${PYTHON_BIN}" -m pip install \
      --upgrade \
      "${PIP_REINSTALL_ARGS[@]}" \
      torch==2.7.1+cu128 \
      torchvision==0.22.1+cu128 \
      --extra-index-url https://download.pytorch.org/whl/cu128
  fi

  if [[ "${FORCE_EVAL_ENV_INSTALL}" == "1" ]] || \
      ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from packaging.version import Version
import transformers
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

assert Version(transformers.__version__) >= Version("4.57.0")
PY
  then
    "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
      "${PIP_REINSTALL_ARGS[@]}" \
      "transformers==5.12.1" \
      "huggingface-hub>=0.34.0" \
      "safetensors>=0.5.3" \
      "timm>=1.0.19"
  fi

  local scipy_spec="scipy==1.10.1"
  local open3d_spec="open3d==0.18.0"
  if "${PYTHON_BIN}" -c \
      'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    scipy_spec="scipy>=1.11.4,<1.14"
    open3d_spec="open3d>=0.19,<0.20"
  fi

  ensure_python_requirement transforms3d "transforms3d==0.4.2"
  ensure_python_requirement accelerate "accelerate==1.12.0"
  ensure_python_requirement av "av==16.0.1"
  ensure_python_requirement boto3 "boto3==1.35.99"
  ensure_python_requirement datasets "datasets==3.6.0"
  ensure_python_requirement einops "einops==0.8.1"
  ensure_python_requirement git "gitpython==3.1.45"
  ensure_python_requirement sapien "sapien==3.0.0b1"
  ensure_python_requirement scipy "${scipy_spec}"
  ensure_python_requirement mplib "mplib==0.2.1"
  ensure_python_requirement gymnasium "gymnasium==0.29.1"
  ensure_python_requirement hydra "hydra-core==1.3.2"
  ensure_python_requirement huggingface_hub "huggingface-hub>=0.34.0"
  ensure_python_requirement jsonlines "jsonlines==4.0.0"
  ensure_python_requirement numpy "numpy==1.26.4"
  ensure_python_requirement omegaconf "omegaconf==2.3.0"
  ensure_python_requirement PIL "pillow==12.0.0"
  ensure_python_requirement pyarrow "pyarrow==23.0.0"
  ensure_python_requirement rich "rich==14.2.0"
  ensure_python_requirement safetensors "safetensors>=0.5.3"
  ensure_python_requirement termcolor "termcolor==2.5.0"
  ensure_python_requirement yaml PyYAML
  ensure_python_requirement trimesh "trimesh==4.4.3"
  ensure_python_requirement open3d "${open3d_spec}"
  ensure_python_requirement imageio "imageio==2.34.2"
  ensure_python_requirement moviepy.editor "moviepy==1.0.3"
  ensure_python_requirement warp "warp-lang==1.12.1"
  ensure_python_requirement zarr "zarr<3"
  ensure_python_requirement h5py h5py
  ensure_python_requirement pyglet "pyglet<2"
  ensure_python_requirement cv2 "opencv-python-headless==4.10.0.84"
  ensure_python_requirement ffmpeg ffmpeg-python
  ensure_python_requirement imageio_ffmpeg imageio-ffmpeg
  ensure_python_requirement timm "timm>=1.0.19"
  ensure_python_requirement tqdm "tqdm==4.66.5"
}

install_python_dependencies() {
  local marker="${PYTHON_ENV_PREFIX}/.fastwam_robotwin_h100_eval_v3"
  if [[ "${FORCE_EVAL_ENV_INSTALL}" != "1" && \
        -f "${marker}" ]] && \
      curobo_checkout_ready && \
      curobo_python_ready && \
      python_environment_ready; then
    echo "[h100-setup] Reusing Python environment: ${PYTHON_ENV_PREFIX}"
    return
  fi

  if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV}" == "1" ]]; then
    install_current_python_dependencies
  else
    if ! command -v ffmpeg >/dev/null 2>&1; then
      conda install -y -p "${FASTWAM_EVAL_ENV}" -c conda-forge "ffmpeg=7.1"
    fi
    "${PYTHON_BIN}" -m pip install --upgrade \
      pip "${SETUPTOOLS_REQUIREMENT}" wheel ninja
    "${PYTHON_BIN}" -m pip install \
      "${PIP_REINSTALL_ARGS[@]}" \
      torch==2.7.1+cu128 \
      torchvision==0.22.1+cu128 \
      --extra-index-url https://download.pytorch.org/whl/cu128
    "${PYTHON_BIN}" -m pip install "${PIP_REINSTALL_ARGS[@]}" -e "${REPO_ROOT}"
    "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
      "${PIP_REINSTALL_ARGS[@]}" \
      "transformers==5.12.1" \
      "huggingface-hub>=0.34.0" \
      "safetensors>=0.5.3" \
      "timm>=1.0.19"
    "${PYTHON_BIN}" -m pip uninstall -y torchaudio || true
    "${PYTHON_BIN}" -m pip install --no-cache-dir \
      "${PIP_REINSTALL_ARGS[@]}" \
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
  fi

  if [[ ! -d "${CUROBO_ROOT}/.git" ]]; then
    if [[ -e "${CUROBO_ROOT}" ]]; then
      echo "[h100-setup] ERROR: non-git cuRobo path exists: ${CUROBO_ROOT}" >&2
      exit 1
    fi
    git clone --branch "${CUROBO_REVISION}" --depth 1 \
      https://github.com/NVlabs/curobo.git "${CUROBO_ROOT}"
  fi
  if ! curobo_checkout_ready; then
    echo "[h100-setup] ERROR: cuRobo checkout is not ${CUROBO_REVISION}." >&2
    exit 1
  fi
  if [[ "${FORCE_EVAL_ENV_INSTALL}" == "1" ]] || ! curobo_python_ready; then
    install_cuda_toolkit_for_torch
    "${PYTHON_BIN}" -m pip install --upgrade \
      "${SETUPTOOLS_REQUIREMENT}" wheel ninja
    "${PYTHON_BIN}" -m pip install "${PIP_REINSTALL_ARGS[@]}" \
      -e "${CUROBO_ROOT}" --no-build-isolation
    if ! curobo_python_ready; then
      echo "[h100-setup] ERROR: cuRobo compiled modules failed to import after installation." >&2
      curobo_python_ready --verbose || true
      exit 1
    fi
  else
    echo "[h100-setup] Reusing cuRobo ${CUROBO_REVISION} from ${CUROBO_ROOT}."
  fi

  "${PYTHON_BIN}" - <<'PY'
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
  if [[ "${USE_SYSTEM_NVIDIA_GRAPHICS}" == "1" ]]; then
    local system_vulkan
    if system_vulkan="$(vulkaninfo --summary 2>&1)" && \
        grep -q "NVIDIA" <<<"${system_vulkan}"; then
      echo "[h100-setup] Reusing the system NVIDIA Vulkan/EGL stack."
      return
    fi
    echo "[h100-setup] System Vulkan is unavailable; preparing matching NVIDIA userspace libraries."
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
  mkdir -p \
    "$(dirname "${VJEPA21_CHECKPOINT}")" \
    "$(dirname "${VJEPA21_VITL_CHECKPOINT}")"
  if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
    curl --fail --location --retry 5 --continue-at - \
      "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt" \
      --output "${VJEPA21_CHECKPOINT}.download"
    mv "${VJEPA21_CHECKPOINT}.download" "${VJEPA21_CHECKPOINT}"
  fi
  local vjepa_size
  vjepa_size="$(stat -c '%s' "${VJEPA21_CHECKPOINT}")"
  if [[ "${vjepa_size}" != "${VJEPA21_VITG_EXPECTED_SIZE}" ]]; then
    echo "[h100-setup] ERROR: invalid V-JEPA checkpoint size ${vjepa_size}." >&2
    exit 1
  fi
  if [[ ! -f "${VJEPA21_VITL_CHECKPOINT}" ]]; then
    curl --fail --location --retry 5 --continue-at - \
      "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt" \
      --output "${VJEPA21_VITL_CHECKPOINT}.download"
    mv "${VJEPA21_VITL_CHECKPOINT}.download" "${VJEPA21_VITL_CHECKPOINT}"
  fi
  local vjepa_vitl_size
  vjepa_vitl_size="$(stat -c '%s' "${VJEPA21_VITL_CHECKPOINT}")"
  if [[ "${vjepa_vitl_size}" != "${VJEPA21_VITL_EXPECTED_SIZE}" ]]; then
    echo "[h100-setup] ERROR: invalid V-JEPA ViT-L checkpoint size ${vjepa_vitl_size}." >&2
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

  if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  fi
  export DINOV3_MODEL_PATH DINOV3_REVISION
  if [[ ! -f "${DINOV3_MODEL_PATH}/config.json" || \
        ! -f "${DINOV3_MODEL_PATH}/model.safetensors" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "[h100-setup] ERROR: DINOv3 is gated; set HF_TOKEN or ~/.hf_token." >&2
      exit 1
    fi
    "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
    revision=os.environ["DINOV3_REVISION"],
    local_dir=os.environ["DINOV3_MODEL_PATH"],
    allow_patterns=["config.json", "model.safetensors"],
    token=os.environ["HF_TOKEN"],
)
PY
  fi

  export SIGLIP2_MODEL_PATH
  if [[ ! -f "${SIGLIP2_MODEL_PATH}/config.json" || \
        ! -f "${SIGLIP2_MODEL_PATH}/model.safetensors" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-so400m-patch16-384",
    local_dir=os.environ["SIGLIP2_MODEL_PATH"],
    allow_patterns=["config.json", "model.safetensors"],
)
PY
  fi

  export SIGLIP2_LARGE_MODEL_PATH SIGLIP2_LARGE_REVISION
  if [[ ! -f "${SIGLIP2_LARGE_MODEL_PATH}/config.json" || \
        ! -f "${SIGLIP2_LARGE_MODEL_PATH}/model.safetensors" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-large-patch16-384",
    revision=os.environ["SIGLIP2_LARGE_REVISION"],
    local_dir=os.environ["SIGLIP2_LARGE_MODEL_PATH"],
    allow_patterns=["config.json", "model.safetensors"],
)
PY
  fi

  DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints" \
  DIFFSYNTH_SKIP_DOWNLOAD=false \
  DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}" \
  "${PYTHON_BIN}" - <<'PY'
from fastwam.models.wan22.helpers.loader import _resolve_configs

_, _, vae_config, _ = _resolve_configs(
    model_id="Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
    redirect_common_files=True,
)
vae_config.download_if_necessary()
print(f"[h100-setup] Wan VAE: {vae_config.path}")
PY

  export QWEN_DIR QWEN_REVISION
  if ! "${PYTHON_BIN}" - <<'PY'
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
    "${PYTHON_BIN}" - <<'PY'
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

ensure_vjepa21_normalise_stats() {
  if [[ -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
    return
  fi
  local legacy_stats="${REPO_ROOT}/data/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt"
  if [[ -f "${legacy_stats}" ]]; then
    mkdir -p "$(dirname "${VJEPA21_NORMALISE_STATS_PATH}")"
    cp "${legacy_stats}" "${VJEPA21_NORMALISE_STATS_PATH}"
    return
  fi
  if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
    echo "[h100-setup] ERROR: cannot generate V-JEPA stats; WebDataset is missing:" >&2
    echo "  ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
    exit 1
  fi
  echo "[h100-setup] Computing missing V-JEPA2.1 ViT-G global statistics."
  ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT}" \
  VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT}" \
  VJEPA21_REPO="${VJEPA21_REPO}" \
  VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH}" \
    bash scripts/precompute_robotwin_vjepa21_global_stats_single8.sh
}

install_system_dependencies
if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV}" != "1" ]]; then
  install_miniforge
fi
activate_eval_environment
install_python_dependencies
setup_nvidia_graphics
setup_robotwin
download_model_assets
ensure_vjepa21_normalise_stats

export HF_HOME TORCH_HOME ROBOTWIN_ROOT VJEPA21_CHECKPOINT
export VJEPA21_VITL_CHECKPOINT VJEPA21_REPO QWEN_DIR
export DINOV3_MODEL_PATH SIGLIP2_MODEL_PATH SIGLIP2_LARGE_MODEL_PATH
export ROBOTWIN_WEBDATASET_ROOT VJEPA21_NORMALISE_STATS_PATH
"${PYTHON_BIN}" scripts/check_robotwin_h100_eval_env.py \
  --robotwin-root "${ROBOTWIN_ROOT}"

cat <<EOF
[h100-setup] Environment is ready.
PYTHON_BIN=${PYTHON_BIN}
FASTWAM_EVAL_ENV=${FASTWAM_EVAL_ENV}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
VJEPA21_CHECKPOINT=${VJEPA21_CHECKPOINT}
VJEPA21_VITL_CHECKPOINT=${VJEPA21_VITL_CHECKPOINT}
VJEPA21_REPO=${VJEPA21_REPO}
VJEPA21_SOURCE_REVISION=${VJEPA21_SOURCE_REVISION}
DINOV3_MODEL_PATH=${DINOV3_MODEL_PATH}
SIGLIP2_MODEL_PATH=${SIGLIP2_MODEL_PATH}
SIGLIP2_LARGE_MODEL_PATH=${SIGLIP2_LARGE_MODEL_PATH}
VJEPA21_NORMALISE_STATS_PATH=${VJEPA21_NORMALISE_STATS_PATH}
QWEN_DIR=${QWEN_DIR}
QWEN_REVISION=${QWEN_REVISION}
NVIDIA_GRAPHICS_ENV=${NVIDIA_GRAPHICS_ENV}
EOF
