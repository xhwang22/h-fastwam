#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


BUNDLE_TYPE = "robotwin_eval_bundle"
MANIFEST_VERSION = 1
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.resolve()
DEFAULT_BUNDLE_DIR = "artifacts/robotwin_eval_bundle"
DEFAULT_DOWNLOAD_DIR = "checkpoints/robotwin_eval_bundle"
DEFAULT_REPO_REVISION = "main"
DEFAULT_UPLOAD_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = 8
SHA256_CHUNK_SIZE = 1024 * 1024
VJEPA21_CHECKPOINT_ENV = "${oc.env:VJEPA21_CHECKPOINT}"
VJEPA21_REPO_ENV = "${oc.env:VJEPA21_REPO}"

DEFAULT_MODEL_RUNS: "OrderedDict[str, str]" = OrderedDict(
    [
        (
            "qwen3vl",
            "runs/robotwin_hfastwam/robotwin_qwen3vl_causal_tubelet_32gpu_b48_cudnn_overlap_efa",
        ),
        (
            "vjepa21",
            "runs/robotwin_hfastwam/robotwin_vjepa21_causal_tubelet_32gpu_b48_cudnn_overlap_efa",
        ),
        (
            "vjepa21_predictor",
            "runs/robotwin_hfastwam/robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa",
        ),
    ]
)
EXPECTED_MODEL_NAMES = tuple(DEFAULT_MODEL_RUNS.keys())
STEP_PREFIX = "step_"
CHECKPOINT_SUFFIX = ".pt"


class BundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceState:
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class FileDigest:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SelectedRun:
    model_name: str
    run_path: Path
    run_name: str
    config_source: Path
    checkpoint_source: Path
    step_int: int
    step_name: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got: {value}")
    return parsed


def resolve_user_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def bundle_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise BundleError(f"Manifest path must use forward slashes only: {value}")
    rel = PurePosixPath(value)
    if rel.is_absolute():
        raise BundleError(f"Manifest path must be relative: {value}")
    if str(rel) in {"", ".", ".."}:
        raise BundleError(f"Invalid manifest path: {value}")
    if any(part in {"", ".", ".."} for part in rel.parts):
        raise BundleError(f"Manifest path contains traversal: {value}")
    return rel


def materialize_bundle_path(bundle_dir: Path, relative_value: str) -> Path:
    rel = bundle_relative_path(relative_value)
    return bundle_dir.joinpath(*rel.parts)


def safe_model_name(name: str) -> str:
    value = str(name).strip()
    if value not in DEFAULT_MODEL_RUNS:
        raise argparse.ArgumentTypeError(
            f"Unknown model name `{name}`. Expected one of: {', '.join(EXPECTED_MODEL_NAMES)}"
        )
    return value


def parse_run_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, raw_path = value.split("=", 1)
    model_name = safe_model_name(name)
    run_path = raw_path.strip()
    if not run_path:
        raise argparse.ArgumentTypeError("Run path must not be empty.")
    return model_name, run_path


def quoted(path: str | Path) -> str:
    return shlex.quote(str(path))


def temp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    tmp_path = temp_sibling(path)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_executable_text_atomic(path: Path, text: str) -> FileDigest:
    ensure_directory(path.parent)
    tmp_path = temp_sibling(path)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.chmod(tmp_path, 0o755)
        sha256, size = sha256_file(tmp_path)
        tmp_path.replace(path)
        os.chmod(path, 0o755)
        return FileDigest(path=str(path), size=size, sha256=sha256)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def describe_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def capture_source_state(path: Path, label: str) -> SourceState:
    if not path.is_file():
        raise BundleError(f"Required {label} file is missing: {path}")
    stat_result = path.stat()
    if stat_result.st_size <= 0:
        raise BundleError(f"Required {label} file is empty: {path}")
    return SourceState(size=stat_result.st_size, mtime_ns=stat_result.st_mtime_ns)


def ensure_source_stable(path: Path, before: SourceState, label: str) -> None:
    after = capture_source_state(path, label)
    if after != before:
        raise BundleError(
            f"{label} changed while staging: {path} "
            f"(before size={before.size} mtime_ns={before.mtime_ns}, "
            f"after size={after.size} mtime_ns={after.mtime_ns})"
        )


def stage_file_atomic(src: Path, dst: Path, label: str) -> tuple[FileDigest, str]:
    ensure_directory(dst.parent)
    before = capture_source_state(src, label)
    tmp_path = temp_sibling(dst)
    method = "copy2"
    try:
        shutil.copy2(src, tmp_path)
        sha256, size = sha256_file(tmp_path)
        ensure_source_stable(src, before, label)
        tmp_path.replace(dst)
        return FileDigest(path=str(dst), size=size, sha256=sha256), method
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def rewrite_vjepa_paths(cfg: Any) -> list[str]:
    rewritten: list[str] = []

    def visit(node: Any, path_parts: tuple[str, ...], under_visual_encoder: bool) -> None:
        if OmegaConf.is_dict(node):
            for key in list(node.keys()):
                key_str = str(key)
                value = node[key]
                next_under_visual = under_visual_encoder or key_str == "visual_encoder_config"
                if next_under_visual and key_str == "checkpoint_path":
                    if value != VJEPA21_CHECKPOINT_ENV:
                        node[key] = VJEPA21_CHECKPOINT_ENV
                        rewritten.append(".".join((*path_parts, key_str)))
                    continue
                if next_under_visual and key_str == "repo_path":
                    if value != VJEPA21_REPO_ENV:
                        node[key] = VJEPA21_REPO_ENV
                        rewritten.append(".".join((*path_parts, key_str)))
                    continue
                visit(value, (*path_parts, key_str), next_under_visual)
        elif OmegaConf.is_list(node):
            for index, value in enumerate(node):
                visit(value, (*path_parts, str(index)), under_visual_encoder)

    visit(cfg, tuple(), False)
    return rewritten


def stage_config_atomic(src: Path, dst: Path) -> tuple[FileDigest, list[str]]:
    ensure_directory(dst.parent)
    before = capture_source_state(src, "config")
    tmp_path = temp_sibling(dst)
    try:
        shutil.copy2(src, tmp_path)
        cfg = OmegaConf.load(tmp_path)
        if not OmegaConf.is_config(cfg) or not OmegaConf.is_dict(cfg):
            raise BundleError(f"Config must be a YAML mapping: {src}")
        rewritten_keys = rewrite_vjepa_paths(cfg)
        OmegaConf.save(config=cfg, f=str(tmp_path))
        sha256, size = sha256_file(tmp_path)
        if size <= 0:
            raise BundleError(f"Rewritten config became empty: {src}")
        ensure_source_stable(src, before, "config")
        tmp_path.replace(dst)
        return FileDigest(path=str(dst), size=size, sha256=sha256), rewritten_keys
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run_git(args: Sequence[str], repo_root: Path = REPO_ROOT) -> str:
    cmd = ["git", "-C", str(repo_root), "--no-pager", *args]
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise BundleError(f"Git command failed: {' '.join(cmd)}\n{message}")
    return completed.stdout.strip()


def resolve_code_info(
    requested_revision: str | None,
    *,
    allow_dirty: bool,
) -> dict[str, Any]:
    repo_root = Path(run_git(["rev-parse", "--show-toplevel"])).resolve(strict=False)
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root=repo_root)
    dirty = bool(run_git(["status", "--porcelain"], repo_root=repo_root))
    if dirty and not allow_dirty:
        raise BundleError(
            "Git worktree has tracked or untracked changes. Commit/stash them, "
            "or pass --allow-dirty if the mismatch is intentional."
        )
    revision = (requested_revision or "HEAD").strip()
    if not revision:
        revision = "HEAD"
    run_git(["cat-file", "-e", f"{revision}^{{commit}}"], repo_root=repo_root)
    sha = run_git(["rev-parse", f"{revision}^{{commit}}"], repo_root=repo_root)
    return {
        "requested_revision": revision,
        "resolved_sha": sha,
        "sha": sha,
        "dirty": dirty,
        "branch": branch,
        "source_repo_root": str(repo_root),
    }


def resolve_bundle_guarded(bundle_dir_arg: str, *, allow_existing: bool = True) -> Path:
    bundle_dir = resolve_user_path(bundle_dir_arg)
    forbidden_paths = {
        REPO_ROOT.resolve(strict=False),
        Path.home().resolve(strict=False),
        Path("/").resolve(strict=False),
    }
    if bundle_dir in forbidden_paths:
        raise BundleError(f"Refusing dangerous bundle directory: {bundle_dir}")
    if bundle_dir.exists() and not bundle_dir.is_dir():
        raise BundleError(f"Bundle path exists but is not a directory: {bundle_dir}")
    if bundle_dir.exists() and not allow_existing:
        raise BundleError(f"Output directory already exists: {bundle_dir}")
    return bundle_dir


def reject_unexpected_bundle_files(
    bundle_dir: Path,
    expected_relative_paths: set[str],
) -> None:
    if not bundle_dir.is_dir():
        return
    unexpected = []
    for path in bundle_dir.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(bundle_dir).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        if relative not in expected_relative_paths:
            unexpected.append(relative)
    if unexpected:
        raise BundleError(
            "Bundle directory contains unexpected files that will not be packed: "
            + ", ".join(sorted(unexpected)[:20])
        )


def resolve_stats_path(stats_path_arg: str | None) -> Path:
    if stats_path_arg:
        path = resolve_user_path(stats_path_arg)
        capture_source_state(path, "dataset stats")
        return path

    candidates = [
        REPO_ROOT / "data/robotwin2.0_webdataset/dataset_stats.json",
        REPO_ROOT / "data/robotwin2.0/dataset_stats.json",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve(strict=False)
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise BundleError(
        "Could not find dataset stats JSON. Checked, in order:\n"
        f"{formatted}\nUse --stats-path to set it explicitly."
    )


def resolve_runs(run_overrides: Sequence[tuple[str, str]] | None) -> OrderedDict[str, Path]:
    runs: OrderedDict[str, Path] = OrderedDict(
        (name, resolve_user_path(rel_path)) for name, rel_path in DEFAULT_MODEL_RUNS.items()
    )
    for model_name, run_path in run_overrides or []:
        runs[model_name] = resolve_user_path(run_path)
    return runs


def parse_checkpoint_step(value: str) -> int:
    text = str(value).strip()
    if text in {"latest-common", "latest-each"}:
        raise BundleError(f"Expected explicit step, got selector: {value}")
    candidate = text[len(STEP_PREFIX):] if text.startswith(STEP_PREFIX) else text
    if not candidate.isdigit():
        raise BundleError(f"Invalid step selector: {value}")
    return int(candidate)


def checkpoint_step_name(step_int: int) -> str:
    return f"{STEP_PREFIX}{step_int:06d}"


def discover_checkpoint_steps(run_path: Path) -> dict[int, Path]:
    weights_dir = run_path / "checkpoints" / "weights"
    if not weights_dir.is_dir():
        raise BundleError(f"Checkpoint directory not found: {weights_dir}")
    found: dict[int, Path] = {}
    for path in sorted(weights_dir.glob(f"{STEP_PREFIX}*{CHECKPOINT_SUFFIX}")):
        if not path.is_file():
            continue
        name = path.name
        if not (name.startswith(STEP_PREFIX) and name.endswith(CHECKPOINT_SUFFIX)):
            continue
        step_text = name[len(STEP_PREFIX):-len(CHECKPOINT_SUFFIX)]
        if not step_text.isdigit():
            continue
        found[int(step_text)] = path.resolve(strict=False)
    if not found:
        raise BundleError(f"No checkpoint files found under: {weights_dir}")
    return found


def select_runs(
    run_mapping: Mapping[str, Path],
    step_selector: str,
) -> tuple[list[SelectedRun], dict[str, Any]]:
    steps_by_model: OrderedDict[str, dict[int, Path]] = OrderedDict()
    for model_name, run_path in run_mapping.items():
        if not run_path.is_dir():
            raise BundleError(f"Run directory not found for {model_name}: {run_path}")
        steps_by_model[model_name] = discover_checkpoint_steps(run_path)

    selector = step_selector.strip()
    selection: dict[str, Any]
    if selector == "latest-common":
        common_steps: set[int] | None = None
        for step_map in steps_by_model.values():
            step_set = set(step_map.keys())
            common_steps = step_set if common_steps is None else common_steps & step_set
        if not common_steps:
            latest_lines = []
            for model_name, step_map in steps_by_model.items():
                latest = max(step_map) if step_map else None
                latest_lines.append(f"{model_name}={checkpoint_step_name(latest) if latest is not None else 'none'}")
            raise BundleError(
                "No common checkpoint step exists across all runs. "
                f"Latest available per run: {', '.join(latest_lines)}"
            )
        selected_step = max(common_steps)
        selection = {"requested": selector, "mode": "latest-common"}
        chosen = {model_name: step_map[selected_step] for model_name, step_map in steps_by_model.items()}
    elif selector == "latest-each":
        selection = {
            "requested": selector,
            "mode": "latest-each",
            "warning": "latest-each selected different checkpoint steps across models; comparison is not fair.",
        }
        chosen = {model_name: step_map[max(step_map)] for model_name, step_map in steps_by_model.items()}
    else:
        explicit_step = parse_checkpoint_step(selector)
        missing = [model_name for model_name, step_map in steps_by_model.items() if explicit_step not in step_map]
        if missing:
            missing_text = ", ".join(missing)
            raise BundleError(
                f"Requested step {checkpoint_step_name(explicit_step)} is missing in: {missing_text}"
            )
        selection = {"requested": selector, "mode": "explicit"}
        chosen = {model_name: step_map[explicit_step] for model_name, step_map in steps_by_model.items()}

    selected_runs: list[SelectedRun] = []
    for model_name, run_path in run_mapping.items():
        checkpoint_path = chosen[model_name]
        step_text = checkpoint_path.stem
        step_int = parse_checkpoint_step(step_text)
        config_path = run_path / "config.yaml"
        capture_source_state(config_path, "config")
        capture_source_state(checkpoint_path, "checkpoint")
        selected_runs.append(
            SelectedRun(
                model_name=model_name,
                run_path=run_path,
                run_name=run_path.name,
                config_source=config_path.resolve(strict=False),
                checkpoint_source=checkpoint_path.resolve(strict=False),
                step_int=step_int,
                step_name=step_text,
            )
        )
    return selected_runs, selection


def make_file_record(path: str, digest: FileDigest) -> dict[str, Any]:
    return {"path": path, "size": digest.size, "sha256": digest.sha256}


def make_auxiliary_record(name: str, path: str, digest: FileDigest, *, executable: bool) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "size": digest.size,
        "sha256": digest.sha256,
        "executable": executable,
    }


def build_run_eval_script_text() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail

        fail() {
          echo "[run_eval] ERROR: $*" >&2
          exit 1
        }

        require_positive_int() {
          local name="$1"
          local value="$2"
          [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer, got: $value"
        }

        normalize_bool() {
          local value
          value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
          case "$value" in
            1|true|yes|y) printf 'true\\n' ;;
            0|false|no|n) printf 'false\\n' ;;
            *) fail "Invalid boolean value: $1" ;;
          esac
        }

        SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
        BUNDLE_DIR="$SCRIPT_DIR"
        MANIFEST_PATH="$BUNDLE_DIR/manifest.json"
        PYTHON_BIN="${PYTHON:-python3}"
        FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-$(pwd)}"
        REPO_ROOT="$(cd "$FASTWAM_REPO_ROOT" && pwd)"

        [[ -f "$MANIFEST_PATH" ]] || fail "Manifest not found: $MANIFEST_PATH"
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"
        [[ -f "$REPO_ROOT/experiments/robotwin/run_robotwin_manager.py" ]] || fail \
          "run_robotwin_manager.py not found under FASTWAM_REPO_ROOT=$REPO_ROOT"

        mapfile -t MANIFEST_INFO < <(
          "$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
        import hashlib
        import json
        import os
        import sys
        from pathlib import Path, PurePosixPath

        manifest_path = sys.argv[1]
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        bundle_dir = Path(manifest_path).resolve().parent

        def verify_file(record, label):
            raw_path = str(record["path"])
            if "\\\\" in raw_path:
                raise SystemExit(f"{label}: backslashes are not allowed in manifest paths")
            relative = PurePosixPath(raw_path)
            if (
                relative.is_absolute()
                or str(relative) in {"", ".", ".."}
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise SystemExit(f"{label}: invalid bundle-relative path: {raw_path}")
            path = bundle_dir.joinpath(*relative.parts)
            if not path.is_file():
                raise SystemExit(f"{label}: file not found: {path}")
            stat = path.stat()
            if stat.st_size != int(record["size"]):
                raise SystemExit(
                    f"{label}: size mismatch for {path}: "
                    f"{stat.st_size} != {record['size']}"
                )
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != str(record["sha256"]):
                raise SystemExit(f"{label}: sha256 mismatch for {path}")

        if manifest.get("manifest_version") != 1 or manifest.get("bundle_type") != "robotwin_eval_bundle":
            raise SystemExit("Unsupported or invalid bundle manifest")
        verify_file(manifest["dataset_stats"], "dataset_stats")
        for model in manifest["models"]:
            verify_file(model["checkpoint"], f"{model['name']} checkpoint")
            verify_file(model["config"], f"{model['name']} config")
        for auxiliary in manifest["auxiliary_files"]:
            verify_file(auxiliary, f"auxiliary {auxiliary['name']}")

        code = manifest["code"]
        print(code["resolved_sha"])
        print(manifest["dataset_stats"]["path"])
        print(",".join(model["name"] for model in manifest["models"]))
        PY
        )
        [[ "${#MANIFEST_INFO[@]}" -ge 3 ]] || fail "Failed to read required metadata from manifest"
        CODE_RESOLVED_SHA="${MANIFEST_INFO[0]}"
        STATS_REL_PATH="${MANIFEST_INFO[1]}"
        AVAILABLE_MODELS_CSV="${MANIFEST_INFO[2]}"

        if [[ "${ALLOW_CODE_MISMATCH:-0}" != "1" ]]; then
          git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail \
            "FASTWAM_REPO_ROOT is not a git checkout: $REPO_ROOT"
          HEAD_SHA="$(git -C "$REPO_ROOT" --no-pager rev-parse HEAD)"
          [[ "$HEAD_SHA" == "$CODE_RESOLVED_SHA" ]] || fail \
            "Git HEAD ($HEAD_SHA) does not match bundle code.resolved_sha ($CODE_RESOLVED_SHA). " \
            "Run 'git checkout $CODE_RESOLVED_SHA' or set ALLOW_CODE_MISMATCH=1."
        fi
        if [[ "${ALLOW_DIRTY_WORKTREE:-0}" != "1" ]]; then
          [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]] || fail \
            "Tracked files differ from the checked-out revision. Commit/stash them or set ALLOW_DIRTY_WORKTREE=1."
        fi

        export HF_HOME="${HF_HOME:-$REPO_ROOT/checkpoints/hf_cache}"
        export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/checkpoints/torch_hub}"
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
        export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
        export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}"
        export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
        export VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-$TORCH_HOME/hub/checkpoints/vjepa2_1_vitG_384.pt}"
        export VJEPA21_REPO="${VJEPA21_REPO:-$TORCH_HOME/hub/facebookresearch_vjepa2_main}"
        ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$REPO_ROOT/third_party/RoboTwin}"
        mkdir -p "$HF_HOME" "$TORCH_HOME"
        [[ -d "$ROBOTWIN_ROOT" ]] || fail "RoboTwin root not found: $ROBOTWIN_ROOT"

        MODEL="${MODEL:-all}"
        EVAL_MODE="${EVAL_MODE:-full}"
        NUM_GPUS="${NUM_GPUS:-1}"
        MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
        require_positive_int NUM_GPUS "$NUM_GPUS"
        require_positive_int MAX_TASKS_PER_GPU "$MAX_TASKS_PER_GPU"

        case "$EVAL_MODE" in
          smoke)
            EVAL_EPISODES="${EVAL_EPISODES:-2}"
            EVAL_VIDEO_LOG="$(normalize_bool "${EVAL_VIDEO_LOG:-false}")"
            SMOKE_TASK="${SMOKE_TASK:-$(
              "$PYTHON_BIN" - "$ROBOTWIN_ROOT/task_config/_eval_step_limit.yml" <<'PY'
        import sys

        task_file = sys.argv[1]
        with open(task_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line[:1].isspace():
                    continue
                key = stripped.split(":", 1)[0].strip().strip("'").strip('"')
                if key:
                    print(key)
                    break
            else:
                raise SystemExit("No smoke task found in " + task_file)
        PY
            )}"
            ;;
          full)
            EVAL_EPISODES="${EVAL_EPISODES:-100}"
            EVAL_VIDEO_LOG="$(normalize_bool "${EVAL_VIDEO_LOG:-false}")"
            SMOKE_TASK="${SMOKE_TASK:-}"
            ;;
          *)
            fail "Unsupported EVAL_MODE: $EVAL_MODE (expected smoke or full)"
            ;;
        esac
        require_positive_int EVAL_EPISODES "$EVAL_EPISODES"

        STATS_PATH="$BUNDLE_DIR/$STATS_REL_PATH"
        [[ -f "$STATS_PATH" ]] || fail "Bundled dataset stats not found: $STATS_PATH"

        while IFS=$'\\t' read -r MODEL_NAME CHECKPOINT_REL CONFIG_REL; do
          [[ -n "$MODEL_NAME" ]] || continue
          CHECKPOINT_PATH="$BUNDLE_DIR/$CHECKPOINT_REL"
          CONFIG_PATH="$BUNDLE_DIR/$CONFIG_REL"
          [[ -f "$CHECKPOINT_PATH" ]] || fail "Checkpoint missing for $MODEL_NAME: $CHECKPOINT_PATH"
          [[ -f "$CONFIG_PATH" ]] || fail "Config missing for $MODEL_NAME: $CONFIG_PATH"

          RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
          OUTPUT_TAG="bundle_${MODEL_NAME}_${EVAL_MODE}_${RUN_TS}"
          CMD=(
            "$PYTHON_BIN"
            "$REPO_ROOT/experiments/robotwin/run_robotwin_manager.py"
            "ckpt=$CHECKPOINT_PATH"
            "EVALUATION.train_config_path=$CONFIG_PATH"
            "EVALUATION.dataset_stats_path=$STATS_PATH"
            "EVALUATION.robotwin_root=$ROBOTWIN_ROOT"
            "MULTIRUN.num_gpus=$NUM_GPUS"
            "MULTIRUN.max_tasks_per_gpu=$MAX_TASKS_PER_GPU"
            "EVALUATION.eval_num_episodes=$EVAL_EPISODES"
            "EVALUATION.eval_video_log=$EVAL_VIDEO_LOG"
            "EVALUATION.output_dir=./evaluate_results/robotwin_bundle/$OUTPUT_TAG"
          )
          if [[ "$EVAL_MODE" == "smoke" ]]; then
            CMD+=("EVALUATION.task_name=$SMOKE_TASK")
          fi

          echo "[run_eval] Repo root: $REPO_ROOT"
          echo "[run_eval] Bundle dir: $BUNDLE_DIR"
          echo "[run_eval] Model: $MODEL_NAME"
          echo "[run_eval] Mode: $EVAL_MODE episodes=$EVAL_EPISODES num_gpus=$NUM_GPUS max_tasks_per_gpu=$MAX_TASKS_PER_GPU"
          if [[ "$EVAL_MODE" == "smoke" ]]; then
            echo "[run_eval] Smoke task: $SMOKE_TASK"
          fi
          (
            cd "$REPO_ROOT"
            "${CMD[@]}"
          )
        done < <(
          "$PYTHON_BIN" - "$MANIFEST_PATH" "$MODEL" <<'PY'
        import json
        import sys

        manifest_path = sys.argv[1]
        requested_model = sys.argv[2]
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        models = {model["name"]: model for model in manifest["models"]}
        if requested_model == "all":
            selected_names = [model["name"] for model in manifest["models"]]
        else:
            if requested_model not in models:
                available = ",".join(model["name"] for model in manifest["models"])
                raise SystemExit(
                    f"Unknown MODEL={requested_model}. Available models: {available}"
                )
            selected_names = [requested_model]

        for name in selected_names:
            model = models[name]
            print(
                "\\t".join(
                    [
                        name,
                        str(model["checkpoint"]["path"]),
                        str(model["config"]["path"]),
                    ]
                )
            )
        PY
        )

        echo "[run_eval] Completed models: $MODEL (available: $AVAILABLE_MODELS_CSV)"
        """
    )


def build_readme(manifest: Mapping[str, Any], bundle_dir: Path) -> str:
    code_sha = manifest["code"]["resolved_sha"]
    warning_line = ""
    selection_warning = manifest["selection"].get("warning")
    if selection_warning:
        warning_line = f"\nWarning: {selection_warning}\n"

    model_lines = []
    for model in manifest["models"]:
        model_lines.append(
            f"- {model['name']}: {model['step']} -> {model['checkpoint']['path']}"
        )
    example_eval = (
        "python experiments/robotwin/run_robotwin_manager.py "
        "task=robotwin_uncond_3cam_384_1e-4 "
        f"ckpt={quoted(bundle_dir / manifest['models'][0]['checkpoint']['path'])} "
        f"EVALUATION.train_config_path={quoted(bundle_dir / manifest['models'][0]['config']['path'])} "
        f"EVALUATION.dataset_stats_path={quoted(bundle_dir / manifest['dataset_stats']['path'])} "
        "MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1"
    )
    launcher_example = f"(cd /path/to/fastwam_checkout && MODEL=all {quoted(bundle_dir / 'run_eval.sh')})"
    return textwrap.dedent(
        f"""\
        # RoboTwin evaluation bundle

        This bundle contains staged checkpoints, configs, and dataset stats for RoboTwin evaluation.

        Required code checkout:
          git checkout {code_sha}

        Verify locally:
          python {quoted(SCRIPT_PATH)} verify --bundle-dir {quoted(bundle_dir)}

        Models:
        {os.linesep.join(model_lines)}
        {warning_line}If a bundled config references V-JEPA 2.1 assets, set:
          export VJEPA21_CHECKPOINT=/path/to/vjepa2_1_vitG_384.pt
          export VJEPA21_REPO=/path/to/facebookresearch_vjepa2_main

        Bundled launcher:
          {launcher_example}

        Example evaluation command:
          {example_eval}

        Replace <robotwin_task> with a valid Hydra task override.
        """
    ).rstrip() + "\n"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    separator = "-+-".join("-" * width for width in widths)
    lines = [" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)), separator]
    for row in rows:
        lines.append(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def pack_bundle(args: argparse.Namespace) -> int:
    bundle_dir = resolve_bundle_guarded(args.bundle_dir)
    expected_paths = {
        ".gitattributes",
        "manifest.json",
        "README.md",
        "dataset_stats.json",
        "run_eval.sh",
        *{
            f"models/{model_name}/{filename}"
            for model_name in EXPECTED_MODEL_NAMES
            for filename in ("checkpoint.pt", "config.yaml")
        },
    }
    reject_unexpected_bundle_files(bundle_dir, expected_paths)
    stats_source = resolve_stats_path(args.stats_path)
    selected_runs, selection = select_runs(resolve_runs(args.run), args.step)
    code_info = resolve_code_info(
        args.code_revision,
        allow_dirty=bool(args.allow_dirty),
    )

    models_dir = bundle_dir / "models"
    ensure_directory(models_dir)

    dataset_stats_dst = bundle_dir / "dataset_stats.json"
    stats_digest, stats_method = stage_file_atomic(
        stats_source,
        dataset_stats_dst,
        label="dataset stats",
    )

    model_rows: list[list[str]] = []
    manifest_models: list[dict[str, Any]] = []
    for selected in selected_runs:
        model_dir = models_dir / selected.model_name
        checkpoint_dst = model_dir / "checkpoint.pt"
        config_dst = model_dir / "config.yaml"

        checkpoint_digest, checkpoint_method = stage_file_atomic(
            selected.checkpoint_source,
            checkpoint_dst,
            label=f"{selected.model_name} checkpoint",
        )
        config_digest, rewritten_keys = stage_config_atomic(selected.config_source, config_dst)

        manifest_models.append(
            {
                "name": selected.model_name,
                "run_name": selected.run_name,
                "original_paths": {
                    "run": str(selected.run_path),
                    "config": str(selected.config_source),
                    "checkpoint": str(selected.checkpoint_source),
                },
                "step": selected.step_name,
                "step_int": selected.step_int,
                "checkpoint": make_file_record(
                    f"models/{selected.model_name}/checkpoint.pt",
                    checkpoint_digest,
                ),
                "config": make_file_record(
                    f"models/{selected.model_name}/config.yaml",
                    config_digest,
                ),
                "rewritten_config_keys": rewritten_keys,
            }
        )
        model_rows.append(
            [
                selected.model_name,
                selected.step_name,
                describe_size(checkpoint_digest.size),
                checkpoint_digest.sha256[:12],
                ",".join(rewritten_keys) if rewritten_keys else "-",
                checkpoint_method,
            ]
        )

    run_eval_digest = write_executable_text_atomic(bundle_dir / "run_eval.sh", build_run_eval_script_text())

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "bundle_type": BUNDLE_TYPE,
        "created_at_utc": utc_now_iso(),
        "code": code_info,
        "selection": selection,
        "dataset_stats": make_file_record("dataset_stats.json", stats_digest),
        "models": manifest_models,
        "auxiliary_files": [
            make_auxiliary_record(
                "run_eval.sh",
                "run_eval.sh",
                run_eval_digest,
                executable=True,
            )
        ],
    }

    readme_text = build_readme(manifest, bundle_dir)
    atomic_write_json(bundle_dir / "manifest.json", manifest)
    atomic_write_text(bundle_dir / "README.md", readme_text)
    verify_bundle_impl(bundle_dir, allow_subset=False)

    print(f"Packed bundle: {bundle_dir}")
    print(
        render_table(
            headers=["model", "step", "checkpoint size", "ckpt sha256", "rewritten keys", "ckpt stage"],
            rows=model_rows,
        )
    )
    print(
        f"dataset_stats: {describe_size(stats_digest.size)} sha256={stats_digest.sha256[:12]} stage={stats_method}"
    )
    print(f"auxiliary: run_eval.sh {describe_size(run_eval_digest.size)} sha256={run_eval_digest.sha256[:12]}")
    if "warning" in selection:
        print(f"WARNING: {selection['warning']}")
    next_cmd = (
        f"python {quoted(SCRIPT_PATH)} upload --bundle-dir {quoted(bundle_dir)} "
        "--repo-id <namespace/repo>"
    )
    print(f"Next upload command:\n{next_cmd}")
    return 0


def require_mapping(obj: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(obj, dict):
        raise BundleError(f"{label} must be an object, got {type(obj).__name__}")
    return obj


def require_list(obj: Any, label: str) -> list[Any]:
    if not isinstance(obj, list):
        raise BundleError(f"{label} must be a list, got {type(obj).__name__}")
    return obj


def require_exact_keys(
    obj: Mapping[str, Any],
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    keys = set(obj.keys())
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise BundleError(f"{label} is missing keys: {', '.join(sorted(missing))}")
    if extra:
        raise BundleError(f"{label} has unexpected keys: {', '.join(sorted(extra))}")


def require_str_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError(f"{label} must be a non-empty string")
    return value


def require_int_field(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise BundleError(f"{label} must be an integer")
    return value


def require_sha256_field(value: Any, label: str) -> str:
    text = require_str_field(value, label)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise BundleError(f"{label} must be a lowercase 64-character sha256 hex digest")
    return text


def require_git_sha_field(value: Any, label: str) -> str:
    text = require_str_field(value, label)
    if len(text) < 7 or any(ch not in "0123456789abcdef" for ch in text):
        raise BundleError(f"{label} must be a lowercase git commit sha")
    return text


def validate_manifest_file_record(entry: Mapping[str, Any], label: str) -> None:
    require_str_field(entry["path"], f"{label}.path")
    size = require_int_field(entry["size"], f"{label}.size")
    if size <= 0:
        raise BundleError(f"{label}.size must be > 0")
    require_sha256_field(entry["sha256"], f"{label}.sha256")


def validate_manifest_auxiliary_file(entry: Mapping[str, Any], label: str) -> None:
    require_exact_keys(entry, label, required={"name", "path", "size", "sha256", "executable"})
    require_str_field(entry["name"], f"{label}.name")
    validate_manifest_file_record(entry, label)
    if not isinstance(entry["executable"], bool):
        raise BundleError(f"{label}.executable must be a boolean")


def load_manifest(bundle_dir: Path) -> Mapping[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"Manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"Invalid JSON in manifest: {manifest_path}: {exc}") from exc
    manifest = require_mapping(payload, "manifest")
    require_exact_keys(
        manifest,
        "manifest",
        required={
            "manifest_version",
            "bundle_type",
            "created_at_utc",
            "code",
            "selection",
            "dataset_stats",
            "models",
            "auxiliary_files",
        },
    )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise BundleError(
            f"Unsupported manifest_version={manifest['manifest_version']}, expected {MANIFEST_VERSION}"
        )
    if manifest["bundle_type"] != BUNDLE_TYPE:
        raise BundleError(f"Unexpected bundle_type={manifest['bundle_type']}, expected {BUNDLE_TYPE}")
    require_str_field(manifest["created_at_utc"], "manifest.created_at_utc")
    try:
        datetime.strptime(str(manifest["created_at_utc"]), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise BundleError("manifest.created_at_utc must be UTC ISO format YYYY-MM-DDTHH:MM:SSZ") from exc

    code = require_mapping(manifest["code"], "manifest.code")
    require_exact_keys(
        code,
        "manifest.code",
        required={"requested_revision", "resolved_sha", "dirty", "branch", "source_repo_root"},
        optional={"sha"},
    )
    require_str_field(code["requested_revision"], "manifest.code.requested_revision")
    require_git_sha_field(code["resolved_sha"], "manifest.code.resolved_sha")
    if not isinstance(code["dirty"], bool):
        raise BundleError("manifest.code.dirty must be a boolean")
    require_str_field(code["branch"], "manifest.code.branch")
    require_str_field(code["source_repo_root"], "manifest.code.source_repo_root")
    if "sha" in code:
        require_git_sha_field(code["sha"], "manifest.code.sha")
        if code["sha"] != code["resolved_sha"]:
            raise BundleError("manifest.code.sha must match manifest.code.resolved_sha when present")

    selection = require_mapping(manifest["selection"], "manifest.selection")
    require_exact_keys(
        selection,
        "manifest.selection",
        required={"requested", "mode"},
        optional={"warning"},
    )
    require_str_field(selection["requested"], "manifest.selection.requested")
    if selection["mode"] not in {"latest-common", "latest-each", "explicit"}:
        raise BundleError(f"Invalid manifest.selection.mode: {selection['mode']}")
    if "warning" in selection:
        require_str_field(selection["warning"], "manifest.selection.warning")

    dataset_stats = require_mapping(manifest["dataset_stats"], "manifest.dataset_stats")
    require_exact_keys(dataset_stats, "manifest.dataset_stats", required={"path", "size", "sha256"})
    validate_manifest_file_record(dataset_stats, "manifest.dataset_stats")

    models = require_list(manifest["models"], "manifest.models")
    if not models:
        raise BundleError("manifest.models must not be empty")
    for index, model in enumerate(models):
        model_map = require_mapping(model, f"manifest.models[{index}]")
        require_exact_keys(
            model_map,
            f"manifest.models[{index}]",
            required={
                "name",
                "run_name",
                "original_paths",
                "step",
                "step_int",
                "checkpoint",
                "config",
                "rewritten_config_keys",
            },
        )
        original_paths = require_mapping(model_map["original_paths"], f"manifest.models[{index}].original_paths")
        require_exact_keys(
            original_paths,
            f"manifest.models[{index}].original_paths",
            required={"run", "config", "checkpoint"},
        )
        require_str_field(model_map["name"], f"manifest.models[{index}].name")
        require_str_field(model_map["run_name"], f"manifest.models[{index}].run_name")
        step_name = require_str_field(model_map["step"], f"manifest.models[{index}].step")
        if not (step_name.startswith(STEP_PREFIX) and step_name[len(STEP_PREFIX):].isdigit()):
            raise BundleError(f"manifest.models[{index}].step must look like step_000123")
        step_int = require_int_field(model_map["step_int"], f"manifest.models[{index}].step_int")
        if step_int != parse_checkpoint_step(step_name):
            raise BundleError(
                f"manifest.models[{index}] step mismatch: step={step_name}, step_int={step_int}"
            )
        for key, value in original_paths.items():
            require_str_field(value, f"manifest.models[{index}].original_paths.{key}")
        checkpoint = require_mapping(model_map["checkpoint"], f"manifest.models[{index}].checkpoint")
        config = require_mapping(model_map["config"], f"manifest.models[{index}].config")
        require_exact_keys(checkpoint, f"manifest.models[{index}].checkpoint", required={"path", "size", "sha256"})
        require_exact_keys(config, f"manifest.models[{index}].config", required={"path", "size", "sha256"})
        validate_manifest_file_record(checkpoint, f"manifest.models[{index}].checkpoint")
        validate_manifest_file_record(config, f"manifest.models[{index}].config")
        if not isinstance(model_map["rewritten_config_keys"], list):
            raise BundleError(f"manifest.models[{index}].rewritten_config_keys must be a list")
        for item_index, rewritten_key in enumerate(model_map["rewritten_config_keys"]):
            require_str_field(
                rewritten_key,
                f"manifest.models[{index}].rewritten_config_keys[{item_index}]",
            )
    auxiliary_files = require_list(manifest["auxiliary_files"], "manifest.auxiliary_files")
    if not auxiliary_files:
        raise BundleError("manifest.auxiliary_files must not be empty")
    for index, auxiliary_file in enumerate(auxiliary_files):
        auxiliary_map = require_mapping(auxiliary_file, f"manifest.auxiliary_files[{index}]")
        validate_manifest_auxiliary_file(auxiliary_map, f"manifest.auxiliary_files[{index}]")
    return manifest


def verify_file(bundle_dir: Path, entry: Mapping[str, Any], label: str) -> tuple[Path, str]:
    rel_path = str(entry["path"])
    path = materialize_bundle_path(bundle_dir, rel_path)
    if not path.is_file():
        raise BundleError(f"{label} file missing: {path}")
    actual_sha256, actual_size = sha256_file(path)
    expected_size = int(entry["size"])
    expected_sha256 = str(entry["sha256"])
    if actual_size != expected_size:
        raise BundleError(f"{label} size mismatch for {path}: expected {expected_size}, got {actual_size}")
    if actual_sha256 != expected_sha256:
        raise BundleError(f"{label} sha256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}")
    return path, actual_sha256


def verify_bundle_impl(bundle_dir: Path, *, allow_subset: bool) -> tuple[Mapping[str, Any], str]:
    bundle_dir = resolve_bundle_guarded(str(bundle_dir))
    manifest = load_manifest(bundle_dir)
    expected_paths = {
        ".gitattributes",
        "manifest.json",
        "README.md",
        str(manifest["dataset_stats"]["path"]),
        *{
            str(record["path"])
            for model in manifest["models"]
            for record in (model["checkpoint"], model["config"])
        },
        *{str(auxiliary["path"]) for auxiliary in manifest["auxiliary_files"]},
    }
    reject_unexpected_bundle_files(bundle_dir, expected_paths)
    model_names = [str(model["name"]) for model in manifest["models"]]
    if len(set(model_names)) != len(model_names):
        raise BundleError(f"Duplicate model names in manifest: {model_names}")
    expected_set = set(EXPECTED_MODEL_NAMES)
    model_set = set(model_names)
    if allow_subset:
        if not model_set.issubset(expected_set):
            raise BundleError(
                f"Unexpected model names {sorted(model_set - expected_set)}; expected subset of {EXPECTED_MODEL_NAMES}"
            )
    else:
        if model_set != expected_set:
            raise BundleError(
                f"Expected models {EXPECTED_MODEL_NAMES}, found {tuple(model_names)}. "
                "Use --allow-subset to relax this check."
            )

    verify_file(bundle_dir, require_mapping(manifest["dataset_stats"], "dataset_stats"), "dataset stats")
    for auxiliary_file in manifest["auxiliary_files"]:
        auxiliary_map = require_mapping(auxiliary_file, "auxiliary_file")
        path, _ = verify_file(bundle_dir, auxiliary_map, f"auxiliary file {auxiliary_map['name']}")
        if auxiliary_map["executable"] and not os.access(path, os.X_OK):
            raise BundleError(f"Auxiliary file is not executable: {path}")

    rows: list[list[str]] = []
    for model in manifest["models"]:
        model_map = require_mapping(model, "model")
        config_path, _ = verify_file(bundle_dir, require_mapping(model_map["config"], "config"), f"{model_map['name']} config")
        checkpoint_path, checkpoint_sha = verify_file(
            bundle_dir,
            require_mapping(model_map["checkpoint"], "checkpoint"),
            f"{model_map['name']} checkpoint",
        )

        try:
            cfg = OmegaConf.load(config_path)
        except Exception as exc:  # pragma: no cover - OmegaConf raises varied subclasses
            raise BundleError(f"Failed to parse YAML config {config_path}: {exc}") from exc
        if not OmegaConf.is_config(cfg) or not OmegaConf.is_dict(cfg):
            raise BundleError(f"Config is not a YAML mapping: {config_path}")
        if OmegaConf.select(cfg, "model") is None:
            raise BundleError(f"Config missing `model`: {config_path}")
        if OmegaConf.select(cfg, "data.train.processor") is None:
            raise BundleError(f"Config missing `data.train.processor`: {config_path}")

        rows.append(
            [
                str(model_map["name"]),
                str(model_map["step"]),
                describe_size(int(model_map["checkpoint"]["size"])),
                checkpoint_sha[:12],
                ",".join(model_map["rewritten_config_keys"]) if model_map["rewritten_config_keys"] else "-",
            ]
        )

    table = render_table(
        headers=["model", "step", "checkpoint size", "ckpt sha256", "rewritten keys"],
        rows=rows,
    )
    return manifest, table


def verify_bundle(args: argparse.Namespace) -> int:
    bundle_dir = resolve_bundle_guarded(args.bundle_dir)
    manifest, table = verify_bundle_impl(bundle_dir, allow_subset=args.allow_subset)
    print(f"Verified bundle: {bundle_dir}")
    print(table)
    selection = manifest["selection"]
    if "warning" in selection:
        print(f"WARNING: {selection['warning']}")
    return 0


def resolve_repo_id(repo_id_arg: str | None) -> str:
    repo_id = (repo_id_arg or os.environ.get("HF_EVAL_REPO_ID") or "").strip()
    if not repo_id:
        raise BundleError("Hugging Face repo id is required via --repo-id or HF_EVAL_REPO_ID.")
    return repo_id


def require_hf_auth() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return
    try:
        from huggingface_hub import HfFolder
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise BundleError("huggingface_hub is required for upload/download commands.") from exc
    if not HfFolder.get_token():
        raise BundleError("Hugging Face auth required. Run `huggingface-cli login` or set HF_TOKEN.")


def upload_bundle(args: argparse.Namespace) -> int:
    bundle_dir = resolve_bundle_guarded(args.bundle_dir)
    repo_id = resolve_repo_id(args.repo_id)
    verify_bundle_impl(bundle_dir, allow_subset=False)
    require_hf_auth()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise BundleError("huggingface_hub is required for upload.") from exc

    api = HfApi()
    if not hasattr(api, "upload_large_folder"):
        raise BundleError("Installed huggingface_hub does not provide HfApi.upload_large_folder.")
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_large_folder(
        repo_id,
        str(bundle_dir),
        repo_type="model",
        revision=args.revision,
        allow_patterns=[
            "manifest.json",
            "README.md",
            "dataset_stats.json",
            "run_eval.sh",
            "models/*/checkpoint.pt",
            "models/*/config.yaml",
        ],
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )
    print(f"Uploaded bundle to https://huggingface.co/{repo_id}/tree/{args.revision}")
    return 0


def example_eval_instruction(bundle_dir: Path, manifest: Mapping[str, Any]) -> str:
    first_model = manifest["models"][0]
    return (
        "python experiments/robotwin/eval_robotwin_single.py "
        "task=<robotwin_task> "
        f"ckpt={quoted(bundle_dir / first_model['checkpoint']['path'])} "
        f"EVALUATION.train_config_path={quoted(bundle_dir / first_model['config']['path'])} "
        f"EVALUATION.dataset_stats_path={quoted(bundle_dir / manifest['dataset_stats']['path'])}"
    )


def download_bundle(args: argparse.Namespace) -> int:
    output_dir = resolve_bundle_guarded(args.output_dir)
    repo_id = resolve_repo_id(args.repo_id)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise BundleError("huggingface_hub is required for download.") from exc

    ensure_directory(output_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=[
            ".gitattributes",
            "manifest.json",
            "README.md",
            "dataset_stats.json",
            "run_eval.sh",
            "models/*/checkpoint.pt",
            "models/*/config.yaml",
        ],
        max_workers=args.workers,
        force_download=args.force_download,
    )
    manifest, table = verify_bundle_impl(output_dir, allow_subset=False)
    print(f"Downloaded and verified bundle: {output_dir}")
    print(table)
    print(f"Code checkout: git checkout {manifest['code']['resolved_sha']}")
    if any(model["rewritten_config_keys"] for model in manifest["models"]):
        print("Set VJEPA21_CHECKPOINT and VJEPA21_REPO before evaluating V-JEPA models.")
    print(f"Bundled launcher: (cd /path/to/fastwam_checkout && MODEL=all {quoted(output_dir / 'run_eval.sh')})")
    print(f"Example eval: {example_eval_instruction(output_dir, manifest)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    examples = textwrap.dedent(
        f"""\
        Examples:
          python {quoted(SCRIPT_PATH)} pack
          python {quoted(SCRIPT_PATH)} pack --step 120000 --code-revision HEAD~1
          python {quoted(SCRIPT_PATH)} pack --run qwen3vl=/path/to/run --run vjepa21=/path/to/run2
          HF_EVAL_REPO_ID=org/repo python {quoted(SCRIPT_PATH)} upload --bundle-dir {quoted(DEFAULT_BUNDLE_DIR)}
          python {quoted(SCRIPT_PATH)} download --repo-id org/repo --output-dir {quoted(DEFAULT_DOWNLOAD_DIR)}
          python {quoted(SCRIPT_PATH)} verify --bundle-dir {quoted(DEFAULT_BUNDLE_DIR)}
        """
    )
    parser = argparse.ArgumentParser(
        description="Pack, upload, download, and verify RoboTwin evaluation bundles for Hugging Face.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser(
        "pack",
        help="Stage dataset stats, configs, and checkpoints into a portable bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pack_parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help=f"Bundle output directory. Default: {DEFAULT_BUNDLE_DIR}",
    )
    pack_parser.add_argument(
        "--stats-path",
        default=None,
        help=(
            "Optional dataset_stats.json path. Default search order: "
            "data/robotwin2.0_webdataset/dataset_stats.json, then data/robotwin2.0/dataset_stats.json."
        ),
    )
    pack_parser.add_argument(
        "--step",
        default="latest-common",
        help="Checkpoint selector: latest-common (default), latest-each, integer, or step_000123.",
    )
    pack_parser.add_argument(
        "--run",
        action="append",
        type=parse_run_override,
        default=[],
        help="Override a default run path with NAME=PATH. Repeatable.",
    )
    pack_parser.add_argument(
        "--code-revision",
        default=None,
        help="Git revision representing the training/eval code. Default: current HEAD.",
    )
    pack_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow packing from a dirty git worktree (not recommended).",
    )
    pack_parser.set_defaults(func=pack_bundle)

    upload_parser = subparsers.add_parser(
        "upload",
        help="Verify locally, create a model repo if needed, then upload the bundle.",
    )
    upload_parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face model repo id. Default: HF_EVAL_REPO_ID environment variable.",
    )
    upload_parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help=f"Bundle directory to upload. Default: {DEFAULT_BUNDLE_DIR}",
    )
    privacy_group = upload_parser.add_mutually_exclusive_group()
    privacy_group.add_argument("--private", dest="private", action="store_true", help="Upload to a private repo.")
    privacy_group.add_argument("--public", dest="private", action="store_false", help="Upload to a public repo.")
    upload_parser.set_defaults(private=True, func=upload_bundle)
    upload_parser.add_argument(
        "--revision",
        default=DEFAULT_REPO_REVISION,
        help=f"Target repo revision/branch. Default: {DEFAULT_REPO_REVISION}",
    )
    upload_parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_UPLOAD_WORKERS,
        help=f"Parallel upload workers. Default: {DEFAULT_UPLOAD_WORKERS}",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download a bundle snapshot to disk and verify it.",
    )
    download_parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face model repo id. Default: HF_EVAL_REPO_ID environment variable.",
    )
    download_parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Destination directory. Default: {DEFAULT_DOWNLOAD_DIR}",
    )
    download_parser.add_argument(
        "--revision",
        default=DEFAULT_REPO_REVISION,
        help=f"Repo revision/branch. Default: {DEFAULT_REPO_REVISION}",
    )
    download_parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help=f"Parallel download workers. Default: {DEFAULT_DOWNLOAD_WORKERS}",
    )
    download_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download files even if cached locally.",
    )
    download_parser.set_defaults(func=download_bundle)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Strictly validate a bundle manifest and staged files.",
    )
    verify_parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help=f"Bundle directory to verify. Default: {DEFAULT_BUNDLE_DIR}",
    )
    verify_parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow a subset of the default three model names instead of requiring all of them.",
    )
    verify_parser.set_defaults(func=verify_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
