#!/usr/bin/env python3
"""Build a compact video-only manifest for heterogeneous LeRobot v3 sources."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


MANIFEST_VERSION = 2
WINDOW_SECONDS = 3.2


def _load_registry(path: Path) -> dict[str, Any]:
    config = OmegaConf.load(path)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Registry must resolve to a mapping: {path}")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Registry must define a non-empty `sources` list: {path}")
    return payload


def _discover_dataset_roots(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    roots = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in {"data", "videos", ".fastwam_multisource"}
        ]
        if Path(current).name == "meta" and "info.json" in files:
            roots.append(Path(current).parent)
    return sorted(roots)


def _resolve_camera_key(
    features: dict[str, Any],
    candidates: list[str] | None,
) -> str | None:
    for key in candidates or []:
        feature = features.get(str(key))
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            return str(key)
    return None


def _episode_columns(camera_keys: dict[str, str | None]) -> list[str]:
    columns = ["episode_index", "tasks", "length"]
    for camera_key in camera_keys.values():
        if camera_key is None:
            continue
        columns.extend(
            [
                f"videos/{camera_key}/chunk_index",
                f"videos/{camera_key}/file_index",
                f"videos/{camera_key}/from_timestamp",
            ]
        )
    return list(dict.fromkeys(columns))


def _read_episode_rows(dataset_root: Path, camera_keys: dict[str, str | None]) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {dataset_root}.")
    wanted_columns = _episode_columns(camera_keys)
    rows = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = [column for column in wanted_columns if column not in available]
        if missing:
            raise ValueError(f"{path} is missing episode columns: {missing}")
        payload = parquet.read(columns=wanted_columns).to_pydict()
        for row_index in range(len(payload["episode_index"])):
            rows.append({key: payload[key][row_index] for key in wanted_columns})
    return rows


def _read_removed_episode_ids(source_root: Path, patterns: list[str]) -> set[int]:
    removed: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int):
            removed.add(int(value))
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                removed.add(int(stripped))
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for pattern in patterns:
        for path in sorted(source_root.glob(pattern)):
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    episode_keys = (
                        "dirty_episodes",
                        "removed_episodes",
                        "episode_ids",
                        "episodes",
                    )
                    for key in episode_keys:
                        if key in payload:
                            collect(payload[key])
                else:
                    collect(payload)
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    collect(line)
    return removed


def _camera_values(
    row: dict,
    role: str,
    camera_key: str | None,
) -> tuple[int, int, float]:
    if camera_key is None:
        return -1, -1, math.nan
    chunk = row.get(f"videos/{camera_key}/chunk_index")
    file_index = row.get(f"videos/{camera_key}/file_index")
    from_timestamp = row.get(f"videos/{camera_key}/from_timestamp")
    if chunk is None or file_index is None or from_timestamp is None:
        return -1, -1, math.nan
    return int(chunk), int(file_index), float(from_timestamp)


def _keep_episode(family: str, camera_values: dict[str, tuple[int, int, float]]) -> bool:
    valid = {role: values[1] >= 0 for role, values in camera_values.items()}
    if valid["head"]:
        return True
    if family == "dual":
        return valid["left"] and valid["right"]
    if family == "single":
        return valid["left"] or valid["right"]
    return False


def _write_array(output_dir: Path, name: str, values, dtype) -> None:
    np.save(output_dir / f"{name}.npy", np.asarray(values, dtype=dtype))


def build_manifest(
    registry_path: Path,
    output_dir: Path,
    force: bool = False,
    max_datasets_per_source: int | None = None,
) -> None:
    registry_path = registry_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    registry = _load_registry(registry_path)
    registry_sha256 = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    lock_path = output_dir.with_name(output_dir.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        done_path = output_dir / "done.json"
        if done_path.is_file() and not force:
            with done_path.open("r", encoding="utf-8") as handle:
                done = json.load(handle)
            if (
                int(done.get("version", -1)) == MANIFEST_VERSION
                and Path(done.get("registry_path", "")).resolve() == registry_path
                and done.get("registry_sha256") == registry_sha256
            ):
                print(f"Multisource video manifest already exists: {output_dir}")
                return

        arrays: dict[str, list] = defaultdict(list)
        datasets = []
        sources = []
        tasks: list[str] = []
        task_to_id: dict[str, int] = {}
        excluded = []
        shard_to_id: dict[tuple, int] = {}

        for source_config in registry["sources"]:
            if not bool(source_config.get("enabled", True)):
                continue
            if str(source_config.get("route", "")).upper() != "VIDEO_ONLY":
                continue
            source_name = str(source_config["source_id"])
            source_root = Path(str(source_config["root"])).expanduser().resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(
                    f"VIDEO_ONLY source `{source_name}` does not exist: {source_root}"
                )
            source_id = len(sources)
            source_meta = {
                "source_id": source_name,
                "root": str(source_root),
                "weight": float(source_config.get("weight", 1.0)),
                "qa_status": str(source_config.get("qa_status", "unverified")),
                "mapping_evidence": source_config.get("mapping_evidence"),
            }
            sources.append(source_meta)
            family = str(source_config.get("family", "unknown"))
            camera_candidates = source_config.get("cameras") or {}
            removed_patterns = [
                str(value) for value in source_config.get("removed_episode_files", [])
            ]

            dataset_roots = _discover_dataset_roots(source_root)
            if not dataset_roots:
                excluded.append(
                    {
                        "source_id": source_name,
                        "relative_root": ".",
                        "reason": "no_lerobot_v3_roots",
                    }
                )
                continue

            valid_dataset_count = 0
            for dataset_root in dataset_roots:
                relative_root = str(dataset_root.relative_to(source_root))
                try:
                    removed_episode_ids = _read_removed_episode_ids(
                        dataset_root,
                        removed_patterns,
                    )
                    with (dataset_root / "meta" / "info.json").open(
                        "r", encoding="utf-8"
                    ) as handle:
                        info = json.load(handle)
                    features = info["features"]
                    fps = float(info["fps"])
                    if fps <= 0:
                        raise ValueError(f"Invalid fps={fps}.")
                    camera_keys = {
                        role: _resolve_camera_key(
                            features,
                            [str(value) for value in camera_candidates.get(role, [])],
                        )
                        for role in ("head", "left", "right")
                    }
                    if not any(camera_keys.values()):
                        raise ValueError("No registered camera key is present.")
                    episode_rows = _read_episode_rows(dataset_root, camera_keys)
                except Exception as exc:
                    excluded.append(
                        {
                            "source_id": source_name,
                            "relative_root": relative_root,
                            "reason": str(exc),
                        }
                    )
                    continue

                dataset_id = len(datasets)
                dataset_meta = {
                    "source_index": source_id,
                    "source_id": source_name,
                    "root": str(dataset_root),
                    "relative_root": relative_root,
                    "robot_type": info.get("robot_type"),
                    "fps": fps,
                    "family": family,
                    "camera_keys": camera_keys,
                }
                valid_episodes = 0
                for row in episode_rows:
                    episode_index = int(row["episode_index"])
                    if episode_index in removed_episode_ids:
                        continue
                    length = int(row["length"])
                    start_count = length - int(math.ceil(WINDOW_SECONDS * fps))
                    if start_count <= 0:
                        continue
                    camera_values = {
                        role: _camera_values(row, role, camera_keys[role])
                        for role in ("head", "left", "right")
                    }
                    if not _keep_episode(family, camera_values):
                        continue
                    task_values = row.get("tasks") or [
                        dataset_root.name.replace("_", " ")
                    ]
                    task = str(task_values[0])
                    task_id = task_to_id.get(task)
                    if task_id is None:
                        task_id = len(tasks)
                        task_to_id[task] = task_id
                        tasks.append(task)

                    shard_key = (
                        dataset_id,
                        camera_values["head"][:2],
                        camera_values["left"][:2],
                        camera_values["right"][:2],
                    )
                    shard_id = shard_to_id.setdefault(shard_key, len(shard_to_id))
                    arrays["source_id"].append(source_id)
                    arrays["dataset_id"].append(dataset_id)
                    arrays["episode_index"].append(episode_index)
                    arrays["length"].append(length)
                    arrays["start_count"].append(start_count)
                    arrays["task_id"].append(task_id)
                    arrays["shard_id"].append(shard_id)
                    for role in ("head", "left", "right"):
                        chunk, file_index, from_timestamp = camera_values[role]
                        arrays[f"{role}_chunk"].append(chunk)
                        arrays[f"{role}_file"].append(file_index)
                        arrays[f"{role}_from_timestamp"].append(from_timestamp)
                    valid_episodes += 1
                if valid_episodes:
                    datasets.append(dataset_meta)
                    valid_dataset_count += 1
                    if (
                        max_datasets_per_source is not None
                        and valid_dataset_count >= max(int(max_datasets_per_source), 0)
                    ):
                        break

        if not arrays["episode_index"]:
            raise RuntimeError("No valid VIDEO_ONLY episodes were indexed.")

        active_source_ids = sorted(set(int(value) for value in arrays["source_id"]))
        source_id_remap = {
            old_source_id: new_source_id
            for new_source_id, old_source_id in enumerate(active_source_ids)
        }
        sources = [sources[source_id] for source_id in active_source_ids]
        arrays["source_id"] = [
            source_id_remap[int(source_id)]
            for source_id in arrays["source_id"]
        ]
        for dataset in datasets:
            dataset["source_index"] = source_id_remap[int(dataset["source_index"])]

        sort_order = np.lexsort(
            (
                np.asarray(arrays["shard_id"], dtype=np.int64),
                np.asarray(arrays["source_id"], dtype=np.int64),
            )
        )
        tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        array_dtypes = {
            "source_id": np.int16,
            "dataset_id": np.int32,
            "episode_index": np.int32,
            "length": np.int32,
            "start_count": np.int32,
            "task_id": np.int32,
            "shard_id": np.int64,
            "head_chunk": np.int16,
            "head_file": np.int32,
            "head_from_timestamp": np.float64,
            "left_chunk": np.int16,
            "left_file": np.int32,
            "left_from_timestamp": np.float64,
            "right_chunk": np.int16,
            "right_file": np.int32,
            "right_from_timestamp": np.float64,
        }
        for name, dtype in array_dtypes.items():
            values = np.asarray(arrays[name], dtype=dtype)[sort_order]
            _write_array(tmp_dir, name, values, dtype)

        for name, payload in (
            ("datasets.json", datasets),
            ("sources.json", sources),
            ("tasks.json", tasks),
        ):
            with (tmp_dir / name).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=name == "tasks.json", indent=2)

        start_counts = np.asarray(arrays["start_count"], dtype=np.int64)
        source_ids = np.asarray(arrays["source_id"], dtype=np.int64)
        source_episode_counts = {
            sources[source_id]["source_id"]: int((source_ids == source_id).sum())
            for source_id in range(len(sources))
        }
        source_clip_counts = {
            sources[source_id]["source_id"]: int(
                start_counts[source_ids == source_id].sum()
            )
            for source_id in range(len(sources))
        }
        done = {
            "version": MANIFEST_VERSION,
            "registry_path": str(registry_path),
            "registry_sha256": registry_sha256,
            "dataset_count": len(datasets),
            "source_count": len(sources),
            "episode_count": int(start_counts.size),
            "native_start_count": int(start_counts.sum()),
            "source_episode_counts": source_episode_counts,
            "source_native_start_counts": source_clip_counts,
            "excluded_dataset_count": len(excluded),
            "excluded_datasets": excluded,
        }
        with (tmp_dir / "done.json").open("w", encoding="utf-8") as handle:
            json.dump(done, handle, ensure_ascii=True, indent=2)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(tmp_dir, output_dir)
        print(json.dumps(done, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-datasets-per-source", type=int, default=None)
    args = parser.parse_args()
    build_manifest(
        registry_path=Path(args.registry),
        output_dir=Path(args.output),
        force=args.force,
        max_datasets_per_source=args.max_datasets_per_source,
    )


if __name__ == "__main__":
    main()
