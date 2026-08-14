#!/usr/bin/env python3
"""Build a compact canonical manifest for heterogeneous LeRobot v3 sources."""

from __future__ import annotations

import argparse
import bisect
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


MANIFEST_VERSION = 5
WINDOW_SECONDS = 3.2
ROUTE_FULL = 0
ROUTE_VIDEO_ONLY = 1


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


def _episode_columns(
    camera_keys: dict[str, str | None],
    adapter: dict[str, Any],
) -> list[str]:
    columns = [
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
    ]
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
    columns.extend(adapter.get("route_stats_columns", []))
    return list(dict.fromkeys(columns))


def _read_episode_rows(
    dataset_root: Path,
    camera_keys: dict[str, str | None],
    adapter: dict[str, Any],
) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {dataset_root}.")
    wanted_columns = _episode_columns(camera_keys, adapter)
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


def _data_file_intervals(
    dataset_root: Path,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    import pyarrow.parquet as pq

    intervals = []
    cumulative_start = 0
    paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files under {dataset_root}.")
    for path in paths:
        chunk = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
        row_count = int(pq.ParquetFile(path).metadata.num_rows)
        cumulative_end = cumulative_start + row_count
        intervals.append((cumulative_start, cumulative_end, chunk, file_index))
        cumulative_start = cumulative_end
    return [interval[1] for interval in intervals], intervals


def _resolve_data_file(
    dataset_from_index: int,
    episode_length: int,
    file_ends: list[int],
    intervals: list[tuple[int, int, int, int]],
) -> tuple[int, int, int]:
    interval_index = bisect.bisect_right(file_ends, int(dataset_from_index))
    if interval_index >= len(intervals):
        raise ValueError(
            f"dataset_from_index={dataset_from_index} is outside data files."
        )
    file_start, file_end, chunk, file_index = intervals[interval_index]
    if int(dataset_from_index) + int(episode_length) > file_end:
        raise ValueError(
            "Episode crosses a parquet boundary: "
            f"start={dataset_from_index}, length={episode_length}, "
            f"file=[{file_start},{file_end})."
        )
    return chunk, file_index, file_start


def _robocoin_base_keys(features: dict[str, Any]) -> list[str]:
    keys = []
    for key, value in features.items():
        names = value.get("names") if isinstance(value, dict) else None
        text = f"{key} {names}".lower()
        if any(
            marker in text
            for marker in (
                "robot_pos_",
                "robot_quat_",
                "base_",
                "chassis",
                "wheel",
            )
        ):
            keys.append(key)
    return keys


def _adapter_from_features(
    adapter_name: str,
    features: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    if adapter_name == "video_only":
        return {"type": "video_only", "route_default": ROUTE_VIDEO_ONLY}
    if adapter_name == "agibot_eef":
        required = {
            "observation.states.end.position",
            "observation.states.end.orientation",
            "actions.end.position",
            "actions.end.orientation",
            "actions.effector.position",
        }
        if not required <= set(features):
            return {"type": "video_only", "route_default": ROUTE_VIDEO_ONLY}
        return {
            "type": "agibot_eef",
            "route_default": ROUTE_FULL,
            "columns": sorted(required | {"actions.robot.velocity"}),
            "gripper_valid": [False, False],
            "route_motion_key": "actions.robot.velocity",
            "route_motion_threshold": 1e-4,
            "route_stats_columns": [
                "stats/actions.robot.velocity/min",
                "stats/actions.robot.velocity/max",
            ],
        }
    if adapter_name == "droid_eef":
        required = {
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            "action.cartesian_position",
            "action.gripper_position",
        }
        if not required <= set(features):
            return {"type": "video_only", "route_default": ROUTE_VIDEO_ONLY}
        return {
            "type": "droid_eef",
            "route_default": ROUTE_FULL,
            "columns": sorted(required),
            "gripper_valid": [False, False],
        }
    if adapter_name == "oxe_eef":
        state = features.get("observation.state", {})
        if (
            float(fps) >= 9.999
            and list(state.get("shape", [])) == [8]
        ):
            return {
                "type": "oxe_euler_state",
                "route_default": ROUTE_FULL,
                "columns": ["observation.state"],
                "gripper_valid": [False, False],
            }
        return {"type": "video_only", "route_default": ROUTE_VIDEO_ONLY}
    if adapter_name == "galaxea_eef":
        required = {
            "observation.state.left_ee_pose",
            "observation.state.right_ee_pose",
            "observation.state.left_gripper",
            "observation.state.right_gripper",
            "observation.state.chassis",
        }
        if not required <= set(features):
            return {"type": "video_only", "route_default": ROUTE_VIDEO_ONLY}
        return {
            "type": "galaxea_eef",
            "route_default": ROUTE_FULL,
            "columns": sorted(required),
            "route_motion_key": "observation.state.chassis",
            "route_motion_slice": [3, 6],
            "route_motion_stat_threshold": 0.03,
            "route_stats_columns": [
                "stats/observation.state.chassis/mean",
                "stats/observation.state.chassis/std",
            ],
        }
    if adapter_name == "robocoin_eef":
        required = {"eef_sim_pose_state", "eef_sim_pose_action"}
        if not required <= set(features):
            return {"type": "video_only", "route_default": ROUTE_VIDEO_ONLY}
        columns = set(required)
        has_gripper = {
            "gripper_open_scale_state",
            "gripper_open_scale_action",
        } <= set(features)
        if has_gripper:
            columns.update(
                {
                    "gripper_open_scale_state",
                    "gripper_open_scale_action",
                }
            )
        base_keys = _robocoin_base_keys(features)
        return {
            "type": "robocoin_eef",
            "route_default": (
                ROUTE_VIDEO_ONLY if base_keys else ROUTE_FULL
            ),
            "columns": sorted(columns),
            "has_gripper": has_gripper,
            "base_keys": base_keys,
        }
    raise ValueError(f"Unknown adapter type: {adapter_name}")


class _ParquetColumnCache:
    def __init__(self):
        self.path: Path | None = None
        self.columns: tuple[str, ...] = ()
        self.payload: dict[str, np.ndarray] = {}

    def get(self, path: Path, columns: list[str]) -> dict[str, np.ndarray]:
        wanted = tuple(columns)
        if self.path == path and self.columns == wanted:
            return self.payload
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=list(wanted))
        payload = {}
        for column in wanted:
            array = table[column].combine_chunks()
            if pa.types.is_fixed_size_list(array.type):
                child = array.values.to_numpy(zero_copy_only=False)
                payload[column] = child.reshape(
                    len(array),
                    int(array.type.list_size),
                )
            else:
                payload[column] = np.asarray(array.to_pylist())
        self.path = path
        self.columns = wanted
        self.payload = payload
        return payload


def _episode_route(
    adapter: dict[str, Any],
    episode_row: dict[str, Any],
    dataset_root: Path,
    data_chunk: int,
    data_file: int,
    local_start: int,
    length: int,
    cache: _ParquetColumnCache,
) -> int:
    route_default = int(adapter["route_default"])
    motion_key = adapter.get("route_motion_key")
    if route_default == ROUTE_VIDEO_ONLY or motion_key is None:
        return route_default
    stats_columns = adapter.get("route_stats_columns", [])
    if len(stats_columns) == 2 and all(
        column in episode_row for column in stats_columns
    ):
        first = np.asarray(episode_row[stats_columns[0]], dtype=np.float64)
        second = np.asarray(episode_row[stats_columns[1]], dtype=np.float64)
        motion_slice = adapter.get("route_motion_slice")
        if motion_slice is not None:
            first = first[int(motion_slice[0]) : int(motion_slice[1])]
            second = second[int(motion_slice[0]) : int(motion_slice[1])]
        stat_threshold = adapter.get("route_motion_stat_threshold")
        if stat_threshold is not None:
            score = float(np.abs(first).sum() + second.sum())
            return (
                ROUTE_FULL
                if score < float(stat_threshold)
                else ROUTE_VIDEO_ONLY
            )
        threshold = float(adapter.get("route_motion_threshold", 0.0))
        max_abs = float(
            max(
                np.max(np.abs(first), initial=0.0),
                np.max(np.abs(second), initial=0.0),
            )
        )
        return ROUTE_FULL if max_abs <= threshold else ROUTE_VIDEO_ONLY
    data_path = dataset_root / (
        f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
    )
    payload = cache.get(data_path, [motion_key])
    motion = np.asarray(
        payload[motion_key][local_start : local_start + length],
        dtype=np.float64,
    )
    if not np.isfinite(motion).all():
        return ROUTE_VIDEO_ONLY
    motion_slice = adapter.get("route_motion_slice")
    if motion_slice is not None:
        motion = motion[..., int(motion_slice[0]) : int(motion_slice[1])]
    stat_threshold = adapter.get("route_motion_stat_threshold")
    if stat_threshold is not None:
        score = float(
            np.abs(motion.mean(axis=0)).sum()
            + motion.std(axis=0).sum()
        )
        return ROUTE_FULL if score < float(stat_threshold) else ROUTE_VIDEO_ONLY
    threshold = float(adapter.get("route_motion_threshold", 0.0))
    return (
        ROUTE_FULL
        if float(np.max(np.abs(motion), initial=0.0)) <= threshold
        else ROUTE_VIDEO_ONLY
    )


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
        video_exists_cache: dict[str, bool] = {}
        missing_video_files: set[str] = set()
        missing_video_episode_count = 0

        for source_config in registry["sources"]:
            if not bool(source_config.get("enabled", True)):
                continue
            source_name = str(source_config["source_id"])
            source_root = Path(str(source_config["root"])).expanduser().resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(
                    f"Source `{source_name}` does not exist: {source_root}"
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
            adapter_name = str(source_config.get("adapter", "video_only"))
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
            route_cache = _ParquetColumnCache()
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
                    adapter = _adapter_from_features(
                        adapter_name,
                        features,
                        fps,
                    )
                    episode_rows = _read_episode_rows(
                        dataset_root,
                        camera_keys,
                        adapter,
                    )
                    file_ends, data_intervals = _data_file_intervals(dataset_root)
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
                    "adapter": adapter,
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
                    episode_video_missing = False
                    for role, camera_key in camera_keys.items():
                        if camera_key is None:
                            continue
                        chunk, file_index, _ = camera_values[role]
                        video_path = dataset_root / (
                            f"videos/{camera_key}/chunk-{chunk:03d}/"
                            f"file-{file_index:03d}.mp4"
                        )
                        cache_key = str(video_path)
                        exists = video_exists_cache.get(cache_key)
                        if exists is None:
                            exists = video_path.is_file()
                            video_exists_cache[cache_key] = exists
                        if not exists:
                            episode_video_missing = True
                            missing_video_files.add(
                                f"{source_name}/"
                                f"{video_path.relative_to(source_root)}"
                            )
                    if episode_video_missing:
                        missing_video_episode_count += 1
                        continue
                    data_chunk, data_file, data_file_from = _resolve_data_file(
                        dataset_from_index=int(row["dataset_from_index"]),
                        episode_length=length,
                        file_ends=file_ends,
                        intervals=data_intervals,
                    )
                    data_from = int(row["dataset_from_index"])
                    local_start = data_from - data_file_from
                    route_id = _episode_route(
                        adapter=adapter,
                        episode_row=row,
                        dataset_root=dataset_root,
                        data_chunk=data_chunk,
                        data_file=data_file,
                        local_start=local_start,
                        length=length,
                        cache=route_cache,
                    )
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
                    arrays["route_id"].append(route_id)
                    arrays["data_chunk"].append(data_chunk)
                    arrays["data_file"].append(data_file)
                    arrays["data_from"].append(data_from)
                    arrays["data_file_from"].append(data_file_from)
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
            "route_id": np.int8,
            "data_chunk": np.int16,
            "data_file": np.int32,
            "data_from": np.int64,
            "data_file_from": np.int64,
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
        route_ids = np.asarray(arrays["route_id"], dtype=np.int64)
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
        source_route_clip_counts = {
            sources[source_id]["source_id"]: {
                "FULL": int(
                    start_counts[
                        (source_ids == source_id)
                        & (route_ids == ROUTE_FULL)
                    ].sum()
                ),
                "VIDEO_ONLY": int(
                    start_counts[
                        (source_ids == source_id)
                        & (route_ids == ROUTE_VIDEO_ONLY)
                    ].sum()
                ),
            }
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
            "source_route_native_start_counts": source_route_clip_counts,
            "excluded_dataset_count": len(excluded),
            "excluded_datasets": excluded,
            "missing_video_episode_count": missing_video_episode_count,
            "missing_video_file_count": len(missing_video_files),
            "missing_video_files": sorted(missing_video_files)[:1000],
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
