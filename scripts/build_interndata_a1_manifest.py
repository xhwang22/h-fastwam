#!/usr/bin/env python3
"""Build a compact runtime index for InternData-A1 LeRobot v3 roots."""

from __future__ import annotations

import argparse
import bisect
import fcntl
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


MANIFEST_VERSION = 4
NATIVE_WINDOW_FRAMES = 97
MIN_EPISODE_FRAMES = NATIVE_WINDOW_FRAMES


def _discover_dataset_roots(root: Path) -> list[Path]:
    roots = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in {"data", "videos", ".fastwam_intern_a1"}
        ]
        if Path(current).name == "meta" and "info.json" in files:
            roots.append(Path(current).parent)
    return sorted(roots)


def _schema_from_features(features: dict) -> dict | None:
    dual_required = {
        "states.left_ee_to_left_armbase_pose",
        "states.right_ee_to_right_armbase_pose",
        "actions.left_ee_to_left_armbase_pose",
        "actions.right_ee_to_right_armbase_pose",
        "states.left_gripper.position",
        "states.right_gripper.position",
        "actions.left_gripper.position",
        "actions.right_gripper.position",
    }
    single_required = {
        "states.ee_to_armbase_pose",
        "actions.ee_to_armbase_pose",
        "states.gripper.position",
    }
    feature_keys = set(features)
    if dual_required <= feature_keys:
        if {
            "master_actions.left_gripper.openness",
            "master_actions.right_gripper.openness",
        } <= feature_keys:
            action_gripper_keys = [
                "master_actions.left_gripper.openness",
                "master_actions.right_gripper.openness",
            ]
        elif {
            "actions.left_gripper.openness",
            "actions.right_gripper.openness",
        } <= feature_keys:
            action_gripper_keys = [
                "actions.left_gripper.openness",
                "actions.right_gripper.openness",
            ]
        else:
            return None
        return {
            "family": "dual",
            "state_pose_keys": [
                "states.left_ee_to_left_armbase_pose",
                "states.right_ee_to_right_armbase_pose",
            ],
            "action_pose_keys": [
                "actions.left_ee_to_left_armbase_pose",
                "actions.right_ee_to_right_armbase_pose",
            ],
            "action_gripper_keys": action_gripper_keys,
            "camera_keys": {
                "head": "images.rgb.head",
                "left": "images.rgb.hand_left",
                "right": "images.rgb.hand_right",
            },
        }
    if single_required <= feature_keys:
        if "actions.gripper.openness" not in feature_keys:
            return None
        return {
            "family": "single",
            "state_pose_keys": ["states.ee_to_armbase_pose"],
            "action_pose_keys": ["actions.ee_to_armbase_pose"],
            "action_gripper_keys": ["actions.gripper.openness"],
            "camera_keys": {
                "head": "images.rgb.head",
                "left": "images.rgb.hand",
                "right": None,
            },
        }
    return None


def _validate_canonical_gripper_stats(stats: dict, keys: list[str]) -> None:
    for key in keys:
        entry = stats.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing canonical gripper stats for `{key}`.")
        minimum = np.asarray(entry["min"], dtype=np.float32).reshape(-1)
        maximum = np.asarray(entry["max"], dtype=np.float32).reshape(-1)
        if minimum.size != 1 or maximum.size != 1:
            raise ValueError(f"Canonical gripper `{key}` must be scalar.")
        if float(minimum[0]) < -1e-4 or float(maximum[0]) > 1.0001:
            raise ValueError(
                f"Canonical gripper `{key}` is outside [0,1]: "
                f"min={minimum[0]}, max={maximum[0]}."
            )


def _episode_columns(camera_keys: dict) -> list[str]:
    columns = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "tasks",
        "length",
    ]
    for camera_key in camera_keys.values():
        if camera_key is None:
            continue
        columns.extend(
            [
                f"videos/{camera_key}/chunk_index",
                f"videos/{camera_key}/file_index",
                f"videos/{camera_key}/from_timestamp",
                f"videos/{camera_key}/to_timestamp",
            ]
        )
    return columns


def _read_episode_rows(dataset_root: Path, camera_keys: dict) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {dataset_root}.")

    rows = []
    wanted_columns = _episode_columns(camera_keys)
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


def _data_file_intervals(dataset_root: Path) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    import pyarrow.parquet as pq

    intervals = []
    cumulative_start = 0
    data_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files under {dataset_root}.")
    for path in data_paths:
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
            f"dataset_from_index={dataset_from_index} is outside the parquet file ranges."
        )
    file_start, file_end, chunk, file_index = intervals[interval_index]
    if int(dataset_from_index) + int(episode_length) > file_end:
        raise ValueError(
            "InternData episode crosses a parquet file boundary: "
            f"start={dataset_from_index}, length={episode_length}, "
            f"file_range=[{file_start}, {file_end})."
        )
    return chunk, file_index, file_start


def _write_array(output_dir: Path, name: str, values, dtype) -> None:
    np.save(output_dir / f"{name}.npy", np.asarray(values, dtype=dtype))


def build_manifest(root: Path, output_dir: Path, force: bool = False) -> None:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
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
                and Path(done.get("source_root", "")).resolve() == root
            ):
                print(f"InternData manifest already exists: {output_dir}")
                return

        dataset_roots = _discover_dataset_roots(root)
        if not dataset_roots:
            raise FileNotFoundError(f"No LeRobot v3 datasets found under {root}.")

        arrays: dict[str, list] = defaultdict(list)
        datasets = []
        task_to_id: dict[str, int] = {}
        tasks: list[str] = []
        shard_to_id: dict[tuple, int] = {}
        excluded = []

        for source_root in dataset_roots:
            with (source_root / "meta" / "info.json").open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            schema = _schema_from_features(info["features"])
            if schema is None:
                excluded.append(str(source_root.relative_to(root)))
                continue
            with (source_root / "meta" / "stats.json").open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
            _validate_canonical_gripper_stats(
                stats,
                schema["action_gripper_keys"],
            )
            dataset_id = len(datasets)
            dataset_meta = {
                "relative_root": str(source_root.relative_to(root)),
                "robot_type": info.get("robot_type"),
                "fps": int(info["fps"]),
                **schema,
            }
            if dataset_meta["fps"] != 30:
                excluded.append(str(source_root.relative_to(root)))
                continue

            episode_rows = _read_episode_rows(source_root, schema["camera_keys"])
            file_ends, data_intervals = _data_file_intervals(source_root)

            valid_episodes = 0
            for row in episode_rows:
                length = int(row["length"])
                if length < MIN_EPISODE_FRAMES:
                    continue
                task_values = row["tasks"] or [source_root.name.replace("_", " ")]
                task = str(task_values[0])
                task_id = task_to_id.get(task)
                if task_id is None:
                    task_id = len(tasks)
                    task_to_id[task] = task_id
                    tasks.append(task)

                camera_values = {}
                for role, camera_key in schema["camera_keys"].items():
                    if camera_key is None:
                        camera_values[role] = (-1, -1, np.nan)
                        continue
                    camera_values[role] = (
                        int(row[f"videos/{camera_key}/chunk_index"]),
                        int(row[f"videos/{camera_key}/file_index"]),
                        float(row[f"videos/{camera_key}/from_timestamp"]),
                    )

                data_chunk, data_file, data_file_from = _resolve_data_file(
                    dataset_from_index=int(row["dataset_from_index"]),
                    episode_length=length,
                    file_ends=file_ends,
                    intervals=data_intervals,
                )
                shard_key = (
                    dataset_id,
                    data_chunk,
                    data_file,
                    camera_values["head"][:2],
                    camera_values["left"][:2],
                    camera_values["right"][:2],
                )
                shard_id = shard_to_id.setdefault(shard_key, len(shard_to_id))

                arrays["dataset_id"].append(dataset_id)
                arrays["episode_index"].append(int(row["episode_index"]))
                arrays["length"].append(length)
                arrays["data_chunk"].append(data_chunk)
                arrays["data_file"].append(data_file)
                arrays["data_from"].append(int(row["dataset_from_index"]))
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

        if not arrays["episode_index"]:
            raise RuntimeError("No valid InternData episodes were indexed.")

        sort_order = np.argsort(np.asarray(arrays["shard_id"], dtype=np.int64), kind="stable")
        tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        array_dtypes = {
            "dataset_id": np.int32,
            "episode_index": np.int32,
            "length": np.int32,
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

        with (tmp_dir / "datasets.json").open("w", encoding="utf-8") as handle:
            json.dump(datasets, handle, ensure_ascii=True, indent=2)
        with (tmp_dir / "tasks.json").open("w", encoding="utf-8") as handle:
            json.dump(tasks, handle, ensure_ascii=False)

        lengths = np.asarray(arrays["length"], dtype=np.int64)
        done = {
            "version": MANIFEST_VERSION,
            "source_root": str(root),
            "native_fps": 30,
            "target_control_hz": 10,
            "native_window_frames": NATIVE_WINDOW_FRAMES,
            "dataset_count": len(datasets),
            "episode_count": int(lengths.size),
            "clip_count_full_horizon": int(
                np.maximum(lengths - (NATIVE_WINDOW_FRAMES - 1), 0).sum()
            ),
            "shard_count": len(shard_to_id),
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
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output = (
        Path(args.output)
        if args.output is not None
        else root / ".fastwam_intern_a1" / "manifest_v4_10hz"
    )
    build_manifest(root=root, output_dir=output, force=args.force)


if __name__ == "__main__":
    main()
