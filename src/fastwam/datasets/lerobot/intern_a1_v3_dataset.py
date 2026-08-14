"""Streaming InternData-A1 LeRobot v3 dataset for FastWAM pretraining."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as transforms_F

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
_ENCODED_INDEX_MARKER = 1 << 62
_EPISODE_ROW_MASK = (1 << 30) - 1
_START_MASK = (1 << 32) - 1
NATIVE_FPS = 30
TARGET_CONTROL_HZ = 10
NATIVE_CONTROL_STRIDE = NATIVE_FPS // TARGET_CONTROL_HZ
NATIVE_WINDOW_FRAMES = 97
NATIVE_HORIZON = NATIVE_WINDOW_FRAMES - 1
STATE_NATIVE_INDICES = np.arange(0, NATIVE_HORIZON, NATIVE_CONTROL_STRIDE, dtype=np.int64)
ACTION_NATIVE_INDICES = np.arange(
    NATIVE_CONTROL_STRIDE - 1,
    NATIVE_HORIZON,
    NATIVE_CONTROL_STRIDE,
    dtype=np.int64,
)
VIDEO_NATIVE_INDICES = np.arange(
    0,
    NATIVE_WINDOW_FRAMES,
    NATIVE_CONTROL_STRIDE * 4,
    dtype=np.int64,
)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Encountered a near-zero quaternion in InternData.")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)


def _canonical_gripper_open(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) * 2.0 - 1.0


class _ParquetShardCache:
    def __init__(self, max_open_shards: int):
        self.max_open_shards = max(int(max_open_shards), 1)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: Path, columns: list[str]) -> dict[str, np.ndarray]:
        cache_key = str(path)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            self._cache[cache_key] = cached
            return cached

        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns)
        payload = {}
        for column in columns:
            values = table[column].combine_chunks().to_pylist()
            array = np.asarray(values)
            if array.dtype == object:
                array = np.asarray(
                    [
                        [value] if np.isscalar(value) else value
                        for value in values
                    ],
                    dtype=np.float32,
                )
            elif np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32, copy=False)
            payload[column] = array

        self._cache[cache_key] = payload
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return payload


class _PyAVShardDecoder:
    def __init__(
        self,
        max_open_videos: int,
        decode_threads: int,
        max_frame_caches: int,
        prefetch_frames: int,
    ):
        self.max_open_videos = max(int(max_open_videos), 1)
        self.decode_threads = max(int(decode_threads), 1)
        self.max_frame_caches = max(int(max_frame_caches), 0)
        self.prefetch_frames = max(int(prefetch_frames), 0)
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._frame_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _open(self, path: Path):
        import av

        cache_key = str(path)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            self._cache[cache_key] = cached
            return cached

        container = av.open(cache_key, mode="r")
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        stream.thread_count = self.decode_threads
        value = (container, stream)
        self._cache[cache_key] = value
        while len(self._cache) > self.max_open_videos:
            _, (old_container, _) = self._cache.popitem(last=False)
            old_container.close()
        return value

    @staticmethod
    def _select_cached_frames(
        cached_times: np.ndarray,
        cached_frames: np.ndarray,
        timestamps: list[float],
        tolerance: float,
    ) -> Optional[torch.Tensor]:
        if cached_times.size == 0:
            return None
        query = np.asarray(timestamps, dtype=np.float64)
        if (
            query.min() < cached_times[0] - tolerance
            or query.max() > cached_times[-1] + tolerance
        ):
            return None
        distances = np.abs(query[:, None] - cached_times[None, :])
        indices = distances.argmin(axis=1)
        if np.any(distances[np.arange(query.size), indices] > tolerance):
            return None
        return torch.from_numpy(cached_frames[indices].copy())

    def decode(self, path: Path, timestamps: list[float], fps: float) -> torch.Tensor:
        timestamps = [float(value) for value in timestamps]
        tolerance = 0.5 / float(fps) + 0.002
        cache_key = str(path)
        cached_frames = self._frame_cache.pop(cache_key, None)
        if cached_frames is not None:
            selected = self._select_cached_frames(
                cached_times=cached_frames[0],
                cached_frames=cached_frames[1],
                timestamps=timestamps,
                tolerance=tolerance,
            )
            self._frame_cache[cache_key] = cached_frames
            if selected is not None:
                return selected

        last_error = None
        for attempt in range(2):
            try:
                container, stream = self._open(path)
                time_base = float(stream.time_base)
                seek_time = max(min(timestamps) - 2.0 / float(fps), 0.0)
                container.seek(
                    max(int(seek_time / time_base), 0),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )

                loaded_frames = []
                loaded_times = []
                stop_time = (
                    max(timestamps)
                    + (self.prefetch_frames + 2) / float(fps)
                )
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    frame_time = float(frame.pts * frame.time_base)
                    if frame_time + tolerance < min(timestamps):
                        continue
                    loaded_frames.append(frame.to_ndarray(format="rgb24"))
                    loaded_times.append(frame_time)
                    if frame_time >= stop_time:
                        break
                if not loaded_frames:
                    raise RuntimeError(f"No frames decoded from {path}.")

                loaded_times_np = np.asarray(loaded_times, dtype=np.float64)
                loaded_frames_np = np.stack(loaded_frames, axis=0)
                selected = self._select_cached_frames(
                    cached_times=loaded_times_np,
                    cached_frames=loaded_frames_np,
                    timestamps=timestamps,
                    tolerance=tolerance,
                )
                if selected is None:
                    raise RuntimeError(
                        f"Decoded frames do not cover requested timestamps for {path}."
                    )
                if self.max_frame_caches > 0:
                    self._frame_cache[cache_key] = (
                        loaded_times_np,
                        loaded_frames_np,
                    )
                    while len(self._frame_cache) > self.max_frame_caches:
                        self._frame_cache.popitem(last=False)
                return selected
            except Exception as exc:
                last_error = exc
                cached = self._cache.pop(str(path), None)
                if cached is not None:
                    cached[0].close()
                self._frame_cache.pop(str(path), None)
        raise RuntimeError(f"Failed to decode {path}: {last_error}") from last_error

    def close(self):
        self._frame_cache.clear()
        while self._cache:
            _, (container, _) = self._cache.popitem(last=False)
            container.close()


class InternDataA1V3Dataset(Dataset):
    """Map-style virtual epoch over InternData-A1 with shard-local sample order."""

    def __init__(
        self,
        root: str,
        manifest_dir: Optional[str] = None,
        samples_per_epoch: Optional[int] = None,
        epoch_size_multiple: int = 1,
        is_training_set: bool = True,
        val_set_proportion: float = 0.01,
        seed: int = 42,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        video_size: tuple[int, int] | list[int] = (384, 320),
        clips_per_episode: int = 8,
        locality_stride: int = 12,
        max_open_parquet_shards: int = 2,
        max_open_video_shards: int = 6,
        video_decode_threads: int = 2,
        video_frame_cache_entries: int = 3,
        video_prefetch_frames: int = 24,
        resized_frame_cache_entries: int = 3,
        resized_frames_per_entry: int = 128,
        max_retries: int = 3,
        processor=None,
    ):
        del processor
        self.root = Path(root).expanduser().resolve()
        self.manifest_dir = (
            Path(manifest_dir).expanduser().resolve()
            if manifest_dir is not None
            else self.root / ".fastwam_intern_a1" / "manifest_v5_10hz"
        )
        done_path = self.manifest_dir / "done.json"
        if not done_path.is_file():
            raise FileNotFoundError(
                f"InternData manifest is missing: {done_path}. "
                "Run scripts/build_interndata_a1_manifest.py first."
            )
        with done_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if int(self.manifest.get("version", -1)) != 5:
            raise ValueError(
                "InternData 10Hz pretraining requires manifest version 5. "
                "Rebuild it with scripts/build_interndata_a1_manifest.py."
            )
        with (self.manifest_dir / "datasets.json").open("r", encoding="utf-8") as handle:
            self.datasets = json.load(handle)
        with (self.manifest_dir / "tasks.json").open("r", encoding="utf-8") as handle:
            self.tasks = json.load(handle)

        array_names = [
            "dataset_id",
            "episode_index",
            "length",
            "data_chunk",
            "data_file",
            "data_from",
            "data_file_from",
            "task_id",
            "shard_id",
            "head_chunk",
            "head_file",
            "head_from_timestamp",
            "left_chunk",
            "left_file",
            "left_from_timestamp",
            "right_chunk",
            "right_file",
            "right_from_timestamp",
        ]
        self.arrays = {
            name: np.load(self.manifest_dir / f"{name}.npy", mmap_mode="r")
            for name in array_names
        }

        if int(num_frames) != 33:
            raise ValueError("InternData pretraining currently requires num_frames=33.")
        if int(action_video_freq_ratio) != 4:
            raise ValueError(
                "InternData pretraining currently requires action_video_freq_ratio=4."
            )
        if tuple(video_size) != (384, 320):
            raise ValueError("InternData pretraining uses the RoboTwin 384x320 canvas.")

        self.enumerate_full_epoch = samples_per_epoch is None
        self.samples_per_epoch = None if self.enumerate_full_epoch else int(samples_per_epoch)
        self.epoch_size_multiple = max(int(epoch_size_multiple), 1)
        self.is_training_set = bool(is_training_set)
        self.val_set_proportion = float(val_set_proportion)
        self.seed = int(seed)
        self.video_indices = VIDEO_NATIVE_INDICES
        self.clips_per_episode = max(int(clips_per_episode), 1)
        self.locality_stride = max(int(locality_stride), 1)
        self.max_retries = max(int(max_retries), 0)

        episode_hash = (
            self.arrays["dataset_id"].astype(np.int64) * 1_000_003
            + self.arrays["episode_index"].astype(np.int64) * 97_409
            + self.seed
        ) % 10_000
        val_cutoff = int(round(self.val_set_proportion * 10_000))
        split_mask = episode_hash < val_cutoff
        if self.is_training_set:
            split_mask = ~split_mask
        self.episode_rows = np.flatnonzero(split_mask).astype(np.int64)
        if self.episode_rows.size == 0:
            raise ValueError("InternData split contains no episodes.")
        self.full_epoch_clip_count = int(
            np.maximum(
                self.arrays["length"][self.episode_rows].astype(np.int64)
                - NATIVE_HORIZON,
                0,
            ).sum()
        )
        if self.samples_per_epoch is None:
            self.samples_per_epoch = (
                self.full_epoch_clip_count // self.epoch_size_multiple
            ) * self.epoch_size_multiple
            if self.samples_per_epoch <= 0:
                raise ValueError(
                    "InternData split has fewer clips than epoch_size_multiple="
                    f"{self.epoch_size_multiple}."
                )

        shard_values = self.arrays["shard_id"][self.episode_rows]
        boundaries = np.flatnonzero(
            np.concatenate(([True], shard_values[1:] != shard_values[:-1], [True]))
        )
        self.group_starts = boundaries[:-1]
        self.group_ends = boundaries[1:]

        self._parquet_cache = _ParquetShardCache(max_open_parquet_shards)
        self._video_decoder = _PyAVShardDecoder(
            max_open_videos=max_open_video_shards,
            decode_threads=video_decode_threads,
            max_frame_caches=video_frame_cache_entries,
            prefetch_frames=video_prefetch_frames,
        )
        self.max_resized_frame_caches = max(int(resized_frame_cache_entries), 0)
        self.resized_frames_per_entry = max(int(resized_frames_per_entry), 1)
        self._resized_frame_cache: OrderedDict[
            tuple[str, int, int],
            OrderedDict[int, torch.Tensor],
        ] = OrderedDict()

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _encode_index(episode_row: int, start: int) -> int:
        if episode_row > _EPISODE_ROW_MASK:
            raise ValueError(f"Episode row is too large to encode: {episode_row}")
        return _ENCODED_INDEX_MARKER | (int(episode_row) << 32) | int(start)

    def _decode_index(self, index: int) -> tuple[int, int]:
        if int(index) & _ENCODED_INDEX_MARKER:
            episode_row = (int(index) >> 32) & _EPISODE_ROW_MASK
            start = int(index) & _START_MASK
            return episode_row, start

        rng = np.random.default_rng(self.seed + int(index))
        episode_row = int(self.episode_rows[int(rng.integers(self.episode_rows.size))])
        clip_count = int(self.arrays["length"][episode_row]) - NATIVE_HORIZON
        start = int(rng.integers(max(clip_count, 1)))
        return episode_row, start

    def _iter_group_batches(
        self,
        group_index: int,
        rng: np.random.Generator,
        batch_size: int,
    ):
        start_index = int(self.group_starts[group_index])
        end_index = int(self.group_ends[group_index])
        rows = self.episode_rows[start_index:end_index].copy()
        rng.shuffle(rows)
        states = [
            [
                int(row),
                0,
                max(int(self.arrays["length"][row]) - NATIVE_HORIZON, 0),
            ]
            for row in rows
            if int(self.arrays["length"][row]) > NATIVE_HORIZON
        ]
        cursor = 0
        batch = []
        while states:
            if cursor >= len(states):
                cursor = 0
            episode_row, clip_start, clip_count = states[cursor]
            remaining = clip_count - clip_start
            take = min(
                self.clips_per_episode,
                remaining,
                batch_size - len(batch),
            )
            batch.extend(
                self._encode_index(episode_row, start)
                for start in range(clip_start, clip_start + take)
            )
            clip_start += take
            if clip_start >= clip_count:
                states.pop(cursor)
            else:
                states[cursor][1] = clip_start
                cursor += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        return batch

    def iter_epoch_indices(
        self,
        epoch_seed: int,
        batch_size: Optional[int] = None,
        num_processes: Optional[int] = None,
    ) -> Iterator[int]:
        rng = np.random.default_rng(int(epoch_seed))
        batch_size = 1 if batch_size is None else int(batch_size)
        del num_processes
        if self.enumerate_full_epoch:
            yielded = 0
            group_iterators = {
                int(group_index): iter(
                    self._iter_group_batches(
                        int(group_index),
                        rng,
                        batch_size,
                    )
                )
                for group_index in range(len(self.group_starts))
            }
            active_groups = list(group_iterators)
            tail = []
            while active_groups:
                rng.shuffle(active_groups)
                next_active = []
                for group_index in active_groups:
                    iterator = group_iterators[group_index]
                    try:
                        batch = next(iterator)
                    except StopIteration as stop:
                        if stop.value:
                            tail.extend(stop.value)
                        continue
                    for encoded_index in batch:
                        yield encoded_index
                        yielded += 1
                        if yielded >= self.samples_per_epoch:
                            return
                    next_active.append(group_index)
                active_groups = next_active
            rng.shuffle(tail)
            for encoded_index in tail:
                yield encoded_index
                yielded += 1
                if yielded >= self.samples_per_epoch:
                    return
            return

        yielded = 0
        while yielded < self.samples_per_epoch:
            group_order = rng.permutation(len(self.group_starts))
            for group_index in group_order:
                start_index = int(self.group_starts[group_index])
                end_index = int(self.group_ends[group_index])
                rows = self.episode_rows[start_index:end_index].copy()
                rng.shuffle(rows)
                for episode_row in rows:
                    clip_count = (
                        int(self.arrays["length"][episode_row]) - NATIVE_HORIZON
                    )
                    if clip_count <= 0:
                        continue
                    base_start = int(rng.integers(clip_count))
                    for local_index in range(min(self.clips_per_episode, clip_count)):
                        clip_start = (
                            base_start + local_index * self.locality_stride
                        ) % clip_count
                        yield self._encode_index(int(episode_row), clip_start)
                        yielded += 1
                        if yielded >= self.samples_per_epoch:
                            return

    def _dataset_meta(self, episode_row: int) -> tuple[int, dict, Path]:
        dataset_id = int(self.arrays["dataset_id"][episode_row])
        metadata = self.datasets[dataset_id]
        dataset_root = self.root / metadata["relative_root"]
        return dataset_id, metadata, dataset_root

    @staticmethod
    def _data_columns(metadata: dict) -> list[str]:
        return list(
            dict.fromkeys(
                metadata["state_pose_keys"]
                + metadata["action_pose_keys"]
                + metadata["action_gripper_keys"]
            )
        )

    def _load_kinematics(
        self,
        episode_row: int,
        clip_start: int,
        metadata: dict,
        dataset_root: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_chunk = int(self.arrays["data_chunk"][episode_row])
        data_file = int(self.arrays["data_file"][episode_row])
        data_path = dataset_root / (
            f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
        )
        payload = self._parquet_cache.get(data_path, self._data_columns(metadata))
        local_start = (
            int(self.arrays["data_from"][episode_row])
            - int(self.arrays["data_file_from"][episode_row])
            + int(clip_start)
        )
        state_indices = local_start + STATE_NATIVE_INDICES
        action_indices = local_start + ACTION_NATIVE_INDICES

        arm_count = 2 if metadata["family"] == "dual" else 1
        state_pose = np.stack(
            [
                np.asarray(payload[key][state_indices], dtype=np.float32)
                for key in metadata["state_pose_keys"]
            ],
            axis=1,
        )
        action_pose = np.stack(
            [
                np.asarray(payload[key][action_indices], dtype=np.float32)
                for key in metadata["action_pose_keys"]
            ],
            axis=1,
        )
        action_gripper = np.stack(
            [
                _canonical_gripper_open(payload[key][action_indices]).reshape(-1)
                for key in metadata["action_gripper_keys"]
            ],
            axis=1,
        )
        state_gripper = np.zeros_like(action_gripper)
        if clip_start > 0:
            state_gripper_indices = local_start + STATE_NATIVE_INDICES - 1
            state_gripper = np.stack(
                [
                    _canonical_gripper_open(
                        payload[key][state_gripper_indices]
                    ).reshape(-1)
                    for key in metadata["action_gripper_keys"]
                ],
                axis=1,
            )
        else:
            # action openness at row t is the target state at t+1. Shift it
            # back for proprio. At the episode start, repeat the first command;
            # this affects only one clip per episode and avoids an arbitrary
            # half-open proprio value.
            state_gripper[0] = action_gripper[0]
            initial_episode_state_indices = (
                local_start + STATE_NATIVE_INDICES[1:] - 1
            )
            state_gripper[1:] = np.stack(
                [
                    _canonical_gripper_open(
                        payload[key][initial_episode_state_indices]
                    ).reshape(-1)
                    for key in metadata["action_gripper_keys"]
                ],
                axis=1,
            )

        if (
            state_pose.shape != (32, arm_count, 7)
            or action_pose.shape != (32, arm_count, 7)
        ):
            raise ValueError(
                f"Unexpected EEF pose shapes: state={state_pose.shape}, "
                f"action={action_pose.shape}."
            )

        state_rotation = _quaternion_wxyz_to_matrix(state_pose[..., 3:7])
        action_rotation = _quaternion_wxyz_to_matrix(action_pose[..., 3:7])
        initial_position = state_pose[0, :, :3]
        initial_rotation = state_rotation[0]

        delta_position = np.einsum(
            "aij,taj->tai",
            np.swapaxes(initial_rotation, -1, -2),
            action_pose[..., :3] - initial_position[None],
        )
        delta_rotation = np.einsum(
            "aij,tajk->taik",
            np.swapaxes(initial_rotation, -1, -2),
            action_rotation,
        )
        action_per_arm = np.concatenate(
            [
                delta_position / 0.25,
                _rotation_to_6d(delta_rotation),
                action_gripper[..., None],
            ],
            axis=-1,
        )
        state_per_arm = np.concatenate(
            [
                state_pose[..., :3],
                _rotation_to_6d(state_rotation),
                state_gripper[..., None],
            ],
            axis=-1,
        )

        action = np.zeros((32, 20), dtype=np.float32)
        proprio = np.zeros((32, 20), dtype=np.float32)
        action[:, : arm_count * 10] = action_per_arm.reshape(32, arm_count * 10)
        proprio[:, : arm_count * 10] = state_per_arm.reshape(32, arm_count * 10)
        action = np.clip(action, -5.0, 5.0)
        proprio = np.clip(proprio, -5.0, 5.0)

        action_dim_is_pad = np.ones((20,), dtype=np.bool_)
        action_dim_is_pad[: arm_count * 10] = False
        if not np.isfinite(action).all() or not np.isfinite(proprio).all():
            raise ValueError("InternData sample contains NaN/Inf after canonicalization.")
        return (
            torch.from_numpy(action),
            torch.from_numpy(proprio),
            torch.from_numpy(action_dim_is_pad),
        )

    def _video_path(
        self,
        dataset_root: Path,
        camera_key: str,
        chunk: int,
        file_index: int,
    ) -> Path:
        return dataset_root / (
            f"videos/{camera_key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        )

    def _decode_camera(
        self,
        episode_row: int,
        clip_start: int,
        role: str,
        camera_key: Optional[str],
        dataset_root: Path,
        fps: int,
        output_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        file_index = int(self.arrays[f"{role}_file"][episode_row])
        if camera_key is None or file_index < 0:
            return None
        chunk = int(self.arrays[f"{role}_chunk"][episode_row])
        from_timestamp = float(self.arrays[f"{role}_from_timestamp"][episode_row])
        timestamps = (
            from_timestamp + (clip_start + self.video_indices) / float(fps)
        ).tolist()
        video_path = self._video_path(
            dataset_root,
            camera_key,
            chunk,
            file_index,
        )
        frame_indices = [int(round(timestamp * fps)) for timestamp in timestamps]
        height, width = map(int, output_size)
        cache_key = (str(video_path), height, width)
        frame_cache = self._resized_frame_cache.pop(cache_key, None)
        if frame_cache is None:
            frame_cache = OrderedDict()
        self._resized_frame_cache[cache_key] = frame_cache
        while len(self._resized_frame_cache) > self.max_resized_frame_caches:
            self._resized_frame_cache.popitem(last=False)

        for frame_index in frame_indices:
            if frame_index in frame_cache:
                frame_cache.move_to_end(frame_index)
        missing_positions = [
            position
            for position, frame_index in enumerate(frame_indices)
            if frame_index not in frame_cache
        ]
        if missing_positions:
            missing_timestamps = [timestamps[position] for position in missing_positions]
            missing_frames = self._video_decoder.decode(
                video_path,
                timestamps=missing_timestamps,
                fps=fps,
            ).permute(0, 3, 1, 2).contiguous()
            missing_frames = transforms_F.resize(
                missing_frames,
                size=[height, width],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for position, frame in zip(
                missing_positions,
                missing_frames,
                strict=True,
            ):
                frame_index = frame_indices[position]
                frame_cache[frame_index] = frame
                frame_cache.move_to_end(frame_index)
            while len(frame_cache) > self.resized_frames_per_entry:
                frame_cache.popitem(last=False)

        frames = []
        for frame_index in frame_indices:
            frame = frame_cache.get(frame_index)
            if frame is None:
                raise RuntimeError(
                    f"Resized frame cache lost required frame {frame_index} for {video_path}."
                )
            frame_cache.move_to_end(frame_index)
            frames.append(frame)
        return torch.stack(frames, dim=0)

    def _load_video(
        self,
        episode_row: int,
        clip_start: int,
        metadata: dict,
        dataset_root: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_keys = metadata["camera_keys"]
        fps = int(metadata["fps"])
        head = self._decode_camera(
            episode_row,
            clip_start,
            "head",
            camera_keys["head"],
            dataset_root,
            fps,
            (256, 320),
        )
        left = self._decode_camera(
            episode_row,
            clip_start,
            "left",
            camera_keys["left"],
            dataset_root,
            fps,
            (128, 160),
        )
        right = self._decode_camera(
            episode_row,
            clip_start,
            "right",
            camera_keys["right"],
            dataset_root,
            fps,
            (128, 160),
        )
        view_role_valid_mask = torch.tensor(
            [head is not None, left is not None, right is not None],
            dtype=torch.bool,
        )
        if head is None:
            raise ValueError("InternData sample has no head camera.")

        if left is None:
            left = torch.zeros((9, 3, 128, 160), dtype=torch.uint8)
        if right is None:
            right = torch.zeros((9, 3, 128, 160), dtype=torch.uint8)
        bottom = torch.cat([left, right], dim=-1)
        canvas = torch.cat([head, bottom], dim=-2)
        video = (canvas.float() * (2.0 / 255.0) - 1.0).permute(1, 0, 2, 3)
        return video, view_role_valid_mask

    def _get_encoded(self, episode_row: int, clip_start: int, sample_index: int) -> dict:
        dataset_id, metadata, dataset_root = self._dataset_meta(episode_row)
        action, proprio, action_dim_is_pad = self._load_kinematics(
            episode_row,
            clip_start,
            metadata,
            dataset_root,
        )
        video, view_role_valid_mask = self._load_video(
            episode_row,
            clip_start,
            metadata,
            dataset_root,
        )
        video_spatial_valid_mask = torch.zeros((384, 320), dtype=torch.bool)
        if bool(view_role_valid_mask[0]):
            video_spatial_valid_mask[:256, :] = True
        if bool(view_role_valid_mask[1]):
            video_spatial_valid_mask[256:, :160] = True
        if bool(view_role_valid_mask[2]):
            video_spatial_valid_mask[256:, 160:] = True
        task = self.tasks[int(self.arrays["task_id"][episode_row])]
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": DEFAULT_PROMPT.format(task=task),
            "sample_idx": torch.tensor(sample_index, dtype=torch.long),
            "image_is_pad": torch.zeros((9,), dtype=torch.bool),
            "action_is_pad": torch.zeros((32,), dtype=torch.bool),
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_is_pad": torch.zeros((32,), dtype=torch.bool),
            "proprio_dim_is_pad": action_dim_is_pad.clone(),
            "view_role_valid_mask": view_role_valid_mask,
            "video_spatial_valid_mask": video_spatial_valid_mask,
            "dataset_id": torch.tensor(dataset_id, dtype=torch.int32),
        }

    def __getitem__(self, index: int) -> dict:
        episode_row, clip_start = self._decode_index(int(index))
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._get_encoded(episode_row, clip_start, int(index))
            except Exception as exc:
                last_error = exc
                if self.enumerate_full_epoch:
                    logger.warning(
                        "InternData exact-epoch sample failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    continue
                digest = hashlib.sha256(
                    f"{index}:{attempt}:{self.seed}".encode("utf-8")
                ).digest()
                fallback = int.from_bytes(digest[:8], "little")
                episode_row = int(
                    self.episode_rows[fallback % int(self.episode_rows.size)]
                )
                clip_count = (
                    int(self.arrays["length"][episode_row]) - NATIVE_HORIZON
                )
                clip_start = (fallback >> 32) % max(clip_count, 1)
                logger.warning(
                    "InternData sample failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
        raise RuntimeError(
            f"Failed to load InternData sample after {self.max_retries + 1} attempts."
        ) from last_error

    def __del__(self):
        decoder = getattr(self, "_video_decoder", None)
        if decoder is not None:
            decoder.close()
        resized_cache = getattr(self, "_resized_frame_cache", None)
        if resized_cache is not None:
            resized_cache.clear()
