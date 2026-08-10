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


def _scale_gripper(value: np.ndarray, value_range: list[float]) -> np.ndarray:
    minimum, maximum = map(float, value_range)
    denominator = maximum - minimum
    if denominator < 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    scaled = (np.asarray(value, dtype=np.float32) - minimum) / denominator
    return np.clip(scaled * 2.0 - 1.0, -1.0, 1.0)


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
    def __init__(self, max_open_videos: int, decode_threads: int):
        self.max_open_videos = max(int(max_open_videos), 1)
        self.decode_threads = max(int(decode_threads), 1)
        self._cache: OrderedDict[str, tuple] = OrderedDict()

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

    def decode(self, path: Path, timestamps: list[float], fps: float) -> torch.Tensor:
        timestamps = [float(value) for value in timestamps]
        tolerance = 0.5 / float(fps) + 0.002
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
                stop_time = max(timestamps) + 2.0 / float(fps)
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
                selected = []
                for timestamp in timestamps:
                    index = int(np.argmin(np.abs(loaded_times_np - timestamp)))
                    error = abs(float(loaded_times_np[index]) - timestamp)
                    if error > tolerance:
                        raise RuntimeError(
                            f"Video timestamp error {error:.6f}s exceeds {tolerance:.6f}s "
                            f"for {path}."
                        )
                    selected.append(loaded_frames[index])
                return torch.from_numpy(np.stack(selected, axis=0))
            except Exception as exc:
                last_error = exc
                cached = self._cache.pop(str(path), None)
                if cached is not None:
                    cached[0].close()
        raise RuntimeError(f"Failed to decode {path}: {last_error}") from last_error

    def close(self):
        while self._cache:
            _, (container, _) = self._cache.popitem(last=False)
            container.close()


class InternDataA1V3Dataset(Dataset):
    """Map-style virtual epoch over InternData-A1 with shard-local sample order."""

    def __init__(
        self,
        root: str,
        manifest_dir: Optional[str] = None,
        samples_per_epoch: int = 2_000_000,
        is_training_set: bool = True,
        val_set_proportion: float = 0.01,
        seed: int = 42,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        video_size: tuple[int, int] | list[int] = (384, 320),
        clips_per_episode: int = 8,
        locality_stride: int = 4,
        max_open_parquet_shards: int = 2,
        max_open_video_shards: int = 6,
        video_decode_threads: int = 2,
        max_retries: int = 3,
        processor=None,
    ):
        del processor
        self.root = Path(root).expanduser().resolve()
        self.manifest_dir = (
            Path(manifest_dir).expanduser().resolve()
            if manifest_dir is not None
            else self.root / ".fastwam_intern_a1" / "manifest_v1"
        )
        done_path = self.manifest_dir / "done.json"
        if not done_path.is_file():
            raise FileNotFoundError(
                f"InternData manifest is missing: {done_path}. "
                "Run scripts/build_interndata_a1_manifest.py first."
            )
        with done_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
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

        self.samples_per_epoch = int(samples_per_epoch)
        self.is_training_set = bool(is_training_set)
        self.val_set_proportion = float(val_set_proportion)
        self.seed = int(seed)
        self.video_indices = np.arange(0, 33, 4, dtype=np.int64)
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
        )

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
        clip_count = int(self.arrays["length"][episode_row]) - 32
        start = int(rng.integers(max(clip_count, 1)))
        return episode_row, start

    def iter_epoch_indices(self, epoch_seed: int) -> Iterator[int]:
        rng = np.random.default_rng(int(epoch_seed))
        yielded = 0
        while yielded < self.samples_per_epoch:
            group_order = rng.permutation(len(self.group_starts))
            for group_index in group_order:
                start_index = int(self.group_starts[group_index])
                end_index = int(self.group_ends[group_index])
                rows = self.episode_rows[start_index:end_index].copy()
                rng.shuffle(rows)
                for episode_row in rows:
                    clip_count = int(self.arrays["length"][episode_row]) - 32
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
                + metadata["state_gripper_keys"]
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
        state_slice = slice(local_start, local_start + 32)
        action_slice = slice(local_start, local_start + 32)

        arm_count = 2 if metadata["family"] == "dual" else 1
        state_pose = np.stack(
            [
                np.asarray(payload[key][state_slice], dtype=np.float32)
                for key in metadata["state_pose_keys"]
            ],
            axis=1,
        )
        action_pose = np.stack(
            [
                np.asarray(payload[key][action_slice], dtype=np.float32)
                for key in metadata["action_pose_keys"]
            ],
            axis=1,
        )
        state_gripper = np.stack(
            [
                _scale_gripper(
                    payload[key][state_slice],
                    metadata["state_gripper_ranges"][arm_index],
                ).reshape(-1)
                for arm_index, key in enumerate(metadata["state_gripper_keys"])
            ],
            axis=1,
        )
        action_gripper = np.stack(
            [
                _scale_gripper(
                    payload[key][action_slice],
                    metadata["action_gripper_ranges"][arm_index],
                ).reshape(-1)
                for arm_index, key in enumerate(metadata["action_gripper_keys"])
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
    ) -> Optional[torch.Tensor]:
        file_index = int(self.arrays[f"{role}_file"][episode_row])
        if camera_key is None or file_index < 0:
            return None
        chunk = int(self.arrays[f"{role}_chunk"][episode_row])
        from_timestamp = float(self.arrays[f"{role}_from_timestamp"][episode_row])
        timestamps = (
            from_timestamp + (clip_start + self.video_indices) / float(fps)
        ).tolist()
        frames = self._video_decoder.decode(
            self._video_path(dataset_root, camera_key, chunk, file_index),
            timestamps=timestamps,
            fps=fps,
        )
        return frames.permute(0, 3, 1, 2).contiguous()

    def _load_video(
        self,
        episode_row: int,
        clip_start: int,
        metadata: dict,
        dataset_root: Path,
    ) -> torch.Tensor:
        camera_keys = metadata["camera_keys"]
        fps = int(metadata["fps"])
        head = self._decode_camera(
            episode_row, clip_start, "head", camera_keys["head"], dataset_root, fps
        )
        left = self._decode_camera(
            episode_row, clip_start, "left", camera_keys["left"], dataset_root, fps
        )
        right = self._decode_camera(
            episode_row, clip_start, "right", camera_keys["right"], dataset_root, fps
        )
        if head is None:
            raise ValueError("InternData sample has no head camera.")

        head = transforms_F.resize(
            head,
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        if left is None:
            left = torch.zeros((9, 3, 128, 160), dtype=torch.uint8)
        else:
            left = transforms_F.resize(
                left,
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
        if right is None:
            right = torch.zeros((9, 3, 128, 160), dtype=torch.uint8)
        else:
            right = transforms_F.resize(
                right,
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
        bottom = torch.cat([left, right], dim=-1)
        canvas = torch.cat([head, bottom], dim=-2)
        return (canvas.float() * (2.0 / 255.0) - 1.0).permute(1, 0, 2, 3)

    def _get_encoded(self, episode_row: int, clip_start: int, sample_index: int) -> dict:
        dataset_id, metadata, dataset_root = self._dataset_meta(episode_row)
        action, proprio, action_dim_is_pad = self._load_kinematics(
            episode_row,
            clip_start,
            metadata,
            dataset_root,
        )
        video = self._load_video(episode_row, clip_start, metadata, dataset_root)
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
                digest = hashlib.sha256(
                    f"{index}:{attempt}:{self.seed}".encode("utf-8")
                ).digest()
                fallback = int.from_bytes(digest[:8], "little")
                episode_row = int(
                    self.episode_rows[fallback % int(self.episode_rows.size)]
                )
                clip_count = int(self.arrays["length"][episode_row]) - 32
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
