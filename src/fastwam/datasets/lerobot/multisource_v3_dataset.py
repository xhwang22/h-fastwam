"""Route-safe mixed InternData and heterogeneous LeRobot v3 pretraining."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision.transforms import functional as transforms_F

from .intern_a1_v3_dataset import (
    DEFAULT_PROMPT,
    InternDataA1V3Dataset,
    _ParquetShardCache,
    _PyAVShardDecoder,
    _quaternion_wxyz_to_matrix,
    _rotation_to_6d,
)


logger = logging.getLogger(__name__)

_MIXED_INDEX_MARKER = 1 << 62
_VIDEO_ONLY_MARKER = 1 << 61
_SOURCE_SHIFT = 52
_SOURCE_MASK = (1 << 9) - 1
_PAYLOAD_MASK = (1 << _SOURCE_SHIFT) - 1
_VIDEO_OFFSETS_SECONDS = np.arange(9, dtype=np.float64) * 0.4
_ACTION_OFFSETS_SECONDS = np.arange(1, 33, dtype=np.float64) * 0.1
ROUTE_FULL = 0
ROUTE_VIDEO_ONLY = 1


def _config_to_dict(value) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping config, got {type(value)}.")
    return dict(value)


def _spatial_valid_mask(view_role_valid_mask: torch.Tensor) -> torch.Tensor:
    valid = view_role_valid_mask.to(dtype=torch.bool).reshape(3)
    mask = torch.zeros((384, 320), dtype=torch.bool)
    if bool(valid[0]):
        mask[:256, :] = True
    if bool(valid[1]):
        mask[256:, :160] = True
    if bool(valid[2]):
        mask[256:, 160:] = True
    return mask


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    return _quaternion_wxyz_to_matrix(
        quaternion[..., [3, 0, 1, 2]]
    )


def _euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=np.float32)
    roll, pitch, yaw = np.moveaxis(euler, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    flat = matrix.reshape(-1, 3, 3)
    output = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            output[index] = [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(
                    1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
                ) * 2.0
                output[index] = [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            elif axis == 1:
                scale = np.sqrt(
                    1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
                ) * 2.0
                output[index] = [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            else:
                scale = np.sqrt(
                    1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
                ) * 2.0
                output[index] = [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
    norms = np.linalg.norm(output, axis=-1, keepdims=True)
    output /= np.maximum(norms, 1e-12)
    return output.reshape(matrix.shape[:-2] + (4,)).astype(np.float32)


def _slerp(
    quaternion_0: np.ndarray,
    quaternion_1: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    quaternion_0 = np.asarray(quaternion_0, dtype=np.float64)
    quaternion_1 = np.asarray(quaternion_1, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    dot = np.sum(quaternion_0 * quaternion_1, axis=-1, keepdims=True)
    quaternion_1 = np.where(dot < 0.0, -quaternion_1, quaternion_1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    linear = sin_angle < 1e-6
    weight_0 = np.sin((1.0 - fraction) * angle) / np.maximum(sin_angle, 1e-12)
    weight_1 = np.sin(fraction * angle) / np.maximum(sin_angle, 1e-12)
    result = weight_0 * quaternion_0 + weight_1 * quaternion_1
    result = np.where(
        linear,
        (1.0 - fraction) * quaternion_0 + fraction * quaternion_1,
        result,
    )
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
    return result.astype(np.float32)


def _interpolate_linear(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    output = np.stack(
        [
            np.interp(target_times, source_times, flat[:, dimension])
            for dimension in range(flat.shape[1])
        ],
        axis=-1,
    )
    return output.reshape((target_times.size,) + values.shape[1:]).astype(np.float32)


def _interpolate_rotation(
    rotation: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    quaternion = _matrix_to_quaternion_wxyz(rotation)
    right = np.searchsorted(source_times, target_times, side="right")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    denominator = source_times[right] - source_times[left]
    fraction = (
        (target_times - source_times[left])
        / np.maximum(denominator, 1e-12)
    )
    fraction = fraction.reshape((-1,) + (1,) * (quaternion.ndim - 2) + (1,))
    interpolated = _slerp(
        quaternion[left],
        quaternion[right],
        fraction,
    )
    return _quaternion_wxyz_to_matrix(interpolated)


def _eef20(
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    target_gripper_open: np.ndarray,
    arm_valid: np.ndarray,
    gripper_valid: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_position = np.einsum(
        "aij,taj->tai",
        np.swapaxes(initial_rotation, -1, -2),
        target_position - initial_position[None],
    )
    delta_rotation = np.einsum(
        "aij,tajk->taik",
        np.swapaxes(initial_rotation, -1, -2),
        target_rotation,
    )
    gripper = np.clip(target_gripper_open, 0.0, 1.0) * 2.0 - 1.0
    action_per_arm = np.concatenate(
        [
            delta_position / 0.25,
            _rotation_to_6d(delta_rotation),
            gripper[..., None],
        ],
        axis=-1,
    )
    action = action_per_arm.reshape(32, 20).astype(np.float32)
    action = np.clip(action, -5.0, 5.0)
    dim_is_pad = np.ones((20,), dtype=np.bool_)
    for arm_index in range(2):
        if arm_valid[arm_index]:
            dim_is_pad[arm_index * 10 : arm_index * 10 + 9] = False
            if gripper_valid[arm_index]:
                dim_is_pad[arm_index * 10 + 9] = False
    action[:, dim_is_pad] = 0.0
    return (
        torch.from_numpy(action),
        torch.zeros((32,), dtype=torch.bool),
        torch.from_numpy(dim_is_pad),
    )


def _proprio20(
    position: np.ndarray,
    rotation: np.ndarray,
    gripper_open: np.ndarray,
    arm_valid: np.ndarray,
    gripper_valid: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    gripper = np.clip(gripper_open, 0.0, 1.0) * 2.0 - 1.0
    per_arm = np.concatenate(
        [position, _rotation_to_6d(rotation), gripper[..., None]],
        axis=-1,
    )
    proprio = np.zeros((32, 20), dtype=np.float32)
    proprio[:] = per_arm.reshape(20)[None]
    dim_is_pad = np.ones((20,), dtype=np.bool_)
    for arm_index in range(2):
        if arm_valid[arm_index]:
            dim_is_pad[arm_index * 10 : arm_index * 10 + 9] = False
            if gripper_valid[arm_index]:
                dim_is_pad[arm_index * 10 + 9] = False
    proprio[:, dim_is_pad] = 0.0
    return torch.from_numpy(np.clip(proprio, -5.0, 5.0)), torch.from_numpy(dim_is_pad)


class CanonicalLeRobotV3Dataset(Dataset):
    """Lazy canonical access to heterogeneous LeRobot v3 sources."""

    def __init__(
        self,
        manifest_dir: str,
        is_training_set: bool = True,
        val_set_proportion: float = 0.01,
        seed: int = 42,
        video_size: tuple[int, int] | list[int] = (384, 320),
        max_open_video_shards: int = 6,
        max_open_parquet_shards: int = 2,
        video_decode_threads: int = 2,
        video_frame_cache_entries: int = 3,
        video_prefetch_frames: int = 12,
        resized_frame_cache_entries: int = 3,
        resized_frames_per_entry: int = 96,
        max_retries: int = 3,
    ):
        self.manifest_dir = Path(manifest_dir).expanduser().resolve()
        done_path = self.manifest_dir / "done.json"
        if not done_path.is_file():
            raise FileNotFoundError(
                f"Multisource video manifest is missing: {done_path}. "
                "Run scripts/build_multisource_video_manifest.py first."
            )
        with done_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if int(self.manifest.get("version", -1)) != 4:
            raise ValueError(
                "Unsupported multisource video manifest version: "
                f"{self.manifest.get('version')}. Rebuild the manifest."
            )
        with (self.manifest_dir / "datasets.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.datasets = json.load(handle)
        with (self.manifest_dir / "sources.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.sources = json.load(handle)
        with (self.manifest_dir / "tasks.json").open("r", encoding="utf-8") as handle:
            self.tasks = json.load(handle)

        array_names = [
            "source_id",
            "dataset_id",
            "episode_index",
            "length",
            "start_count",
            "route_id",
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
        if tuple(video_size) != (384, 320):
            raise ValueError("Multisource pretraining uses the RoboTwin 384x320 canvas.")

        self.is_training_set = bool(is_training_set)
        self.val_set_proportion = float(val_set_proportion)
        self.seed = int(seed)
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
            raise ValueError("Multisource canonical split contains no episodes.")

        self.source_route_episode_rows: dict[tuple[int, int], np.ndarray] = {}
        self.source_route_clip_cumulative: dict[tuple[int, int], np.ndarray] = {}
        self.source_route_clip_counts: dict[tuple[int, int], int] = {}
        self.source_weights = []
        for source_id, source in enumerate(self.sources):
            rows = self.episode_rows[
                self.arrays["source_id"][self.episode_rows] == source_id
            ]
            if rows.size == 0:
                raise ValueError(
                    f"Source `{source['source_id']}` has no episodes in this split."
                )
            for route_id in (ROUTE_FULL, ROUTE_VIDEO_ONLY):
                route_rows = rows[
                    self.arrays["route_id"][rows] == route_id
                ]
                key = (source_id, route_id)
                self.source_route_episode_rows[key] = route_rows
                if route_rows.size == 0:
                    self.source_route_clip_cumulative[key] = np.empty(
                        (0,), dtype=np.int64
                    )
                    self.source_route_clip_counts[key] = 0
                    continue
                clip_counts = self.arrays["start_count"][route_rows].astype(np.int64)
                cumulative = np.cumsum(clip_counts, dtype=np.int64)
                self.source_route_clip_cumulative[key] = cumulative
                self.source_route_clip_counts[key] = int(cumulative[-1])
            self.source_weights.append(float(source.get("weight", 1.0)))
        self.dataset_id_offset = 0

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
        return int(self.arrays["start_count"][self.episode_rows].astype(np.int64).sum())

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
        native_start: int,
        role: str,
        camera_key: Optional[str],
        dataset_root: Path,
        fps: float,
        output_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        file_index = int(self.arrays[f"{role}_file"][episode_row])
        if camera_key is None or file_index < 0:
            return None
        chunk = int(self.arrays[f"{role}_chunk"][episode_row])
        from_timestamp = float(self.arrays[f"{role}_from_timestamp"][episode_row])
        timestamps = (
            from_timestamp
            + float(native_start) / float(fps)
            + _VIDEO_OFFSETS_SECONDS
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
                    f"Resized frame cache lost frame {frame_index} for {video_path}."
                )
            frame_cache.move_to_end(frame_index)
            frames.append(frame)
        return torch.stack(frames, dim=0)

    def _load_video(
        self,
        episode_row: int,
        native_start: int,
        metadata: dict,
        dataset_root: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_keys = metadata["camera_keys"]
        fps = float(metadata["fps"])
        head = self._decode_camera(
            episode_row,
            native_start,
            "head",
            camera_keys["head"],
            dataset_root,
            fps,
            (256, 320),
        )
        left = self._decode_camera(
            episode_row,
            native_start,
            "left",
            camera_keys["left"],
            dataset_root,
            fps,
            (128, 160),
        )
        right = self._decode_camera(
            episode_row,
            native_start,
            "right",
            camera_keys["right"],
            dataset_root,
            fps,
            (128, 160),
        )
        valid = torch.tensor(
            [head is not None, left is not None, right is not None],
            dtype=torch.bool,
        )
        if head is None:
            head = torch.zeros((9, 3, 256, 320), dtype=torch.uint8)
        if left is None:
            left = torch.zeros((9, 3, 128, 160), dtype=torch.uint8)
        if right is None:
            right = torch.zeros((9, 3, 128, 160), dtype=torch.uint8)
        bottom = torch.cat([left, right], dim=-1)
        canvas = torch.cat([head, bottom], dim=-2)
        video = (canvas.float() * (2.0 / 255.0) - 1.0).permute(1, 0, 2, 3)
        return video, valid

    @staticmethod
    def _adapter_columns(metadata: dict) -> list[str]:
        return list(metadata["adapter"].get("columns", []))

    def _load_native_payload(
        self,
        episode_row: int,
        native_start: int,
        metadata: dict,
        dataset_root: Path,
    ) -> tuple[dict[str, np.ndarray], slice, np.ndarray]:
        data_chunk = int(self.arrays["data_chunk"][episode_row])
        data_file = int(self.arrays["data_file"][episode_row])
        data_path = dataset_root / (
            f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
        )
        columns = self._adapter_columns(metadata)
        payload = self._parquet_cache.get(data_path, columns)
        fps = float(metadata["fps"])
        native_count = int(np.ceil(3.2 * fps)) + 1
        local_start = (
            int(self.arrays["data_from"][episode_row])
            - int(self.arrays["data_file_from"][episode_row])
            + int(native_start)
        )
        native_slice = slice(local_start, local_start + native_count)
        source_times = np.arange(native_count, dtype=np.float64) / fps
        return payload, native_slice, source_times

    @staticmethod
    def _identity_rotation(count: int) -> np.ndarray:
        return np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (count, 3, 3),
        ).copy()

    def _load_kinematics(
        self,
        episode_row: int,
        native_start: int,
        metadata: dict,
        dataset_root: Path,
    ) -> dict:
        payload, native_slice, source_times = self._load_native_payload(
            episode_row,
            native_start,
            metadata,
            dataset_root,
        )
        adapter = metadata["adapter"]
        adapter_type = adapter["type"]
        arm_valid = np.asarray([True, True], dtype=np.bool_)
        gripper_valid = np.asarray(
            adapter.get("gripper_valid", [True, True]),
            dtype=np.bool_,
        ).reshape(-1)
        if gripper_valid.size == 1:
            gripper_valid = np.repeat(gripper_valid, 2)
        if gripper_valid.shape != (2,):
            raise ValueError(
                f"Adapter `{adapter_type}` gripper_valid must have 2 values."
            )

        if adapter_type == "agibot_eef":
            state_position = np.asarray(
                payload["observation.states.end.position"][native_slice],
                dtype=np.float32,
            )
            state_rotation = _quaternion_xyzw_to_matrix(
                np.asarray(
                    payload["observation.states.end.orientation"][native_slice],
                    dtype=np.float32,
                )
            )
            action_position = np.asarray(
                payload["actions.end.position"][native_slice],
                dtype=np.float32,
            )
            action_rotation = _quaternion_xyzw_to_matrix(
                np.asarray(
                    payload["actions.end.orientation"][native_slice],
                    dtype=np.float32,
                )
            )
            action_gripper = np.asarray(
                payload["actions.effector.position"][native_slice],
                dtype=np.float32,
            )
            initial_position = state_position[0]
            initial_rotation = state_rotation[0]
            initial_gripper = np.clip(action_gripper[0], 0.0, 1.0)
            target_position = _interpolate_linear(
                action_position,
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_rotation = _interpolate_rotation(
                action_rotation,
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_gripper = _interpolate_linear(
                action_gripper,
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
        elif adapter_type == "droid_eef":
            state_pose = np.asarray(
                payload["observation.state.cartesian_position"][native_slice],
                dtype=np.float32,
            )
            action_pose = np.asarray(
                payload["action.cartesian_position"][native_slice],
                dtype=np.float32,
            )
            state_gripper = np.asarray(
                payload["observation.state.gripper_position"][native_slice],
                dtype=np.float32,
            ).reshape(-1)
            action_gripper = np.asarray(
                payload["action.gripper_position"][native_slice],
                dtype=np.float32,
            ).reshape(-1)
            initial_position = np.zeros((2, 3), dtype=np.float32)
            initial_rotation = self._identity_rotation(2)
            initial_gripper = np.zeros((2,), dtype=np.float32)
            initial_position[0] = state_pose[0, :3]
            initial_rotation[0] = _euler_xyz_to_matrix(state_pose[0, 3:6])
            initial_gripper[0] = state_gripper[0]
            target_position = np.zeros((32, 2, 3), dtype=np.float32)
            target_rotation = np.broadcast_to(
                np.eye(3, dtype=np.float32),
                (32, 2, 3, 3),
            ).copy()
            target_gripper = np.zeros((32, 2), dtype=np.float32)
            target_position[:, 0] = _interpolate_linear(
                action_pose[:, :3],
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_rotation[:, 0] = _interpolate_rotation(
                _euler_xyz_to_matrix(action_pose[:, 3:6]),
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_gripper[:, 0] = _interpolate_linear(
                action_gripper[:, None],
                source_times,
                _ACTION_OFFSETS_SECONDS,
            ).reshape(-1)
            arm_valid[1] = False
            gripper_valid[1] = False
        elif adapter_type == "oxe_euler_state":
            state = np.asarray(
                payload["observation.state"][native_slice],
                dtype=np.float32,
            )
            initial_position = np.zeros((2, 3), dtype=np.float32)
            initial_rotation = self._identity_rotation(2)
            initial_gripper = np.zeros((2,), dtype=np.float32)
            initial_position[0] = state[0, :3]
            initial_rotation[0] = _euler_xyz_to_matrix(state[0, 3:6])
            target_position = np.zeros((32, 2, 3), dtype=np.float32)
            target_rotation = np.broadcast_to(
                np.eye(3, dtype=np.float32),
                (32, 2, 3, 3),
            ).copy()
            target_gripper = np.zeros((32, 2), dtype=np.float32)
            target_position[:, 0] = _interpolate_linear(
                state[:, :3],
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_rotation[:, 0] = _interpolate_rotation(
                _euler_xyz_to_matrix(state[:, 3:6]),
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            arm_valid[1] = False
            gripper_valid[:] = False
        elif adapter_type == "robocoin_eef":
            state_pose = np.asarray(
                payload["eef_sim_pose_state"][native_slice],
                dtype=np.float32,
            ).reshape(-1, 2, 6)
            action_pose = np.asarray(
                payload["eef_sim_pose_action"][native_slice],
                dtype=np.float32,
            ).reshape(-1, 2, 6)
            initial_position = state_pose[0, :, :3]
            initial_rotation = _euler_xyz_to_matrix(state_pose[0, :, 3:6])
            target_position = _interpolate_linear(
                action_pose[:, :, :3],
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_rotation = _interpolate_rotation(
                _euler_xyz_to_matrix(action_pose[:, :, 3:6]),
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            if adapter.get("has_gripper", False):
                state_gripper = np.asarray(
                    payload["gripper_open_scale_state"][native_slice],
                    dtype=np.float32,
                )
                action_gripper = np.asarray(
                    payload["gripper_open_scale_action"][native_slice],
                    dtype=np.float32,
                )
                initial_gripper = np.clip(state_gripper[0], 0.0, 1.0)
                target_gripper = _interpolate_linear(
                    action_gripper,
                    source_times,
                    _ACTION_OFFSETS_SECONDS,
                )
            else:
                initial_gripper = np.zeros((2,), dtype=np.float32)
                target_gripper = np.zeros((32, 2), dtype=np.float32)
                gripper_valid[:] = False
        elif adapter_type == "galaxea_eef":
            left_pose = np.asarray(
                payload["observation.state.left_ee_pose"][native_slice],
                dtype=np.float32,
            )
            right_pose = np.asarray(
                payload["observation.state.right_ee_pose"][native_slice],
                dtype=np.float32,
            )
            state_position = np.stack(
                [left_pose[:, :3], right_pose[:, :3]],
                axis=1,
            )
            state_rotation = _quaternion_xyzw_to_matrix(
                np.stack(
                    [left_pose[:, 3:7], right_pose[:, 3:7]],
                    axis=1,
                )
            )
            state_gripper = np.stack(
                [
                    np.asarray(
                        payload["observation.state.left_gripper"][native_slice],
                        dtype=np.float32,
                    ).reshape(-1),
                    np.asarray(
                        payload["observation.state.right_gripper"][native_slice],
                        dtype=np.float32,
                    ).reshape(-1),
                ],
                axis=1,
            )
            state_gripper = 1.0 - np.clip(state_gripper, 0.0, 100.0) / 100.0
            initial_position = state_position[0]
            initial_rotation = state_rotation[0]
            initial_gripper = state_gripper[0]
            target_position = _interpolate_linear(
                state_position,
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_rotation = _interpolate_rotation(
                state_rotation,
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
            target_gripper = _interpolate_linear(
                state_gripper,
                source_times,
                _ACTION_OFFSETS_SECONDS,
            )
        else:
            raise ValueError(
                f"Adapter `{adapter_type}` cannot produce FULL kinematics."
            )

        action, action_is_pad, action_dim_is_pad = _eef20(
            initial_position=initial_position,
            initial_rotation=initial_rotation,
            target_position=target_position,
            target_rotation=target_rotation,
            target_gripper_open=target_gripper,
            arm_valid=arm_valid,
            gripper_valid=gripper_valid,
        )
        proprio, proprio_dim_is_pad = _proprio20(
            position=initial_position,
            rotation=initial_rotation,
            gripper_open=initial_gripper,
            arm_valid=arm_valid,
            gripper_valid=gripper_valid,
        )
        if (
            not torch.isfinite(action).all()
            or not torch.isfinite(proprio).all()
        ):
            raise ValueError(
                f"{adapter_type} produced NaN/Inf canonical kinematics."
            )
        return {
            "action": action,
            "proprio": proprio,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_is_pad": torch.zeros((32,), dtype=torch.bool),
            "proprio_dim_is_pad": proprio_dim_is_pad,
        }

    def _get_sample(
        self,
        source_id: int,
        route_id: int,
        sample_key: int,
        sample_index: int,
    ) -> dict:
        pool_key = (source_id, route_id)
        source_rows = self.source_route_episode_rows[pool_key]
        total_clips = self.source_route_clip_counts[pool_key]
        if total_clips <= 0:
            raise ValueError(
                f"Source {source_id} has no clips for route {route_id}."
            )
        clip_ordinal = int(sample_key) % total_clips
        cumulative = self.source_route_clip_cumulative[pool_key]
        source_episode_index = int(
            np.searchsorted(cumulative, clip_ordinal, side="right")
        )
        episode_row = int(source_rows[source_episode_index])
        previous_end = (
            0
            if source_episode_index == 0
            else int(cumulative[source_episode_index - 1])
        )
        native_start = clip_ordinal - previous_end
        dataset_id = int(self.arrays["dataset_id"][episode_row])
        metadata = self.datasets[dataset_id]
        dataset_root = Path(metadata["root"])
        video, view_role_valid_mask = self._load_video(
            episode_row,
            native_start,
            metadata,
            dataset_root,
        )
        task = self.tasks[int(self.arrays["task_id"][episode_row])]
        sample = {
            "video": video,
            "prompt": DEFAULT_PROMPT.format(task=task),
            "sample_idx": torch.tensor(sample_index, dtype=torch.long),
            "image_is_pad": torch.zeros((9,), dtype=torch.bool),
            "view_role_valid_mask": view_role_valid_mask,
            "video_spatial_valid_mask": _spatial_valid_mask(view_role_valid_mask),
            "dataset_id": torch.tensor(
                self.dataset_id_offset + dataset_id,
                dtype=torch.int32,
            ),
            "source_id": torch.tensor(source_id + 1, dtype=torch.int16),
            "route_id": torch.tensor(route_id, dtype=torch.int8),
        }
        if route_id == ROUTE_FULL:
            sample.update(
                self._load_kinematics(
                    episode_row,
                    native_start,
                    metadata,
                    dataset_root,
                )
            )
        return sample

    def get_source_sample(
        self,
        source_id: int,
        route_id: int,
        sample_key: int,
        sample_index: int,
    ) -> dict:
        if source_id < 0 or source_id >= len(self.sources):
            raise IndexError(f"Invalid VIDEO_ONLY source id: {source_id}")
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._get_sample(
                    source_id,
                    route_id,
                    sample_key,
                    sample_index,
                )
            except Exception as exc:
                last_error = exc
                digest = hashlib.sha256(
                    f"{source_id}:{route_id}:{sample_key}:{attempt}:{self.seed}".encode("utf-8")
                ).digest()
                sample_key = int.from_bytes(digest[:8], "little") & _PAYLOAD_MASK
                logger.warning(
                    "Multisource sample failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
        raise RuntimeError(
            "Failed to load multisource sample after "
            f"{self.max_retries + 1} attempts."
        ) from last_error

    def __getitem__(self, index: int) -> dict:
        source_id = int(index) % len(self.sources)
        route_id = (
            ROUTE_FULL
            if self.source_route_clip_counts[(source_id, ROUTE_FULL)] > 0
            else ROUTE_VIDEO_ONLY
        )
        return self.get_source_sample(
            source_id,
            route_id,
            int(index),
            int(index),
        )

    def __del__(self):
        decoder = getattr(self, "_video_decoder", None)
        if decoder is not None:
            decoder.close()
        cache = getattr(self, "_resized_frame_cache", None)
        if cache is not None:
            cache.clear()


class MultiSourceRobotV3Dataset(Dataset):
    """Mix canonical FULL and VIDEO_ONLY samples in route-homogeneous batches."""

    def __init__(
        self,
        intern_a1: dict,
        external_sources: dict,
        samples_per_epoch: int,
        epoch_size_multiple: int = 1,
        full_batch_fraction: Optional[float] = None,
        intern_a1_weight: float = 0.20,
        video_clips_per_episode: int = 8,
        video_locality_stride: Optional[int] = None,
        seed: int = 42,
        processor=None,
    ):
        del processor
        intern_config = _config_to_dict(intern_a1)
        external_config = _config_to_dict(external_sources)
        intern_config["samples_per_epoch"] = int(samples_per_epoch)
        intern_config["epoch_size_multiple"] = 1
        self.intern_a1 = InternDataA1V3Dataset(**intern_config)
        self.external_sources = CanonicalLeRobotV3Dataset(**external_config)
        self.external_sources.dataset_id_offset = len(self.intern_a1.datasets)

        self.epoch_size_multiple = max(int(epoch_size_multiple), 1)
        requested_samples = int(samples_per_epoch)
        self.samples_per_epoch = (
            requested_samples // self.epoch_size_multiple
        ) * self.epoch_size_multiple
        if self.samples_per_epoch <= 0:
            raise ValueError(
                f"samples_per_epoch={samples_per_epoch} is smaller than "
                f"epoch_size_multiple={self.epoch_size_multiple}."
            )
        self.seed = int(seed)
        self.video_clips_per_episode = max(int(video_clips_per_episode), 1)
        self.video_locality_stride = (
            None
            if video_locality_stride is None
            else max(int(video_locality_stride), 1)
        )

        weights = np.asarray(self.external_sources.source_weights, dtype=np.float64)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError(f"Invalid source weights: {weights.tolist()}")
        self.intern_a1_weight = float(intern_a1_weight)
        if self.intern_a1_weight <= 0:
            raise ValueError(
                f"intern_a1_weight must be positive, got {intern_a1_weight}."
            )
        full_masses = [self.intern_a1_weight]
        full_source_ids = [-1]
        video_masses = []
        video_source_ids = []
        source_effective_weights = {"interndata_a1": self.intern_a1_weight}
        for source_id, (source, source_weight) in enumerate(
            zip(self.external_sources.sources, weights, strict=True)
        ):
            full_count = self.external_sources.source_route_clip_counts[
                (source_id, ROUTE_FULL)
            ]
            video_count = self.external_sources.source_route_clip_counts[
                (source_id, ROUTE_VIDEO_ONLY)
            ]
            total_count = full_count + video_count
            if total_count <= 0:
                raise ValueError(
                    f"Source `{source['source_id']}` has no clips."
                )
            full_mass = float(source_weight) * full_count / total_count
            video_mass = float(source_weight) * video_count / total_count
            if full_mass > 0:
                full_source_ids.append(source_id)
                full_masses.append(full_mass)
            if video_mass > 0:
                video_source_ids.append(source_id)
                video_masses.append(video_mass)
            source_effective_weights[source["source_id"]] = float(source_weight)

        self.full_source_ids = np.asarray(full_source_ids, dtype=np.int64)
        self.full_source_probabilities = np.asarray(
            full_masses, dtype=np.float64
        )
        self.video_source_ids = np.asarray(video_source_ids, dtype=np.int64)
        self.video_source_probabilities = np.asarray(
            video_masses, dtype=np.float64
        )
        full_mass_total = float(self.full_source_probabilities.sum())
        video_mass_total = float(self.video_source_probabilities.sum())
        total_mass = full_mass_total + video_mass_total
        if full_batch_fraction is None:
            self.full_batch_fraction = full_mass_total / total_mass
        else:
            self.full_batch_fraction = float(full_batch_fraction)
        if not 0.0 < self.full_batch_fraction <= 1.0:
            raise ValueError(
                f"Invalid FULL route fraction: {self.full_batch_fraction}."
            )
        self.full_source_probabilities /= full_mass_total
        if video_mass_total > 0:
            self.video_source_probabilities /= video_mass_total
        logger.info(
            "Multisource declared weights=%s; route fractions FULL=%.3f "
            "VIDEO_ONLY=%.3f; FULL sources=%s; VIDEO_ONLY sources=%s",
            source_effective_weights,
            self.full_batch_fraction,
            1.0 - self.full_batch_fraction,
            self.full_source_ids.tolist(),
            self.video_source_ids.tolist(),
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _encode_external(
        route_id: int,
        source_id: int,
        sample_key: int,
    ) -> int:
        if source_id > _SOURCE_MASK:
            raise ValueError(f"Too many external sources to encode: {source_id}")
        packed = (
            (int(route_id) << 61)
            | (int(source_id) << _SOURCE_SHIFT)
            | (int(sample_key) & _PAYLOAD_MASK)
        )
        return -1 - packed

    @staticmethod
    def _decode_mixed_index(index: int) -> tuple[bool, int, int, int]:
        if int(index) >= 0:
            return False, ROUTE_FULL, -1, int(index)
        packed = -1 - int(index)
        route_id = (packed >> 61) & 1
        source_id = (packed >> _SOURCE_SHIFT) & _SOURCE_MASK
        sample_key = packed & _PAYLOAD_MASK
        return True, route_id, source_id, sample_key

    def _external_locality_group(
        self,
        source_id: int,
        route_id: int,
        rng: np.random.Generator,
        group_size: int,
    ) -> Iterator[int]:
        pool_key = (source_id, route_id)
        cumulative = self.external_sources.source_route_clip_cumulative[pool_key]
        total_clips = self.external_sources.source_route_clip_counts[pool_key]
        clip_ordinal = int(rng.integers(total_clips))
        source_episode_index = int(
            np.searchsorted(cumulative, clip_ordinal, side="right")
        )
        previous_end = (
            0
            if source_episode_index == 0
            else int(cumulative[source_episode_index - 1])
        )
        episode_clip_count = int(
            cumulative[source_episode_index] - previous_end
        )
        base_start = clip_ordinal - previous_end
        episode_row = int(
            self.external_sources.source_route_episode_rows[pool_key][
                source_episode_index
            ]
        )
        dataset_id = int(self.external_sources.arrays["dataset_id"][episode_row])
        source_fps = float(self.external_sources.datasets[dataset_id]["fps"])
        locality_stride = (
            self.video_locality_stride
            if self.video_locality_stride is not None
            else max(int(round(source_fps * 0.4)), 1)
        )
        for local_index in range(group_size):
            episode_start = (
                base_start + local_index * locality_stride
            ) % episode_clip_count
            yield self._encode_external(
                route_id,
                source_id,
                previous_end + episode_start,
            )

    def _route_schedule(
        self,
        num_global_batches: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        video_fraction = 1.0 - self.full_batch_fraction
        video_batch_count = int(round(num_global_batches * video_fraction))
        schedule = np.zeros((num_global_batches,), dtype=np.bool_)
        if video_batch_count <= 0:
            return schedule
        positions = np.floor(
            (np.arange(video_batch_count, dtype=np.float64) + 0.5)
            * num_global_batches
            / video_batch_count
        ).astype(np.int64)
        schedule[np.clip(positions, 0, num_global_batches - 1)] = True
        schedule = np.roll(schedule, int(rng.integers(num_global_batches)))
        return schedule

    def iter_epoch_indices(
        self,
        epoch_seed: int,
        batch_size: Optional[int] = None,
        num_processes: Optional[int] = None,
    ) -> Iterator[int]:
        batch_size = 1 if batch_size is None else int(batch_size)
        num_processes = 1 if num_processes is None else int(num_processes)
        global_batch_size = batch_size * num_processes
        if self.samples_per_epoch % global_batch_size != 0:
            raise ValueError(
                f"samples_per_epoch={self.samples_per_epoch} must be divisible by "
                f"global_batch_size={global_batch_size}."
            )

        rng = np.random.default_rng(int(epoch_seed))
        num_global_batches = self.samples_per_epoch // global_batch_size
        route_schedule = self._route_schedule(num_global_batches, rng)
        intern_indices = iter(
            self.intern_a1.iter_epoch_indices(
                epoch_seed,
                batch_size=batch_size,
                num_processes=num_processes,
            )
        )
        for is_video_only in route_schedule:
            route_id = (
                ROUTE_VIDEO_ONLY if bool(is_video_only) else ROUTE_FULL
            )
            yielded = 0
            while yielded < global_batch_size:
                group_size = min(
                    self.video_clips_per_episode,
                    global_batch_size - yielded,
                )
                if route_id == ROUTE_FULL:
                    source_id = int(
                        rng.choice(
                            self.full_source_ids,
                            p=self.full_source_probabilities,
                        )
                    )
                    if source_id < 0:
                        for _ in range(group_size):
                            yield next(intern_indices)
                    else:
                        yield from self._external_locality_group(
                            source_id,
                            route_id,
                            rng,
                            group_size,
                        )
                else:
                    source_id = int(
                        rng.choice(
                            self.video_source_ids,
                            p=self.video_source_probabilities,
                        )
                    )
                    yield from self._external_locality_group(
                        source_id,
                        route_id,
                        rng,
                        group_size,
                    )
                yielded += group_size

    def __getitem__(self, index: int) -> dict:
        is_external, route_id, source_id, sample_key = self._decode_mixed_index(
            int(index)
        )
        if is_external:
            return self.external_sources.get_source_sample(
                source_id=source_id,
                route_id=route_id,
                sample_key=sample_key,
                sample_index=int(index),
            )
        sample = self.intern_a1[sample_key]
        sample["source_id"] = torch.tensor(0, dtype=torch.int16)
        sample["route_id"] = torch.tensor(0, dtype=torch.int8)
        return sample


# Backward-compatible import for manifests/scripts created during the pilot.
VideoOnlyLeRobotV3Dataset = CanonicalLeRobotV3Dataset
