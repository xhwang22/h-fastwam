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
    _PyAVShardDecoder,
)


logger = logging.getLogger(__name__)

_MIXED_INDEX_MARKER = 1 << 62
_VIDEO_ONLY_MARKER = 1 << 61
_SOURCE_SHIFT = 52
_SOURCE_MASK = (1 << 9) - 1
_PAYLOAD_MASK = (1 << _SOURCE_SHIFT) - 1
_VIDEO_OFFSETS_SECONDS = np.arange(9, dtype=np.float64) * 0.4


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


class VideoOnlyLeRobotV3Dataset(Dataset):
    """Lazy video-only access to a manifest spanning several LeRobot v3 sources."""

    def __init__(
        self,
        manifest_dir: str,
        is_training_set: bool = True,
        val_set_proportion: float = 0.01,
        seed: int = 42,
        video_size: tuple[int, int] | list[int] = (384, 320),
        max_open_video_shards: int = 6,
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
        if int(self.manifest.get("version", -1)) != 2:
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
            raise ValueError("Multisource VIDEO_ONLY split contains no episodes.")

        self.source_episode_rows = []
        self.source_clip_cumulative = []
        self.source_clip_counts = []
        self.source_weights = []
        for source_id, source in enumerate(self.sources):
            rows = self.episode_rows[
                self.arrays["source_id"][self.episode_rows] == source_id
            ]
            if rows.size == 0:
                raise ValueError(
                    f"Source `{source['source_id']}` has no episodes in this split."
                )
            self.source_episode_rows.append(rows)
            clip_counts = self.arrays["start_count"][rows].astype(np.int64)
            cumulative = np.cumsum(clip_counts, dtype=np.int64)
            self.source_clip_cumulative.append(cumulative)
            self.source_clip_counts.append(int(cumulative[-1]))
            self.source_weights.append(float(source.get("weight", 1.0)))
        self.dataset_id_offset = 0

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

    def _get_sample(
        self,
        source_id: int,
        sample_key: int,
        sample_index: int,
    ) -> dict:
        source_rows = self.source_episode_rows[source_id]
        clip_ordinal = int(sample_key) % self.source_clip_counts[source_id]
        cumulative = self.source_clip_cumulative[source_id]
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
        return {
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
            "route_id": torch.tensor(1, dtype=torch.int8),
        }

    def get_source_sample(
        self,
        source_id: int,
        sample_key: int,
        sample_index: int,
    ) -> dict:
        if source_id < 0 or source_id >= len(self.sources):
            raise IndexError(f"Invalid VIDEO_ONLY source id: {source_id}")
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._get_sample(source_id, sample_key, sample_index)
            except Exception as exc:
                last_error = exc
                digest = hashlib.sha256(
                    f"{source_id}:{sample_key}:{attempt}:{self.seed}".encode("utf-8")
                ).digest()
                sample_key = int.from_bytes(digest[:8], "little") & _PAYLOAD_MASK
                logger.warning(
                    "Multisource VIDEO_ONLY sample failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
        raise RuntimeError(
            "Failed to load multisource VIDEO_ONLY sample after "
            f"{self.max_retries + 1} attempts."
        ) from last_error

    def __getitem__(self, index: int) -> dict:
        source_id = int(index) % len(self.sources)
        return self.get_source_sample(source_id, int(index), int(index))

    def __del__(self):
        decoder = getattr(self, "_video_decoder", None)
        if decoder is not None:
            decoder.close()
        cache = getattr(self, "_resized_frame_cache", None)
        if cache is not None:
            cache.clear()


class MultiSourceRobotV3Dataset(Dataset):
    """Mix FULL InternData and route-homogeneous VIDEO_ONLY global batches."""

    def __init__(
        self,
        intern_a1: dict,
        video_only: dict,
        samples_per_epoch: int,
        epoch_size_multiple: int = 1,
        full_batch_fraction: float = 0.85,
        video_clips_per_episode: int = 8,
        video_locality_stride: int = 1,
        seed: int = 42,
        processor=None,
    ):
        del processor
        intern_config = _config_to_dict(intern_a1)
        video_config = _config_to_dict(video_only)
        intern_config["samples_per_epoch"] = int(samples_per_epoch)
        intern_config["epoch_size_multiple"] = 1
        self.intern_a1 = InternDataA1V3Dataset(**intern_config)
        self.video_only = VideoOnlyLeRobotV3Dataset(**video_config)
        self.video_only.dataset_id_offset = len(self.intern_a1.datasets)

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
        self.full_batch_fraction = float(full_batch_fraction)
        if not 0.0 < self.full_batch_fraction <= 1.0:
            raise ValueError(
                f"full_batch_fraction must be in (0,1], got {full_batch_fraction}."
            )
        self.seed = int(seed)
        self.video_clips_per_episode = max(int(video_clips_per_episode), 1)
        self.video_locality_stride = max(int(video_locality_stride), 1)

        weights = np.asarray(self.video_only.source_weights, dtype=np.float64)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError(f"Invalid VIDEO_ONLY source weights: {weights.tolist()}")
        self.video_source_probabilities = weights / weights.sum()
        effective_video_fraction = 1.0 - self.full_batch_fraction
        effective_weights = {
            source["source_id"]: effective_video_fraction * float(probability)
            for source, probability in zip(
                self.video_only.sources,
                self.video_source_probabilities,
                strict=True,
            )
        }
        logger.info(
            "Multisource routes: InternData FULL=%.3f, VIDEO_ONLY=%.3f; "
            "effective VIDEO_ONLY source weights=%s",
            self.full_batch_fraction,
            effective_video_fraction,
            effective_weights,
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _encode_video(source_id: int, sample_key: int) -> int:
        if source_id > _SOURCE_MASK:
            raise ValueError(f"Too many VIDEO_ONLY sources to encode: {source_id}")
        return (
            _MIXED_INDEX_MARKER
            | _VIDEO_ONLY_MARKER
            | (int(source_id) << _SOURCE_SHIFT)
            | (int(sample_key) & _PAYLOAD_MASK)
        )

    @staticmethod
    def _decode_mixed_index(index: int) -> tuple[bool, int, int]:
        is_video_only = bool(
            int(index) & _MIXED_INDEX_MARKER
            and int(index) & _VIDEO_ONLY_MARKER
        )
        if not is_video_only:
            # FULL indices are passed through from InternData, including its
            # own encoded episode/start marker.
            return False, 0, int(index)
        source_id = (int(index) >> _SOURCE_SHIFT) & _SOURCE_MASK
        sample_key = int(index) & _PAYLOAD_MASK
        return True, source_id, sample_key

    def _video_batch_indices(
        self,
        rng: np.random.Generator,
        global_batch_size: int,
    ) -> Iterator[int]:
        yielded = 0
        while yielded < global_batch_size:
            source_id = int(
                rng.choice(
                    len(self.video_only.sources),
                    p=self.video_source_probabilities,
                )
            )
            cumulative = self.video_only.source_clip_cumulative[source_id]
            total_clips = self.video_only.source_clip_counts[source_id]
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
            group_size = min(
                self.video_clips_per_episode,
                global_batch_size - yielded,
            )
            for local_index in range(group_size):
                episode_start = (
                    base_start + local_index * self.video_locality_stride
                ) % episode_clip_count
                yield self._encode_video(
                    source_id,
                    previous_end + episode_start,
                )
                yielded += 1

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
            if not bool(is_video_only):
                for _ in range(global_batch_size):
                    yield next(intern_indices)
                continue
            yield from self._video_batch_indices(rng, global_batch_size)

    def __getitem__(self, index: int) -> dict:
        is_video_only, source_id, sample_key = self._decode_mixed_index(int(index))
        if is_video_only:
            return self.video_only.get_source_sample(
                source_id=source_id,
                sample_key=sample_key,
                sample_index=int(index),
            )
        sample = self.intern_a1[sample_key]
        sample["source_id"] = torch.tensor(0, dtype=torch.int16)
        sample["route_id"] = torch.tensor(0, dtype=torch.int8)
        return sample
