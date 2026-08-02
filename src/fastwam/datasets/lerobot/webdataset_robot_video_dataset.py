from __future__ import annotations

import bisect
import io
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.utils.logging_config import get_logger

from .robot_video_dataset import (
    InterleavedRobotVideoDataset,
    RobotVideoDataset,
)
from .utils.normalizer import load_dataset_stats_from_json
from ..dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from fastwam.utils.video_latent_cache import load_video_latent_cache_manifest


logger = get_logger(__name__)


class IndexedWebDatasetRobotVideoDataset(RobotVideoDataset):
    """Map-style RoboTwin dataset backed by indexed WebDataset tar shards."""

    _MAX_OPEN_SHARDS = 8

    def __init__(
        self,
        preprocessed_root: str,
        shape_meta,
        num_frames: int = 33,
        video_size=(384, 320),
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        load_text_context: bool = False,
        context_len: int = 128,
        pretrained_norm_stats: Optional[str] = None,
        val_set_proportion: float = 0.01,
        is_training_set: bool = True,
        seed: int = 42,
        action_video_freq_ratio: int = 4,
        concat_multi_camera: str = "robotwin",
        override_instruction: Optional[str] = None,
        num_segments: int = 1,
        segment_stride: Optional[int] = None,
    ):
        self.preprocessed_root = Path(preprocessed_root).expanduser().resolve()
        manifest_path = self.preprocessed_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"WebDataset manifest does not exist: {manifest_path}. "
                "Run scripts/preprocess_robotwin_webdataset.py first."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if (
            self.manifest.get("format") != "robotwin-webdataset"
            or int(self.manifest.get("version", -1)) != 1
        ):
            raise ValueError(
                "Unsupported WebDataset manifest format/version: "
                f"{self.manifest.get('format')!r}/"
                f"{self.manifest.get('version')!r}"
            )

        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if self.num_frames <= 1:
            raise ValueError(f"`num_frames` must be greater than 1, got {num_frames}")
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                "`action_video_freq_ratio` must be positive, "
                f"got {action_video_freq_ratio}"
            )
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "`num_frames - 1` must be divisible by `action_video_freq_ratio`, "
                f"got {self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        self.video_sample_indices = list(
            range(0, self.num_frames, self.action_video_freq_ratio)
        )

        manifest_camera_keys = list(
            self.manifest.get(
                "camera_feature_keys",
                self.manifest.get("camera_keys", []),
            )
        )
        if manifest_camera_keys == [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ]:
            manifest_camera_keys = [
                f"observation.images.{key}" for key in manifest_camera_keys
            ]
        expected_camera_keys = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        if manifest_camera_keys != expected_camera_keys:
            raise ValueError(
                "WebDataset camera order mismatch: "
                f"expected {expected_camera_keys}, got {manifest_camera_keys}"
            )
        self.camera_keys = manifest_camera_keys
        self.raw_height = int(self.manifest.get("raw_height", 0))
        self.raw_width = int(self.manifest.get("raw_width", 0))
        if self.raw_height <= 0 or self.raw_width <= 0:
            raise ValueError(
                "Manifest must contain positive `raw_height` and `raw_width`."
            )

        self._shards: dict[int, dict[str, Any]] = {}
        self._episodes: dict[int, dict[str, Any]] = {}
        shards_dir = self.preprocessed_root / "shards"
        manifest_shards = self.manifest.get("shards")
        if not isinstance(manifest_shards, list) or not manifest_shards:
            raise ValueError("Manifest must contain a non-empty `shards` list.")
        for shard_summary in sorted(
            manifest_shards,
            key=lambda row: int(row["shard_index"]),
        ):
            shard_id = int(shard_summary["shard_index"])
            stem = str(
                shard_summary.get("shard_name", f"shard-{shard_id:05d}")
            )
            paths = {
                "tar": shards_dir / f"{stem}.tar",
                "offsets": shards_dir / f"{stem}.offsets.npy",
                "sizes": shards_dir / f"{stem}.sizes.npy",
                "state": shards_dir / f"{stem}.state.npy",
                "action": shards_dir / f"{stem}.action.npy",
                "task_index": shards_dir / f"{stem}.task_index.npy",
                "episodes": shards_dir / f"{stem}.episodes.json",
                "done": shards_dir / f"{stem}.done",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Completed shard {stem} is missing files: {missing}"
                )
            with paths["episodes"].open("r", encoding="utf-8") as handle:
                episode_payload = json.load(handle)
            episode_rows = (
                episode_payload["episodes"]
                if isinstance(episode_payload, dict)
                else episode_payload
            )
            if not episode_rows:
                raise ValueError(f"Shard {stem} has no episode metadata.")
            frame_count = sum(int(row["length"]) for row in episode_rows)
            self._shards[shard_id] = {
                "id": shard_id,
                "stem": stem,
                "frame_count": frame_count,
                **paths,
            }
            for row in episode_rows:
                episode_id = int(row["episode_id"])
                if episode_id in self._episodes:
                    raise ValueError(
                        f"Episode {episode_id} appears in multiple shards."
                    )
                normalized = {
                    "episode_id": episode_id,
                    "original_global_start": int(row["original_global_start"]),
                    "shard_local_start": int(row["shard_local_start"]),
                    "length": int(row["length"]),
                    "shard_id": shard_id,
                }
                if normalized["length"] <= 0:
                    raise ValueError(
                        f"Episode {episode_id} has invalid length "
                        f"{normalized['length']}."
                    )
                self._episodes[episode_id] = normalized

        available_episode_ids = sorted(self._episodes)
        converted_episode_count = int(
            self.manifest.get("converted_episodes", len(available_episode_ids))
        )
        if converted_episode_count != len(available_episode_ids):
            raise ValueError(
                "Manifest/shard episode count mismatch: "
                f"manifest={converted_episode_count}, shards={len(available_episode_ids)}"
            )
        expected_ids = list(range(converted_episode_count))
        if available_episode_ids != expected_ids:
            raise ValueError(
                "Converted episode IDs must be contiguous from zero. "
                f"First mismatch near {available_episode_ids[:10]}."
            )

        val_set_proportion = float(val_set_proportion)
        if not 0.0 <= val_set_proportion < 1.0:
            raise ValueError(
                "`val_set_proportion` must be in [0, 1), "
                f"got {val_set_proportion}"
            )
        episode_ids = list(available_episode_ids)
        if val_set_proportion >= 1e-6:
            rng = np.random.default_rng(int(seed))
            rng.shuffle(episode_ids)
            split_idx = int(len(episode_ids) * (1.0 - val_set_proportion))
            episode_ids = (
                episode_ids[:split_idx]
                if is_training_set
                else episode_ids[split_idx:]
            )
        self.selected_episode_ids = episode_ids
        self._selected_episode_starts = []
        self._selected_episode_ends = []
        running = 0
        for episode_id in self.selected_episode_ids:
            self._selected_episode_starts.append(running)
            running += self._episodes[episode_id]["length"]
            self._selected_episode_ends.append(running)
        self._length = running
        if self._length <= 0:
            raise ValueError("Selected WebDataset split is empty.")

        tasks_payload = self.manifest.get("tasks")
        if isinstance(tasks_payload, dict):
            tasks_payload = tasks_payload.get("by_index")
        if not isinstance(tasks_payload, list) or not tasks_payload:
            raise ValueError(
                "Manifest must contain a non-empty `tasks` list or "
                "`tasks.by_index` list."
            )
        self.tasks = {}
        for row in tasks_payload:
            if isinstance(row, dict):
                task_index = int(row["task_index"])
                task = str(row["task"])
            else:
                task_index = len(self.tasks)
                task = str(row)
            self.tasks[task_index] = task

        self.camera_key = camera_key
        self.video_size = tuple(video_size)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.load_text_context = bool(load_text_context)
        self.context_len = int(context_len)
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.video_latent_cache_dir = None
        self.video_latent_cache_manifest = None
        self.drop_video_when_cached = False
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )

        if processor is None:
            raise ValueError("`processor` is required for WebDataset training.")
        if isinstance(processor, DictConfig):
            processor = instantiate(processor)
        set_num_image_obs_steps = getattr(
            processor,
            "set_num_image_obs_steps",
            None,
        )
        if callable(set_num_image_obs_steps):
            set_num_image_obs_steps(len(self.video_sample_indices))
        stats_path = (
            Path(pretrained_norm_stats).expanduser().resolve()
            if pretrained_norm_stats
            else self.preprocessed_root / "dataset_stats.json"
        )
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Normalization statistics do not exist: {stats_path}"
            )
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(stats_path))
        )
        if is_training_set:
            processor.train()
        else:
            processor.eval()
        self.processor = processor

        self.num_segments = int(num_segments)
        if self.num_segments <= 0:
            raise ValueError(f"`num_segments` must be positive, got {num_segments}")
        self.segment_stride = (
            int(segment_stride)
            if segment_stride is not None
            else self.num_frames - 1
        )
        if self.segment_stride <= 0:
            raise ValueError(
                f"`segment_stride` must be positive, got {self.segment_stride}"
            )

        self._runtime_shards: OrderedDict[int, dict[str, Any]] = OrderedDict()
        logger.info(
            "Indexed WebDataset split: root=%s episodes=%d frames=%d "
            "shards=%d image_frames=%d",
            self.preprocessed_root,
            len(self.selected_episode_ids),
            self._length,
            len(self._shards),
            len(self.video_sample_indices),
        )

    def __len__(self) -> int:
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime_shards"] = OrderedDict()
        return state

    def __del__(self):
        for runtime in getattr(self, "_runtime_shards", {}).values():
            fd = runtime.get("fd")
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _runtime_shard(self, shard_id: int) -> dict[str, Any]:
        runtime = self._runtime_shards.get(shard_id)
        if runtime is not None:
            self._runtime_shards.move_to_end(shard_id)
            return runtime
        shard = self._shards[shard_id]
        runtime = {
            "fd": os.open(shard["tar"], os.O_RDONLY),
            "offsets": np.load(shard["offsets"], mmap_mode="r"),
            "sizes": np.load(shard["sizes"], mmap_mode="r"),
            "state": np.load(shard["state"], mmap_mode="r"),
            "action": np.load(shard["action"], mmap_mode="r"),
            "task_index": np.load(shard["task_index"], mmap_mode="r"),
        }
        expected = int(shard["frame_count"])
        for key in ("offsets", "sizes", "state", "action", "task_index"):
            if len(runtime[key]) != expected:
                os.close(runtime["fd"])
                raise ValueError(
                    f"Shard {shard['stem']} sidecar `{key}` length "
                    f"{len(runtime[key])} != {expected}"
                )
        self._runtime_shards[shard_id] = runtime
        while len(self._runtime_shards) > self._MAX_OPEN_SHARDS:
            _, evicted = self._runtime_shards.popitem(last=False)
            os.close(evicted["fd"])
        return runtime

    def _resolve_index(
        self,
        idx: int,
    ) -> tuple[dict[str, Any], int, int, int]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds for length {self._length}.")
        episode_pos = bisect.bisect_right(self._selected_episode_ends, idx)
        episode_id = self.selected_episode_ids[episode_pos]
        virtual_start = self._selected_episode_starts[episode_pos]
        local_index = idx - virtual_start
        return self._episodes[episode_id], local_index, episode_pos, idx

    def _read_combined_frame(
        self,
        episode: dict[str, Any],
        local_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        runtime = self._runtime_shard(episode["shard_id"])
        shard_index = episode["shard_local_start"] + int(local_index)
        offset = int(runtime["offsets"][shard_index])
        size = int(runtime["sizes"][shard_index])
        payload = os.pread(runtime["fd"], size, offset)
        if len(payload) != size:
            raise IOError(
                f"Short tar read for episode={episode['episode_id']} "
                f"frame={local_index}: expected {size}, got {len(payload)}"
            )
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        expected_shape = (self.raw_height, 3 * self.raw_width, 3)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"Combined PNG shape mismatch for episode={episode['episode_id']} "
                f"frame={local_index}: expected {expected_shape}, got {array.shape}"
            )
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        return tuple(
            tensor[:, :, index * self.raw_width : (index + 1) * self.raw_width]
            for index in range(3)
        )

    @staticmethod
    def _window_indices(
        start: int,
        offsets: list[int],
        episode_length: int,
    ) -> tuple[list[int], torch.Tensor]:
        raw_indices = [start + offset for offset in offsets]
        indices = [min(max(index, 0), episode_length - 1) for index in raw_indices]
        padding = torch.tensor(
            [index < 0 or index >= episode_length for index in raw_indices],
            dtype=torch.bool,
        )
        return indices, padding

    def _get_single(self, idx: int) -> dict:
        episode, local_index, _, normalized_idx = self._resolve_index(idx)
        episode_length = int(episode["length"])
        image_indices, image_is_pad = self._window_indices(
            local_index,
            self.video_sample_indices,
            episode_length,
        )
        state_indices, state_is_pad = self._window_indices(
            local_index,
            list(range(self.num_frames)),
            episode_length,
        )
        action_indices, action_is_pad = self._window_indices(
            local_index,
            list(range(self.num_frames - 1)),
            episode_length,
        )

        images = None
        if not (
            self.video_latent_cache_dir is not None
            and self.drop_video_when_cached
        ):
            camera_frames = [[], [], []]
            for frame_index in image_indices:
                frames = self._read_combined_frame(episode, frame_index)
                for camera_idx, frame in enumerate(frames):
                    camera_frames[camera_idx].append(frame)
            images = {
                "cam_high": torch.stack(camera_frames[0], dim=0),
                "cam_left_wrist": torch.stack(camera_frames[1], dim=0),
                "cam_right_wrist": torch.stack(camera_frames[2], dim=0),
            }

        runtime = self._runtime_shard(episode["shard_id"])
        shard_start = int(episode["shard_local_start"])
        state = torch.from_numpy(
            np.asarray(
                runtime["state"][
                    [shard_start + index for index in state_indices]
                ],
                dtype=np.float32,
            ).copy()
        )
        action = torch.from_numpy(
            np.asarray(
                runtime["action"][
                    [shard_start + index for index in action_indices]
                ],
                dtype=np.float32,
            ).copy()
        )
        current_shard_index = shard_start + local_index
        task_index = int(runtime["task_index"][current_shard_index])
        try:
            task = self.tasks[task_index]
        except KeyError as exc:
            raise KeyError(
                f"Task index {task_index} is missing from the manifest."
            ) from exc

        raw_sample = {
            "idx": normalized_idx,
            "task": task,
            "state": {"default": state},
            "action": {"default": action},
            "image_is_pad": image_is_pad,
            "state_is_pad": state_is_pad,
            "action_is_pad": action_is_pad,
        }
        if images is not None:
            raw_sample["images"] = images
        processed = self.processor.preprocess(raw_sample)
        return self._format_processed_sample(processed, normalized_idx)

    def _get(self, idx: int) -> dict:
        return self._get_single(idx)

    def set_video_latent_cache(
        self,
        *,
        cache_dir: str | os.PathLike,
        expected_length: Optional[int] = None,
        drop_video: bool = False,
    ):
        expected_length = len(self) if expected_length is None else int(expected_length)
        manifest = load_video_latent_cache_manifest(
            cache_dir,
            expected_length=expected_length,
        )
        self.video_latent_cache_dir = os.path.realpath(
            os.path.expanduser(str(cache_dir))
        )
        self.video_latent_cache_manifest = manifest
        self.drop_video_when_cached = bool(drop_video)
        logger.info(
            "Configured indexed WebDataset video latent cache: "
            "dir=%s length=%d drop_video=%s",
            self.video_latent_cache_dir,
            expected_length,
            self.drop_video_when_cached,
        )
        return self

    def _get_segment_indices(self, idx: int) -> list[int]:
        episode, _, episode_pos, normalized_idx = self._resolve_index(idx)
        episode_end = (
            self._selected_episode_starts[episode_pos] + int(episode["length"])
        )
        return [
            min(normalized_idx + segment_idx * self.segment_stride, episode_end - 1)
            for segment_idx in range(self.num_segments)
        ]

    def __getitem__(self, idx: int) -> dict:
        segment_indices = self._get_segment_indices(idx)
        segment_samples = [self._get_single(index) for index in segment_indices]
        segments = {
            key: InterleavedRobotVideoDataset._stack_segment_values(
                [sample[key] for sample in segment_samples]
            )
            for key in segment_samples[0]
        }
        segments["segment_mask"] = torch.ones(
            len(segment_samples),
            dtype=torch.bool,
        )
        return {"segments": segments}
