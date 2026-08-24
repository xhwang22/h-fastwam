import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from fastwam.utils.video_latent_cache import (
    VideoLatentCacheError,
    load_video_latent,
    load_video_latent_cache_manifest,
)
from fastwam.utils.latent_action_cache import (
    LatentActionCacheError,
    load_latent_action,
    load_latent_action_cache_manifest,
)
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DATASET_STATS_FILENAME = "dataset_stats.json"
DEFAULT_STATS_SYNC_TIMEOUT_SECONDS = 3600.0


def make_robotwin_canvas(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5 or video.shape[0] != 3:
        raise ValueError(
            "RoboTwin canvas requires video shaped [3,T,C,H,W], "
            f"got {tuple(video.shape)}."
        )
    cam_top = transforms_F.resize(
        video[0],
        size=[256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        video[1],
        size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        video[2],
        size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    return torch.cat([cam_top, bottom], dim=-2)


def _dataset_stats_path() -> str:
    return os.path.join(misc.get_work_dir(), DATASET_STATS_FILENAME)


def _resolve_stats_sync_timeout(timeout_s: Optional[float]) -> float:
    if timeout_s is not None:
        return float(timeout_s)
    return float(os.environ.get("FASTWAM_DATASET_STATS_SYNC_TIMEOUT", DEFAULT_STATS_SYNC_TIMEOUT_SECONDS))


def _wait_for_dataset_stats(stats_path: str, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(stats_path):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s:.1f}s waiting for dataset stats at {stats_path}. "
                "Check the main rank for normalization-stat calculation errors, or increase "
                "FASTWAM_DATASET_STATS_SYNC_TIMEOUT."
            )
        time.sleep(min(1.0, remaining))
    return load_dataset_stats_from_json(stats_path)


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        load_text_context: bool = True,
        context_len=128,
        pretrained_norm_stats=None,
        stats_sync_timeout: Optional[float] = None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        video_latent_cache_dir: Optional[str] = None,
        drop_video_when_cached: bool = False,
        latent_action_cache_dir: Optional[str] = None,
        latent_action_cache_expected_signature: Optional[str] = None,
    ):
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "`num_frames - 1` must be divisible by `action_video_freq_ratio`, "
                f"got {self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "Video transitions must be divisible by 4 for tokenization, "
                f"got {(self.num_frames - 1) // self.action_video_freq_ratio}"
            )
        self.video_sample_indices = list(
            range(0, self.num_frames, self.action_video_freq_ratio)
        )

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_sample_indices=self.video_sample_indices,
        )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.load_text_context = bool(load_text_context)
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.video_latent_cache_dir = None
        self.video_latent_cache_manifest = None
        self.drop_video_when_cached = bool(drop_video_when_cached)
        self.latent_action_cache_dir = None
        self.latent_action_cache_manifest = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            set_num_image_obs_steps = getattr(
                processor,
                "set_num_image_obs_steps",
                None,
            )
            if callable(set_num_image_obs_steps):
                set_num_image_obs_steps(len(self.video_sample_indices))
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                stats_path = _dataset_stats_path()
                distributed_state = PartialState()
                if distributed_state.is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    save_dataset_stats_to_json(dataset_stats, stats_path)
                else:
                    dataset_stats = _wait_for_dataset_stats(
                        stats_path,
                        _resolve_stats_sync_timeout(stats_sync_timeout),
                    )
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    save_dataset_stats_to_json(dataset_stats, _dataset_stats_path())

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
            self.processor = processor

        if video_latent_cache_dir is not None:
            self.set_video_latent_cache(
                cache_dir=video_latent_cache_dir,
                expected_length=len(self),
                drop_video=self.drop_video_when_cached,
            )
        if latent_action_cache_dir is not None:
            self.set_latent_action_cache(
                cache_dir=latent_action_cache_dir,
                expected_length=len(self),
                expected_signature=latent_action_cache_expected_signature,
            )
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _select_video_timeline(
        self,
        value: torch.Tensor,
        *,
        time_dim: int,
        name: str,
    ) -> torch.Tensor:
        actual_frames = int(value.shape[time_dim])
        expected_frames = len(self.video_sample_indices)
        if actual_frames == expected_frames:
            return value
        if actual_frames == self.num_frames:
            indices = torch.as_tensor(
                self.video_sample_indices,
                device=value.device,
                dtype=torch.long,
            )
            return value.index_select(time_dim, indices)
        raise ValueError(
            f"`{name}` has {actual_frames} frames; expected "
            f"{expected_frames} pre-sampled frames or {self.num_frames} full frames."
        )

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
        self.video_latent_cache_dir = os.path.realpath(os.path.expanduser(str(cache_dir)))
        self.video_latent_cache_manifest = manifest
        self.drop_video_when_cached = bool(drop_video)
        self.lerobot_dataset._set_return_images(not self.drop_video_when_cached)
        logger.info(
            "Configured video latent cache: dir=%s length=%d drop_video=%s",
            self.video_latent_cache_dir,
            expected_length,
            self.drop_video_when_cached,
        )
        return self

    def set_latent_action_cache(
        self,
        *,
        cache_dir: str | os.PathLike,
        expected_length: Optional[int] = None,
        expected_signature: Optional[str] = None,
    ):
        expected_length = len(self) if expected_length is None else int(expected_length)
        manifest = load_latent_action_cache_manifest(
            cache_dir,
            expected_length=expected_length,
            expected_signature=expected_signature,
        )
        self.latent_action_cache_dir = os.path.realpath(os.path.expanduser(str(cache_dir)))
        self.latent_action_cache_manifest = manifest
        logger.info(
            "Configured latent action cache: dir=%s length=%d signature=%s",
            self.latent_action_cache_dir,
            expected_length,
            manifest["signature"],
        )
        return self

    @staticmethod
    def _expected_latent_action_is_pad(
        image_is_pad: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> torch.Tensor:
        image_is_pad = torch.as_tensor(image_is_pad, dtype=torch.bool)
        action_is_pad = torch.as_tensor(action_is_pad, dtype=torch.bool)
        if image_is_pad.shape != (9,):
            raise LatentActionCacheError(
                f"Latent action alignment requires 9 image padding flags, got {tuple(image_is_pad.shape)}."
            )
        if action_is_pad.shape != (32,):
            raise LatentActionCacheError(
                f"Latent action alignment requires 32 physical action padding flags, got {tuple(action_is_pad.shape)}."
            )
        return image_is_pad[:-1] | image_is_pad[1:] | action_is_pad.view(8, 4).any(dim=1)

    def _attach_cached_latent_action(self, data: dict, sample_idx: int) -> None:
        if self.latent_action_cache_dir is None:
            return
        try:
            latent_action, cached_is_pad = load_latent_action(
                self.latent_action_cache_dir,
                self.latent_action_cache_manifest,
                sample_idx,
            )
        except Exception as exc:
            raise LatentActionCacheError(
                f"Failed to load cached latent action for sample {sample_idx} "
                f"from {self.latent_action_cache_dir}"
            ) from exc
        expected_is_pad = self._expected_latent_action_is_pad(
            data["image_is_pad"],
            data["action_is_pad"],
        )
        if not torch.equal(cached_is_pad, expected_is_pad):
            raise LatentActionCacheError(
                f"Cached latent action padding mismatch for sample {sample_idx}: "
                f"cached={cached_is_pad.tolist()} expected={expected_is_pad.tolist()}."
            )
        data["latent_action"] = latent_action
        data["latent_action_is_pad"] = cached_is_pad | expected_is_pad

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        sample_idx = int(sample.get("idx", sample_idx))
        return self._format_processed_sample(sample, sample_idx)

    def _format_processed_sample(self, sample: dict, sample_idx: int) -> dict:
        if self.video_latent_cache_dir is not None and self.drop_video_when_cached:
            task = sample["instruction"]
            if self.override_instruction is not None:
                task = self.override_instruction
            instruction = DEFAULT_PROMPT.format(task=task)
            data = {
                "video_latents": self._load_cached_video_latent(sample_idx),
                "action": sample["action"],
                "proprio": sample["proprio"][:-1, :],
                "prompt": instruction,
                "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
                "image_is_pad": self._select_video_timeline(
                    sample["image_is_pad"],
                    time_dim=0,
                    name="image_is_pad",
                ),
                "action_is_pad": sample["action_is_pad"],
                "proprio_is_pad": sample["proprio_is_pad"],
            }
            self._attach_cached_latent_action(data, sample_idx)
            if self.load_text_context:
                context, context_mask = self._get_cached_text_context(instruction)
                context[~context_mask] = 0.0
                context_mask = torch.ones_like(context_mask)
                data["context"] = context
                data["context_mask"] = context_mask
            return data
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = self._select_video_timeline(
                video,
                time_dim=1,
                name="pixel_values",
            )
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = self._select_video_timeline(
                video,
                time_dim=0,
                name="pixel_values",
            )
            T_video, C, H, W = video.shape
        image_is_pad = self._select_video_timeline(
            image_is_pad,
            time_dim=0,
            name="image_is_pad",
        )

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            video = make_robotwin_canvas(video)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.video_latent_cache_dir is not None:
            data["video_latents"] = self._load_cached_video_latent(sample_idx)
            if self.drop_video_when_cached:
                del data["video"]
        self._attach_cached_latent_action(data, sample_idx)
        if self.load_text_context:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
        return data

    def _load_cached_video_latent(self, sample_idx: int) -> torch.Tensor:
        try:
            return load_video_latent(
                self.video_latent_cache_dir,
                self.video_latent_cache_manifest,
                sample_idx,
            )
        except Exception as exc:
            raise VideoLatentCacheError(
                f"Failed to load cached video latent for sample {sample_idx} "
                f"from {self.video_latent_cache_dir}"
            ) from exc

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except (VideoLatentCacheError, LatentActionCacheError):
            raise
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data


class InterleavedRobotVideoDataset(RobotVideoDataset):
    def __init__(
        self,
        *args,
        num_segments: int = 2,
        segment_stride: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_segments = int(num_segments)
        if self.num_segments <= 0:
            raise ValueError(f"`num_segments` must be positive, got {num_segments}")
        self.segment_stride = int(segment_stride) if segment_stride is not None else int(self.num_frames - 1)
        if self.segment_stride <= 0:
            raise ValueError(f"`segment_stride` must be positive, got {self.segment_stride}")

    @staticmethod
    def _stack_segment_values(values):
        first = values[0]
        if torch.is_tensor(first):
            return torch.stack(values, dim=0)
        if isinstance(first, str):
            return list(values)
        if isinstance(first, dict):
            return {
                key: InterleavedRobotVideoDataset._stack_segment_values([value[key] for value in values])
                for key in first
            }
        if isinstance(first, (int, float, bool, np.integer, np.floating, np.bool_)):
            return torch.as_tensor(values)
        return list(values)

    def _get_segment_indices(self, idx: int) -> list[int]:
        starts = self.lerobot_dataset.episode_data_index["from"]
        ends = self.lerobot_dataset.episode_data_index["to"]
        idx_tensor = torch.as_tensor(idx, device=starts.device)
        matches = ((starts <= idx_tensor) & (idx_tensor < ends)).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return [min(idx + i * self.segment_stride, len(self) - 1) for i in range(self.num_segments)]

        episode_idx = int(matches[0].item())
        ep_end = int(ends[episode_idx].item())
        return [
            min(idx + i * self.segment_stride, ep_end - 1)
            for i in range(self.num_segments)
        ]

    def _get_segments(self, idx: int):
        segment_indices = self._get_segment_indices(idx)
        segment_samples = [self._get(segment_idx) for segment_idx in segment_indices]
        segments = {
            key: self._stack_segment_values([sample[key] for sample in segment_samples])
            for key in segment_samples[0]
        }
        segments["segment_mask"] = torch.ones(len(segment_samples), dtype=torch.bool)
        return {"segments": segments}

    def __getitem__(self, idx):
        try:
            return self._get_segments(idx)
        except (VideoLatentCacheError, LatentActionCacheError):
            raise
        except Exception as e:
            print(f"Error processing interleaved sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            return self._get_segments(random_idx)
