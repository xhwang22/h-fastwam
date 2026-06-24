"""
Navigation Video Dataset for FastWAM.

Reads VLN trajectory data in LeRobot format (parquet + jpg images) and produces
samples compatible with the FastWAM training pipeline.

Each sample contains:
  - video: [C, T_video, H, W] — dual-camera (horizontal concat) RGB video
  - action: [T_action, 3] — relative (x, y, theta) trajectory waypoints
  - action_is_pad: [T_action] — padding mask for action
  - context: [context_len, text_dim] — cached T5 text embedding
  - context_mask: [context_len] — text mask
  - image_is_pad: [T_video] — video frame padding mask
"""

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from fastwam.utils.logging_config import get_logger
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize

logger = get_logger(__name__)

DEFAULT_PROMPT = "A video recorded from a navigation agent's point of view executing the following instruction: {task}"


class NavVideoDataset(torch.utils.data.Dataset):
    """
    Dataset for VLN navigation trajectories.

    Args:
        dataset_dirs: List of scene root directories (each contains multiple scene folders).
        camera_keys: Camera angle names, e.g. ["125cm_0deg", "125cm_30deg"].
        num_frames: Total frames including initial (33 = 32 action steps + 1).
        action_video_freq_ratio: Ratio between action freq and video freq. Default 4.
        video_size: [H, W] after concatenation and resize.
        concat_multi_camera: How to concat cameras ("horizontal" or "vertical").
        text_embedding_cache_dir: Path to pre-computed text embeddings.
        context_len: Text context length (128).
        sample_stride: Stride for sampling start frames within episodes.
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        camera_keys: List[str] = None,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        video_size: List[int] = None,
        concat_multi_camera: str = "horizontal",
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        sample_stride: int = 1,
        # Unused kwargs for compatibility with hydra config
        **kwargs,
    ):
        super().__init__()
        if camera_keys is None:
            camera_keys = ["125cm_0deg", "125cm_30deg"]
        if video_size is None:
            video_size = [224, 448]

        self.camera_keys = camera_keys
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.video_size = video_size
        self.concat_multi_camera = concat_multi_camera
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.sample_stride = sample_stride

        # Video sampling indices
        assert (num_frames - 1) % action_video_freq_ratio == 0
        self.num_action_steps = num_frames - 1  # 32
        self.num_video_frames = (num_frames - 1) // action_video_freq_ratio + 1  # 9
        self.video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))

        # Image transforms
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )

        # Build index: list of (scene_path, episode_idx, start_frame, episode_length, instruction)
        self.samples = []
        self._build_index(dataset_dirs)
        logger.info(
            f"NavVideoDataset: {len(self.samples)} samples from {len(dataset_dirs)} dataset dirs, "
            f"cameras={camera_keys}, num_frames={num_frames}, action_video_freq_ratio={action_video_freq_ratio}"
        )

    def _build_index(self, dataset_dirs: List[str]):
        """Scan all scenes and episodes to build sample index."""
        for dataset_dir in dataset_dirs:
            if not os.path.isdir(dataset_dir):
                logger.warning(f"Dataset dir not found: {dataset_dir}")
                continue

            # Each sub-directory is a scene
            scene_names = sorted([
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ])

            for scene_name in scene_names:
                scene_path = os.path.join(dataset_dir, scene_name)
                episodes_file = os.path.join(scene_path, "meta", "episodes.jsonl")
                if not os.path.isfile(episodes_file):
                    continue

                # Read episodes metadata
                episodes = []
                with open(episodes_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episodes.append(json.loads(line))

                for ep_info in episodes:
                    ep_idx = ep_info["episode_index"]
                    ep_length = ep_info["length"]
                    # Get instruction (first task)
                    tasks = ep_info.get("tasks", [])
                    instruction = tasks[0] if tasks else ""

                    # Skip very short episodes
                    if ep_length < 10:
                        continue

                    # Create samples with stride
                    # Allow start_frame to go up to ep_length-1 so that
                    # samples near the end learn "stop" behavior (trajectory
                    # padded with final pose = zero relative motion)
                    max_start = ep_length - 1
                    for start in range(0, max_start + 1, self.sample_stride):
                        self.samples.append({
                            "scene_path": scene_path,
                            "episode_idx": ep_idx,
                            "start_frame": start,
                            "episode_length": ep_length,
                            "instruction": instruction,
                        })

    def __len__(self):
        return len(self.samples)

    def _load_image(self, scene_path: str, camera_key: str, episode_idx: int, frame_idx: int) -> torch.Tensor:
        """Load a single image as tensor [C, H, W] in [0, 1]."""
        img_dir = os.path.join(
            scene_path, "videos", "chunk-000",
            f"observation.images.rgb.{camera_key}"
        )
        img_path = os.path.join(img_dir, f"episode_{episode_idx:06d}_{frame_idx}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms_F.to_tensor(img)  # [C, H, W] in [0, 1]
        return img_tensor

    def _load_poses(self, scene_path: str, episode_idx: int, camera_key: str) -> np.ndarray:
        """Load all poses for an episode. Returns [N, 4, 4] array."""
        # Determine chunk (assume chunk-000 for most cases)
        parquet_path = os.path.join(
            scene_path, "data", "chunk-000",
            f"episode_{episode_idx:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
        poses_raw = df[f"pose.{camera_key}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])  # [N, 4, 4]
        return poses

    def _compute_relative_actions(
        self, poses: np.ndarray, start_idx: int, num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relative (x, y, theta) actions from pose sequence.

        When start_idx + i exceeds the trajectory length, the last pose is
        repeated. This means the relative action becomes the same as the final
        waypoint (i.e. "stay at goal" = zero additional motion), teaching the
        model to output a stationary trajectory when the goal is reached.

        Args:
            poses: [N, 4, 4] absolute pose matrices.
            start_idx: Current frame index.
            num_steps: Number of action steps to compute.

        Returns:
            actions: [num_steps, 3] relative (x, y, theta) waypoints.
            is_pad: [num_steps] boolean mask (True = beyond trajectory end).
        """
        T_base = poses[start_idx]
        T_base_inv = np.linalg.inv(T_base)
        last_pose_idx = len(poses) - 1

        actions = np.zeros((num_steps, 3), dtype=np.float32)
        is_pad = np.zeros(num_steps, dtype=bool)

        for i in range(num_steps):
            frame_j = start_idx + i + 1
            # Clamp to last frame: beyond trajectory end, use final pose
            # This makes relative action converge to a fixed point (= stop)
            actual_j = min(frame_j, last_pose_idx)
            if frame_j > last_pose_idx:
                is_pad[i] = True

            T_rel = T_base_inv @ poses[actual_j]
            local_pos = T_rel[:3, 3]
            R_rel = T_rel[:3, :3]
            theta = np.arctan2(R_rel[0, 2], R_rel[2, 2])

            actions[i] = [local_pos[0], local_pos[2], theta]

        return actions, is_pad

    def _get(self, idx: int) -> dict:
        """Get a single sample."""
        sample_info = self.samples[idx]
        scene_path = sample_info["scene_path"]
        episode_idx = sample_info["episode_idx"]
        start_frame = sample_info["start_frame"]
        instruction = sample_info["instruction"]

        # Load poses for first camera (for action computation)
        poses = self._load_poses(scene_path, episode_idx, self.camera_keys[0])

        # Compute relative actions for all num_frames-1 steps
        actions, action_is_pad = self._compute_relative_actions(
            poses, start_frame, self.num_action_steps
        )

        # Load video frames (at video sample rate)
        frame_indices = [start_frame + i for i in self.video_sample_indices]
        video_frames = []  # [T_video, C, H, W]
        image_is_pad = []

        for fidx in frame_indices:
            # Clamp frame index to last valid frame (repeat last frame for "stop" samples)
            actual_fidx = min(fidx, sample_info["episode_length"] - 1)
            is_pad_frame = (fidx >= sample_info["episode_length"])

            # Load and concat cameras
            cam_frames = []
            for cam_key in self.camera_keys:
                img = self._load_image(scene_path, cam_key, episode_idx, actual_fidx)
                # Resize individual camera to target single-camera size
                target_h = self.video_size[0]
                target_w = self.video_size[1] // len(self.camera_keys)
                img = transforms_F.resize(
                    img, [target_h, target_w],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                cam_frames.append(img)

            if self.concat_multi_camera == "horizontal":
                frame = torch.cat(cam_frames, dim=-1)  # [C, H, W*num_cams]
            elif self.concat_multi_camera == "vertical":
                frame = torch.cat(cam_frames, dim=-2)  # [C, H*num_cams, W]
            else:
                frame = cam_frames[0]

            video_frames.append(frame)
            image_is_pad.append(is_pad_frame)

        video = torch.stack(video_frames, dim=0)  # [T_video, C, H, W]

        # Normalize video to [-1, 1]
        video = video * 2.0 - 1.0  # from [0,1] to [-1,1]

        # Permute to [C, T_video, H, W]
        video = video.permute(1, 0, 2, 3)

        # Build text context
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)
        # Keep consistent with wan2.2 behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        # Convert to tensors
        action_tensor = torch.from_numpy(actions).float()  # [32, 3]
        action_is_pad_tensor = torch.from_numpy(action_is_pad).bool()  # [32]
        image_is_pad_tensor = torch.tensor(image_is_pad, dtype=torch.bool)  # [T_video]

        data = {
            "video": video,
            "action": action_tensor,
            "action_is_pad": action_is_pad_tensor,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad_tensor,
            "prompt": prompt,
        }
        return data

    def _get_cached_text_context(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load pre-computed text embedding from cache."""
        if self.text_embedding_cache_dir is None:
            # Return zeros if no cache dir specified (debug mode)
            context = torch.zeros(self.context_len, 4096)
            context_mask = torch.ones(self.context_len, dtype=torch.bool)
            return context, context_mask
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            # Fallback: return zeros with warning (allows training to proceed)
            logger.warning(
                f"Missing text embedding cache (using zeros): {cache_path}. "
                "Run scripts/precompute_nav_text_embeds.py to pre-compute all embeddings."
            )
            context = torch.zeros(self.context_len, 4096)
            context_mask = torch.ones(self.context_len, dtype=torch.bool)
            return context, context_mask
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]  # [context_len, text_dim]
        context_mask = payload["mask"].bool()  # [context_len]
        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            logger.warning(f"Error processing sample idx {idx}: {e}")
            logger.warning(traceback.format_exc())
            # Return a random valid sample
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
