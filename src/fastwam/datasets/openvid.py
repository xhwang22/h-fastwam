"""OpenVid-1M dataset for text-conditioned video generation pretraining.

OpenVid-1M provides ~1M text-video pairs. Each sample yields:
    - ``video``: Tensor ``[3, T, H, W]`` in ``[-1, 1]``
    - ``prompt``: caption string  **or**
    - ``context`` / ``context_mask``: precomputed text embeddings

Directory layout expected::

    data_root/
        OpenVid-1M.csv          # CSV with columns: video, caption, ...
        video/                  # directory containing .mp4 files
            xxx.mp4
            yyy.mp4
            ...

Alternatively, if ``precomputed_text_embeds_dir`` is given, text embeddings are
loaded from ``{precomputed_text_embeds_dir}/{video_stem}.pt`` and the text
encoder is not required at runtime.

Usage::

    ds = OpenVidDataset(
        data_root="/path/to/OpenVid-1M",
        num_frames=17,
        height=480,
        width=832,
    )
    sample = ds[0]
    # sample["video"]: [3, 17, 480, 832]
    # sample["prompt"]: "A cat sitting on ..."
"""

from __future__ import annotations

import csv
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _read_csv(csv_path: str) -> list[dict]:
    """Read OpenVid-1M CSV and return list of dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_video_frames(
    video_path: str,
    num_frames: int,
    height: int,
    width: int,
    fps_subsample: int = 1,
) -> Optional[torch.Tensor]:
    """Load *num_frames* frames from a video file.

    Returns Tensor ``[3, T, H, W]`` in ``[-1, 1]`` or ``None`` on failure.
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
    except ImportError:
        raise ImportError(
            "decord is required for video loading. Install with: pip install decord"
        )

    try:
        vr = decord.VideoReader(video_path, num_threads=1)
    except Exception as e:
        logger.warning("Failed to open video %s: %s", video_path, e)
        return None

    total_frames = len(vr)
    if total_frames < num_frames:
        return None

    # Sample frames uniformly with subsample
    total_needed = num_frames * fps_subsample
    if total_needed > total_frames:
        # Fall back to uniform sampling without subsample
        indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
    else:
        start = random.randint(0, total_frames - total_needed)
        indices = list(range(start, start + total_needed, fps_subsample))

    try:
        frames = vr.get_batch(indices)  # [T, H, W, C] uint8
    except Exception as e:
        logger.warning("Failed to decode frames from %s: %s", video_path, e)
        return None

    # [T, H, W, C] -> [T, C, H, W]
    frames = frames.permute(0, 3, 1, 2).float()  # [T, 3, H, W]

    # Resize and center crop
    T, C, H_orig, W_orig = frames.shape
    scale = max(height / H_orig, width / W_orig)
    new_h = max(int(H_orig * scale + 0.5), height)
    new_w = max(int(W_orig * scale + 0.5), width)

    # Resize all frames
    frames_resized = torch.nn.functional.interpolate(
        frames, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Center crop
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    frames_cropped = frames_resized[:, :, top : top + height, left : left + width]

    # Normalize to [-1, 1]
    frames_cropped = frames_cropped / 127.5 - 1.0

    # [T, C, H, W] -> [C, T, H, W]
    video = frames_cropped.permute(1, 0, 2, 3).contiguous()
    return video


class OpenVidDataset(Dataset):
    """OpenVid-1M text-video dataset for pretraining.

    Args:
        data_root: Path containing ``OpenVid-1M.csv`` and ``video/`` directory.
        num_frames: Number of frames to sample per clip (must satisfy T%%4==1).
        height: Target frame height (must be multiple of 16).
        width: Target frame width (must be multiple of 16).
        csv_name: CSV filename inside ``data_root``.
        video_subdir: Subdirectory containing .mp4 files.
        fps_subsample: Temporal subsampling factor when picking frames.
        precomputed_text_embeds_dir: If set, load ``{stem}.pt`` files instead
            of raw captions.  Each ``.pt`` file should contain a dict with
            ``context`` ([L, D] tensor) and ``context_mask`` ([L] tensor).
        max_samples: Cap dataset size (useful for debugging).
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 17,
        height: int = 480,
        width: int = 832,
        csv_name: str = "OpenVid-1M.csv",
        video_subdir: str = "video",
        fps_subsample: int = 1,
        precomputed_text_embeds_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = str(data_root)
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.video_subdir = str(video_subdir)
        self.fps_subsample = int(fps_subsample)
        self.precomputed_text_embeds_dir = precomputed_text_embeds_dir

        if self.num_frames % 4 != 1:
            raise ValueError(f"`num_frames` must satisfy T%%4==1, got {self.num_frames}")
        if self.height % 16 != 0:
            raise ValueError(f"`height` must be multiple of 16, got {self.height}")
        if self.width % 16 != 0:
            raise ValueError(f"`width` must be multiple of 16, got {self.width}")

        csv_path = os.path.join(self.data_root, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.entries = _read_csv(csv_path)
        if max_samples is not None:
            self.entries = self.entries[: int(max_samples)]

        logger.info(
            "OpenVidDataset: %d entries, %d frames, %dx%d, fps_sub=%d",
            len(self.entries),
            self.num_frames,
            self.width,
            self.height,
            self.fps_subsample,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _get_video_path(self, entry: dict) -> str:
        # OpenVid-1M CSV has a "video" column with the filename/relative path
        video_rel = entry.get("video", entry.get("video_path", ""))
        # If it looks like a bare filename, prepend the video subdir
        if not os.path.isabs(video_rel):
            return os.path.join(self.data_root, self.video_subdir, video_rel)
        return video_rel

    def _get_caption(self, entry: dict) -> str:
        return entry.get("caption", entry.get("text", ""))

    def __getitem__(self, idx: int) -> dict:
        # Try to load the requested index; on failure, try random fallbacks
        for attempt in range(10):
            if attempt == 0:
                entry_idx = idx
            else:
                entry_idx = random.randint(0, len(self.entries) - 1)

            entry = self.entries[entry_idx]
            video_path = self._get_video_path(entry)

            video = _load_video_frames(
                video_path=video_path,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                fps_subsample=self.fps_subsample,
            )
            if video is not None:
                break
        else:
            raise RuntimeError(
                f"Failed to load video after 10 attempts. Last tried index={entry_idx}"
            )

        sample = {"video": video}

        # Text embeddings: precomputed or raw caption
        if self.precomputed_text_embeds_dir is not None:
            video_stem = Path(video_path).stem
            embed_path = os.path.join(self.precomputed_text_embeds_dir, f"{video_stem}.pt")
            if os.path.exists(embed_path):
                embed_data = torch.load(embed_path, map_location="cpu")
                sample["context"] = embed_data["context"]       # [L, D]
                sample["context_mask"] = embed_data["context_mask"]  # [L]
            else:
                # Fallback to raw caption
                sample["prompt"] = self._get_caption(entry)
        else:
            sample["prompt"] = self._get_caption(entry)

        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate that handles both prompt and context modes."""
        videos = torch.stack([b["video"] for b in batch])
        result = {"video": videos}

        has_context = "context" in batch[0]
        if has_context:
            # Pad context to same length
            contexts = [b["context"] for b in batch]
            masks = [b["context_mask"] for b in batch]
            max_len = max(c.shape[0] for c in contexts)
            dim = contexts[0].shape[1]

            padded_contexts = []
            padded_masks = []
            for c, m in zip(contexts, masks):
                pad_len = max_len - c.shape[0]
                if pad_len > 0:
                    c = torch.cat([c, torch.zeros(pad_len, dim, dtype=c.dtype)], dim=0)
                    m = torch.cat([m, torch.zeros(pad_len, dtype=m.dtype)], dim=0)
                padded_contexts.append(c)
                padded_masks.append(m)

            result["context"] = torch.stack(padded_contexts)
            result["context_mask"] = torch.stack(padded_masks)
        else:
            result["prompt"] = [b["prompt"] for b in batch]

        return result
