"""Precompute DINO feature statistics (per-channel mean & std) for normalisation.

Runs frozen DINO ViT-L over the training dataset (OpenVid-1M or LIBERO) and
accumulates running per-channel mean and variance using Welford's online
algorithm. Saves a .pt file with ``{"mean": [D], "std": [D]}`` that can be
loaded at training/inference time for fixed normalisation.

Why offline stats instead of batch standardise?
  - Batch standardise (`_channel_standardise`) uses the current batch's stats,
    which is noisy and non-reproducible across different batch sizes / GPU counts.
  - Fixed global stats are deterministic and allow exact un-standardisation at
    inference time (for reconstruction / visualisation).

Usage::

    # Single GPU (will take a while on OpenVid-1M)
    python scripts/precompute_dino_stats.py \
        --data_root /path/to/OpenVid_Data \
        --csv_name data/train/OpenVid-1M.csv \
        --output_path ./data/dino_vitl_stats.pt \
        --num_frames 17 --height 480 --width 832 \
        --max_samples 10000

    # Multi-GPU (8 GPUs, faster)
    torchrun --nproc_per_node=8 scripts/precompute_dino_stats.py \
        --data_root /path/to/OpenVid_Data \
        --csv_name data/train/OpenVid-1M.csv \
        --output_path ./data/dino_vitl_stats.pt \
        --num_frames 17 --height 480 --width 832 \
        --max_samples 10000

    # On LIBERO data (smaller, single GPU is fine)
    python scripts/precompute_dino_stats.py \
        --data_root /path/to/libero_data \
        --dataset_type libero \
        --output_path ./data/dino_vitl_stats_libero.pt \
        --max_samples 5000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ========================================================================== #
# Distributed helpers
# ========================================================================== #

def _init_distributed():
    """Initialize distributed process group if running under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        logger.info(
            "Distributed init: rank=%d world_size=%d device=cuda:%s",
            dist.get_rank(), dist.get_world_size(), os.environ["LOCAL_RANK"],
        )
    else:
        logger.info("Running in single-GPU mode.")


def _get_rank_world():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ========================================================================== #
# Welford's online algorithm for mean/variance
# ========================================================================== #

class WelfordAccumulator:
    """Online per-channel mean/variance accumulator (numerically stable).

    Tracks statistics over the (B, T, H, W) dimensions for each of D channels.
    """

    def __init__(self, num_channels: int, device: str = "cpu"):
        self.n = 0
        self.mean = torch.zeros(num_channels, dtype=torch.float64, device=device)
        self.M2 = torch.zeros(num_channels, dtype=torch.float64, device=device)

    def update(self, latents: torch.Tensor):
        """Update with a batch of latents [B, D, T, H, W]."""
        # Reshape to [N, D] where N = B*T*H*W
        B, D, T, H, W = latents.shape
        x = latents.to(dtype=torch.float64).permute(0, 2, 3, 4, 1).reshape(-1, D)
        # x: [N, D]

        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)  # [D]
        batch_var = x.var(dim=0, unbiased=False)  # [D]

        # Combine with running stats (parallel Welford)
        new_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_n / new_n)
        self.M2 = self.M2 + batch_var * batch_n + delta.pow(2) * (self.n * batch_n / new_n)
        self.n = new_n

    @property
    def variance(self) -> torch.Tensor:
        if self.n < 2:
            return torch.zeros_like(self.mean)
        return self.M2 / self.n

    @property
    def std(self) -> torch.Tensor:
        return self.variance.sqrt().clamp(min=1e-6)

    def merge(self, other: "WelfordAccumulator"):
        """Merge another accumulator into this one (for distributed reduction)."""
        if other.n == 0:
            return
        new_n = self.n + other.n
        delta = other.mean - self.mean
        self.mean = self.mean + delta * (other.n / new_n)
        self.M2 = self.M2 + other.M2 + delta.pow(2) * (self.n * other.n / new_n)
        self.n = new_n


# ========================================================================== #
# Dataset loading
# ========================================================================== #

def _build_openvid_dataset(args):
    """Build OpenVid dataset for feature extraction."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from fastwam.datasets.openvid import OpenVidDataset

    ds = OpenVidDataset(
        data_root=args.data_root,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        csv_name=args.csv_name,
        video_subdir=args.video_subdir,
        fps_subsample=args.fps_subsample,
        precomputed_text_embeds_dir=None,
        max_samples=args.max_samples,
    )
    return ds


def _build_simple_video_dataset(args):
    """Simple dataset that just returns video tensors from a directory."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    # For LIBERO or other datasets, try to use the configured dataset
    # Fall back to OpenVid format
    return _build_openvid_dataset(args)


# ========================================================================== #
# Main
# ========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Precompute DINO per-channel mean/std")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--csv_name", type=str, default="data/train/OpenVid-1M.csv",
                        help="CSV file relative to data_root (for OpenVid)")
    parser.add_argument("--video_subdir", type=str, default="video",
                        help="Video subdirectory under data_root")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output .pt file path for stats")
    parser.add_argument("--num_frames", type=int, default=17,
                        help="Number of frames to sample per video")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps_subsample", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max videos to process (None = all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for DINO encoding")
    parser.add_argument("--dino_model", type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="DINO model name")
    parser.add_argument("--temporal_downsample", type=int, default=4)
    parser.add_argument("--spatial_downsample", type=int, default=16)
    args = parser.parse_args()

    _init_distributed()
    rank, world_size = _get_rank_world()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    # Build DINO encoder (skip_projection mode — raw features)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from fastwam.models.wan22.visual_encoder import DINOEncoder

    encoder = DINOEncoder(
        model_name=args.dino_model,
        skip_projection=True,
        freeze_backbone=True,
        spatial_downsample=args.spatial_downsample,
        temporal_downsample=args.temporal_downsample,
        standardise_output=False,  # We want raw features to compute stats
        torch_dtype=torch.bfloat16,
    ).to(device)
    encoder.eval()

    num_channels = encoder.output_dim  # 1024 for ViT-L
    logger.info("DINO encoder output_dim = %d", num_channels)

    # Build dataset
    dataset = _build_openvid_dataset(args)
    total = len(dataset)
    logger.info("Dataset size: %d", total)

    # Shard across GPUs
    indices = list(range(total))
    indices = indices[rank::world_size]
    logger.info("Rank %d processing %d / %d samples", rank, len(indices), total)

    # Accumulate stats
    accumulator = WelfordAccumulator(num_channels, device="cpu")
    num_processed = 0
    num_failed = 0

    # Process in batches
    batch_videos = []
    pbar = tqdm(indices, desc=f"[rank {rank}] Computing DINO stats", disable=(rank != 0))

    for idx in pbar:
        try:
            sample = dataset[idx]
            video = sample["video"]  # [3, T, H, W] in [-1, 1]
            batch_videos.append(video)
        except Exception as e:
            num_failed += 1
            continue

        if len(batch_videos) >= args.batch_size:
            _process_batch(batch_videos, encoder, accumulator, device)
            num_processed += len(batch_videos)
            batch_videos = []
            if rank == 0:
                pbar.set_postfix(processed=num_processed, failed=num_failed)

    # Process remaining
    if batch_videos:
        _process_batch(batch_videos, encoder, accumulator, device)
        num_processed += len(batch_videos)

    logger.info("Rank %d: processed=%d, failed=%d, total_pixels=%d",
                rank, num_processed, num_failed, accumulator.n)

    # All-reduce across GPUs
    if world_size > 1:
        # Gather all accumulators to rank 0
        gathered_n = [torch.tensor(0, dtype=torch.long) for _ in range(world_size)]
        gathered_mean = [torch.zeros(num_channels, dtype=torch.float64) for _ in range(world_size)]
        gathered_M2 = [torch.zeros(num_channels, dtype=torch.float64) for _ in range(world_size)]

        dist.all_gather_object(gathered_n, torch.tensor(accumulator.n, dtype=torch.long))
        dist.all_gather_object(gathered_mean, accumulator.mean.cpu())
        dist.all_gather_object(gathered_M2, accumulator.M2.cpu())

        if rank == 0:
            final_acc = WelfordAccumulator(num_channels)
            for i in range(world_size):
                other = WelfordAccumulator(num_channels)
                other.n = int(gathered_n[i])
                other.mean = gathered_mean[i]
                other.M2 = gathered_M2[i]
                final_acc.merge(other)
            accumulator = final_acc
        dist.barrier()

    # Save on rank 0
    if rank == 0:
        mean = accumulator.mean.float()
        std = accumulator.std.float()

        logger.info("Global stats computed over %d pixels:", accumulator.n)
        logger.info("  mean range: [%.4f, %.4f]", mean.min().item(), mean.max().item())
        logger.info("  std  range: [%.4f, %.4f]", std.min().item(), std.max().item())

        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        torch.save({
            "mean": mean,       # [D]
            "std": std,         # [D]
            "num_samples": accumulator.n,
            "dino_model": args.dino_model,
            "num_channels": num_channels,
        }, args.output_path)
        logger.info("Saved stats to: %s", args.output_path)

    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def _process_batch(batch_videos, encoder, accumulator, device):
    """Encode a batch of videos and update the accumulator."""
    # Stack: [B, 3, T, H, W]
    videos = torch.stack(batch_videos, dim=0).to(device=device, dtype=torch.bfloat16)

    # Encode with DINO (skip_projection, no standardise)
    latents = encoder.encode(videos, device=device)  # [B, D, T_lat, H_lat, W_lat]

    # Update accumulator on CPU to avoid GPU memory pressure
    accumulator.update(latents.cpu())


if __name__ == "__main__":
    main()
