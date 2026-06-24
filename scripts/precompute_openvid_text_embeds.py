"""Precompute text embeddings for OpenVid-1M dataset.

This script encodes all captions in the CSV to tensors and saves them to disk,
so training can load precomputed embeddings instead of running the text encoder.

Supports multi-GPU via torchrun for parallel processing.

Usage::

    # Single GPU
    python scripts/precompute_openvid_text_embeds.py \\
        --csv_path /path/to/OpenVid-1M/OpenVid-1M.csv \\
        --output_dir /path/to/OpenVid-1M/text_embeds \\
        --batch_size 64

    # Multi-GPU (e.g., 8 GPUs)
    torchrun --nproc_per_node=8 scripts/precompute_openvid_text_embeds.py \\
        --csv_path /path/to/OpenVid-1M/OpenVid-1M.csv \\
        --output_dir /path/to/OpenVid-1M/text_embeds \\
        --batch_size 64
"""

from __future__ import annotations
from tqdm import tqdm
import argparse
import csv
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _get_rank_and_world_size():
    """Get distributed rank and world size. Returns (0, 1) if not in distributed mode."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    # Support torchrun env vars even before dist.init
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def _init_distributed():
    """Initialize distributed process group if running under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return rank, world_size
    return 0, 1


def main():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for OpenVid-1M")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to OpenVid-1M.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer_model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--tokenizer_max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--redirect_common_files", action="store_true", default=True)
    parser.add_argument("--video_column", type=str, default="video", help="CSV column for video filename")
    parser.add_argument("--caption_column", type=str, default="caption", help="CSV column for caption text")
    args = parser.parse_args()

    # Initialize distributed
    rank, world_size = _init_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info("Running with %d GPU(s)", world_size)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text encoder and tokenizer
    from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components

    if rank == 0:
        logger.info("Loading text encoder and tokenizer...")
    components = load_wan22_ti2v_5b_components(
        device=str(device),
        torch_dtype=torch_dtype,
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        tokenizer_max_len=args.tokenizer_max_len,
        redirect_common_files=args.redirect_common_files,
        dit_config={
            "hidden_dim": 3072, "in_dim": 48, "ffn_dim": 14336, "out_dim": 48,
            "text_dim": 4096, "freq_dim": 256, "num_heads": 24, "attn_head_dim": 128,
            "num_layers": 30, "eps": 1e-6, "patch_size": [1, 2, 2],
            "has_image_input": False, "seperated_timestep": True,
            "require_vae_embedding": False, "require_clip_embedding": False,
            "fuse_vae_embedding_in_latents": True,
        },
        skip_dit_load_from_pretrain=True,  # Don't need DiT weights
        load_text_encoder=True,
        skip_vae_load=True,  # Don't need VAE
    )

    text_encoder = components.text_encoder
    tokenizer = components.tokenizer
    text_encoder.eval()

    # Read CSV
    if rank == 0:
        logger.info("Reading CSV: %s", args.csv_path)
    entries = []
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    if rank == 0:
        logger.info("Total entries: %d", len(entries))

    # Filter already processed
    remaining = []
    for entry in tqdm(entries):
        video_name = entry.get(args.video_column, "")
        stem = Path(video_name).stem
        out_path = os.path.join(args.output_dir, f"{stem}.pt")
        if not os.path.exists(out_path):
            remaining.append(entry)
    if rank == 0:
        logger.info("Remaining to process: %d (skipping %d already done)", len(remaining), len(entries) - len(remaining))

    # Shard remaining entries across GPUs
    shard = remaining[rank::world_size]
    if rank == 0:
        logger.info("Rank %d processing %d entries (total remaining: %d across %d GPUs)", rank, len(shard), len(remaining), world_size)

    # Process in batches
    pbar = tqdm(range(0, len(shard), args.batch_size), desc=f"Rank {rank}", disable=(rank != 0))
    for batch_start in pbar:
        batch_entries = shard[batch_start : batch_start + args.batch_size]
        captions = [e.get(args.caption_column, e.get("text", "")) for e in batch_entries]
        video_names = [e.get(args.video_column, "") for e in batch_entries]

        with torch.no_grad():
            ids, mask = tokenizer(captions, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device, dtype=torch.bool)
            prompt_emb = text_encoder(ids, mask)

            # Zero padding
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, v in enumerate(seq_lens):
                prompt_emb[i, v:] = 0
            mask = torch.ones_like(mask)

        # Save each embedding
        for i, video_name in enumerate(video_names):
            stem = Path(video_name).stem
            out_path = os.path.join(args.output_dir, f"{stem}.pt")
            torch.save(
                {
                    "context": prompt_emb[i].cpu(),           # [L, D]
                    "context_mask": mask[i].cpu(),             # [L]
                },
                out_path,
            )

    # Wait for all ranks to finish
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        logger.info("Done! Embeddings saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
