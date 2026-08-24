#!/usr/bin/env python3
"""Fail-fast validation of RoboTwin timestamps, padding, and latent cache integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.utils.latent_action_cache import (  # noqa: E402
    load_latent_action,
    load_latent_action_cache_manifest,
    sha256_file,
)

EXPECTED_FRAME_OFFSETS = list(range(0, 33, 4))
EXPECTED_DREAMDOJO_PROVENANCE = {
    "git_revision": "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff",
    "checkpoint_repo": "nvidia/DreamDojo",
    "checkpoint_revision": "89d029e10816d2995d700cb8ba06f171e0504203",
    "checkpoint_filename": "LAM_400k.ckpt",
    "checkpoint_sha256": "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f",
    "source_license": "Apache-2.0",
    "checkpoint_license": "NVIDIA Open Model License",
    "checkpoint_license_accepted": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Original RoboTwin LeRobot root containing meta/info.json and parquet.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs/data/robotwin_interleaved_webdataset.yaml")
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--expected-signature", default=None)
    parser.add_argument("--family-cache", type=Path, default=None, help="Other split cache whose signature and train-derived normalization must match.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=256, help="Padding rows to inspect; 0 checks every row. All shard checksums are always verified.")
    return parser.parse_args()


def validate_timestamp_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "robotwin-webdataset" or manifest.get("version") != 1:
        raise RuntimeError(f"Unsupported source manifest format/version in {path}")
    fps = manifest.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise RuntimeError(f"Source manifest has invalid fps: {fps!r}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("Source manifest contains no shard summaries.")
    for shard in shards:
        validation = shard.get("validation")
        if not isinstance(validation, dict):
            raise RuntimeError(f"Shard {shard.get('shard_index')} has no validation record.")
        for key in (
            "frame_index_is_sequential",
            "episode_index_is_constant_and_correct",
            "timestamp_round_fps_matches_frame_index",
        ):
            if validation.get(key) is not True:
                raise RuntimeError(f"Shard {shard.get('shard_index')} failed or omitted `{key}`.")
        camera_fps = validation.get("decoder_average_fps_by_camera")
        if not isinstance(camera_fps, dict) or any(abs(float(value) - fps) > 1e-6 for value in camera_fps.values()):
            raise RuntimeError(f"Shard {shard.get('shard_index')} camera FPS disagrees with dataset fps={fps}.")
    return manifest


def validate_source_action_semantics(source_root: Path, source_manifest: dict) -> dict:
    source_root = source_root.expanduser().resolve()
    info_path = source_root / "meta/info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    fps = int(source_manifest["fps"])
    if int(info.get("fps", 0)) != fps:
        raise RuntimeError("Original source and WebDataset fps differ.")
    template = info.get("data_path")
    chunks_size = int(info.get("chunks_size", 0))
    digest = hashlib.sha256()
    transitions = 0
    for shard in source_manifest["shards"]:
        for raw_episode_id in shard.get("episode_ids", []):
            episode_id = int(raw_episode_id)
            path = source_root / template.format(
                episode_chunk=episode_id // chunks_size,
                episode_index=episode_id,
            )
            table = pq.read_table(
                path,
                columns=["observation.state", "action", "timestamp", "frame_index"],
            )
            count = table.num_rows
            states = np.asarray(
                table.column("observation.state").combine_chunks().values.to_numpy(),
                dtype=np.float32,
            ).reshape(count, 14)
            actions = np.asarray(
                table.column("action").combine_chunks().values.to_numpy(),
                dtype=np.float32,
            ).reshape(count, 14)
            timestamps = np.asarray(table.column("timestamp").combine_chunks().to_numpy(), dtype=np.float64)
            indices = np.asarray(table.column("frame_index").combine_chunks().to_numpy(), dtype=np.int64)
            expected = np.arange(count, dtype=np.int64)
            if not np.array_equal(indices, expected) or not np.array_equal(np.rint(timestamps * fps).astype(np.int64), expected):
                raise RuntimeError(f"Episode {episode_id} timestamp/frame alignment failed.")
            if not np.allclose(actions[:-1], states[1:], rtol=0.0, atol=1e-6):
                raise RuntimeError(f"Episode {episode_id} does not satisfy action[t] == state[t+1].")
            digest.update(episode_id.to_bytes(8, "little", signed=False))
            digest.update(timestamps.tobytes())
            digest.update(actions[:-1].tobytes())
            digest.update(states[1:].tobytes())
            transitions += count - 1
    return {
        "source_info_sha256": sha256_file(info_path),
        "validated_transitions": transitions,
        "alignment_sha256": digest.hexdigest(),
        "action_semantics": "action[t] == observation.state[t+1]",
    }


def build_dataset(args: argparse.Namespace):
    data_config = OmegaConf.load(args.data_config)
    config = OmegaConf.create({"seed": args.seed, "data": data_config})
    split = config.data[args.split]
    split.preprocessed_root = str(args.preprocessed_root.expanduser().resolve())
    split.pretrained_norm_stats = str((args.stats or args.preprocessed_root / "dataset_stats.json").expanduser().resolve())
    split.latent_action_cache_dir = None
    split.latent_action_cache_expected_signature = None
    split.num_segments = 1
    OmegaConf.resolve(config)
    dataset = instantiate(split)
    if dataset.video_sample_indices != EXPECTED_FRAME_OFFSETS:
        raise RuntimeError(f"Frame offsets mismatch: expected {EXPECTED_FRAME_OFFSETS}, got {dataset.video_sample_indices}")
    return dataset


def sample_indices(dataset, maximum: int) -> list[int]:
    length = len(dataset)
    if maximum == 0 or maximum >= length:
        return list(range(length))
    chosen = {0, length - 1}
    for episode_start, episode_end in zip(dataset._selected_episode_starts, dataset._selected_episode_ends):
        chosen.update((episode_start, max(episode_start, episode_end - 33), episode_end - 1))
        if len(chosen) >= maximum:
            break
    if len(chosen) < maximum:
        chosen.update(round(index * (length - 1) / max(maximum - 1, 1)) for index in range(maximum))
    return sorted(chosen)[:maximum]


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    source_root = args.preprocessed_root.expanduser().resolve()
    source_manifest = validate_timestamp_manifest(source_root)
    source_alignment = validate_source_action_semantics(args.source_root, source_manifest)
    dataset = build_dataset(args)
    cache = args.cache.expanduser().resolve()
    manifest = load_latent_action_cache_manifest(
        cache,
        expected_length=len(dataset),
        expected_signature=args.expected_signature,
    )
    if manifest.get("split") != args.split:
        raise RuntimeError(f"Cache split mismatch: expected {args.split!r}, got {manifest.get('split')!r}")
    signature_payload = manifest.get("signature_payload")
    if not isinstance(signature_payload, dict):
        raise RuntimeError("Cache manifest is missing signature_payload.")
    preprocessing = signature_payload.get("preprocessing", {})
    expected_preprocessing = {
        "frame_offsets": EXPECTED_FRAME_OFFSETS,
        "actions_per_latent": 4,
        "latent_target": "z_mu",
    }
    for key, value in expected_preprocessing.items():
        if preprocessing.get(key) != value:
            raise RuntimeError(f"Cache preprocessing `{key}` mismatch: expected {value!r}, got {preprocessing.get(key)!r}")
    source_digest = signature_payload.get("dataset_source", {}).get("manifest_sha256")
    actual_digest = sha256_file(source_root / "manifest.json")
    if source_digest != actual_digest:
        raise RuntimeError(f"Cache/source manifest checksum mismatch: cache={source_digest}, source={actual_digest}")
    signed_source = signature_payload.get("dataset_source", {})
    for key, value in source_alignment.items():
        if signed_source.get(key) != value:
            raise RuntimeError(
                f"Signed source alignment `{key}` mismatch: cache={signed_source.get(key)!r}, source={value!r}"
            )
    stats_path = (args.stats or source_root / "dataset_stats.json").expanduser().resolve()
    signed_files = {
        "data_config_sha256": sha256_file(args.data_config.expanduser().resolve()),
        "normalization_stats_sha256": sha256_file(stats_path),
    }
    for key, value in signed_files.items():
        if signed_source.get(key) != value:
            raise RuntimeError(
                f"Signed input `{key}` mismatch: cache={signed_source.get(key)!r}, source={value!r}"
            )
    dreamdojo = signature_payload.get("dreamdojo")
    if not isinstance(dreamdojo, dict):
        raise RuntimeError("Cache signature payload is missing DreamDojo provenance.")
    for key, value in EXPECTED_DREAMDOJO_PROVENANCE.items():
        if dreamdojo.get(key) != value:
            raise RuntimeError(
                f"DreamDojo provenance `{key}` mismatch: cache={dreamdojo.get(key)!r}, expected={value!r}"
            )
    if signature_payload.get("normalization") != manifest.get("normalization"):
        raise RuntimeError("Top-level and signed cache normalization differ.")
    if args.family_cache is not None:
        family = load_latent_action_cache_manifest(args.family_cache.expanduser().resolve())
        if family["signature"] != manifest["signature"]:
            raise RuntimeError(
                f"Cache-family signature mismatch: {manifest['signature']} != {family['signature']}"
            )
        if family["normalization"] != manifest["normalization"]:
            raise RuntimeError("Cache-family normalization mismatch.")
        if family.get("split") == manifest.get("split"):
            raise RuntimeError(f"Cache-family manifests both declare split {manifest.get('split')!r}.")
    checked = sample_indices(dataset, args.max_samples)
    padded = 0
    for index in checked:
        sample = dataset._get(index)
        image_pad = torch.as_tensor(sample["image_is_pad"], dtype=torch.bool)
        action_pad = torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
        if image_pad.shape != (9,) or action_pad.shape != (32,):
            raise RuntimeError(f"Sample {index} shapes mismatch: image padding {tuple(image_pad.shape)}, action padding {tuple(action_pad.shape)}")
        expected = image_pad[:-1] | image_pad[1:] | action_pad.view(8, 4).any(dim=1)
        latent, cached = load_latent_action(cache, manifest, index)
        if latent.shape != (8, 32) or cached.shape != (8,):
            raise RuntimeError(f"Sample {index} cache shapes mismatch: {tuple(latent.shape)}, {tuple(cached.shape)}")
        if not torch.equal(cached, expected):
            raise RuntimeError(f"Sample {index} padding mismatch: cached={cached.tolist()}, expected={expected.tolist()}")
        padded += int(cached.sum())

    print(
        f"alignment OK: split={args.split} samples={len(dataset)} checked={len(checked)} "
        f"padded_transitions={padded} source_shards={len(source_manifest['shards'])} "
        f"cache_shards={manifest['num_shards']} signature={manifest['signature']}"
    )


if __name__ == "__main__":
    main()
