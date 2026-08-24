#!/usr/bin/env python3
"""Offline DreamDojo z_mu extraction into FastWAM's versioned cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.utils.latent_action_cache import (  # noqa: E402
    CACHE_FORMAT,
    CACHE_VERSION,
    LATENT_ACTION_IS_PAD_SHAPE,
    LATENT_ACTION_SHAPE,
    MANIFEST_FILENAME,
    PARTIAL_MANIFEST_FILENAME,
    atomic_write_json,
    canonical_signature,
    load_latent_action_cache_manifest,
    sha256_file,
    write_latent_action_shard,
)

DREAMDOJO_COMMIT = "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"
DREAMDOJO_HF_REPO = "nvidia/DreamDojo"
DREAMDOJO_HF_REVISION = "89d029e10816d2995d700cb8ba06f171e0504203"
DREAMDOJO_LAM_FILENAME = "LAM_400k.ckpt"
DREAMDOJO_LAM_SHA256 = "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f"
DREAMDOJO_LAM_SIZE = 8_518_404_954
DREAMDOJO_SOURCE_LICENSE = "Apache-2.0"
DREAMDOJO_CHECKPOINT_LICENSE = "NVIDIA Open Model License"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dreamdojo-root", type=Path, required=True, help="Local official DreamDojo git checkout (no download is attempted).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local official LAM_400k.ckpt path.")
    parser.add_argument("--accept-nvidia-open-model-license", action="store_true", help="Confirm acceptance of the checkpoint's NVIDIA Open Model License.")
    parser.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs/data/robotwin_interleaved_webdataset.yaml")
    parser.add_argument("--preprocessed-root", type=Path, required=True, help="Local indexed RoboTwin WebDataset root.")
    parser.add_argument("--source-root", type=Path, required=True, help="Original RoboTwin LeRobot root used to verify timestamp/action semantics from parquet.")
    parser.add_argument("--stats", type=Path, default=None, help="Normalization stats; defaults to PREPROCESSED_ROOT/dataset_stats.json.")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-manifest", type=Path, default=None, help="Required for val: completed train cache manifest whose mean/std are reused.")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--cache-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_external_assets(args: argparse.Namespace) -> None:
    import os

    if not args.accept_nvidia_open_model_license or os.environ.get("ACCEPT_NVIDIA_OPEN_MODEL_LICENSE") != "1":
        raise SystemExit("Refusing to load weights: set ACCEPT_NVIDIA_OPEN_MODEL_LICENSE=1 and pass --accept-nvidia-open-model-license after reviewing the license.")
    root = args.dreamdojo_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (root / ".git").is_dir() or not (root / "external/lam/model.py").is_file():
        raise FileNotFoundError(f"Not an official DreamDojo checkout with the LAM API: {root}")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != DREAMDOJO_COMMIT:
        raise RuntimeError(f"DreamDojo revision mismatch: expected {DREAMDOJO_COMMIT}, got {revision}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DreamDojo checkpoint does not exist: {checkpoint}")
    if checkpoint.stat().st_size != DREAMDOJO_LAM_SIZE:
        raise RuntimeError(f"DreamDojo checkpoint size mismatch: expected {DREAMDOJO_LAM_SIZE}, got {checkpoint.stat().st_size}")
    digest = sha256_file(checkpoint)
    if digest != DREAMDOJO_LAM_SHA256:
        raise RuntimeError(f"DreamDojo checkpoint checksum mismatch: expected {DREAMDOJO_LAM_SHA256}, got {digest}")


def build_dataset(args: argparse.Namespace):
    data_cfg = OmegaConf.load(args.data_config)
    root_cfg = OmegaConf.create({"seed": args.seed, "data": data_cfg})
    dataset_cfg = root_cfg.data[args.split]
    dataset_cfg.preprocessed_root = str(args.preprocessed_root.expanduser().resolve())
    dataset_cfg.pretrained_norm_stats = str((args.stats or args.preprocessed_root / "dataset_stats.json").expanduser().resolve())
    dataset_cfg.latent_action_cache_dir = None
    dataset_cfg.latent_action_cache_expected_signature = None
    dataset_cfg.num_segments = 1
    OmegaConf.resolve(root_cfg)
    return instantiate(dataset_cfg), OmegaConf.to_container(dataset_cfg, resolve=True)


def _fixed_width_column(table, name: str, width: int) -> np.ndarray:
    column = table.column(name).combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    if values.size != table.num_rows * width:
        raise RuntimeError(f"Parquet column {name!r} is not width {width}.")
    return np.asarray(values, dtype=np.float32).reshape(table.num_rows, width)


def validate_source_alignment(preprocessed_root: Path, source_root: Path) -> dict[str, Any]:
    manifest_path = preprocessed_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "robotwin-webdataset" or manifest.get("version") != 1:
        raise RuntimeError("Unsupported RoboTwin WebDataset manifest; timestamp provenance cannot be established.")
    fps = int(manifest.get("fps", 0))
    if fps <= 0:
        raise RuntimeError("RoboTwin WebDataset manifest has no positive fps.")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("RoboTwin WebDataset manifest has no shards.")
    episode_ids: list[int] = []
    for shard in shards:
        validation = shard.get("validation", {})
        required = (
            "frame_index_is_sequential",
            "episode_index_is_constant_and_correct",
            "timestamp_round_fps_matches_frame_index",
        )
        if any(validation.get(key) is not True for key in required):
            raise RuntimeError(f"Shard {shard.get('shard_index')} lacks successful timestamp/frame alignment evidence.")
        ids = shard.get("episode_ids")
        if not isinstance(ids, list) or not ids:
            raise RuntimeError(f"Shard {shard.get('shard_index')} has no episode IDs for source validation.")
        episode_ids.extend(int(value) for value in ids)

    source_root = source_root.expanduser().resolve()
    info_path = source_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing original RoboTwin metadata: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if int(info.get("fps", 0)) != fps:
        raise RuntimeError(f"Source/preprocessed fps mismatch: {info.get('fps')} != {fps}")
    template = info.get("data_path")
    chunks_size = int(info.get("chunks_size", 0))
    if not isinstance(template, str) or chunks_size <= 0:
        raise RuntimeError("Original RoboTwin metadata has invalid data_path/chunks_size.")

    alignment_digest = hashlib.sha256()
    transition_count = 0
    for episode_id in episode_ids:
        parquet_path = source_root / template.format(
            episode_chunk=episode_id // chunks_size,
            episode_index=episode_id,
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing source parquet for episode {episode_id}: {parquet_path}")
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "timestamp", "frame_index", "episode_index"],
        )
        if table.num_rows < 2:
            raise RuntimeError(f"Episode {episode_id} has fewer than two rows.")
        states = _fixed_width_column(table, "observation.state", 14)
        actions = _fixed_width_column(table, "action", 14)
        timestamps = np.asarray(table.column("timestamp").combine_chunks().to_numpy(), dtype=np.float64)
        frame_indices = np.asarray(table.column("frame_index").combine_chunks().to_numpy(), dtype=np.int64)
        stored_episode_ids = np.asarray(table.column("episode_index").combine_chunks().to_numpy(), dtype=np.int64)
        expected_indices = np.arange(table.num_rows, dtype=np.int64)
        if not np.array_equal(frame_indices, expected_indices):
            raise RuntimeError(f"Episode {episode_id} source frame indices are not sequential.")
        if not np.array_equal(np.rint(timestamps * fps).astype(np.int64), expected_indices):
            raise RuntimeError(f"Episode {episode_id} source timestamps do not map to frame indices at {fps} fps.")
        if not np.all(stored_episode_ids == episode_id):
            raise RuntimeError(f"Episode {episode_id} source episode_index column disagrees.")
        if not np.allclose(actions[:-1], states[1:], rtol=0.0, atol=1e-6):
            mismatch = float(np.max(np.abs(actions[:-1] - states[1:])))
            raise RuntimeError(
                f"Episode {episode_id} does not establish action[t] == state[t+1]; max error {mismatch}."
            )
        alignment_digest.update(episode_id.to_bytes(8, "little", signed=False))
        alignment_digest.update(timestamps.tobytes())
        alignment_digest.update(actions[:-1].tobytes())
        alignment_digest.update(states[1:].tobytes())
        transition_count += table.num_rows - 1
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "source_info_sha256": sha256_file(info_path),
        "fps": fps,
        "validated_shards": len(shards),
        "validated_episodes": len(episode_ids),
        "validated_transitions": transition_count,
        "action_semantics": "action[t] == observation.state[t+1]",
        "alignment_sha256": alignment_digest.hexdigest(),
    }


def prepare_lam_video(video: torch.Tensor) -> torch.Tensor:
    if tuple(video.shape[:2]) != (3, 9) or tuple(video.shape[-2:]) != (384, 320):
        raise RuntimeError(f"Expected the repository's [3,9,384,320] RoboTwin canvas, got {tuple(video.shape)}")
    video = video.float().add(1.0).mul(0.5).clamp_(0.0, 1.0).permute(1, 0, 2, 3)
    video = F.interpolate(video, size=(240, 320), mode="bilinear", align_corners=False, antialias=True)
    return video.permute(0, 2, 3, 1).contiguous()  # official LAM API: [T,H,W,C], [0,1]


def expected_padding(sample: dict[str, Any]) -> torch.Tensor:
    image_is_pad = torch.as_tensor(sample["image_is_pad"], dtype=torch.bool)
    action_is_pad = torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
    if image_is_pad.shape != (9,) or action_is_pad.shape != (32,):
        raise RuntimeError(f"Alignment requires 9 frames and 32 actions, got {tuple(image_is_pad.shape)} and {tuple(action_is_pad.shape)}")
    return image_is_pad[:-1] | image_is_pad[1:] | action_is_pad.view(8, 4).any(dim=1)


def load_lam(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    sys.path.insert(0, str(args.dreamdojo_root.expanduser().resolve()))
    try:
        from external.lam.model import LAM
    except ImportError as exc:
        raise RuntimeError("DreamDojo dependencies are unavailable. Install its pinned environment outside FastWAM first.") from exc
    model = LAM(
        image_channels=3,
        lam_model_dim=1024,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=24,
        lam_dec_blocks=24,
        lam_num_heads=16,
        ckpt_path=None,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError("Official LAM checkpoint has no state_dict.")
    model.load_state_dict(state_dict, strict=True, assign=True)
    if model.lam.latent_dim != 32:
        raise RuntimeError(f"Loaded LAM latent_dim is {model.lam.latent_dim}, expected 32.")
    return model.eval().to(device=device, dtype=dtype)


def completed_shard_record(path: Path, shard_id: int, start: int, stop: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"sample_indices", "latent_action", "latent_action_is_pad"}:
                return None
            indices = handle.get_tensor("sample_indices")
            latent = handle.get_slice("latent_action")
            padding = handle.get_slice("latent_action_is_pad")
            expected = torch.arange(start, stop, dtype=torch.int64)
            if not torch.equal(indices, expected):
                return None
            if tuple(latent.get_shape()) != (stop - start, 8, 32):
                return None
            if tuple(padding.get_shape()) != (stop - start, 8):
                return None
    except Exception:
        return None
    return {
        "shard_id": shard_id,
        "filename": path.name,
        "index_start": start,
        "index_stop": stop,
        "sample_count": stop - start,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0 or args.batch_size <= 0:
        raise ValueError("--shard-size and --batch-size must be positive.")
    validate_external_assets(args)
    dataset, resolved_data_config = build_dataset(args)
    if getattr(dataset, "video_sample_indices", None) != list(range(0, 33, 4)):
        raise RuntimeError(f"Expected frame offsets [0,4,...,32], got {getattr(dataset, 'video_sample_indices', None)}")
    preprocessed_root = args.preprocessed_root.expanduser().resolve()
    timestamp_evidence = validate_source_alignment(preprocessed_root, args.source_root)
    stats_path = (args.stats or preprocessed_root / "dataset_stats.json").expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization stats do not exist: {stats_path}")
    data_config_path = args.data_config.expanduser().resolve()
    if not data_config_path.is_file():
        raise FileNotFoundError(f"Data config does not exist: {data_config_path}")
    dataset_length = len(dataset)
    num_shards = math.ceil(dataset_length / args.shard_size)
    output = args.output.expanduser().resolve()
    if args.split == "val" and args.normalization_manifest is None:
        raise RuntimeError("--normalization-manifest is required for val so train statistics are reused.")
    train_normalization = None
    normalization_manifest = None
    if args.split == "train" and args.normalization_manifest is not None:
        raise RuntimeError("--normalization-manifest is only valid for val; train must compute its own statistics.")
    if args.normalization_manifest is not None:
        normalization_path = args.normalization_manifest.expanduser().resolve()
        if normalization_path.name != MANIFEST_FILENAME or not normalization_path.is_file():
            raise FileNotFoundError(f"Normalization manifest does not exist: {normalization_path}")
        normalization_manifest = load_latent_action_cache_manifest(normalization_path.parent)
        if normalization_manifest.get("split") != "train":
            raise RuntimeError(f"Normalization manifest is not a train cache: {normalization_path}")
        train_normalization = normalization_manifest.get("normalization")
        if not isinstance(train_normalization, dict) or train_normalization.get("type") != "standardize":
            raise RuntimeError("Training manifest has no valid standardize normalization statistics.")
        train_normalization = dict(train_normalization)

    # This payload deliberately excludes split, split length, shard layout, and local
    # paths. Train and val are one cache family and therefore must share it exactly.
    family_payload_base = {
        "cache_format": CACHE_FORMAT,
        "cache_version": CACHE_VERSION,
        "dataset_source": {
            **timestamp_evidence,
            "data_config_sha256": sha256_file(data_config_path),
            "normalization_stats_sha256": sha256_file(stats_path),
            "split_seed": int(args.seed),
            "val_set_proportion": float(resolved_data_config["val_set_proportion"]),
            "split_policy": "numpy.default_rng.shuffle_then_floor_v1",
        },
        "dreamdojo": {
            "git_revision": DREAMDOJO_COMMIT,
            "checkpoint_repo": DREAMDOJO_HF_REPO,
            "checkpoint_revision": DREAMDOJO_HF_REVISION,
            "checkpoint_filename": DREAMDOJO_LAM_FILENAME,
            "checkpoint_sha256": DREAMDOJO_LAM_SHA256,
            "source_license": DREAMDOJO_SOURCE_LICENSE,
            "checkpoint_license": DREAMDOJO_CHECKPOINT_LICENSE,
            "checkpoint_license_accepted": True,
        },
        "preprocessing": {
            "view": "robotwin-canvas",
            "source_shape": [3, 9, 384, 320],
            "lam_pair_shape": [2, 240, 320, 3],
            "input_range": [0.0, 1.0],
            "interpolation": "bilinear_antialias",
            "frame_offsets": list(range(0, 33, 4)),
            "actions_per_latent": 4,
            "latent_target": "z_mu",
            "lam_call": "model.lam({'videos': pairs})",
            "physical_action_slice": "action[4k:4k+4]",
        },
        "compute_dtype": args.dtype,
        "cache_dtype": args.cache_dtype,
    }
    if normalization_manifest is not None:
        expected_train_payload = {**family_payload_base, "normalization": train_normalization}
        if normalization_manifest.get("signature_payload") != expected_train_payload:
            raise RuntimeError("Training normalization manifest uses a different cache-family extraction contract.")

    extraction_signature = canonical_signature(family_payload_base)
    manifest_path = output / MANIFEST_FILENAME
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_payload = existing.get("signature_payload")
        if not isinstance(existing_payload, dict):
            raise RuntimeError(f"Existing cache has no signature payload: {manifest_path}")
        existing_base = dict(existing_payload)
        existing_normalization = existing_base.pop("normalization", None)
        if existing_base != family_payload_base or existing.get("split") != args.split:
            raise RuntimeError(f"Existing cache family or split mismatch: {manifest_path}")
        if train_normalization is not None and existing_normalization != train_normalization:
            raise RuntimeError(f"Existing cache does not use the requested train normalization: {manifest_path}")
        manifest = load_latent_action_cache_manifest(
            output,
            expected_length=dataset_length,
            expected_signature=existing.get("signature"),
        )
        print(f"Complete matching cache already exists: {manifest_path} ({manifest['signature']})")
        return
    partial_path = output / PARTIAL_MANIFEST_FILENAME
    if partial_path.is_file():
        with partial_path.open("r", encoding="utf-8") as handle:
            partial = json.load(handle)
        if (
            partial.get("extraction_signature") != extraction_signature
            or partial.get("split") != args.split
            or partial.get("dataset_length") != dataset_length
        ):
            raise RuntimeError(
                f"Partial cache family, split, or dataset length mismatch at {partial_path}; "
                "use a new output directory."
            )
    else:
        if output.is_dir() and any(output.glob("shard_*.safetensors")):
            raise RuntimeError(f"Orphaned shards found in {output}; use a new output directory.")
        atomic_write_json(
            partial_path,
            {
                "format": CACHE_FORMAT,
                "version": CACHE_VERSION,
                "complete": False,
                "split": args.split,
                "dataset_length": dataset_length,
                "extraction_signature": extraction_signature,
                "family_payload_base": family_payload_base,
            },
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The pinned DreamDojo LAM implementation requires a CUDA device.")
    torch.cuda.set_device(device)
    compute_dtype = getattr(torch, args.dtype)
    cache_dtype = getattr(torch, args.cache_dtype)
    model = load_lam(args, device, compute_dtype)
    shard_records: list[dict[str, Any]] = []
    stat_sum = torch.zeros(32, dtype=torch.float64)
    stat_sq_sum = torch.zeros(32, dtype=torch.float64)
    stat_count = 0
    for shard_id in range(num_shards):
        start = shard_id * args.shard_size
        stop = min(start + args.shard_size, dataset_length)
        shard_path = output / f"shard_{shard_id:08d}.safetensors"
        existing_record = completed_shard_record(shard_path, shard_id, start, stop)
        if existing_record is not None:
            shard_records.append(existing_record)
            print(f"validated existing {existing_record['filename']} ({stop}/{dataset_length})", flush=True)
            continue
        if shard_path.exists():
            raise RuntimeError(f"Existing shard is incomplete or invalid: {shard_path}")
        latent_chunks, padding_chunks = [], []
        for batch_start in range(start, stop, args.batch_size):
            batch_stop = min(batch_start + args.batch_size, stop)
            samples = [dataset._get(index) for index in range(batch_start, batch_stop)]
            videos = torch.stack([prepare_lam_video(sample["video"]) for sample in samples])
            pairs = torch.stack([videos[:, :-1], videos[:, 1:]], dim=2).flatten(0, 1)
            with torch.inference_mode():
                outputs = model.lam({"videos": pairs.to(device=device, dtype=compute_dtype)})
            if not isinstance(outputs, dict) or "z_mu" not in outputs:
                raise RuntimeError("Official DreamDojo LAM call did not return a dict containing `z_mu`.")
            if tuple(outputs["z_mu"].shape) != (len(samples) * 8, 32):
                raise RuntimeError(
                    "Official DreamDojo LAM z_mu shape mismatch: "
                    f"expected {(len(samples) * 8, 32)}, got {tuple(outputs['z_mu'].shape)}."
                )
            z_mu = outputs["z_mu"].reshape(len(samples), 8, 32).float().cpu()
            if not torch.isfinite(z_mu).all():
                raise RuntimeError(f"Non-finite z_mu produced for samples [{batch_start}, {batch_stop}).")
            padding = torch.stack([expected_padding(sample) for sample in samples])
            valid = z_mu[~padding]
            stat_sum += valid.double().sum(dim=0)
            stat_sq_sum += valid.double().square().sum(dim=0)
            stat_count += int(valid.shape[0])
            latent_chunks.append(z_mu.to(cache_dtype))
            padding_chunks.append(padding)
        record = write_latent_action_shard(
            output,
            shard_id,
            range(start, stop),
            torch.cat(latent_chunks),
            torch.cat(padding_chunks),
        )
        shard_records.append(record)
        print(f"wrote {record['filename']} ({stop}/{dataset_length})", flush=True)
    if args.split == "val":
        if train_normalization is None:
            raise RuntimeError("--normalization-manifest is required for val so train statistics are reused.")
        normalization = train_normalization
    else:
        # Re-scan every shard so resumed and newly computed rows contribute identically.
        stat_sum.zero_()
        stat_sq_sum.zero_()
        stat_count = 0
        for record in shard_records:
            with safe_open(str(output / record["filename"]), framework="pt", device="cpu") as handle:
                latents = handle.get_tensor("latent_action").float()
                padding = handle.get_tensor("latent_action_is_pad").bool()
            valid = latents[~padding].double()
            stat_sum += valid.sum(dim=0)
            stat_sq_sum += valid.square().sum(dim=0)
            stat_count += int(valid.shape[0])
        if stat_count == 0:
            raise RuntimeError("No valid latent transitions were produced.")
        mean = stat_sum / stat_count
        variance = (stat_sq_sum / stat_count - mean.square()).clamp_min(0)
        normalization = {
            "type": "standardize",
            "valid_count": stat_count,
            "mean": mean.tolist(),
            "std": variance.sqrt().clamp_min(1e-6).tolist(),
        }
    signature_payload = {**family_payload_base, "normalization": normalization}
    signature = canonical_signature(signature_payload)
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "complete": True,
        "split": args.split,
        "signature": signature,
        "signature_payload": signature_payload,
        "dataset_length": dataset_length,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "latent_action_shape": list(LATENT_ACTION_SHAPE),
        "latent_action_is_pad_shape": list(LATENT_ACTION_IS_PAD_SHAPE),
        "normalization": normalization,
        "shards": shard_records,
    }
    atomic_write_json(manifest_path, manifest)
    partial_path.unlink(missing_ok=True)
    load_latent_action_cache_manifest(output, expected_length=dataset_length, expected_signature=signature)
    print(f"completed cache: {manifest_path}\nsignature: {signature}")


if __name__ == "__main__":
    main()
