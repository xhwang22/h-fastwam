from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


def _dreamdojo_sdpa(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=self.scale,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dreamdojo_assets(
    dreamdojo_root: Path,
    checkpoint: Path,
    *,
    expected_source_revision: str,
    expected_checkpoint_sha256: str,
) -> None:
    actual_revision = subprocess.run(
        ["git", "-C", str(dreamdojo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != expected_source_revision:
        raise ValueError(
            "DreamDojo source revision mismatch: "
            f"got {actual_revision}, expected {expected_source_revision}."
        )
    dirty_files = subprocess.run(
        [
            "git",
            "-C",
            str(dreamdojo_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty_files:
        raise ValueError(
            "DreamDojo checkout has modified tracked files; online targets "
            f"require the pinned clean source tree: {dirty_files}"
        )

    stat = checkpoint.stat()
    cache_key = hashlib.sha256(
        (
            f"{checkpoint}:{stat.st_dev}:{stat.st_ino}:"
            f"{stat.st_size}:{stat.st_mtime_ns}:{expected_checkpoint_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    verification_dir = Path("/tmp/fastwam-dreamdojo-verification")
    verification_dir.mkdir(parents=True, exist_ok=True)
    marker = verification_dir / f"{cache_key}.json"
    lock_path = verification_dir / f"{cache_key}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip()
            == expected_checkpoint_sha256
        ):
            return
        actual_checkpoint_sha256 = _sha256(checkpoint)
        if actual_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(
                "DreamDojo checkpoint SHA256 mismatch: "
                f"got {actual_checkpoint_sha256}, "
                f"expected {expected_checkpoint_sha256}."
            )
        temporary = marker.with_name(
            f".{marker.name}.{os.getpid()}.tmp"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"{actual_checkpoint_sha256}\n")
        os.replace(temporary, marker)


def load_dreamdojo_lam(
    dreamdojo_root: str | Path,
    checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    *,
    expected_source_revision: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> torch.nn.Module:
    dreamdojo_root = Path(dreamdojo_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    lam_source = dreamdojo_root / "external" / "lam" / "modules" / "lam.py"
    if not lam_source.is_file():
        raise FileNotFoundError(
            f"DreamDojo LAM source is missing: {lam_source}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DreamDojo LAM checkpoint is missing: {checkpoint}"
        )
    if (
        expected_source_revision is None
        or expected_checkpoint_sha256 is None
    ):
        if expected_source_revision is not None:
            raise ValueError(
                "`expected_checkpoint_sha256` is required when validating "
                "DreamDojo assets."
            )
        if expected_checkpoint_sha256 is not None:
            raise ValueError(
                "`expected_source_revision` is required when validating "
                "DreamDojo assets."
            )
    else:
        _verify_dreamdojo_assets(
            dreamdojo_root,
            checkpoint,
            expected_source_revision=expected_source_revision,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

    if device.type == "cuda":
        torch.cuda.set_device(device)
    sys.path.insert(0, str(dreamdojo_root))
    from external.lam.modules.lam import LatentActionModel

    model = LatentActionModel(
        in_dim=3,
        model_dim=1024,
        latent_dim=32,
        patch_size=16,
        enc_blocks=24,
        dec_blocks=24,
        num_heads=16,
        dropout=0.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"),
        dict,
    ):
        raise ValueError(
            "DreamDojo checkpoint must contain a `state_dict` mapping."
        )
    wrapper_state = payload["state_dict"]
    non_lam_keys = [
        key for key in wrapper_state if not key.startswith("lam.")
    ]
    if non_lam_keys:
        raise ValueError(
            "DreamDojo checkpoint contains unexpected non-LAM state keys: "
            f"{non_lam_keys[:20]}."
        )
    lam_state = {
        key.removeprefix("lam."): value
        for key, value in wrapper_state.items()
    }
    missing, unexpected = model.load_state_dict(
        lam_state,
        strict=False,
        assign=True,
    )
    if missing or unexpected:
        raise ValueError(
            "DreamDojo LAM checkpoint does not exactly match the official "
            f"1024D/24+24/32D architecture: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}."
        )

    del model.patch_up
    del model.action_up
    del model.decoder
    patched_attention_layers = 0
    for module in model.modules():
        if (
            module.__class__.__name__ == "SelfAttention"
            and hasattr(module, "scaled_dot_product_attention")
            and hasattr(module, "scale")
        ):
            module.scaled_dot_product_attention = types.MethodType(
                _dreamdojo_sdpa,
                module,
            )
            patched_attention_layers += 1
    if patched_attention_layers == 0:
        raise RuntimeError(
            "DreamDojo LAM contains no compatible SelfAttention layers for "
            "the optimized SDPA path."
        )
    model.eval()
    model.requires_grad_(False)
    return model.to(device=device, dtype=dtype)


@torch.inference_mode()
def encode_dreamdojo_latent_actions(
    model: torch.nn.Module,
    video: torch.Tensor,
    *,
    pair_batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype = torch.bfloat16,
    preprocess_all_frames: bool = False,
) -> torch.Tensor:
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError(
            "DreamDojo online encoding requires video [B,3,33,H,W], "
            f"got {type(video)} with shape {getattr(video, 'shape', None)}."
        )
    if tuple(video.shape[1:3]) != (3, 33):
        raise ValueError(
            "DreamDojo online encoding requires exactly 33 RGB frames, "
            f"got {tuple(video.shape)}."
        )
    if pair_batch_size <= 0:
        raise ValueError(
            f"`pair_batch_size` must be positive, got {pair_batch_size}."
        )

    batch_size = int(video.shape[0])
    num_transitions = 32
    num_pairs = batch_size * num_transitions
    frames = None
    if preprocess_all_frames:
        frames = (
            video.permute(0, 2, 1, 3, 4)
            .flatten(0, 1)
            .to(device=device, dtype=torch.float32, non_blocking=True)
        )
        frames = F.interpolate(
            frames.clamp(-1.0, 1.0),
            size=(240, 320),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        frames = (
            ((frames + 1.0) * 0.5)
            .to(dtype=model_dtype)
            .reshape(batch_size, 33, 3, 240, 320)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    outputs = []
    for start in range(0, num_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, num_pairs)
        index_device = device if frames is not None else video.device
        pair_indices = torch.arange(start, stop, device=index_device)
        batch_indices = torch.div(
            pair_indices,
            num_transitions,
            rounding_mode="floor",
        )
        transition_indices = pair_indices.remainder(num_transitions)
        if frames is not None:
            pair_frames = torch.stack(
                (
                    frames[batch_indices, transition_indices],
                    frames[batch_indices, transition_indices + 1],
                ),
                dim=1,
            )
        else:
            raw_pair_frames = torch.stack(
                (
                    video[batch_indices, :, transition_indices],
                    video[batch_indices, :, transition_indices + 1],
                ),
                dim=1,
            )
            pair_count = int(raw_pair_frames.shape[0])
            resized = F.interpolate(
                raw_pair_frames.flatten(0, 1)
                .to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                .clamp(-1.0, 1.0),
                size=(240, 320),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            pair_frames = (
                ((resized + 1.0) * 0.5)
                .reshape(pair_count, 2, 3, 240, 320)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
                .to(dtype=model_dtype)
            )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=model_dtype)
            if device.type == "cuda"
            and model_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_context:
            encoded = model.encode(pair_frames)
        z_rep = encoded.get("z_rep")
        if not torch.is_tensor(z_rep) or z_rep.shape[1:] != (1, 1, 32):
            raise ValueError(
                "DreamDojo LAM must return z_rep [B,1,1,32], "
                f"got {type(z_rep)} with shape "
                f"{getattr(z_rep, 'shape', None)}."
            )
        outputs.append(z_rep[:, 0, 0].float())

    latent_actions = torch.cat(outputs, dim=0).reshape(batch_size, 32, 32)
    if not bool(torch.isfinite(latent_actions).all().item()):
        raise ValueError("DreamDojo LAM produced non-finite latent actions.")
    return latent_actions
