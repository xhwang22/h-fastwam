"""Visual encoders — drop-in replacements for WanVideoVAE38 encoder.

Supported backends:

* **DINOv3 / DINOv2** — frozen image ViT, per-frame encoding + temporal stride.
* **Qwen3-VL vision** — frozen SigLIP2-initialized video ViT.
* **Xiaomi Robotics-1 vision** — XR-1-tuned Qwen3-VL-4B vision tower.
* **V-JEPA 2** — frozen video ViT, native spatiotemporal encoding.
* **V-JEPA 2.1** — frozen dense-feature video ViT.

Both produce latents with the same shape convention as the VAE encoder:
``[B, output_dim, T_lat, H_lat, W_lat]`` so the downstream DiT / MoT
pipeline requires zero changes.

Usage::

    # DINOv3
    encoder = DINOEncoder(
        model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        output_dim=48,
    )

    # V-JEPA 2
    encoder = VJEPA2Encoder(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        output_dim=48,
    )

    # videos: [B, 3, T, H, W]  (pixel range [-1, 1])
    latents = encoder.encode(videos, device="cuda")
    # latents: [B, 48, T_lat, H_lat, W_lat]
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ========================================================================== #
# SIGReg loss — anti-collapse regularisation for visual encoder latents
# ========================================================================== #

def sigreg_loss(
    latents: torch.Tensor,
    lam_var: float = 1.0,
    lam_cov: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SIGReg regularisation loss on encoder latents.

    Prevents representation collapse by encouraging each latent channel to have
    high variance (non-degenerate) and low correlation with other channels
    (decorrelation).  Uses sigmoid-based losses for bounded, stable gradients.

    Reference: Eymael & Garrido, "SIGReg: Sigmoid Regularization for
    Self-Supervised Learning", 2025.

    Args:
        latents: ``[B, D, T, H, W]`` — raw encoder output (before standardise).
        lam_var: Weight for variance term.
        lam_cov: Weight for covariance (decorrelation) term.
        eps: Small constant for numerical stability.

    Returns:
        (loss, metrics_dict) where metrics_dict has ``sigreg_var`` and ``sigreg_cov``.
    """
    B, D, T, H, W = latents.shape
    # Reshape to [N, D] where N = B * T * H * W
    x = latents.permute(0, 2, 3, 4, 1).reshape(-1, D)  # [N, D]
    N = x.shape[0]

    # Center
    x = x - x.mean(dim=0, keepdim=True)

    # Covariance matrix [D, D]
    cov = (x.T @ x) / (N - 1 + eps)

    # Variance term: -log(sigmoid(diag)) → pushes variance away from 0
    diag = cov.diagonal()
    loss_var = -F.logsigmoid(diag).mean()

    # Covariance term: -log(sigmoid(-off_diag^2)) → pushes off-diagonal toward 0
    off_diag = cov - torch.diag(diag)
    loss_cov = -F.logsigmoid(-off_diag.pow(2)).mean()

    loss = lam_var * loss_var + lam_cov * loss_cov

    metrics = {
        "sigreg_var": float(loss_var.detach().item()),
        "sigreg_cov": float(loss_cov.detach().item()),
    }
    return loss, metrics

# ========================================================================== #
# Factory
# ========================================================================== #

# Registry: encoder_type → class
_ENCODER_REGISTRY: dict[str, type["BaseVisualEncoder"]] = {}


def build_visual_encoder(
    encoder_type: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> "BaseVisualEncoder":
    """Build a visual encoder by type string.

    Args:
        encoder_type: Registered encoder backend, e.g. ``"dino"``,
            ``"qwen3_vl_vision"``, ``"vjepa2"``, or ``"vjepa2_1"``.
        torch_dtype: model dtype.
        **kwargs: forwarded to the encoder constructor.
    """
    cls = _ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(
            f"Unknown visual encoder type '{encoder_type}'. "
            f"Available: {sorted(_ENCODER_REGISTRY.keys())}"
        )
    return cls(torch_dtype=torch_dtype, **kwargs)


# ========================================================================== #
# Base class
# ========================================================================== #

class BaseVisualEncoder(ABC, nn.Module):
    """Abstract base for VAE-replacement visual encoders.

    Subclasses must set the following attributes in ``__init__``:
        - ``z_dim``  (int)                     — output channel dimension
        - ``upsampling_factor``  (int)          — spatial downsample ratio
        - ``temporal_downsample_factor``  (int)  — temporal downsample ratio
        - ``projection``  (nn.Module)           — trainable projection head

    and implement :meth:`encode`.
    """

    z_dim: int
    upsampling_factor: int
    temporal_downsample_factor: int
    projection: nn.Module
    requires_independent_first_frame: bool = False
    causal_tubelet_encoding: bool = False
    causal_prefix_encoding: bool = False

    @abstractmethod
    def encode(
        self,
        videos: torch.Tensor,
        device: str | torch.device = "cuda",
        tiled: bool = False,
        tile_size: tuple = (30, 52),
        tile_stride: tuple = (15, 26),
        return_pre_standardise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode ``[B, 3, T, H, W]`` → ``[B, z_dim, T_lat, H_lat, W_lat]``.

        If ``return_pre_standardise=True``, also returns the latents before
        channel standardisation (for regularisation losses like SIGReg).
        """
        ...

    # Convenience: mimic ``vae.model.z_dim`` for inference compat.
    @property
    def model(self):
        return SimpleNamespace(z_dim=self.z_dim)

    # ---- ImageNet normalisation helpers ----------------------------------- #
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def _normalise_for_backbone(self, images: torch.Tensor) -> torch.Tensor:
        """``[-1, 1]`` → ImageNet-normalised."""
        images = (images + 1.0) * 0.5
        mean = self._IMAGENET_MEAN.to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = self._IMAGENET_STD.to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    # ---- Common temporal downsample --------------------------------------- #
    @staticmethod
    def _temporal_stride(
        features: torch.Tensor,
        temporal_downsample_factor: int,
    ) -> torch.Tensor:
        """Keep first frame, stride the rest by ``temporal_downsample_factor``.

        Input:  ``[B, D, T, H, W]``
        Output: ``[B, D, T_lat, H, W]``  where ``T_lat = (T-1)//factor + 1``.
        """
        T = features.shape[2]
        if T <= 1:
            return features
        first = features[:, :, 0:1]
        rest = features[:, :, 1:]
        rest_strided = rest[:, :, ::temporal_downsample_factor]
        return torch.cat([first, rest_strided], dim=2)

    @staticmethod
    def _select_temporal_states(
        features: torch.Tensor,
        original_num_frames: int,
        temporal_patch_size: int,
        temporal_downsample_factor: int,
    ) -> torch.Tensor:
        """Select exactly the latent states at times 0, stride, 2*stride, ..."""
        if original_num_frames < 1:
            raise ValueError("Visual encoder input must contain at least one frame.")
        if temporal_patch_size < 1 or temporal_downsample_factor < 1:
            raise ValueError("Temporal patch/downsample factors must be positive.")

        target_frames = (original_num_frames - 1) // temporal_downsample_factor + 1
        source_indices = (
            torch.arange(target_frames, device=features.device)
            * temporal_downsample_factor
            // temporal_patch_size
        ).to(dtype=torch.long)
        if int(source_indices[-1]) >= features.shape[2]:
            raise ValueError(
                "Temporal state selection exceeds encoder output: "
                f"max_index={int(source_indices[-1])}, available={features.shape[2]}."
            )
        return features.index_select(2, source_indices)

    # ---- Per-channel standardisation (anti-collapse) ---------------------- #
    @staticmethod
    def _channel_standardise(
        latents: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-channel zero-mean unit-variance standardisation.

        Prevents representation collapse where the MLP learns to output all
        zeros (trivial solution for flow-matching video loss).

        Input/Output: ``[B, D, T, H, W]`` — standardised along ``(B, T, H, W)``
        so that each of the ``D`` channels has mean≈0 and std≈1.
        """
        # Compute stats over (B, T, H, W), keeping D.
        mean = latents.mean(dim=(0, 2, 3, 4), keepdim=True)   # [1, D, 1, 1, 1]
        var = latents.var(dim=(0, 2, 3, 4), keepdim=True)      # [1, D, 1, 1, 1]
        return (latents - mean) / (var.sqrt() + eps)

    def _standardise_latents(
        self,
        latents: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if not (
            bool(getattr(self, "causal_tubelet_encoding", False))
            or bool(getattr(self, "causal_prefix_encoding", False))
        ):
            return self._channel_standardise(latents, eps=eps)

        # Causal modes normalize each state independently so normalization
        # statistics cannot reveal information from another temporal state.
        mean = latents.mean(dim=(3, 4), keepdim=True)
        var = latents.var(dim=(3, 4), keepdim=True, unbiased=False)
        return (latents - mean) / (var.sqrt() + eps)

    def _configure_fixed_output_normalisation(
        self,
        normalise_stats_path: Optional[str],
        num_channels: int,
        min_std: float = 1e-6,
    ) -> None:
        self._has_fixed_stats = False
        self.normalise_stats_path = normalise_stats_path
        if normalise_stats_path is None:
            return

        stats_path = Path(normalise_stats_path).expanduser().resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Visual encoder normalisation stats not found: {stats_path}"
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
            raise ValueError(
                f"Visual encoder stats must contain `mean` and `std`: {stats_path}"
            )
        expected_metadata = {
            "model_name": getattr(self, "model_name", None),
            "temporal_downsample": getattr(
                self, "temporal_downsample_factor", None
            ),
            "causal_tubelet_encoding": bool(
                getattr(self, "causal_tubelet_encoding", False)
            ),
            "causal_prefix_encoding": bool(
                getattr(self, "causal_prefix_encoding", False)
            ),
            "skip_projection": bool(getattr(self, "skip_projection", False)),
        }
        for key, expected in expected_metadata.items():
            if key in stats and expected is not None and stats[key] != expected:
                raise ValueError(
                    f"Visual encoder stats `{key}` mismatch: "
                    f"file={stats[key]!r}, model={expected!r}, path={stats_path}."
                )
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(-1)
        if mean.numel() != num_channels or std.numel() != num_channels:
            raise ValueError(
                "Visual encoder stats channel mismatch: "
                f"expected {num_channels}, mean={mean.numel()}, std={std.numel()}."
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(f"Visual encoder stats contain NaN/Inf: {stats_path}")
        if torch.any(std <= 0):
            raise ValueError(f"Visual encoder stats contain non-positive std: {stats_path}")

        self.register_buffer("_norm_mean", mean.view(1, -1, 1, 1, 1))
        self.register_buffer(
            "_norm_std",
            std.clamp_min(float(min_std)).view(1, -1, 1, 1, 1),
        )
        self._has_fixed_stats = True
        logger.info(
            "%s: loaded fixed output normalisation from %s "
            "(channels=%d, mean=[%.4f, %.4f], std=[%.4f, %.4f])",
            self.__class__.__name__,
            stats_path,
            num_channels,
            mean.min().item(),
            mean.max().item(),
            std.min().item(),
            std.max().item(),
        )

    def _normalise_encoder_output(self, latents: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self, "_has_fixed_stats", False)):
            mean = self._norm_mean.to(device=latents.device, dtype=torch.float32)
            std = self._norm_std.to(device=latents.device, dtype=torch.float32)
            return ((latents.float() - mean) / std).to(dtype=latents.dtype)
        if self.standardise_output:
            return self._standardise_latents(latents)
        return latents

    def _apply(self, fn):
        result = super()._apply(fn)
        if bool(getattr(self, "_has_fixed_stats", False)):
            self._buffers["_norm_mean"] = self._norm_mean.float()
            self._buffers["_norm_std"] = self._norm_std.float()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        mean_key = prefix + "_norm_mean"
        std_key = prefix + "_norm_std"
        if (
            mean_key in state_dict
            and std_key in state_dict
            and not hasattr(self, "_norm_mean")
            and not hasattr(self, "_norm_std")
        ):
            self.register_buffer("_norm_mean", torch.empty_like(state_dict[mean_key]))
            self.register_buffer("_norm_std", torch.empty_like(state_dict[std_key]))
            self._has_fixed_stats = True
            self.normalise_stats_path = "<checkpoint>"
            logger.info(
                "%s: restored fixed output normalisation buffers from checkpoint.",
                self.__class__.__name__,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ---- train/eval override ---------------------------------------------- #
    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_backbone", False) and hasattr(self, "backbone"):
            self.backbone.eval()
        return self


# ========================================================================== #
# DINOv3 / DINOv2 (image encoder — per-frame)
# ========================================================================== #

_DINO_MODEL_SPECS = {
    # DINOv3
    "facebook/dinov3-vits16-pretrain-lvd1689m": {"hidden_dim": 384, "patch_size": 16},
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {"hidden_dim": 768, "patch_size": 16},
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {"hidden_dim": 1024, "patch_size": 16},
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {"hidden_dim": 1536, "patch_size": 16},
    # DINOv2
    "facebook/dinov2-small": {"hidden_dim": 384, "patch_size": 14},
    "facebook/dinov2-base": {"hidden_dim": 768, "patch_size": 14},
    "facebook/dinov2-large": {"hidden_dim": 1024, "patch_size": 14},
    "facebook/dinov2-giant": {"hidden_dim": 1536, "patch_size": 14},
}


class DINOEncoder(BaseVisualEncoder):
    """Frozen DINOv3/v2 backbone + optional trainable MLP projection.

    Per-frame image encoding with temporal stride to match VAE convention.

    When ``skip_projection=True`` (DiT-side projection mode), the MLP is
    removed entirely and the encoder outputs raw backbone features
    (``hidden_dim``-dimensional, e.g. 1024 for ViT-L).  The downstream
    DiT's ``patch_embedding`` Conv3d then serves as both patchify *and*
    projection (``in_dim=hidden_dim``).  This avoids the 1024→48
    information bottleneck and removes the need for SIGReg / channel
    standardisation.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        output_dim: int = 48,
        mlp_hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        spatial_downsample: int = 16,
        temporal_downsample: int = 4,
        standardise_output: bool = True,
        skip_projection: bool = False,
        normalise_stats_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.standardise_output = standardise_output
        self.skip_projection = skip_projection
        self._freeze_backbone = freeze_backbone

        # Load backbone
        spec = _DINO_MODEL_SPECS.get(model_name)
        self._hidden_dim = spec["hidden_dim"] if spec else None
        self._patch_size = spec["patch_size"] if spec else None

        self.backbone = self._load_hf_model(model_name, torch_dtype)
        if self._hidden_dim is None:
            self._hidden_dim = self._infer_attr(self.backbone, "hidden_size")
        if self._patch_size is None:
            self._patch_size = self._infer_attr(self.backbone, "patch_size")

        # In skip_projection mode, output_dim = backbone hidden_dim (e.g. 1024)
        if skip_projection:
            self.output_dim = self._hidden_dim
        else:
            self.output_dim = output_dim

        # VAE-compat attributes
        self.z_dim = self.output_dim
        self.upsampling_factor = spatial_downsample
        self.temporal_downsample_factor = temporal_downsample

        logger.info(
            "DINOEncoder: model=%s  hidden_dim=%d  patch_size=%d  output_dim=%d  "
            "skip_projection=%s  freeze=%s",
            model_name, self._hidden_dim, self._patch_size, self.output_dim,
            skip_projection, freeze_backbone,
        )

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Trainable MLP (only when NOT in skip_projection mode)
        if skip_projection:
            self.projection = nn.Identity()  # keep attribute for compatibility
        else:
            _mlp_h = mlp_hidden_dim if mlp_hidden_dim is not None else 2 * self._hidden_dim
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_dim, _mlp_h),
                nn.GELU(),
                nn.Linear(_mlp_h, output_dim),
            ).to(dtype=torch_dtype)

            # Xavier init for non-degenerate initial output
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # ---- Fixed normalisation stats (offline-computed mean/std) -------- #
        # When provided, replaces batch _channel_standardise with fixed
        # global stats: x_norm = (x - mean) / std.
        # At inference, un-standardise via: x = x_norm * std + mean.
        self._has_fixed_stats = False
        if normalise_stats_path is not None:
            stats = torch.load(normalise_stats_path, map_location="cpu")
            # stats["mean"]: [D], stats["std"]: [D]
            # Clamp std to avoid division by near-zero (some DINO channels are near-constant)
            std_clamped = stats["std"].clamp(min=0.01)
            self.register_buffer("_norm_mean", stats["mean"].view(1, -1, 1, 1, 1))
            self.register_buffer("_norm_std", std_clamped.view(1, -1, 1, 1, 1))
            self._has_fixed_stats = True
            logger.info(
                "DINOEncoder: loaded fixed normalisation stats from %s "
                "(mean range [%.3f, %.3f], std range [%.3f, %.3f])",
                normalise_stats_path,
                stats["mean"].min().item(), stats["mean"].max().item(),
                stats["std"].min().item(), stats["std"].max().item(),
            )

    # ---- Fixed normalisation helpers ------------------------------------- #
    def normalise(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply fixed normalisation: (x - mean) / std. Shape: [B, D, T, H, W]."""
        if not self._has_fixed_stats:
            if self.standardise_output:
                return self._channel_standardise(latents)
            return latents
        mean = self._norm_mean.to(device=latents.device, dtype=latents.dtype)
        std = self._norm_std.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def unnormalise(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse fixed normalisation: x * std + mean. Shape: [B, D, T, H, W]."""
        if not self._has_fixed_stats:
            raise RuntimeError(
                "Cannot unnormalise without fixed stats. "
                "Provide `normalise_stats_path` or use batch standardise (irreversible)."
            )
        mean = self._norm_mean.to(device=latents.device, dtype=latents.dtype)
        std = self._norm_std.to(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    # ---- encode ----------------------------------------------------------- #
    def encode(self, videos, device="cuda", tiled=False, tile_size=(30, 52), tile_stride=(15, 26), return_pre_standardise=False):
        videos = videos.to(device=device)
        B, C, T, H, W = videos.shape
        H_lat = H // self.upsampling_factor
        W_lat = W // self.upsampling_factor

        # Per-frame: [B*T, 3, H, W]
        frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frames = self._normalise_for_backbone(frames)

        with torch.no_grad():
            out = self.backbone(pixel_values=frames)
            tokens = out.last_hidden_state  # [B*T, 1+num_reg+num_patches, D]

        # DINOv3/v2 prepend CLS and optionally register tokens before patch tokens.
        # Keep only the last (patch_h * patch_w) tokens.
        num_patches = (H // self._patch_size) * (W // self._patch_size)
        patch_tokens = tokens[:, -num_patches:, :]  # [B*T, num_patches, D]

        # Apply projection (MLP or Identity in skip_projection mode)
        projected = self.projection(patch_tokens)  # [B*T, N, D_out]

        patch_h = H // self._patch_size
        patch_w = W // self._patch_size
        projected = projected.reshape(B * T, patch_h, patch_w, self.output_dim)
        projected = projected.permute(0, 3, 1, 2)  # [B*T, D_out, ph, pw]

        if patch_h != H_lat or patch_w != W_lat:
            projected = F.interpolate(
                projected.float(), size=(H_lat, W_lat), mode="bilinear", align_corners=False,
            ).to(dtype=projected.dtype)

        projected = projected.reshape(B, T, self.output_dim, H_lat, W_lat)
        projected = projected.permute(0, 2, 1, 3, 4)  # [B, D_out, T, H_lat, W_lat]

        latents = self._temporal_stride(projected, self.temporal_downsample_factor)

        # In skip_projection mode: no SIGReg needed (no bottleneck).
        # Normalise using fixed stats (if available) or batch standardise.
        if self.skip_projection:
            latents = self.normalise(latents)
            if return_pre_standardise:
                return latents, None
            return latents

        if return_pre_standardise:
            latents_raw = latents
            latents = self.normalise(latents)
            return latents, latents_raw

        latents = self.normalise(latents)
        return latents

    # ---- helpers ---------------------------------------------------------- #
    @staticmethod
    def _load_hf_model(model_name: str, dtype: torch.dtype) -> nn.Module:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name)
        return model.to(dtype=dtype)

    @staticmethod
    def _infer_attr(backbone: nn.Module, attr: str):
        if hasattr(backbone, "config") and hasattr(backbone.config, attr):
            v = getattr(backbone.config, attr)
            return int(v) if not isinstance(v, (list, tuple)) else int(v[0])
        raise ValueError(f"Cannot infer `{attr}` from backbone config.")


_ENCODER_REGISTRY["dino"] = DINOEncoder


# ========================================================================== #
# Qwen3-VL vision tower (SigLIP2-initialized video ViT)
# ========================================================================== #

class Qwen3VLVisualEncoder(BaseVisualEncoder):
    """Frozen Qwen3-VL vision tower with raw patch-token output.

    This loads only ``model.visual.*`` from the Qwen3-VL checkpoint. The
    language-facing spatial mergers and DeepStack projection heads are removed,
    so the output remains in the native 1024-dimensional visual feature space.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        output_dim: int = 48,
        mlp_hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        spatial_downsample: int = 16,
        temporal_downsample: int = 4,
        standardise_output: bool = True,
        skip_projection: bool = False,
        causal_tubelet_encoding: bool = False,
        causal_prefix_encoding: bool = False,
        local_files_only: bool = True,
        checkpoint_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.standardise_output = standardise_output
        self.skip_projection = skip_projection
        self._freeze_backbone = freeze_backbone
        self.requires_independent_first_frame = True
        self.causal_tubelet_encoding = causal_tubelet_encoding
        self.causal_prefix_encoding = causal_prefix_encoding

        self.backbone = self._load_vision_backbone(
            model_name=model_name,
            dtype=torch_dtype,
            local_files_only=local_files_only,
            checkpoint_path=checkpoint_path,
        )
        config = self.backbone.config
        self._hidden_dim = int(config.hidden_size)
        self._patch_size = int(config.patch_size)
        self._temporal_patch = int(config.temporal_patch_size)
        self._spatial_merge = int(config.spatial_merge_size)

        self.output_dim = self._hidden_dim if skip_projection else output_dim
        self.z_dim = self.output_dim
        self.upsampling_factor = spatial_downsample
        self.temporal_downsample_factor = temporal_downsample

        logger.info(
            "%s: model=%s hidden_dim=%d patch=%d temporal_patch=%d "
            "output_dim=%d skip_projection=%s freeze=%s",
            self.__class__.__name__,
            model_name,
            self._hidden_dim,
            self._patch_size,
            self._temporal_patch,
            self.output_dim,
            skip_projection,
            freeze_backbone,
        )

        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if skip_projection:
            self.projection = nn.Identity()
        else:
            hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else 2 * self._hidden_dim
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            ).to(dtype=torch_dtype)
            for module in self.projection.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @staticmethod
    def _weight_files(model_name: str, local_files_only: bool) -> list[str]:
        local_path = Path(model_name)
        if local_path.is_dir():
            index_path = local_path / "model.safetensors.index.json"
            if index_path.is_file():
                with index_path.open("r", encoding="utf-8") as handle:
                    weight_map = json.load(handle).get("weight_map", {})
                filenames = {
                    filename
                    for key, filename in weight_map.items()
                    if key.startswith("model.visual.")
                }
                if not filenames:
                    raise ValueError(f"No Qwen3-VL visual weights found in {index_path}.")
                return [str(local_path / filename) for filename in sorted(filenames)]

            weights_path = local_path / "model.safetensors"
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Qwen3-VL checkpoint directory has no safetensors weights: {local_path}"
                )
            return [str(weights_path)]

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        except Exception as exc:  # pragma: no cover
            raise ImportError("Qwen3-VL visual loading requires huggingface_hub.") from exc

        try:
            index_path = hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors.index.json",
                local_files_only=local_files_only,
            )
        except (EntryNotFoundError, LocalEntryNotFoundError):
            index_path = None

        if index_path is None:
            return [
                hf_hub_download(
                    repo_id=model_name,
                    filename="model.safetensors",
                    local_files_only=local_files_only,
                )
            ]

        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        filenames = {
            filename
            for key, filename in weight_map.items()
            if key.startswith("model.visual.")
        }
        if not filenames:
            raise ValueError(f"No Qwen3-VL visual weights found in {index_path}.")
        return [
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_files_only=local_files_only,
            )
            for filename in sorted(filenames)
        ]

    @classmethod
    def _load_vision_backbone(
        cls,
        model_name: str,
        dtype: torch.dtype,
        local_files_only: bool,
        checkpoint_path: Optional[str] = None,
    ) -> nn.Module:
        del checkpoint_path
        try:
            from accelerate import init_empty_weights
            from safetensors import safe_open
            from transformers import AutoConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Qwen3-VL visual loading requires transformers>=4.57, accelerate, "
                "and safetensors."
            ) from exc

        full_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        vision_config = getattr(full_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(f"{model_name} has no Qwen3-VL `vision_config`.")

        with init_empty_weights():
            backbone = Qwen3VLVisionModel(vision_config)

        # Keep the pre-merge token norm, but remove the language-facing spatial
        # concatenation/projection and DeepStack projection heads.
        backbone.final_norm = backbone.merger.norm
        backbone.merger = nn.Identity()
        backbone.deepstack_merger_list = nn.ModuleList()
        backbone.deepstack_visual_indexes = []

        state_dict = {}
        excluded_prefixes = ("merger.", "deepstack_merger_list.")
        for path in cls._weight_files(model_name, local_files_only):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if not key.startswith("model.visual."):
                        continue
                    relative_key = key.removeprefix("model.visual.")
                    if relative_key.startswith("merger.norm."):
                        relative_key = "final_norm." + relative_key.removeprefix("merger.norm.")
                    if relative_key.startswith(excluded_prefixes):
                        continue
                    state_dict[relative_key] = handle.get_tensor(key)

        if not state_dict:
            raise ValueError(f"No usable Qwen3-VL visual backbone weights found for {model_name}.")

        incompatible = backbone.load_state_dict(state_dict, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Failed to load Qwen3-VL visual backbone cleanly: "
                f"missing={incompatible.missing_keys[:8]} "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        return backbone.to(dtype=dtype)

    def encode(
        self,
        videos,
        device="cuda",
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        return_pre_standardise=False,
    ):
        videos = videos.to(device=device)
        batch_size, channels, num_frames, height, width = videos.shape
        if channels != 3:
            raise ValueError(f"Qwen3VLVisualEncoder expects 3 channels, got {channels}.")
        if height % self._patch_size != 0 or width % self._patch_size != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch size "
                f"{self._patch_size}."
            )

        patch_h = height // self._patch_size
        patch_w = width // self._patch_size
        if patch_h % self._spatial_merge != 0 or patch_w % self._spatial_merge != 0:
            raise ValueError(
                f"Patch grid {(patch_h, patch_w)} must be divisible by Qwen spatial "
                f"merge size {self._spatial_merge}."
            )

        original_frames = num_frames
        pad_frames = (-num_frames) % self._temporal_patch
        if pad_frames:
            videos = torch.cat(
                [videos, videos[:, :, -1:].expand(-1, -1, pad_frames, -1, -1)],
                dim=2,
            )
            num_frames = videos.shape[2]

        grid_t = num_frames // self._temporal_patch
        merge = self._spatial_merge
        patches = videos.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(
            batch_size,
            grid_t,
            self._temporal_patch,
            channels,
            patch_h // merge,
            merge,
            self._patch_size,
            patch_w // merge,
            merge,
            self._patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        patches = patches.reshape(
            batch_size * grid_t * patch_h * patch_w,
            channels * self._temporal_patch * self._patch_size * self._patch_size,
        )
        grid_thw = torch.tensor(
            [[grid_t, patch_h, patch_w]] * batch_size,
            device=videos.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            outputs = self.backbone(patches, grid_thw=grid_thw, return_dict=True)
            tokens = self.backbone.final_norm(outputs.last_hidden_state)

        expected_tokens = batch_size * grid_t * patch_h * patch_w
        if tokens.shape != (expected_tokens, self._hidden_dim):
            raise ValueError(
                "Unexpected Qwen3-VL visual output shape: "
                f"got {tuple(tokens.shape)}, expected "
                f"{(expected_tokens, self._hidden_dim)}."
            )

        tokens = self.projection(tokens)
        tokens = tokens.view(
            batch_size,
            grid_t,
            patch_h // merge,
            patch_w // merge,
            merge,
            merge,
            self.output_dim,
        )
        tokens = tokens.permute(0, 6, 1, 2, 4, 3, 5).reshape(
            batch_size,
            self.output_dim,
            grid_t,
            patch_h,
            patch_w,
        )

        latent_h = height // self.upsampling_factor
        latent_w = width // self.upsampling_factor
        if patch_h != latent_h or patch_w != latent_w:
            tokens = F.interpolate(
                tokens.float(),
                size=(tokens.shape[2], latent_h, latent_w),
                mode="trilinear",
                align_corners=False,
            ).to(dtype=tokens.dtype)

        tokens = self._select_temporal_states(
            tokens,
            original_num_frames=original_frames,
            temporal_patch_size=self._temporal_patch,
            temporal_downsample_factor=self.temporal_downsample_factor,
        )

        if self.skip_projection:
            if self.standardise_output:
                tokens = self._standardise_latents(tokens)
            if return_pre_standardise:
                return tokens, None
            return tokens

        raw_tokens = tokens
        if self.standardise_output:
            tokens = self._standardise_latents(tokens)
        if return_pre_standardise:
            return tokens, raw_tokens
        return tokens


_ENCODER_REGISTRY["qwen3_vl_vision"] = Qwen3VLVisualEncoder
_ENCODER_REGISTRY["siglip2_qwen3vl"] = Qwen3VLVisualEncoder


# ========================================================================== #
# Xiaomi Robotics-1 vision tower
# ========================================================================== #

class XR1VisualEncoder(Qwen3VLVisualEncoder):
    """Frozen XR-1-tuned Qwen3-VL vision tower with raw 1024-d tokens."""

    _VISUAL_PREFIXES = (
        "module.model.vlm.model.visual.",
        "model.vlm.model.visual.",
        "vlm.model.visual.",
    )
    _EXCLUDED_PREFIXES = ("merger.", "deepstack_merger_list.")

    @classmethod
    def _relative_visual_key(cls, key: str) -> Optional[str]:
        relative_key = None
        for prefix in cls._VISUAL_PREFIXES:
            if key.startswith(prefix):
                relative_key = key.removeprefix(prefix)
                break
        if relative_key is None:
            return None
        if relative_key.startswith("merger.norm."):
            return "final_norm." + relative_key.removeprefix("merger.norm.")
        if relative_key.startswith(cls._EXCLUDED_PREFIXES):
            return None
        return relative_key

    @classmethod
    def _load_xr1_visual_state_dict(cls, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = {}

        if checkpoint_path.is_dir():
            model_states_path = checkpoint_path / "model_states.pt"
            if model_states_path.is_file():
                checkpoint_path = model_states_path
            else:
                try:
                    from safetensors import safe_open
                except Exception as exc:  # pragma: no cover
                    raise ImportError("XR-1 safetensors loading requires `safetensors`.") from exc

                index_path = checkpoint_path / "model.safetensors.index.json"
                if index_path.is_file():
                    with index_path.open("r", encoding="utf-8") as handle:
                        weight_map = json.load(handle).get("weight_map", {})
                    filenames = {
                        filename
                        for key, filename in weight_map.items()
                        if cls._relative_visual_key(key) is not None
                    }
                    if not filenames:
                        raise ValueError(f"No XR-1 visual weights found in {index_path}.")
                    weight_files = [checkpoint_path / name for name in sorted(filenames)]
                else:
                    single_file = checkpoint_path / "model.safetensors"
                    if not single_file.is_file():
                        raise FileNotFoundError(
                            "XR-1 model directory must contain model_states.pt, "
                            "model.safetensors, or model.safetensors.index.json: "
                            f"{checkpoint_path}"
                        )
                    weight_files = [single_file]

                for weight_file in weight_files:
                    if not weight_file.is_file():
                        raise FileNotFoundError(f"Missing XR-1 weight shard: {weight_file}")
                    with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
                        for key in handle.keys():
                            relative_key = cls._relative_visual_key(key)
                            if relative_key is not None:
                                state_dict[relative_key] = handle.get_tensor(key)
                return state_dict

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"XR-1 checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        source_state = checkpoint.get("module", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(source_state, dict):
            raise TypeError(
                "XR-1 checkpoint must be a state dict or contain a `module` state dict."
            )
        for key, tensor in source_state.items():
            relative_key = cls._relative_visual_key(str(key))
            if relative_key is not None:
                state_dict[relative_key] = tensor
        return state_dict

    @classmethod
    def _load_vision_backbone(
        cls,
        model_name: str,
        dtype: torch.dtype,
        local_files_only: bool,
        checkpoint_path: Optional[str] = None,
    ) -> nn.Module:
        if checkpoint_path is None:
            raise ValueError("XR1VisualEncoder requires `checkpoint_path`.")

        try:
            from accelerate import init_empty_weights
            from transformers import AutoConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "XR-1 visual loading requires transformers>=4.57 and accelerate."
            ) from exc

        full_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        vision_config = getattr(full_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(f"{model_name} has no Qwen3-VL `vision_config`.")

        with init_empty_weights():
            backbone = Qwen3VLVisionModel(vision_config)
        backbone.final_norm = backbone.merger.norm
        backbone.merger = nn.Identity()
        backbone.deepstack_merger_list = nn.ModuleList()
        backbone.deepstack_visual_indexes = []

        state_dict = cls._load_xr1_visual_state_dict(
            Path(checkpoint_path).expanduser().resolve()
        )
        if not state_dict:
            raise ValueError(f"No usable XR-1 visual weights found in {checkpoint_path}.")

        incompatible = backbone.load_state_dict(state_dict, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Failed to load XR-1 visual backbone cleanly: "
                f"missing={incompatible.missing_keys[:8]} "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        return backbone.to(dtype=dtype)


_ENCODER_REGISTRY["xr1_vision"] = XR1VisualEncoder


# ========================================================================== #
# V-JEPA 2 (video encoder — native spatiotemporal)
# ========================================================================== #

_VJEPA2_MODEL_SPECS = {
    "facebook/vjepa2-vitl-fpc64-256": {
        "hidden_dim": 1024,
        "spatial_patch_size": 16,
        "temporal_patch_size": 2,
    },
    "facebook/vjepa2-vith-fpc64-256": {
        "hidden_dim": 1280,
        "spatial_patch_size": 16,
        "temporal_patch_size": 2,
    },
}


class VJEPA2Encoder(BaseVisualEncoder):
    """Frozen V-JEPA 2 backbone + optional trainable MLP projection.

    V-JEPA 2 is a native **video** encoder — it operates on a clip of frames
    and produces spatiotemporal patch tokens.  The spatial patch size is 16
    and the temporal patch size is 2, so it already does 16× spatial and 2×
    temporal downsampling internally.  An additional temporal stride is applied
    to match the VAE's 4× temporal convention.

    When ``skip_projection=True``, the MLP is removed and raw backbone features
    are output (same DiT-side projection strategy as DINOEncoder).

    Output: ``[B, output_dim, T_lat, H_lat, W_lat]`` matching VAE format.
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        output_dim: int = 48,
        mlp_hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        spatial_downsample: int = 16,
        temporal_downsample: int = 4,
        standardise_output: bool = True,
        skip_projection: bool = False,
        causal_tubelet_encoding: bool = False,
        causal_prefix_encoding: bool = False,
        normalise_stats_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.standardise_output = standardise_output
        self.skip_projection = skip_projection
        self._freeze_backbone = freeze_backbone
        self.requires_independent_first_frame = True
        self.causal_tubelet_encoding = causal_tubelet_encoding
        self.causal_prefix_encoding = causal_prefix_encoding

        # Load backbone
        spec = _VJEPA2_MODEL_SPECS.get(model_name, {})
        self._hidden_dim = spec.get("hidden_dim")
        self._spatial_patch = spec.get("spatial_patch_size", 16)
        self._temporal_patch = spec.get("temporal_patch_size", 2)

        self.backbone, self.processor = self._load_hf_model(model_name, torch_dtype)

        if self._hidden_dim is None:
            self._hidden_dim = self._infer_attr(self.backbone, "hidden_size")

        if skip_projection:
            self.output_dim = self._hidden_dim
        else:
            self.output_dim = output_dim

        # VAE-compat attributes
        self.z_dim = self.output_dim
        self.upsampling_factor = spatial_downsample
        self.temporal_downsample_factor = temporal_downsample
        self._configure_fixed_output_normalisation(
            normalise_stats_path,
            num_channels=self.output_dim,
        )

        logger.info(
            "VJEPA2Encoder: model=%s  hidden_dim=%d  spatial_patch=%d  temporal_patch=%d  "
            "output_dim=%d  skip_projection=%s  freeze=%s",
            model_name, self._hidden_dim, self._spatial_patch, self._temporal_patch,
            self.output_dim, skip_projection, freeze_backbone,
        )

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Trainable MLP (only when NOT in skip_projection mode)
        if skip_projection:
            self.projection = nn.Identity()
        else:
            _mlp_h = mlp_hidden_dim if mlp_hidden_dim is not None else 2 * self._hidden_dim
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_dim, _mlp_h),
                nn.GELU(),
                nn.Linear(_mlp_h, output_dim),
            ).to(dtype=torch_dtype)

    # ---- encode ----------------------------------------------------------- #
    def encode(self, videos, device="cuda", tiled=False, tile_size=(30, 52), tile_stride=(15, 26), return_pre_standardise=False):
        """Encode ``[B, 3, T, H, W]`` → ``[B, output_dim, T_lat, H_lat, W_lat]``.

        V-JEPA 2 processes video clips natively and returns spatiotemporal
        patch tokens ``[B, T_p*H_p*W_p, D]``.  We reshape, project via MLP,
        and apply additional temporal striding to match VAE's T_lat convention.
        """
        videos = videos.to(device=device)
        B, C, T, H, W = videos.shape
        original_frames = T
        H_lat = H // self.upsampling_factor
        W_lat = W // self.upsampling_factor

        # Pad to a complete temporal tubelet. This also turns single-frame
        # inference into the zero-motion clip [f0, f0].
        pad_frames = (-T) % self._temporal_patch
        if pad_frames:
            videos = torch.cat(
                [videos, videos[:, :, -1:].expand(-1, -1, pad_frames, -1, -1)],
                dim=2,
            )
            T = videos.shape[2]

        # V-JEPA 2 expects [B, T, C, H, W] or processor-normalised input.
        # Convert from [-1,1] → ImageNet normalised, then to [B, T, C, H, W].
        frames_bthw = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        # Normalise each frame
        frames_flat = frames_bthw.reshape(B * T, C, H, W)
        frames_normed = self._normalise_for_backbone(frames_flat)
        frames_normed = frames_normed.reshape(B, T, C, H, W)

        # V-JEPA 2 forward (frozen)
        with torch.no_grad():
            patch_tokens = self._extract_tokens(frames_normed)  # [B, N_total, D]

        # Compute spatial/temporal patch grid
        T_p = T // self._temporal_patch
        H_p = H // self._spatial_patch
        W_p = W // self._spatial_patch

        # MLP projection
        projected = self.projection(patch_tokens)  # [B, N_total, output_dim]

        # Reshape to spatiotemporal grid [B, T_p, H_p, W_p, output_dim]
        projected = projected.reshape(B, T_p, H_p, W_p, self.output_dim)
        projected = projected.permute(0, 4, 1, 2, 3)  # [B, output_dim, T_p, H_p, W_p]

        # Spatial interpolation if patch grid != latent grid
        if H_p != H_lat or W_p != W_lat:
            # Reshape for 2D interpolation: [B*T_p, output_dim, H_p, W_p]
            Bt = B * T_p
            projected = projected.permute(0, 2, 1, 3, 4).reshape(Bt, self.output_dim, H_p, W_p)
            projected = F.interpolate(
                projected.float(), size=(H_lat, W_lat), mode="bilinear", align_corners=False,
            ).to(dtype=projected.dtype)
            projected = projected.reshape(B, T_p, self.output_dim, H_lat, W_lat)
            projected = projected.permute(0, 2, 1, 3, 4)  # [B, output_dim, T_p, H_lat, W_lat]

        projected = self._select_temporal_states(
            projected,
            original_num_frames=original_frames,
            temporal_patch_size=self._temporal_patch,
            temporal_downsample_factor=self.temporal_downsample_factor,
        )

        # In skip_projection mode: no SIGReg, but standardise is still
        # useful to normalise features to mean=0 std=1 per channel.
        if self.skip_projection:
            projected = self._normalise_encoder_output(projected)
            if return_pre_standardise:
                return projected, None
            return projected

        if return_pre_standardise:
            latents_raw = projected
            projected = self._normalise_encoder_output(projected)
            return projected, latents_raw

        return self._normalise_encoder_output(projected)

    def _extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract spatiotemporal patch tokens from V-JEPA 2.

        Args:
            frames: ``[B, T, C, H, W]`` ImageNet-normalised.

        Returns:
            ``[B, N_total, D]`` — patch tokens (CLS excluded if present).
        """
        # V-JEPA 2 HuggingFace API uses `pixel_values_videos` (not `pixel_values`).
        # Input shape: [B, T, C, H, W].
        outputs = self.backbone(pixel_values_videos=frames)

        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
            # Skip CLS token if present (usually index 0)
            if hasattr(self.backbone, "config"):
                # V-JEPA 2 may or may not use CLS
                num_patches_expected = (
                    (frames.shape[1] // self._temporal_patch)
                    * (frames.shape[3] // self._spatial_patch)
                    * (frames.shape[4] // self._spatial_patch)
                )
                if tokens.shape[1] > num_patches_expected:
                    tokens = tokens[:, -num_patches_expected:, :]
            return tokens

        raise ValueError(
            "Unexpected V-JEPA 2 output format. "
            "Expected `last_hidden_state` attribute."
        )

    # ---- helpers ---------------------------------------------------------- #
    @staticmethod
    def _load_hf_model(model_name: str, dtype: torch.dtype):
        from transformers import AutoModel, AutoVideoProcessor
        model = AutoModel.from_pretrained(model_name).to(dtype=dtype)
        try:
            processor = AutoVideoProcessor.from_pretrained(model_name)
        except Exception:
            processor = None
            logger.warning("No AutoVideoProcessor found for %s; using manual normalisation.", model_name)
        return model, processor

    @staticmethod
    def _infer_attr(backbone: nn.Module, attr: str):
        if hasattr(backbone, "config") and hasattr(backbone.config, attr):
            v = getattr(backbone.config, attr)
            return int(v) if not isinstance(v, (list, tuple)) else int(v[0])
        raise ValueError(f"Cannot infer `{attr}` from V-JEPA 2 backbone config.")


_ENCODER_REGISTRY["vjepa2"] = VJEPA2Encoder


# ========================================================================== #
# V-JEPA 2-AC (action-conditioned, post-trained on Droid robot data)
# ========================================================================== #
#
# V-JEPA 2-AC shares the V-JEPA 2 ViT-g/16 encoder architecture but uses
# weights post-trained with a latent action-conditioned world-model objective
# on ~62h of Droid robot interaction data (Meta FAIR, 2025).
#
# Unlike V-JEPA 2, the AC variant is NOT distributed on HuggingFace.  Weights
# are obtained via:
#   (a) torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
#   (b) direct download: https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
#
# We reuse ``VJEPA2Encoder.encode`` verbatim and only override:
#   - ``__init__``          : load backbone from torch.hub or local .pt
#   - ``_extract_tokens``   : the hub backbone has a different forward signature
#                             than HuggingFace's ``AutoModel``.
#
# Extending to new AC variants is as simple as adding an entry to
# ``_VJEPA2AC_MODEL_SPECS``.

_VJEPA2AC_MODEL_SPECS = {
    # Official V-JEPA 2-AC ViT-g/16 @ 256px, 8 frames/clip, temporal patch 2
    "vjepa2_ac_vit_giant": {
        "hub_repo":            "facebookresearch/vjepa2",
        "hub_entrypoint":      "vjepa2_ac_vit_giant",
        "direct_url":          "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt",
        "spatial_patch_size":  16,
        "temporal_patch_size": 2,
        # ``hidden_dim`` resolved from ``backbone.embed_dim`` at runtime.
    },
}


class VJEPA2ACEncoder(VJEPA2Encoder):
    """Frozen V-JEPA 2-AC backbone + optional trainable MLP projection.

    Identical I/O contract to :class:`VJEPA2Encoder`
    (``[B,3,T,H,W] → [B, output_dim, T_lat, H_lat, W_lat]``). Only the weight
    provenance and backbone forward API differ.

    Args:
        model_name: key into ``_VJEPA2AC_MODEL_SPECS`` (default: ``vjepa2_ac_vit_giant``).
        checkpoint_source: ``"torch_hub"`` (default) or ``"local"``.
        checkpoint_path: path to a local ``.pt`` — required when
            ``checkpoint_source="local"``.  Either the raw state_dict or the
            full object returned by ``torch.hub.load`` is accepted.
        All other args match :class:`VJEPA2Encoder`.
    """

    def __init__(
        self,
        model_name: str = "vjepa2_ac_vit_giant",
        output_dim: int = 48,
        mlp_hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        spatial_downsample: int = 16,
        temporal_downsample: int = 4,
        standardise_output: bool = True,
        skip_projection: bool = False,
        causal_tubelet_encoding: bool = False,
        causal_prefix_encoding: bool = False,
        checkpoint_source: str = "torch_hub",
        checkpoint_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        # NOTE: we bypass ``VJEPA2Encoder.__init__`` (which loads from HF) and
        # re-implement the setup here so we can plug in a torch.hub backbone.
        # We call ``nn.Module.__init__`` via ``BaseVisualEncoder``'s MRO.
        nn.Module.__init__(self)

        self.model_name = model_name
        self.standardise_output = standardise_output
        self.skip_projection = skip_projection
        self._freeze_backbone = freeze_backbone
        self.requires_independent_first_frame = True
        self.causal_tubelet_encoding = causal_tubelet_encoding
        self.causal_prefix_encoding = causal_prefix_encoding
        self.checkpoint_source = checkpoint_source
        self.checkpoint_path = checkpoint_path

        spec = _VJEPA2AC_MODEL_SPECS.get(model_name, {})
        self._spatial_patch = spec.get("spatial_patch_size", 16)
        self._temporal_patch = spec.get("temporal_patch_size", 2)

        # Load the backbone (predictor is discarded — encoder-only experiment).
        self.backbone = self._load_ac_backbone(
            model_name=model_name,
            source=checkpoint_source,
            local_path=checkpoint_path,
            dtype=torch_dtype,
            spec=spec,
        )
        self.processor = None  # torch.hub backbone needs no HF processor.

        self._hidden_dim = self._infer_hidden_dim(self.backbone)

        if skip_projection:
            self.output_dim = self._hidden_dim
        else:
            self.output_dim = output_dim

        # VAE-compat attributes
        self.z_dim = self.output_dim
        self.upsampling_factor = spatial_downsample
        self.temporal_downsample_factor = temporal_downsample

        logger.info(
            "VJEPA2ACEncoder: model=%s  source=%s  hidden_dim=%d  "
            "spatial_patch=%d  temporal_patch=%d  output_dim=%d  "
            "skip_projection=%s  freeze=%s",
            model_name, checkpoint_source, self._hidden_dim,
            self._spatial_patch, self._temporal_patch, self.output_dim,
            skip_projection, freeze_backbone,
        )

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        if skip_projection:
            self.projection = nn.Identity()
        else:
            _mlp_h = mlp_hidden_dim if mlp_hidden_dim is not None else 2 * self._hidden_dim
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_dim, _mlp_h),
                nn.GELU(),
                nn.Linear(_mlp_h, output_dim),
            ).to(dtype=torch_dtype)

    # ---- checkpoint loading ---------------------------------------------- #
    @staticmethod
    def _load_ac_backbone(
        model_name: str,
        source: str,
        local_path: Optional[str],
        dtype: torch.dtype,
        spec: dict,
    ) -> nn.Module:
        """Load the AC encoder backbone via torch.hub or from a local .pt.

        The AC release from FAIR bundles ``(encoder, predictor)``; we take the
        encoder and discard the predictor.
        """
        if source == "torch_hub":
            hub_repo = spec.get("hub_repo", "facebookresearch/vjepa2")
            hub_entry = spec.get("hub_entrypoint", model_name)
            logger.info(
                "VJEPA2ACEncoder: loading via torch.hub.load('%s', '%s') ...",
                hub_repo, hub_entry,
            )
            loaded = torch.hub.load(hub_repo, hub_entry, trust_repo=True)
            encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            return encoder.to(dtype=dtype)

        if source == "local":
            if local_path is None:
                raise ValueError(
                    "VJEPA2ACEncoder: checkpoint_source='local' requires "
                    "checkpoint_path to be set (path to vjepa2-ac-vitg.pt)."
                )
            logger.info("VJEPA2ACEncoder: loading local checkpoint %s ...", local_path)
            # The user can either provide a raw state_dict or the full hub
            # object pickled to disk.  We try both.
            blob = torch.load(local_path, map_location="cpu")
            if isinstance(blob, (tuple, list)):
                encoder = blob[0]
            elif isinstance(blob, nn.Module):
                encoder = blob
            elif isinstance(blob, dict) and "encoder" in blob and isinstance(blob["encoder"], nn.Module):
                encoder = blob["encoder"]
            else:
                # Raw state_dict — we still need an architecture skeleton.
                # Pull a BARE architecture from torch.hub with pretrained=False
                # so the hubconf does NOT try to download its own weights (the
                # vjepa2 hubconf otherwise fetches from a hard-coded URL that is
                # unreachable here), then load our local state_dict into it.
                logger.info(
                    "Local file appears to be a state_dict; building model "
                    "skeleton from torch.hub (pretrained=False) and loading "
                    "the local state_dict."
                )
                hub_repo = spec.get("hub_repo", "facebookresearch/vjepa2")
                hub_entry = spec.get("hub_entrypoint", model_name)
                skeleton = torch.hub.load(
                    hub_repo, hub_entry, trust_repo=True, pretrained=False,
                )
                encoder = skeleton[0] if isinstance(skeleton, (tuple, list)) else skeleton
                state_dict = blob.get("encoder", blob) if isinstance(blob, dict) else blob
                # The hub encoder stores keys without the "backbone."/"module."
                # prefixes that the checkpoint may carry; strip them best-effort.
                def _strip_prefix(sd):
                    out = {}
                    for k, v in sd.items():
                        nk = k
                        for p in ("module.", "backbone.", "encoder."):
                            if nk.startswith(p):
                                nk = nk[len(p):]
                        out[nk] = v
                    return out
                missing, unexpected = encoder.load_state_dict(
                    _strip_prefix(state_dict), strict=False,
                )
                logger.info(
                    "VJEPA2ACEncoder local load_state_dict: %d missing, %d unexpected keys",
                    len(missing), len(unexpected),
                )
                if len(missing) > 50:
                    logger.warning(
                        "VJEPA2ACEncoder: %d missing keys — encoder weights may "
                        "not have loaded correctly (check checkpoint key names).",
                        len(missing),
                    )
            return encoder.to(dtype=dtype)

        raise ValueError(
            f"VJEPA2ACEncoder: unknown checkpoint_source='{source}'. "
            "Expected 'torch_hub' or 'local'."
        )

    @staticmethod
    def _infer_hidden_dim(backbone: nn.Module) -> int:
        """Introspect the V-JEPA 2-AC backbone for its feature dimension.

        The torch.hub model is a custom ViT (not HF), so ``config.hidden_size``
        may not exist.  We check several common attributes in order.
        """
        for attr in ("embed_dim", "hidden_dim", "hidden_size", "dim"):
            if hasattr(backbone, attr):
                v = getattr(backbone, attr)
                if isinstance(v, int):
                    return v
        if hasattr(backbone, "config"):
            cfg = backbone.config
            for attr in ("hidden_size", "embed_dim", "hidden_dim"):
                if hasattr(cfg, attr):
                    v = getattr(cfg, attr)
                    if isinstance(v, int):
                        return v
        # Last resort: sniff the final norm layer.
        for m in reversed(list(backbone.modules())):
            if isinstance(m, nn.LayerNorm) and isinstance(m.normalized_shape, (tuple, list)):
                if len(m.normalized_shape) == 1:
                    return int(m.normalized_shape[0])
        raise ValueError(
            "VJEPA2ACEncoder: cannot infer hidden_dim from backbone. "
            "Please add it to _VJEPA2AC_MODEL_SPECS or expose `embed_dim`."
        )

    # ---- override token extraction --------------------------------------- #
    def _extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract spatiotemporal patch tokens from the V-JEPA 2-AC backbone.

        The hub backbone expects ``[B, C, T, H, W]`` (channels-first video)
        and returns patch tokens directly — no CLS, no wrapper object.

        Args:
            frames: ``[B, T, C, H, W]`` ImageNet-normalised (produced by
                ``VJEPA2Encoder.encode``).

        Returns:
            ``[B, N_total, D]`` where ``N_total = T_p * H_p * W_p``.
        """
        # Permute [B, T, C, H, W] → [B, C, T, H, W] for the hub backbone.
        video = frames.permute(0, 2, 1, 3, 4).contiguous()

        out = self.backbone(video)

        # The hub backbone may return a raw tensor, a tuple, or a dataclass.
        if isinstance(out, torch.Tensor):
            tokens = out
        elif isinstance(out, (tuple, list)):
            tokens = out[0]
        elif hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        else:
            raise ValueError(
                f"VJEPA2ACEncoder: unexpected backbone output type {type(out)}."
            )

        # Expected shape: [B, N_total, D].  If the backbone prepends a CLS
        # token, strip it (match V-JEPA2 parent behaviour).
        num_patches_expected = (
            (frames.shape[1] // self._temporal_patch)
            * (frames.shape[3] // self._spatial_patch)
            * (frames.shape[4] // self._spatial_patch)
        )
        if tokens.dim() == 3 and tokens.shape[1] > num_patches_expected:
            tokens = tokens[:, -num_patches_expected:, :]
        return tokens


_ENCODER_REGISTRY["vjepa2_ac"] = VJEPA2ACEncoder


# ========================================================================== #
# V-JEPA 2.1 (dense-feature video encoder)
# ========================================================================== #

_VJEPA21_MODEL_SPECS = {
    "vjepa2_1_vit_base_384": {
        "builder": "vit_base",
        "checkpoint_key": "ema_encoder",
        "checkpoint_file": "vjepa2_1_vitb_dist_vitG_384.pt",
        "hidden_dim": 768,
    },
    "vjepa2_1_vit_large_384": {
        "builder": "vit_large",
        "checkpoint_key": "ema_encoder",
        "checkpoint_file": "vjepa2_1_vitl_dist_vitG_384.pt",
        "hidden_dim": 1024,
    },
    "vjepa2_1_vit_giant_384": {
        "builder": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
        "checkpoint_file": "vjepa2_1_vitg_384.pt",
        "hidden_dim": 1408,
    },
    "vjepa2_1_vit_gigantic_384": {
        "builder": "vit_gigantic_xformers",
        "checkpoint_key": "target_encoder",
        "checkpoint_file": "vjepa2_1_vitG_384.pt",
        "hidden_dim": 1664,
    },
}


class VJEPA21Encoder(VJEPA2Encoder):
    """Frozen V-JEPA 2.1 encoder loaded from the official local checkpoint."""

    def __init__(
        self,
        model_name: str = "vjepa2_1_vit_gigantic_384",
        output_dim: int = 48,
        mlp_hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        spatial_downsample: int = 16,
        temporal_downsample: int = 4,
        standardise_output: bool = True,
        skip_projection: bool = False,
        causal_tubelet_encoding: bool = False,
        causal_prefix_encoding: bool = False,
        checkpoint_source: str = "local",
        checkpoint_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        normalise_stats_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        nn.Module.__init__(self)
        if checkpoint_source != "local":
            raise ValueError(
                "VJEPA21Encoder currently requires checkpoint_source='local' so all "
                "distributed ranks load the same pre-downloaded checkpoint safely."
            )

        spec = _VJEPA21_MODEL_SPECS.get(model_name)
        if spec is None:
            raise ValueError(
                f"Unknown V-JEPA 2.1 model '{model_name}'. "
                f"Available: {sorted(_VJEPA21_MODEL_SPECS)}"
            )
        if checkpoint_path is None:
            raise ValueError(
                "VJEPA21Encoder requires checkpoint_path to the official "
                f"{spec['checkpoint_file']} checkpoint."
            )

        self.model_name = model_name
        self.standardise_output = standardise_output
        self.skip_projection = skip_projection
        self._freeze_backbone = freeze_backbone
        self.requires_independent_first_frame = True
        self.causal_tubelet_encoding = causal_tubelet_encoding
        self.causal_prefix_encoding = causal_prefix_encoding
        self.checkpoint_source = checkpoint_source
        self.checkpoint_path = checkpoint_path
        self.repo_path = repo_path
        self._spatial_patch = 16
        self._temporal_patch = 2

        self.backbone = self._load_vjepa21_backbone(
            spec=spec,
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            dtype=torch_dtype,
        )
        self.processor = None
        self._hidden_dim = int(spec["hidden_dim"])
        actual_hidden_dim = int(getattr(self.backbone, "embed_dim", -1))
        if actual_hidden_dim != self._hidden_dim:
            raise ValueError(
                f"V-JEPA 2.1 hidden dim mismatch for {model_name}: "
                f"expected {self._hidden_dim}, got {actual_hidden_dim}."
            )

        self.output_dim = self._hidden_dim if skip_projection else output_dim
        self.z_dim = self.output_dim
        self.upsampling_factor = spatial_downsample
        self.temporal_downsample_factor = temporal_downsample
        self._configure_fixed_output_normalisation(
            normalise_stats_path,
            num_channels=self.output_dim,
        )

        logger.info(
            "VJEPA21Encoder: model=%s hidden_dim=%d output_dim=%d "
            "skip_projection=%s freeze=%s checkpoint=%s",
            model_name,
            self._hidden_dim,
            self.output_dim,
            skip_projection,
            freeze_backbone,
            checkpoint_path,
        )

        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if skip_projection:
            self.projection = nn.Identity()
        else:
            hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else 2 * self._hidden_dim
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            ).to(dtype=torch_dtype)

    @staticmethod
    def _load_vjepa21_backbone(
        spec: dict,
        checkpoint_path: str,
        repo_path: Optional[str],
        dtype: torch.dtype,
    ) -> nn.Module:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"V-JEPA 2.1 checkpoint not found: {checkpoint}")

        if repo_path is None:
            repo = Path(torch.hub.get_dir()) / "facebookresearch_vjepa2_main"
        else:
            repo = Path(repo_path).expanduser()
        repo = repo.resolve()
        model_file = repo / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"V-JEPA 2.1 source tree not found at {repo}. Expected {model_file}."
            )

        repo_string = str(repo)
        sys.path.insert(0, repo_string)
        try:
            module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
            utils_module = importlib.import_module("app.vjepa_2_1.models.utils.modules")
        finally:
            if sys.path[0] == repo_string:
                sys.path.pop(0)
            else:
                sys.path.remove(repo_string)

        module_path = Path(module.__file__).resolve()
        if repo not in module_path.parents:
            raise ImportError(
                f"Loaded V-JEPA 2.1 code from {module_path}, expected source under {repo}."
            )

        # The official RoPE helper computes positions in fp32, which promotes
        # q/k while v stays bf16 and makes SDPA reject the mixed dtypes. Keep
        # the trigonometry in fp32, then restore the attention tensor dtype.
        if not getattr(utils_module, "_fastwam_dtype_safe_rope", False):
            original_rotate = utils_module.rotate_queries_or_keys

            def _dtype_safe_rotate(x, pos, n_registers, has_cls_first):
                rotated = original_rotate(
                    x.float(),
                    pos.float(),
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                )
                return rotated.to(dtype=x.dtype)

            utils_module.rotate_queries_or_keys = _dtype_safe_rotate
            utils_module._fastwam_dtype_safe_rope = True

        builder = getattr(module, spec["builder"], None)
        if builder is None:
            raise ValueError(f"V-JEPA 2.1 source has no builder '{spec['builder']}'.")

        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            backbone = builder(
                patch_size=16,
                img_size=(384, 384),
                num_frames=64,
                tubelet_size=2,
                use_sdpa=True,
                use_SiLU=False,
                wide_SiLU=True,
                uniform_power=False,
                use_rope=True,
                img_temporal_dim_size=1,
                interpolate_rope=True,
            )
        finally:
            torch.set_default_dtype(previous_dtype)

        checkpoint_blob = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        checkpoint_key = spec["checkpoint_key"]
        if checkpoint_key not in checkpoint_blob:
            raise ValueError(
                f"V-JEPA 2.1 checkpoint {checkpoint} has no '{checkpoint_key}' state dict."
            )

        state_dict = {}
        for key, value in checkpoint_blob[checkpoint_key].items():
            clean_key = key
            for prefix in ("module.", "backbone.", "encoder."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            state_dict[clean_key] = value

        incompatible = backbone.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Failed to load V-JEPA 2.1 backbone cleanly: "
                f"missing={incompatible.missing_keys[:8]} "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        return backbone.to(dtype=dtype)

    def _extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        video = frames.permute(0, 2, 1, 3, 4).contiguous()
        output = self.backbone(video)
        if isinstance(output, torch.Tensor):
            tokens = output
        elif isinstance(output, (tuple, list)):
            tokens = output[0]
        elif hasattr(output, "last_hidden_state"):
            tokens = output.last_hidden_state
        else:
            raise ValueError(
                f"VJEPA21Encoder: unexpected backbone output type {type(output)}."
            )

        expected_tokens = (
            (frames.shape[1] // self._temporal_patch)
            * (frames.shape[3] // self._spatial_patch)
            * (frames.shape[4] // self._spatial_patch)
        )
        if tokens.ndim != 3:
            raise ValueError(
                f"VJEPA21Encoder expected [B,N,D] tokens, got {tuple(tokens.shape)}."
            )
        if tokens.shape[1] > expected_tokens:
            tokens = tokens[:, -expected_tokens:, :]
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"VJEPA21Encoder token count mismatch: got {tokens.shape[1]}, "
                f"expected {expected_tokens}."
            )
        return tokens


_ENCODER_REGISTRY["vjepa2_1"] = VJEPA21Encoder
_ENCODER_REGISTRY["vjepa21"] = VJEPA21Encoder
