"""Visual encoders — drop-in replacements for WanVideoVAE38 encoder.

Supported backends:

* **DINOv3 / DINOv2** — frozen image ViT, per-frame encoding + temporal stride.
* **V-JEPA 2** — frozen video ViT, native spatiotemporal encoding.

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

import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

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
        encoder_type: ``"dino"`` or ``"vjepa2"``.
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
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.standardise_output = standardise_output
        self.skip_projection = skip_projection
        self._freeze_backbone = freeze_backbone

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
        H_lat = H // self.upsampling_factor
        W_lat = W // self.upsampling_factor

        # V-JEPA 2 requires at least `_temporal_patch` frames (typically 2).
        # For single-frame input (inference), repeat the frame to meet the
        # minimum and keep only the first temporal output afterwards.
        _single_frame = T < self._temporal_patch
        if _single_frame:
            pad_T = self._temporal_patch
            videos = videos.expand(B, C, pad_T, H, W).contiguous()
            T = pad_T

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

        # Temporal alignment to VAE convention: T_lat = (T-1)//temp_ds + 1
        # V-JEPA already did temporal_patch (2×) downsampling → T_p = T//2.
        # Need to further stride to match target T_lat.
        T_lat_target = (T - 1) // self.temporal_downsample_factor + 1
        T_current = projected.shape[2]

        if T_current > T_lat_target:
            # Additional temporal stride
            extra_stride = max(1, ceil(T_current / T_lat_target))
            projected = projected[:, :, ::extra_stride]
            # Trim to exact T_lat if needed
            projected = projected[:, :, :T_lat_target]
        elif T_current < T_lat_target:
            # Upsample temporally (rare — only if T_p < T_lat)
            projected = F.interpolate(
                projected.float().reshape(B * self.output_dim, 1, T_current, H_lat, W_lat),
                size=(T_lat_target, H_lat, W_lat),
                mode="trilinear",
                align_corners=False,
            ).to(dtype=projected.dtype).reshape(B, self.output_dim, T_lat_target, H_lat, W_lat)

        # Single-frame input: keep only the first temporal slice.
        if _single_frame:
            projected = projected[:, :, :1]

        # In skip_projection mode: no SIGReg, but standardise is still
        # useful to normalise features to mean=0 std=1 per channel.
        if self.skip_projection:
            if self.standardise_output:
                projected = self._channel_standardise(projected)
            if return_pre_standardise:
                return projected, None
            return projected

        if return_pre_standardise:
            latents_raw = projected
            if self.standardise_output:
                projected = self._channel_standardise(projected)
            return projected, latents_raw

        if self.standardise_output:
            projected = self._channel_standardise(projected)
        return projected

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
