"""JEPA predictor expert — deterministic next-frame latent predictor.

This module implements the **video expert** for the V-JEPA 2-AC predictor
recipe (the "W-1" design referenced by the ``run_*_vjepa2ac_predictor*`` launch
scripts).  Unlike :class:`fastwam.models.wan22.wan_video_dit.WanVideoDiT`, this
expert is **not** a flow-matching denoiser: it consumes the *clean*, frozen
V-JEPA 2-AC encoder latents of the context frames and **deterministically
predicts the latents of the next frame** (next-frame prediction in
representation space).  The training objective is an L1 regression loss in the
encoder's latent space (the V-JEPA 2-AC default), not flow matching.

Design choices (settled with the user):

* **Geometry**: native V-JEPA 2-AC predictor dims — ``hidden_dim=1024``,
  ``num_heads=16``, ``attn_head_dim=64`` (1024 attention width), ``24`` layers.
* **MoT-compatible blocks**: we reuse :class:`DiTBlock` so the predictor can
  share per-layer mixed self-attention with the Wan ``ActionDiT`` through
  :class:`fastwam.models.wan22.mot.MoT` running in **non-strict / tail-overlap**
  mode (q/k/v/o projection adapters bridge the 1024 vs 3072 attention widths).
* **No timestep / flow matching**: the predictor is a *plain* transformer.
  We follow the :class:`LanguageExpert` trick — initialise each block's AdaLN
  ``modulation`` so the gate rows are 1 and shift/scale rows are 0, then feed
  ``t_mod = 0``.  ``(modulation + t_mod)`` then makes every ``DiTBlock`` behave
  like a vanilla pre-norm transformer block.
* **Actions reach the predictor via shared MoT attention only** — the predictor
  itself does *not* embed actions; the joint attention mask lets each context
  frame attend to the corresponding action-expert tokens.
* **Projections align dims**: a ``Conv3d`` patchify maps encoder latents
  (``in_dim=1408``) → ``hidden_dim=1024``; a linear head maps ``1024`` →
  ``out_dim=1408`` so predictions live back in encoder-latent space.

The public contract mirrors ``WanVideoDiT`` (``pre_dit`` / ``post_dit`` /
``build_video_to_video_mask`` plus the ``blocks`` / ``num_heads`` /
``attn_head_dim`` / ``use_gradient_checkpointing`` attributes) so that
:class:`MoT` and :class:`FastWAMJepa` can drive it without special-casing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    create_group_causal_attn_mask,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


def precompute_freqs_cis_3d_even(dim: int, end: int = 1024, theta: float = 10000.0):
    """3D RoPE tables with an **even** per-axis split.

    Wan's :func:`precompute_freqs_cis_3d` splits ``attn_head_dim`` as
    ``(dim - 2*(dim//3), dim//3, dim//3)``. When ``dim//3`` is odd (e.g. the
    native V-JEPA 2-AC head dim of 64 → ``64//3 = 21``), each axis contributes
    ``floor(part/2)`` complex pairs and the total falls one short of
    ``dim//2`` — which makes :func:`rope_apply` mis-shape. We round the
    height/width axes down to an even size and give the remainder to the frame
    axis so the three even parts always sum to ``dim`` and their half-counts
    sum to exactly ``dim // 2``.
    """
    base = dim // 3
    if base % 2 != 0:
        base -= 1  # make h/w axes even
    f_dim = dim - 2 * base
    if f_dim % 2 != 0:
        # Shift one unit from frame axis to keep everything even (rare).
        f_dim -= 1
        base += 1
    f_freqs_cis = precompute_freqs_cis(f_dim, end, theta)
    h_freqs_cis = precompute_freqs_cis(base, end, theta)
    w_freqs_cis = precompute_freqs_cis(base, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


class JEPAHead(nn.Module):
    """Plain (time-agnostic) prediction head: LayerNorm → Linear → unpatchify.

    Maps each ``hidden_dim`` token to ``out_dim * prod(patch_size)`` values so
    that :meth:`JEPAPredictor.unpatchify` can restore the original latent grid.
    Unlike :class:`fastwam.models.wan22.wan_video_dit.Head`, there is no AdaLN
    timestep modulation — the predictor is deterministic.
    """

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        import math

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


class JEPAPredictor(nn.Module):
    """Deterministic next-frame latent predictor (MoT-compatible video expert)."""

    _MAX_FREQS_DEVICE_CACHE_ENTRIES = 16

    def _apply(self, fn):
        result = super()._apply(fn)
        cache = getattr(self, "_freqs_device_cache", None)
        if cache is not None:
            cache.clear()
        return result

    def __init__(
        self,
        hidden_dim: int = 1024,
        in_dim: int = 1408,
        out_dim: int = 1408,
        ffn_dim: int = 4096,
        text_dim: int = 4096,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 16,
        attn_head_dim: int = 64,
        num_layers: int = 24,
        video_attention_mask_mode: str = "per_frame_causal",
        action_group_causal_mask_mode: str = "group_diagonal",
        use_gradient_checkpointing: bool = False,
        use_text_context: bool = True,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        if num_heads * attn_head_dim != hidden_dim:
            # Not strictly required (DiTBlock decouples attn width from hidden),
            # but for the native JEPA geometry they coincide; warn loudly.
            logger.warning(
                "JEPAPredictor: hidden_dim=%d != num_heads*attn_head_dim=%d. "
                "The attention width (%d) will differ from the residual width (%d).",
                hidden_dim, num_heads * attn_head_dim, num_heads * attn_head_dim, hidden_dim,
            )

        self.hidden_dim = int(hidden_dim)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.text_dim = int(text_dim)
        self.patch_size = tuple(int(p) for p in patch_size)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.action_group_causal_mask_mode = str(action_group_causal_mask_mode)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_text_context = bool(use_text_context)
        # The deterministic predictor does not fuse / replace the first latent
        # frame with clean VAE features (that is a flow-matching idiom). The
        # model conditions on context frames through the causal attention mask.
        self.fuse_vae_embedding_in_latents = False

        # ---- Input projection (patchify) and output head -------------------- #
        # Conv3d acts as patchify + (1408 -> 1024) projection, mirroring
        # WanVideoDiT.patch_embedding so spatial/temporal handling is identical.
        self.patch_embedding = nn.Conv3d(
            self.in_dim, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.head = JEPAHead(self.hidden_dim, self.out_dim, self.patch_size, eps)

        # ---- Optional text cross-attention conditioning --------------------- #
        if self.use_text_context:
            self.text_embedding = nn.Sequential(
                nn.Linear(self.text_dim, self.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.text_embedding = None

        # ---- MoT-compatible transformer blocks ------------------------------ #
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=int(ffn_dim),
                    eps=eps,
                )
                for _ in range(int(num_layers))
            ]
        )
        # Make each DiTBlock a *plain* transformer block (no timestep semantics):
        # init AdaLN modulation so gate rows = 1 and shift/scale rows = 0, then
        # feed t_mod = 0 in pre_dit. See LanguageExpert for the same trick.
        for block in self.blocks:
            with torch.no_grad():
                block.modulation.zero_()
                block.modulation[:, 2, :].fill_(1.0)  # gate_msa
                block.modulation[:, 5, :].fill_(1.0)  # gate_mlp

        # 3D RoPE frequency tables (frame, height, width). Use the even-split
        # variant so non-multiple-of-6 head dims (e.g. native JEPA's 64) yield
        # exactly attn_head_dim/2 complex pairs.
        self.freqs = precompute_freqs_cis_3d_even(self.attn_head_dim)
        self._freqs_device_cache: OrderedDict[
            tuple[int, int, int, str, int | None], torch.Tensor
        ] = OrderedDict()

        logger.info(
            "JEPAPredictor: hidden=%d in=%d out=%d heads=%d×%d=%d layers=%d patch=%s "
            "video_mask=%s use_text=%s grad_ckpt=%s",
            self.hidden_dim, self.in_dim, self.out_dim, self.num_heads, self.attn_head_dim,
            self.num_heads * self.attn_head_dim, len(self.blocks), self.patch_size,
            self.video_attention_mask_mode, self.use_text_context, self.use_gradient_checkpointing,
        )

    # ------------------------------------------------------------------ #
    # patchify / unpatchify (identical convention to WanVideoDiT)
    # ------------------------------------------------------------------ #
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )

    # ------------------------------------------------------------------ #
    # Video self-attention mask (reused from WanVideoDiT semantics)
    # ------------------------------------------------------------------ #
    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        mode = self.video_attention_mask_mode
        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in "
                    f"`per_frame_causal` mode, got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {mode}")

    def build_video_to_action_mask(
        self,
        num_frames: int,
        tokens_per_frame: int,
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Frame-causal mask: context frame ``i`` attends to action tokens 0..i.

        When ``action_seq_len`` is exactly divisible by ``num_frames`` the mask
        delegates to :func:`create_group_causal_attn_mask` with
        ``self.action_group_causal_mask_mode`` (typically ``"group_diagonal"``
        or ``"causal"``). When *not* divisible — the common case in LIBERO where
        the trajectory has ``T`` actions but only ``T-1`` JEPA context frames —
        we fall back to a simple per-frame causal mask built without any
        divisibility constraint:

            mask[row, col] = (row // tokens_per_frame) >= col

        i.e. frame ``i`` can attend to action tokens ``0 .. i``.  The last
        ``action_seq_len - num_frames`` actions (those beyond the last context
        frame) are never visible to any frame, which is the correct causal
        behaviour.

        Returns shape ``[num_frames * tokens_per_frame, action_seq_len]``.
        """
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}")
        if action_seq_len <= 0:
            raise ValueError(f"`action_seq_len` must be positive, got {action_seq_len}")

        # Fast path: exact group structure available.
        if action_seq_len % num_frames == 0:
            return create_group_causal_attn_mask(
                num_temporal_groups=num_frames,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_seq_len // num_frames,
                mode=self.action_group_causal_mask_mode,
            ).to(device)

        # Fallback: frame-causal mask without divisibility constraint.
        # Frame i (0-indexed) attends to action token j when j <= i.
        video_seq_len = num_frames * tokens_per_frame
        # frame_idx[row] = which context frame this video token belongs to
        frame_idx = torch.arange(video_seq_len, device=device) // tokens_per_frame  # [S_vid]
        # action_idx[col] = action step index (one token per action step)
        action_idx = torch.arange(action_seq_len, device=device)  # [S_act]
        # mask[i, j] = True when frame_idx[i] >= action_idx[j]
        return (frame_idx.unsqueeze(1) >= action_idx.unsqueeze(0))  # [S_vid, S_act]

    # ------------------------------------------------------------------ #
    # MoT pre-/post-hook
    # ------------------------------------------------------------------ #
    def pre_dit(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Tokenise the clean context latents for the MoT self-attention loop.

        Args:
            x: ``[B, in_dim, T, H, W]`` clean encoder latents of the *context*
                frames (i.e. frames ``0 .. T-1`` whose successors are predicted).
            context: optional ``[B, L, text_dim]`` text/proprio context for
                cross-attention. Required iff ``use_text_context`` is True.
            context_mask: optional ``[B, L]`` boolean mask for ``context``.

        Returns:
            Dict with ``tokens``/``freqs``/``t_mod``/``context``/``context_mask``/``meta``.
        """
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by predictor patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )

        x = self.patchify(x)
        f, h, w = x.shape[2:]
        tokens_per_frame = h * w

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs_cache_key = (f, h, w, x_tokens.device.type, x_tokens.device.index)
        freqs = self._freqs_device_cache.get(freqs_cache_key)
        if freqs is None:
            freqs = torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(f * h * w, 1, -1).to(x_tokens.device)
            self._freqs_device_cache[freqs_cache_key] = freqs
            while len(self._freqs_device_cache) > self._MAX_FREQS_DEVICE_CACHE_ENTRIES:
                self._freqs_device_cache.popitem(last=False)
        else:
            self._freqs_device_cache.move_to_end(freqs_cache_key)

        # Plain-transformer modulation: t_mod = 0 with gate-initialised blocks.
        t_mod = torch.zeros((1, 6, self.hidden_dim), dtype=x_tokens.dtype, device=x_tokens.device)

        context_emb = None
        context_attn_mask = None
        if self.use_text_context:
            if context is None:
                raise ValueError("`context` is required when `use_text_context=True`.")
            if context.ndim != 3:
                raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
            if context_mask is None:
                context_mask = torch.ones(
                    (context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device
                )
            context_emb = self.text_embedding(context)
            seq_len = f * h * w
            context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens)
        return self.unpatchify(x, (f, h, w))

    # ------------------------------------------------------------------ #
    # Stand-alone forward (no MoT) — useful for unit tests / ablation.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre = self.pre_dit(x=x, context=context, context_mask=context_mask)
        x_tokens = pre["tokens"]
        freqs = pre["freqs"]
        t_mod = pre["t_mod"]
        ctx = pre["context"]
        ctx_mask = pre["context_mask"]
        self_attn_mask = (
            self.build_video_to_video_mask(
                video_seq_len=x_tokens.shape[1],
                video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),
                device=x_tokens.device,
            )
            if self.video_attention_mask_mode != "bidirectional"
            else None
        )
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, ctx, t_mod, freqs,
                    context_mask=ctx_mask, self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens, ctx, t_mod, freqs,
                    context_mask=ctx_mask, self_attn_mask=self_attn_mask,
                )
        return self.post_dit(x_tokens, pre)


# ====================================================================== #
# Best-effort loader for native V-JEPA 2-AC predictor weights.
# ====================================================================== #
#
# The torch.hub V-JEPA 2-AC bundle returns ``(encoder, predictor)``. The native
# predictor is a plain pre-LN ViT (timm-style ``blocks.N.{norm1,attn.qkv,
# attn.proj,norm2,mlp.fc1,mlp.fc2}`` plus ``predictor_embed`` / ``predictor_proj``
# / ``predictor_norm``), which is structurally different from our ``DiTBlock``
# (separate q/k/v with RMSNorm, AdaLN modulation, cross-attn). We therefore map
# only the structurally-compatible tensors and log coverage. Mapping must be
# validated on a node where the hub source + checkpoint are present.


def load_ac_predictor_weights(
    predictor: "JEPAPredictor",
    native_predictor: nn.Module | dict,
    strict: bool = False,
) -> dict[str, int]:
    """Map native V-JEPA 2-AC predictor weights onto :class:`JEPAPredictor`.

    Args:
        predictor: target module to receive weights (modified in place).
        native_predictor: the ``loaded[1]`` predictor ``nn.Module`` from
            ``torch.hub.load`` or its ``state_dict``.
        strict: if True, raise when coverage is suspiciously low.

    Returns:
        Coverage dict ``{"mapped": int, "total_target": int, "skipped": int}``.
    """
    if isinstance(native_predictor, nn.Module):
        src = dict(native_predictor.state_dict())
    elif isinstance(native_predictor, dict):
        src = dict(native_predictor)
    else:
        raise TypeError(
            f"`native_predictor` must be nn.Module or state_dict, got {type(native_predictor)}"
        )

    tgt = predictor.state_dict()
    new_state = dict(tgt)
    mapped = 0
    notes: list[str] = []

    def _try_copy(tgt_key: str, src_tensor: Optional[torch.Tensor]) -> bool:
        nonlocal mapped
        if src_tensor is None:
            return False
        if tgt_key not in tgt:
            return False
        if tuple(src_tensor.shape) != tuple(tgt[tgt_key].shape):
            notes.append(f"shape mismatch {tgt_key}: src{tuple(src_tensor.shape)} vs tgt{tuple(tgt[tgt_key].shape)}")
            return False
        new_state[tgt_key] = src_tensor.to(dtype=tgt[tgt_key].dtype)
        mapped += 1
        return True

    # Input projection: native ``predictor_embed`` (Linear over patch tokens)
    # vs our Conv3d patchify. Only map when the source is also a Conv3d-style
    # weight; otherwise leave the Conv3d randomly initialised.
    _try_copy("patch_embedding.weight", src.get("predictor_embed.weight"))
    _try_copy("patch_embedding.bias", src.get("predictor_embed.bias"))

    # Per-block mapping (timm ViT -> DiTBlock).
    num_layers = len(predictor.blocks)
    for i in range(num_layers):
        # Combined qkv -> separate q/k/v.
        qkv_w = src.get(f"blocks.{i}.attn.qkv.weight")
        qkv_b = src.get(f"blocks.{i}.attn.qkv.bias")
        if qkv_w is not None and qkv_w.shape[0] % 3 == 0:
            d = qkv_w.shape[0] // 3
            _try_copy(f"blocks.{i}.self_attn.q.weight", qkv_w[:d])
            _try_copy(f"blocks.{i}.self_attn.k.weight", qkv_w[d : 2 * d])
            _try_copy(f"blocks.{i}.self_attn.v.weight", qkv_w[2 * d :])
        if qkv_b is not None and qkv_b.shape[0] % 3 == 0:
            d = qkv_b.shape[0] // 3
            _try_copy(f"blocks.{i}.self_attn.q.bias", qkv_b[:d])
            _try_copy(f"blocks.{i}.self_attn.k.bias", qkv_b[d : 2 * d])
            _try_copy(f"blocks.{i}.self_attn.v.bias", qkv_b[2 * d :])

        _try_copy(f"blocks.{i}.self_attn.o.weight", src.get(f"blocks.{i}.attn.proj.weight"))
        _try_copy(f"blocks.{i}.self_attn.o.bias", src.get(f"blocks.{i}.attn.proj.bias"))

        # MLP.
        _try_copy(f"blocks.{i}.ffn.0.weight", src.get(f"blocks.{i}.mlp.fc1.weight"))
        _try_copy(f"blocks.{i}.ffn.0.bias", src.get(f"blocks.{i}.mlp.fc1.bias"))
        _try_copy(f"blocks.{i}.ffn.2.weight", src.get(f"blocks.{i}.mlp.fc2.weight"))
        _try_copy(f"blocks.{i}.ffn.2.bias", src.get(f"blocks.{i}.mlp.fc2.bias"))

        # NOTE: native norm1/norm2 are affine LayerNorms; our DiTBlock norm1/norm2
        # are elementwise_affine=False (AdaLN handles scale/shift). We cannot copy
        # their affine params without folding them into the adjacent linears, so we
        # leave the predictor's pre-norms as identity-affine. This is a known
        # approximation of the native predictor — validate downstream.

    predictor.load_state_dict(new_state, strict=False)
    total = len(tgt)
    coverage = {"mapped": mapped, "total_target": total, "skipped": total - mapped}
    logger.info(
        "load_ac_predictor_weights: mapped %d/%d target tensors (%.1f%%). %d notes.",
        mapped, total, 100.0 * mapped / max(total, 1), len(notes),
    )
    for n in notes[:20]:
        logger.info("  weight-map note: %s", n)
    if strict and mapped < 0.3 * total:
        raise RuntimeError(
            f"load_ac_predictor_weights: low coverage ({mapped}/{total}). "
            "Native predictor naming likely differs from the timm ViT assumption."
        )
    return coverage
