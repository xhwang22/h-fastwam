"""V-JEPA-AC predictor compatible self-attention with 3D RoPE.

This module faithfully reproduces ``ACRoPEAttention`` from
``facebookresearch/vjepa2``, including its head_dim splitting into
(d_dim, h_dim, w_dim) sub-axes for separate temporal/H/W RoPE rotations,
and the documented "buggy expansion" that V-JEPA 2-AC was trained with.

Structural choices (vs Wan-style ``SelfAttention``):
  * Wan keeps three independent linear modules (q/k/v) and a single output
    projection (o); it then applies a 1D complex-RoPE to the full head dim.
  * V-JEPA-AC uses a *fused* qkv linear and slices head_dim into
    (d_dim, h_dim, w_dim) channels each rotated by a *separate* 2D
    sinusoidal RoPE keyed off frame/H/W positions, with any leftover
    channels left un-rotated.

To keep MoT's joint-attention interface intact (it calls ``block.self_attn.q``,
``block.self_attn.k``, ``block.self_attn.v`` separately, then rope_apply),
we expose ``q``, ``k``, ``v`` as standalone ``nn.Linear`` modules whose
weights mirror slices of the V-JEPA fused ``qkv.weight`` after loading.
The actual RoPE rotation is *not* applied here — instead it is implemented
as a callable ``apply_rope_to_qk`` that MoT will invoke after the q/k linear
projections (replacing the current ``rope_apply`` call for this expert).

This module also provides a ``load_from_vjepa_qkv`` helper that takes a
fused ``qkv.weight`` (shape [3*D, D]) and an ``proj.weight`` from a V-JEPA
predictor block and writes them into the q/k/v/o linear modules here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _vjepa_rope_dims(head_dim: int) -> tuple[int, int, int]:
    """Compute V-JEPA-AC's (d_dim, h_dim, w_dim) split of head_dim.

    Reproduces ``ACRoPEAttention.__init__``::

        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))

    For head_dim=64:  d=h=w=20, leftover=4 (un-rotated).
    For head_dim=72:  d=h=w=24, leftover=0.
    """
    chunk = int(2 * ((head_dim // 3) // 2))
    return chunk, chunk, chunk


def rotate_queries_or_keys_buggy(
    x: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """Reproduce V-JEPA's ``rotate_queries_or_keys`` *with* the documented bug.

    Per upstream comment in ``modules.py``: the expansion duplicates frequencies
    across the (sin/cos) vector pair. Fixing the bug breaks pretrained weights.
    We faithfully reproduce the buggy version.

    Args:
        x:   [B, num_heads, N, D] where D is one sub-axis size (d_dim/h_dim/w_dim).
             D must be even.
        pos: [B, num_heads, N] or broadcastable position indices (float).

    Returns:
        Rotated tensor of same shape as x.
    """
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Sub-axis dim must be even for RoPE."
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega = omega / (D / 2.0)
    omega = 1.0 / (10000.0 ** omega)                           # [D/2]
    freq = torch.einsum("..., f -> ... f", pos.to(x.dtype), omega)  # [..., N, D/2]

    emb_sin = freq.sin()
    emb_cos = freq.cos()
    # buggy expansion: squeeze last dim then repeat-pair.
    # Mirrors upstream:
    #     emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    # which is equivalent (when shapes match) to ``.repeat(1,1,1,2)``.
    emb_sin = emb_sin.repeat(1, 1, 1, 2)
    emb_cos = emb_cos.repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))        # [..., D/2, 2]
    y1, y2 = y.unbind(dim=-1)           # each [..., D/2]
    y = torch.stack((-y2, y1), dim=-1)  # [..., D/2, 2]
    y = y.flatten(-2)                   # [..., D]
    return (x * emb_cos) + (y * emb_sin)


def vjepa_apply_rope_to_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    T: int,
    H: int,
    W: int,
    grid_size: int = 16,
    prefix_per_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply V-JEPA-AC 3D RoPE to query/key tensors.

    Tokens are arranged per-timestep as
        [prefix_0, ..., prefix_{prefix_per_step-1}, vid_patch_0, ..., vid_patch_{H*W-1}]
    repeated for T timesteps, total length T * (prefix_per_step + H*W).

    Frame tokens get full (d_dim, h_dim, w_dim) RoPE; prefix tokens
    only get the temporal (d_dim) rotation, matching how V-JEPA-AC
    handles its action/state/extrinsics prefix tokens.

    Args:
        q, k: [B, S, num_heads * head_dim] with S = T * (prefix_per_step + H*W).
        num_heads, T, H, W: as in V-JEPA-AC's ``ACRoPEAttention.forward``.
        grid_size: 16 by default (V-JEPA's grid normalisation).
        prefix_per_step: number of prefix tokens per timestep (k=3 for our
            text-prefix design; matches V-JEPA-AC's action_tokens=3 path).

    Returns:
        (q_rotated, k_rotated) with same shapes as inputs.
    """
    B, S, full = q.shape
    head_dim = full // num_heads
    d_dim, h_dim, w_dim = _vjepa_rope_dims(head_dim)

    expected_S = T * (prefix_per_step + H * W)
    if S != expected_S:
        raise ValueError(
            f"vjepa_apply_rope_to_qk: token length {S} does not match "
            f"T*(prefix+HW) = {T}*({prefix_per_step}+{H*W}) = {expected_S}."
        )

    # Reshape to [B, num_heads, S, head_dim] (V-JEPA convention).
    def _split_heads(x):
        return x.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(x):
        return x.permute(0, 2, 1, 3).contiguous().view(B, S, num_heads * head_dim)

    q4 = _split_heads(q)  # [B, H, S, D]
    k4 = _split_heads(k)

    # Split prefix vs frame tokens.
    if prefix_per_step > 0:
        # Reshape S into [T, prefix+H*W] so we can pull prefix slots out cleanly.
        q4 = q4.view(B, num_heads, T, prefix_per_step + H * W, head_dim)
        k4 = k4.view(B, num_heads, T, prefix_per_step + H * W, head_dim)
        q_prefix = q4[:, :, :, :prefix_per_step, :]   # [B, H, T, P, D]
        k_prefix = k4[:, :, :, :prefix_per_step, :]
        q_frame = q4[:, :, :, prefix_per_step:, :]    # [B, H, T, H*W, D]
        k_frame = k4[:, :, :, prefix_per_step:, :]

        # Prefix: rotate temporal sub-axis only.
        # pos shape needed: [B, H, T*P]
        prefix_pos = (
            torch.arange(T, device=q.device)
            .view(1, 1, T, 1)
            .expand(B, num_heads, T, prefix_per_step)
            .reshape(B, num_heads, T * prefix_per_step)
            .float()
        )
        q_prefix_flat = q_prefix.reshape(B, num_heads, T * prefix_per_step, head_dim)
        k_prefix_flat = k_prefix.reshape(B, num_heads, T * prefix_per_step, head_dim)
        qd_p = rotate_queries_or_keys_buggy(q_prefix_flat[..., :d_dim], prefix_pos)
        kd_p = rotate_queries_or_keys_buggy(k_prefix_flat[..., :d_dim], prefix_pos)
        q_prefix_flat = torch.cat([qd_p, q_prefix_flat[..., d_dim:]], dim=-1)
        k_prefix_flat = torch.cat([kd_p, k_prefix_flat[..., d_dim:]], dim=-1)
        q_prefix = q_prefix_flat.view(B, num_heads, T, prefix_per_step, head_dim)
        k_prefix = k_prefix_flat.view(B, num_heads, T, prefix_per_step, head_dim)

        # Frame: full 3D RoPE.
        q_frame_flat = q_frame.reshape(B, num_heads, T * H * W, head_dim)
        k_frame_flat = k_frame.reshape(B, num_heads, T * H * W, head_dim)
    else:
        q_frame_flat = q4.view(B, num_heads, T * H * W, head_dim)
        k_frame_flat = k4.view(B, num_heads, T * H * W, head_dim)

    # Build d/h/w masks for frame tokens (same as ACRoPEAttention.forward).
    ids = torch.arange(T * H * W, device=q.device)
    tokens_per_frame = H * W
    frame_ids = (ids // tokens_per_frame).float()                           # [N]
    height_ids = ((ids - tokens_per_frame * (ids // tokens_per_frame)) // W).float()
    width_ids = (
        ids - tokens_per_frame * (ids // tokens_per_frame)
        - W * ((ids - tokens_per_frame * (ids // tokens_per_frame)) // W)
    ).float()
    # Snap spatial positions to grid_size (V-JEPA scaling).
    height_ids = height_ids * (grid_size / H)
    width_ids = width_ids * (grid_size / W)
    # Broadcast to [B, num_heads, N]
    d_mask = frame_ids.view(1, 1, -1).expand(B, num_heads, -1)
    h_mask = height_ids.view(1, 1, -1).expand(B, num_heads, -1)
    w_mask = width_ids.view(1, 1, -1).expand(B, num_heads, -1)

    # Rotate per sub-axis.
    s = 0
    qd = rotate_queries_or_keys_buggy(q_frame_flat[..., s : s + d_dim], d_mask)
    kd = rotate_queries_or_keys_buggy(k_frame_flat[..., s : s + d_dim], d_mask)
    s += d_dim
    qh = rotate_queries_or_keys_buggy(q_frame_flat[..., s : s + h_dim], h_mask)
    kh = rotate_queries_or_keys_buggy(k_frame_flat[..., s : s + h_dim], h_mask)
    s += h_dim
    qw = rotate_queries_or_keys_buggy(q_frame_flat[..., s : s + w_dim], w_mask)
    kw = rotate_queries_or_keys_buggy(k_frame_flat[..., s : s + w_dim], w_mask)
    s += w_dim
    if s < head_dim:
        q_frame_flat = torch.cat([qd, qh, qw, q_frame_flat[..., s:]], dim=-1)
        k_frame_flat = torch.cat([kd, kh, kw, k_frame_flat[..., s:]], dim=-1)
    else:
        q_frame_flat = torch.cat([qd, qh, qw], dim=-1)
        k_frame_flat = torch.cat([kd, kh, kw], dim=-1)

    if prefix_per_step > 0:
        q_frame = q_frame_flat.view(B, num_heads, T, H * W, head_dim)
        k_frame = k_frame_flat.view(B, num_heads, T, H * W, head_dim)
        # Stitch prefix + frame back into [B, H, T, P+HW, D] then flatten T*(P+HW).
        q_full = torch.cat([q_prefix, q_frame], dim=3).reshape(
            B, num_heads, T * (prefix_per_step + H * W), head_dim
        )
        k_full = torch.cat([k_prefix, k_frame], dim=3).reshape(
            B, num_heads, T * (prefix_per_step + H * W), head_dim
        )
    else:
        q_full = q_frame_flat
        k_full = k_frame_flat

    return _merge_heads(q_full), _merge_heads(k_full)
