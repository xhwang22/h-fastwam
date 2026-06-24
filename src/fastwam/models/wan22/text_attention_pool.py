"""Attention pool for compressing variable-length text embeddings to a small
fixed number of conditioning tokens.

Usage in V-JEPA-AC predictor video expert: T5 cache (128 tokens × 4096) is
pooled to k=3 tokens × 1024, mirroring the (action, state, extrinsics) prefix
slot count in the original V-JEPA-AC AC predictor (3 prefix tokens / timestep).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TextAttentionPool(nn.Module):
    """Pool a variable-length [B, L, in_dim] sequence to k learned [B, k, out_dim] tokens
    via a single multi-head attention layer with learned queries.

    Args:
        num_queries: Number of output tokens (k). Default 3, matching V-JEPA-AC's
            (action, state, extrinsics) per-timestep prefix slot count.
        in_dim: Input dim (e.g. 4096 for T5 / Wan2.2 text encoder).
        out_dim: Output dim (e.g. 1024 to match V-JEPA-AC predictor hidden_dim).
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        num_queries: int = 3,
        in_dim: int = 4096,
        out_dim: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, out_dim) * 0.02)
        self.kv_proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            text: [B, L, in_dim] text embeddings (e.g. T5 cache).
            text_mask: [B, L] bool, True where valid (non-pad). If None, all tokens valid.

        Returns:
            [B, num_queries, out_dim]
        """
        B = text.shape[0]
        kv = self.kv_proj(text)                            # [B, L, out_dim]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)    # [B, k, out_dim]
        # nn.MultiheadAttention's key_padding_mask: True positions are *ignored*.
        key_padding_mask = (~text_mask) if text_mask is not None else None
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(out)                              # [B, k, out_dim]
