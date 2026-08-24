from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .action_dit import ActionDiT
from .helpers.gradient import gradient_checkpoint_forward


class LatentActionDiT(ActionDiT):
    """Flow-matching DiT over one latent-action token per transition."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        latent_horizon: int,
        ffn_dim: int,
        context_dim: int,
        context_spatial_pool: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        if latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {latent_dim}.")
        if latent_horizon <= 0:
            raise ValueError(
                f"`latent_horizon` must be positive, got {latent_horizon}."
            )
        if context_dim <= 0:
            raise ValueError(f"`context_dim` must be positive, got {context_dim}.")
        if context_spatial_pool <= 0:
            raise ValueError(
                "`context_spatial_pool` must be positive, "
                f"got {context_spatial_pool}."
            )
        super().__init__(
            hidden_dim=hidden_dim,
            action_dim=latent_dim,
            ffn_dim=ffn_dim,
            text_dim=context_dim,
            freq_dim=freq_dim,
            eps=eps,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.latent_dim = int(latent_dim)
        self.latent_horizon = int(latent_horizon)
        self.context_dim = int(context_dim)
        self.context_spatial_pool = int(context_spatial_pool)
        self.video_attention_mask_mode = "bidirectional"
        self.use_text_context = True
        self.fuse_vae_embedding_in_latents = False

    def _prepare_visual_context(
        self,
        context_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_latents.ndim != 5:
            raise ValueError(
                "`context_latents` must be [B,D,T,H,W], "
                f"got {tuple(context_latents.shape)}."
            )
        if context_latents.shape[1] != self.context_dim:
            raise ValueError(
                f"V-JEPA context dim must be {self.context_dim}, "
                f"got {context_latents.shape[1]}."
            )
        if context_latents.shape[2] != 1:
            raise ValueError(
                "LatentActionDiT requires first-frame context with T=1, "
                f"got T={context_latents.shape[2]}."
            )

        pool = self.context_spatial_pool
        height, width = context_latents.shape[-2:]
        if height % pool != 0 or width % pool != 0:
            raise ValueError(
                "V-JEPA context grid must be divisible by context_spatial_pool: "
                f"grid={(height, width)}, pool={pool}."
            )
        if pool > 1:
            context_latents = F.avg_pool3d(
                context_latents,
                kernel_size=(1, pool, pool),
                stride=(1, pool, pool),
            )
        context = context_latents.flatten(2).transpose(1, 2).contiguous()
        context_mask = torch.ones(
            context.shape[:2],
            dtype=torch.bool,
            device=context.device,
        )
        return context, context_mask

    def pre_dit(
        self,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context_latents: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if latent_tokens.ndim != 3:
            raise ValueError(
                "`latent_tokens` must be [B,T,D], "
                f"got {tuple(latent_tokens.shape)}."
            )
        if latent_tokens.shape[1] != self.latent_horizon:
            raise ValueError(
                f"Latent-action horizon must be {self.latent_horizon}, "
                f"got {latent_tokens.shape[1]}."
            )
        if latent_tokens.shape[2] != self.latent_dim:
            raise ValueError(
                f"Latent-action dim must be {self.latent_dim}, "
                f"got {latent_tokens.shape[2]}."
            )
        if context_mask is not None:
            raise ValueError(
                "`context_mask` is derived from the V-JEPA grid and must be None."
            )
        context, derived_context_mask = self._prepare_visual_context(context_latents)
        state = super().pre_dit(
            action_tokens=latent_tokens,
            timestep=timestep,
            context=context,
            context_mask=derived_context_mask,
        )
        state["meta"]["tokens_per_frame"] = self.latent_horizon
        state["meta"]["context_tokens"] = int(context.shape[1])
        return state

    @staticmethod
    def build_video_to_video_mask(
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(
                f"`video_seq_len` must be positive, got {video_seq_len}."
            )
        if int(video_tokens_per_frame) != int(video_seq_len):
            raise ValueError(
                "A latent-action trajectory is one bidirectional token block: "
                f"tokens_per_frame={video_tokens_per_frame}, seq_len={video_seq_len}."
            )
        return torch.ones(
            (video_seq_len, video_seq_len),
            dtype=torch.bool,
            device=device,
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context_latents: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.pre_dit(
            latent_tokens=latent_tokens,
            timestep=timestep,
            context_latents=context_latents,
            context_mask=context_mask,
        )
        x = state["tokens"]
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    state["context"],
                    state["t_mod"],
                    state["freqs"],
                    context_mask=state["context_mask"],
                )
            else:
                x = block(
                    x,
                    state["context"],
                    state["t_mod"],
                    state["freqs"],
                    context_mask=state["context_mask"],
                )
        return self.post_dit(x, state)
