"""Standalone action decoder for latent action tokens and current state."""

from __future__ import annotations

from numbers import Real

import torch
import torch.nn as nn


class LatentActionDecoder(nn.Module):
    """Decode latent action tokens into fixed-size action substeps.

    The decoder only conditions on the supplied latent tokens and current state.
    ``visual_state`` may use one of these explicit layouts:

    - ``[B, C, 1, H, W]``: channel-first single-frame feature map.
    - ``[B, C, H, W]``: channel-first feature map with the singleton time axis removed.
    - ``[B, N, C]``: flattened, channel-last spatial tokens.
    - ``[B, C, N]``: flattened, channel-first spatial tokens.
    - ``[B, C]``: one pooled visual token.

    Here ``C`` must equal ``visual_dim``. The canonical five-dimensional form
    requires a singleton time axis so future visual context cannot be supplied.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        proprio_dim: int = 14,
        visual_dim: int = 1664,
        num_latents: int = 8,
        substeps_per_latent: int = 4,
        action_dim: int = 14,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        dimensions = {
            "latent_dim": latent_dim,
            "proprio_dim": proprio_dim,
            "visual_dim": visual_dim,
            "num_latents": num_latents,
            "substeps_per_latent": substeps_per_latent,
            "action_dim": action_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")
        if d_model % nhead != 0:
            raise ValueError(f"`d_model` ({d_model}) must be divisible by `nhead` ({nhead}).")
        if not isinstance(dropout, Real) or isinstance(dropout, bool) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout!r}.")

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.visual_dim = visual_dim
        self.num_latents = num_latents
        self.substeps_per_latent = substeps_per_latent
        self.action_dim = action_dim
        self.d_model = d_model

        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.proprio_projection = nn.Linear(proprio_dim, d_model)
        self.visual_projection = nn.Linear(visual_dim, d_model)
        self.latent_positions = nn.Parameter(torch.empty(num_latents, d_model))
        self.substep_queries = nn.Parameter(torch.empty(substeps_per_latent, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.action_projection = nn.Linear(d_model, action_dim)
        nn.init.normal_(self.latent_positions, mean=0.0, std=d_model**-0.5)
        nn.init.normal_(self.substep_queries, mean=0.0, std=d_model**-0.5)

    @staticmethod
    def _require_floating_tensor(name: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value).__name__}.")
        if not value.is_floating_point():
            raise TypeError(f"`{name}` must have a floating-point dtype, got {value.dtype}.")

    def _flatten_visual_state(
        self,
        visual_state: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        self._require_floating_tensor("visual_state", visual_state)
        if visual_state.ndim not in (2, 3, 4, 5):
            raise ValueError(
                "`visual_state` must be [B,C,1,H,W], [B,C,H,W], [B,N,C], "
                f"[B,C,N], or [B,C]; got {tuple(visual_state.shape)}."
            )
        if visual_state.shape[0] != batch_size:
            raise ValueError(
                "`visual_state` batch size must match `latent`: "
                f"got {visual_state.shape[0]} and {batch_size}."
            )

        if visual_state.ndim == 5:
            _, channels, time, height, width = visual_state.shape
            if channels != self.visual_dim or time != 1 or height <= 0 or width <= 0:
                raise ValueError(
                    "`visual_state` with 5 dimensions must have shape "
                    f"[B, {self.visual_dim}, 1, H, W] with H,W > 0, got "
                    f"{tuple(visual_state.shape)}."
                )
            return visual_state.permute(0, 2, 3, 4, 1).reshape(
                batch_size, time * height * width, channels
            )

        if visual_state.ndim == 4:
            _, channels, height, width = visual_state.shape
            if channels != self.visual_dim or height <= 0 or width <= 0:
                raise ValueError(
                    "`visual_state` with 4 dimensions must have shape "
                    f"[B, {self.visual_dim}, H, W] with H,W > 0, got "
                    f"{tuple(visual_state.shape)}."
                )
            return visual_state.permute(0, 2, 3, 1).reshape(
                batch_size, height * width, channels
            )

        if visual_state.ndim == 3:
            token_axis_is_channels = visual_state.shape[1] == self.visual_dim
            last_axis_is_channels = visual_state.shape[2] == self.visual_dim
            if token_axis_is_channels == last_axis_is_channels:
                raise ValueError(
                    "`visual_state` with 3 dimensions must be unambiguously "
                    f"[B, N, {self.visual_dim}] or [B, {self.visual_dim}, N], got "
                    f"{tuple(visual_state.shape)}."
                )
            if visual_state.shape[1] <= 0 or visual_state.shape[2] <= 0:
                raise ValueError("`visual_state` must contain at least one spatial token.")
            return visual_state.transpose(1, 2) if token_axis_is_channels else visual_state

        if visual_state.ndim == 2:
            if visual_state.shape[1] != self.visual_dim:
                raise ValueError(
                    "`visual_state` with 2 dimensions must have shape "
                    f"[B, {self.visual_dim}], got {tuple(visual_state.shape)}."
                )
            return visual_state.unsqueeze(1)

        raise ValueError(
            "`visual_state` must be [B,C,1,H,W], [B,C,H,W], [B,N,C], "
            f"[B,C,N], or [B,C]; got {tuple(visual_state.shape)}."
        )

    def forward(
        self,
        latent: torch.Tensor,
        current_proprio: torch.Tensor,
        visual_state: torch.Tensor,
        *,
        latent_is_pad: torch.Tensor | None = None,
        flatten_output: bool = False,
    ) -> torch.Tensor:
        """Return actions as ``[B, num_latents, substeps, action_dim]``.

        Set ``flatten_output=True`` to return
        ``[B, num_latents * substeps, action_dim]`` instead.
        """
        self._require_floating_tensor("latent", latent)
        self._require_floating_tensor("current_proprio", current_proprio)
        if latent.ndim != 3 or latent.shape[1:] != (self.num_latents, self.latent_dim):
            raise ValueError(
                f"`latent` must have shape [B, {self.num_latents}, {self.latent_dim}], "
                f"got {tuple(latent.shape)}."
            )
        batch_size = latent.shape[0]
        if batch_size <= 0:
            raise ValueError("`latent` batch size must be positive.")
        if current_proprio.shape != (batch_size, self.proprio_dim):
            raise ValueError(
                f"`current_proprio` must have shape [B, {self.proprio_dim}] with B={batch_size}, "
                f"got {tuple(current_proprio.shape)}."
            )
        if not isinstance(flatten_output, bool):
            raise TypeError(f"`flatten_output` must be bool, got {type(flatten_output).__name__}.")

        visual_tokens = self._flatten_visual_state(visual_state, batch_size)
        devices = {latent.device, current_proprio.device, visual_state.device}
        if len(devices) != 1:
            raise ValueError(
                "`latent`, `current_proprio`, and `visual_state` must be on the same device."
            )

        substep_is_pad = None
        safe_substep_is_pad = None
        if latent_is_pad is not None:
            if not torch.is_tensor(latent_is_pad):
                raise TypeError(
                    f"`latent_is_pad` must be a torch.Tensor, got {type(latent_is_pad).__name__}."
                )
            if latent_is_pad.dtype != torch.bool or latent_is_pad.shape != (
                batch_size,
                self.num_latents,
            ):
                raise ValueError(
                    "`latent_is_pad` must be bool with shape "
                    f"[B, {self.num_latents}], got {tuple(latent_is_pad.shape)} "
                    f"with dtype {latent_is_pad.dtype}."
                )
            if latent_is_pad.device != latent.device:
                raise ValueError("`latent_is_pad` must be on the same device as `latent`.")
            substep_is_pad = latent_is_pad.repeat_interleave(
                self.substeps_per_latent,
                dim=1,
            )
            safe_substep_is_pad = substep_is_pad.clone()
            fully_padded = safe_substep_is_pad.all(dim=1)
            safe_substep_is_pad[fully_padded, 0] = False

        latent_queries = (
            self.latent_projection(latent)
            + self.latent_positions.unsqueeze(0)
        ).unsqueeze(2)
        queries = latent_queries + self.substep_queries.view(
            1, 1, self.substeps_per_latent, self.d_model
        )
        queries = queries.reshape(
            batch_size, self.num_latents * self.substeps_per_latent, self.d_model
        )

        proprio_token = self.proprio_projection(current_proprio).unsqueeze(1)
        visual_tokens = self.visual_projection(visual_tokens)
        memory = torch.cat((proprio_token, visual_tokens), dim=1)

        actions = self.action_projection(
            self.decoder(
                tgt=queries,
                memory=memory,
                tgt_key_padding_mask=safe_substep_is_pad,
            )
        )
        if substep_is_pad is not None:
            actions = actions.masked_fill(substep_is_pad.unsqueeze(-1), 0.0)
        if flatten_output:
            return actions
        return actions.reshape(
            batch_size,
            self.num_latents,
            self.substeps_per_latent,
            self.action_dim,
        )


__all__ = ["LatentActionDecoder"]
