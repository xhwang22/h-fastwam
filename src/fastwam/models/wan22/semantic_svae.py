"""Frozen encoder half of semantic-wm's V-JEPA S-VAE adapter.

The module structure intentionally matches the published semantic-wm
checkpoint. Portions are adapted from chandar-lab/semantic-wm under the MIT
license; see ``third_party/semantic-wm/LICENSE``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


SEMANTIC_SVAE_SOURCE_REVISION = "ba06ced314d61e313ff670b0f932cfecad5126a6"
SEMANTIC_SVAE_SOURCE_PATH = "vjepa/adapter_vjepa_image_96_multi.pt"
SEMANTIC_SVAE_SOURCE_SHA256 = (
    "179dc0809262bca042e2a05d834df40f3e5c953271223727c94eb9e4484f6ec8"
)


@dataclass(frozen=True)
class SemanticSVAEConfig:
    input_dim: int = 1024
    latent_dim: int = 96
    num_heads: int = 16
    num_layers: int = 3
    intermediate_size: int = 1024
    layer_norm_eps: float = 1e-12


class _SelfAttention(nn.Module):
    def __init__(self, config: SemanticSVAEConfig):
        super().__init__()
        if config.input_dim % config.num_heads != 0:
            raise ValueError(
                f"input_dim={config.input_dim} must be divisible by "
                f"num_heads={config.num_heads}."
            )
        self.num_attention_heads = config.num_heads
        self.attention_head_size = config.input_dim // config.num_heads
        self.all_head_size = config.input_dim
        self.query = nn.Linear(config.input_dim, config.input_dim, bias=True)
        self.key = nn.Linear(config.input_dim, config.input_dim, bias=True)
        self.value = nn.Linear(config.input_dim, config.input_dim, bias=True)
        self.dropout = nn.Dropout(0.0)

    def _transpose_for_scores(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return tensor.view(shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        query = self._transpose_for_scores(self.query(hidden_states))
        key = self._transpose_for_scores(self.key(hidden_states))
        value = self._transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        probabilities = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(probabilities, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return (context.view(hidden_states.shape),)


class _SelfOutput(nn.Module):
    def __init__(self, config: SemanticSVAEConfig):
        super().__init__()
        self.dense = nn.Linear(config.input_dim, config.input_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        del input_tensor
        return self.dropout(self.dense(hidden_states))


class _Attention(nn.Module):
    def __init__(self, config: SemanticSVAEConfig):
        super().__init__()
        self.attention = _SelfAttention(config)
        self.output = _SelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        self_output = self.attention(hidden_states)[0]
        return (self.output(self_output, hidden_states),)


class _Intermediate(nn.Module):
    def __init__(self, config: SemanticSVAEConfig):
        super().__init__()
        self.dense = nn.Linear(config.input_dim, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))


class _Output(nn.Module):
    def __init__(self, config: SemanticSVAEConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.input_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states)) + input_tensor


class _ViTMAELayer(nn.Module):
    def __init__(self, config: SemanticSVAEConfig):
        super().__init__()
        self.attention = _Attention(config)
        self.intermediate = _Intermediate(config)
        self.output = _Output(config)
        self.layernorm_before = nn.LayerNorm(
            config.input_dim,
            eps=config.layer_norm_eps,
        )
        self.layernorm_after = nn.LayerNorm(
            config.input_dim,
            eps=config.layer_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        attention_output = self.attention(
            self.layernorm_before(hidden_states)
        )[0]
        hidden_states = attention_output + hidden_states
        layer_output = self.intermediate(self.layernorm_after(hidden_states))
        return (self.output(layer_output, hidden_states),)


class SemanticSVAEEncoder(nn.Module):
    """Deterministic posterior-mean encoder from 1024-d features to 96-d."""

    def __init__(self, config: SemanticSVAEConfig = SemanticSVAEConfig()):
        super().__init__()
        self.config = config
        self.enc_blocks = nn.ModuleList(
            [_ViTMAELayer(config) for _ in range(config.num_layers)]
        )
        self.enc_norm = nn.LayerNorm(
            config.input_dim,
            eps=config.layer_norm_eps,
        )
        self.enc_proj = nn.Linear(config.input_dim, config.latent_dim * 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "SemanticSVAEEncoder expects [batch, tokens, channels], "
                f"got {tuple(features.shape)}."
            )
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected feature dim {self.config.input_dim}, "
                f"got {features.shape[-1]}."
            )
        hidden_states = features
        for block in self.enc_blocks:
            hidden_states = block(hidden_states)[0]
        moments = self.enc_proj(self.enc_norm(hidden_states))
        mean, _ = moments.chunk(2, dim=-1)
        return mean


def _strip_compile_prefix(key: str) -> str:
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _validate_adapter_config(config: dict[str, Any] | None) -> None:
    if not config:
        return
    expected = {
        "input_dim": 1024,
        "latent_dim": 96,
        "num_heads": 16,
        "num_layers": 3,
        "intermediate_size": 1024,
        "progressive": False,
        "latent_layers": 0,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if key in config and config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Unsupported semantic S-VAE checkpoint configuration: {mismatches}"
        )


def extract_semantic_svae_encoder_state(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Extract and validate the encoder-only state from a source checkpoint."""
    if "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
    elif "adapter" in checkpoint:
        source_state = checkpoint["adapter"]
    else:
        source_state = checkpoint
    if not isinstance(source_state, dict):
        raise ValueError("Semantic S-VAE checkpoint has no adapter state dict.")

    adapter_config = dict(checkpoint.get("adapter_config") or {})
    _validate_adapter_config(adapter_config)
    state_dict = {}
    for key, value in source_state.items():
        clean_key = _strip_compile_prefix(str(key))
        if clean_key.startswith(("enc_blocks.", "enc_norm.", "enc_proj.")):
            state_dict[clean_key] = value
    if not state_dict:
        raise ValueError("Semantic S-VAE checkpoint contains no encoder weights.")
    return state_dict, adapter_config


def load_semantic_svae_encoder(
    checkpoint_path: str,
    dtype: torch.dtype,
) -> SemanticSVAEEncoder:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Semantic S-VAE checkpoint not found: {path}")
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid semantic S-VAE checkpoint: {path}")
    state_dict, adapter_config = extract_semantic_svae_encoder_state(checkpoint)
    _validate_adapter_config(adapter_config)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        encoder = SemanticSVAEEncoder()
    finally:
        torch.set_default_dtype(previous_dtype)
    encoder.load_state_dict(state_dict, strict=True)
    return encoder.to(dtype=dtype).eval().requires_grad_(False)
