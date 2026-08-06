from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        strict_expert_compat: bool = True,
        layer_alignment_mode: str = "strict",
        shared_attention_expert: str = "video",
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if len(mixtures) < 2:
            raise ValueError("`mixtures` must contain at least two experts.")

        valid_alignment = {"strict", "tail_overlap"}
        if layer_alignment_mode not in valid_alignment:
            raise ValueError(
                f"Invalid `layer_alignment_mode`: {layer_alignment_mode}. Must be one of {valid_alignment}."
            )

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.strict_expert_compat = bool(strict_expert_compat)
        self.layer_alignment_mode = layer_alignment_mode
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        if shared_attention_expert not in self.mixtures:
            raise ValueError(
                f"`shared_attention_expert`={shared_attention_expert} not found in mixtures={self.expert_order}."
            )

        self.expert_num_layers = {name: int(len(expert.blocks)) for name, expert in self.mixtures.items()}
        self.expert_num_heads = {name: int(expert.num_heads) for name, expert in self.mixtures.items()}
        self.expert_attn_head_dim = {name: int(expert.attn_head_dim) for name, expert in self.mixtures.items()}
        self.expert_hidden_dim = {
            name: self.expert_num_heads[name] * self.expert_attn_head_dim[name]
            for name in self.expert_order
        }

        shared_name = shared_attention_expert
        self.num_heads = self.expert_num_heads[shared_name]
        self.attn_head_dim = self.expert_attn_head_dim[shared_name]
        self.shared_hidden_dim = self.num_heads * self.attn_head_dim

        if self.strict_expert_compat:
            first_name = self.expert_order[0]
            self.num_layers = self.expert_num_layers[first_name]
            for name in self.expert_order[1:]:
                if self.expert_num_layers[name] != self.num_layers:
                    raise ValueError(
                        "All experts must have same number of layers in strict mode; "
                        f"got {self.num_layers} and {self.expert_num_layers[name]} ({name})."
                    )
                if self.expert_num_heads[name] != self.num_heads:
                    raise ValueError(
                        "All experts must have same num_heads in strict mode; "
                        f"got {self.num_heads} and {self.expert_num_heads[name]} ({name})."
                    )
                if self.expert_attn_head_dim[name] != self.attn_head_dim:
                    raise ValueError(
                        "All experts must have same attn_head_dim in strict mode; "
                        f"got {self.attn_head_dim} and {self.expert_attn_head_dim[name]} ({name})."
                    )
            self.overlap_num_layers = self.num_layers
            self.layer_start_indices = {name: 0 for name in self.expert_order}
        else:
            self.overlap_num_layers = min(self.expert_num_layers.values())
            if self.overlap_num_layers <= 0:
                raise ValueError("All experts must have at least one transformer block.")
            if self.layer_alignment_mode == "strict":
                # In non-strict shape mode + strict alignment, still only overlap layers are mixed.
                self.layer_start_indices = {name: 0 for name in self.expert_order}
            else:
                # tail-overlap: expert-specific prefix layers run solo, then overlap layers are mixed.
                self.layer_start_indices = {
                    name: self.expert_num_layers[name] - self.overlap_num_layers
                    for name in self.expert_order
                }
            self.num_layers = self.overlap_num_layers

        # Projection adapters for heterogeneous experts (strict mode uses implicit identity).
        self.q_proj_to_shared = nn.ModuleDict()
        self.k_proj_to_shared = nn.ModuleDict()
        self.v_proj_to_shared = nn.ModuleDict()
        self.o_proj_from_shared = nn.ModuleDict()
        if not self.strict_expert_compat:
            for name in self.expert_order:
                in_dim = self.expert_hidden_dim[name]
                start_idx = self.layer_start_indices[name]
                adapter_kwargs = self._module_float_kwargs(self.mixtures[name])
                for overlap_idx in range(self.overlap_num_layers):
                    layer_idx = start_idx + overlap_idx
                    key = f"{name}__{layer_idx}"
                    if in_dim == self.shared_hidden_dim:
                        self.q_proj_to_shared[key] = nn.Identity()
                        self.k_proj_to_shared[key] = nn.Identity()
                        self.v_proj_to_shared[key] = nn.Identity()
                        self.o_proj_from_shared[key] = nn.Identity()
                    else:
                        self.q_proj_to_shared[key] = nn.Linear(
                            in_dim,
                            self.shared_hidden_dim,
                            bias=False,
                            **adapter_kwargs,
                        )
                        self.k_proj_to_shared[key] = nn.Linear(
                            in_dim,
                            self.shared_hidden_dim,
                            bias=False,
                            **adapter_kwargs,
                        )
                        self.v_proj_to_shared[key] = nn.Linear(
                            in_dim,
                            self.shared_hidden_dim,
                            bias=False,
                            **adapter_kwargs,
                        )
                        self.o_proj_from_shared[key] = nn.Linear(
                            self.shared_hidden_dim,
                            in_dim,
                            bias=False,
                            **adapter_kwargs,
                        )

        logger.info(
            "Initialized MoT with experts=%s, strict=%s, align=%s, overlap_layers=%d, shared_heads=%d, shared_head_dim=%d",
            self.expert_order,
            self.strict_expert_compat,
            self.layer_alignment_mode,
            self.overlap_num_layers,
            self.num_heads,
            self.attn_head_dim,
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                "  Expert '%s': params=%.2fB layers=%d heads=%d head_dim=%d",
                name,
                sum(p.numel() for p in expert.parameters()) / 1e9,
                self.expert_num_layers[name],
                self.expert_num_heads[name],
                self.expert_attn_head_dim[name],
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _module_float_kwargs(module: nn.Module) -> dict:
        for tensor in module.parameters():
            if tensor.is_floating_point():
                return {"device": tensor.device, "dtype": tensor.dtype}
        for tensor in module.buffers():
            if tensor.is_floating_point():
                return {"device": tensor.device, "dtype": tensor.dtype}
        return {}

    @staticmethod
    def _proj_key(name: str, layer_idx: int) -> str:
        return f"{name}__{layer_idx}"

    def _project_qkv_to_shared(
        self,
        name: str,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.strict_expert_compat:
            return q, k, v
        key = self._proj_key(name, layer_idx)
        if key not in self.q_proj_to_shared:
            return q, k, v
        return (
            self.q_proj_to_shared[key](q),
            self.k_proj_to_shared[key](k),
            self.v_proj_to_shared[key](v),
        )

    def _project_mixed_from_shared(
        self,
        name: str,
        layer_idx: int,
        mixed: torch.Tensor,
    ) -> torch.Tensor:
        if self.strict_expert_compat:
            return mixed
        key = self._proj_key(name, layer_idx)
        if key not in self.o_proj_from_shared:
            return mixed
        return self.o_proj_from_shared[key](mixed)

    def _expert_layer_idx(self, name: str, overlap_idx: int) -> int:
        return int(self.layer_start_indices[name]) + int(overlap_idx)

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._attention_with_num_heads(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
            num_heads=self.num_heads,
        )

    def _attention_with_num_heads(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=int(num_heads), ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        name: str,
        layer_idx: int,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            name: Expert name.
            layer_idx: Concrete layer index inside this expert.
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        q, k, v = self._project_qkv_to_shared(
            name=name,
            layer_idx=layer_idx,
            q=q,
            k=k,
            v=v,
        )

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if not self.strict_expert_compat or self.layer_alignment_mode != "strict":
            raise NotImplementedError(
                "`prefill_video_cache` currently supports strict-compatible MoT only."
            )
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                name="video",
                layer_idx=layer_idx,
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if not self.strict_expert_compat or self.layer_alignment_mode != "strict":
            raise NotImplementedError(
                "`forward_action_with_video_cache` currently supports strict-compatible MoT only."
            )
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                name="action",
                layer_idx=layer_idx,
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def prefill_fixed_expert_cache(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        fixed_expert_order: tuple[str, ...],
    ) -> dict:
        """Run fixed prefix experts once and cache their overlap-layer K/V."""
        fixed_order = list(fixed_expert_order)
        if not fixed_order:
            raise ValueError("`fixed_expert_order` cannot be empty.")
        if self.expert_order[:len(fixed_order)] != fixed_order:
            raise ValueError(
                "`fixed_expert_order` must be a prefix of the registered expert order: "
                f"got {fixed_order}, registered={self.expert_order}."
            )

        missing = [
            name
            for name in fixed_order
            if name not in embeds_all or name not in freqs_all or name not in t_mod_all
        ]
        if missing:
            raise ValueError(f"Missing fixed-expert inputs for: {missing}")

        tokens_all = {name: embeds_all[name] for name in fixed_order}
        seq_lens = [int(tokens_all[name].shape[1]) for name in fixed_order]
        fixed_seq_len = int(sum(seq_lens))
        if attention_mask.shape != (fixed_seq_len, fixed_seq_len):
            raise ValueError(
                "Fixed-expert attention mask shape mismatch: "
                f"got {tuple(attention_mask.shape)}, expected {(fixed_seq_len, fixed_seq_len)}."
            )

        seq_offsets = {}
        start = 0
        for name, seq_len in zip(fixed_order, seq_lens):
            seq_offsets[name] = (start, start + seq_len)
            start += seq_len

        if not self.strict_expert_compat and self.layer_alignment_mode == "tail_overlap":
            for name in fixed_order:
                prefix_layers = int(self.layer_start_indices[name])
                if prefix_layers <= 0:
                    continue
                expert = self.mixtures[name]
                x = tokens_all[name]
                row_start, row_end = seq_offsets[name]
                self_mask = attention_mask[row_start:row_end, row_start:row_end]
                for layer_idx in range(prefix_layers):
                    block = expert.blocks[layer_idx]
                    (
                        q,
                        k,
                        v,
                        residual_x,
                        gate_msa,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                        use_gradient_checkpointing,
                    ) = self._build_expert_attention_io(
                        name=name,
                        layer_idx=layer_idx,
                        expert=expert,
                        block=block,
                        x=x,
                        freqs=freqs_all[name],
                        t_mod=t_mod_all[name],
                    )
                    mixed = self._attention_with_num_heads(
                        q_cat=q,
                        k_cat=k,
                        v_cat=v,
                        attention_mask=self_mask,
                        num_heads=block.num_heads,
                    )
                    x = self._apply_post_with_optional_checkpoint(
                        block=block,
                        residual_x=residual_x,
                        gate_msa=gate_msa,
                        shift_mlp=shift_mlp,
                        scale_mlp=scale_mlp,
                        gate_mlp=gate_mlp,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        mixed_slice=mixed,
                        context_payload=context_all.get(name),
                    )
                tokens_all[name] = x

        layer_cache = []
        for overlap_idx in range(self.overlap_num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            post_states = {}
            current_seq_lens = []

            for name in fixed_order:
                expert = self.mixtures[name]
                layer_idx = self._expert_layer_idx(name, overlap_idx)
                block = expert.blocks[layer_idx]
                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    name=name,
                    layer_idx=layer_idx,
                    expert=expert,
                    block=block,
                    x=tokens_all[name],
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                current_seq_lens.append(int(tokens_all[name].shape[1]))
                post_states[name] = {
                    "block": block,
                    "layer_idx": layer_idx,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            mixed = self._mixed_attention(
                q_cat=q_cat,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=attention_mask,
            )
            layer_cache.append({"k": k_cat, "v": v_cat})

            start = 0
            for name, seq_len in zip(fixed_order, current_seq_lens):
                end = start + seq_len
                state = post_states[name]
                mixed_slice = self._project_mixed_from_shared(
                    name=name,
                    layer_idx=state["layer_idx"],
                    mixed=mixed[:, start:end],
                )
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=state["block"],
                    residual_x=state["residual_x"],
                    gate_msa=state["gate_msa"],
                    shift_mlp=state["shift_mlp"],
                    scale_mlp=state["scale_mlp"],
                    gate_mlp=state["gate_mlp"],
                    use_gradient_checkpointing=state["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_all.get(name),
                )
                start = end

        return {
            "layers": layer_cache,
            "fixed_expert_order": tuple(fixed_order),
            "fixed_seq_len": fixed_seq_len,
        }

    def forward_target_with_fixed_cache(
        self,
        target_name: str,
        target_tokens: torch.Tensor,
        target_freqs: torch.Tensor,
        target_t_mod: torch.Tensor,
        target_context_payload: Optional[dict],
        fixed_cache: dict,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one target expert while reusing fixed-expert K/V."""
        fixed_order = list(fixed_cache.get("fixed_expert_order", ()))
        if self.expert_order != fixed_order + [target_name]:
            raise ValueError(
                "Cached target inference requires registered order "
                f"{fixed_order + [target_name]}, got {self.expert_order}."
            )
        layers = fixed_cache.get("layers")
        if not isinstance(layers, list) or len(layers) != self.overlap_num_layers:
            raise ValueError(
                "Fixed cache layer count mismatch: "
                f"got {0 if not isinstance(layers, list) else len(layers)}, "
                f"expected {self.overlap_num_layers}."
            )

        fixed_seq_len = int(fixed_cache.get("fixed_seq_len", 0))
        target_seq_len = int(target_tokens.shape[1])
        total_seq_len = fixed_seq_len + target_seq_len
        if attention_mask.shape != (total_seq_len, total_seq_len):
            raise ValueError(
                "Cached target attention mask shape mismatch: "
                f"got {tuple(attention_mask.shape)}, expected {(total_seq_len, total_seq_len)}."
            )

        expert = self.mixtures[target_name]
        x = target_tokens
        target_self_mask = attention_mask[fixed_seq_len:, fixed_seq_len:]
        if not self.strict_expert_compat and self.layer_alignment_mode == "tail_overlap":
            for layer_idx in range(int(self.layer_start_indices[target_name])):
                block = expert.blocks[layer_idx]
                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    name=target_name,
                    layer_idx=layer_idx,
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=target_freqs,
                    t_mod=target_t_mod,
                )
                mixed = self._attention_with_num_heads(
                    q_cat=q,
                    k_cat=k,
                    v_cat=v,
                    attention_mask=target_self_mask,
                    num_heads=block.num_heads,
                )
                x = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    mixed_slice=mixed,
                    context_payload=target_context_payload,
                )

        target_attention_mask = attention_mask[fixed_seq_len:, :]
        for overlap_idx, layer_fixed_cache in enumerate(layers):
            layer_idx = self._expert_layer_idx(target_name, overlap_idx)
            block = expert.blocks[layer_idx]
            (
                q_target,
                k_target,
                v_target,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                name=target_name,
                layer_idx=layer_idx,
                expert=expert,
                block=block,
                x=x,
                freqs=target_freqs,
                t_mod=target_t_mod,
            )
            k_fixed = layer_fixed_cache["k"]
            v_fixed = layer_fixed_cache["v"]
            if k_fixed.shape[1] != fixed_seq_len or v_fixed.shape[1] != fixed_seq_len:
                raise ValueError(
                    f"Fixed cache seq length mismatch at overlap layer {overlap_idx}."
                )
            mixed = self._mixed_attention(
                q_cat=q_target,
                k_cat=torch.cat([k_fixed, k_target], dim=1),
                v_cat=torch.cat([v_fixed, v_target], dim=1),
                attention_mask=target_attention_mask,
            )
            mixed = self._project_mixed_from_shared(
                name=target_name,
                layer_idx=layer_idx,
                mixed=mixed,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=target_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        detach_video_for_action: bool = False,
        detach_kv_experts: Optional[set] = None,
        active_expert_order: Optional[list[str] | tuple[str, ...]] = None,
    ):
        """Run shared multi-expert attention.

        In strict mode, all experts are mixed at every layer.
        In hetero tail-overlap mode, each expert runs its prefix layers solo,
        then participates in mixed attention for the shared overlap suffix.
        """
        expert_order = list(active_expert_order) if active_expert_order is not None else self.expert_order
        if not expert_order:
            raise ValueError("`active_expert_order` cannot be empty.")
        unknown_active = [name for name in expert_order if name not in self.mixtures]
        if unknown_active:
            raise ValueError(
                f"`active_expert_order` contains unknown experts: {unknown_active}. "
                f"Known experts: {self.expert_order}"
            )

        missing = [k for k in expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        if detach_kv_experts is not None:
            unknown = set(detach_kv_experts) - set(expert_order)
            if unknown:
                raise ValueError(
                    f"`detach_kv_experts` contains unknown expert names: {sorted(unknown)}. "
                    f"Known active experts: {expert_order}"
                )

        tokens_all = {k: v for k, v in embeds_all.items()}

        # Sequence offsets inside [expert_0 | expert_1 | ...] for mask slicing.
        seq_lens_init = [int(tokens_all[name].shape[1]) for name in expert_order]
        total_seq_init = int(sum(seq_lens_init))
        if attention_mask.shape[0] != total_seq_init:
            raise ValueError(
                "Attention mask seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs tokens={total_seq_init}"
            )
        seq_offsets = {}
        start = 0
        for name, seq_len in zip(expert_order, seq_lens_init):
            end = start + int(seq_len)
            seq_offsets[name] = (start, end)
            start = end

        # Heterogeneous tail-overlap mode: run expert-specific prefix layers without cross-expert mixing.
        if not self.strict_expert_compat and self.layer_alignment_mode == "tail_overlap":
            for name in expert_order:
                prefix_layers = int(self.layer_start_indices[name])
                if prefix_layers <= 0:
                    continue
                expert = self.mixtures[name]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]
                row_s, row_e = seq_offsets[name]
                self_mask = attention_mask[row_s:row_e, row_s:row_e]

                for layer_idx in range(prefix_layers):
                    block = expert.blocks[layer_idx]
                    (
                        q,
                        k,
                        v,
                        residual_x,
                        gate_msa,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                        use_gradient_checkpointing,
                    ) = self._build_expert_attention_io(
                        name=name,
                        layer_idx=layer_idx,
                        expert=expert,
                        block=block,
                        x=x,
                        freqs=freqs,
                        t_mod=t_mod,
                    )
                    mixed = self._attention_with_num_heads(
                        q_cat=q,
                        k_cat=k,
                        v_cat=v,
                        attention_mask=self_mask,
                        num_heads=block.num_heads,
                    )
                    x = self._apply_post_with_optional_checkpoint(
                        block=block,
                        residual_x=residual_x,
                        gate_msa=gate_msa,
                        shift_mlp=shift_mlp,
                        scale_mlp=scale_mlp,
                        gate_mlp=gate_mlp,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        mixed_slice=mixed,
                        context_payload=context_all.get(name),
                    )
                tokens_all[name] = x

        # Mixed-attention overlap layers.
        for overlap_idx in range(self.overlap_num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in expert_order:
                expert = self.mixtures[name]
                layer_idx = self._expert_layer_idx(name, overlap_idx)
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    name=name,
                    layer_idx=layer_idx,
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "layer_idx": layer_idx,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = int(q_cat.shape[1])
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            if detach_video_for_action and len(expert_order) == 2 and expert_order[0] == "video" and expert_order[1] == "action":
                video_seq_len = seq_lens[0]
                video_mask = attention_mask[:video_seq_len, :total_seq]
                video_mixed = self._mixed_attention(
                    q_cat=q_chunks[0], k_cat=k_cat, v_cat=v_cat, attention_mask=video_mask,
                )

                k_cat_detached = torch.cat([k_chunks[0].detach(), k_chunks[1]], dim=1)
                v_cat_detached = torch.cat([v_chunks[0].detach(), v_chunks[1]], dim=1)
                action_mask = attention_mask[video_seq_len:, :total_seq]
                action_mixed = self._mixed_attention(
                    q_cat=q_chunks[1], k_cat=k_cat_detached, v_cat=v_cat_detached, attention_mask=action_mask,
                )
                mixed = torch.cat([video_mixed, action_mixed], dim=1)
            elif detach_kv_experts:
                detach_set = set(detach_kv_experts)
                per_expert_mixed = []
                q_start = 0
                for q_idx, name_q in enumerate(expert_order):
                    q_seq_len = seq_lens[q_idx]
                    q_end = q_start + q_seq_len
                    q_slice = q_chunks[q_idx]

                    k_pieces = []
                    v_pieces = []
                    for kv_idx, name_kv in enumerate(expert_order):
                        if name_kv in detach_set and name_kv != name_q:
                            k_pieces.append(k_chunks[kv_idx].detach())
                            v_pieces.append(v_chunks[kv_idx].detach())
                        else:
                            k_pieces.append(k_chunks[kv_idx])
                            v_pieces.append(v_chunks[kv_idx])
                    k_for_q = torch.cat(k_pieces, dim=1)
                    v_for_q = torch.cat(v_pieces, dim=1)

                    q_mask = attention_mask[q_start:q_end, :total_seq]
                    per_expert_mixed.append(
                        self._mixed_attention(
                            q_cat=q_slice,
                            k_cat=k_for_q,
                            v_cat=v_for_q,
                            attention_mask=q_mask,
                        )
                    )
                    q_start = q_end
                mixed = torch.cat(per_expert_mixed, dim=1)
            else:
                mixed = self._mixed_attention(
                    q_cat=q_cat,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=attention_mask,
                )

            start = 0
            for name, seq_len in zip(expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                mixed_slice = self._project_mixed_from_shared(
                    name=name,
                    layer_idx=cached_expert["layer_idx"],
                    mixed=mixed_slice,
                )
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
