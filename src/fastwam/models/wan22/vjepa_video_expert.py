"""V-JEPA-AC predictor as a Wan-style video expert (W-1 wrapped design).

This module wraps the V-JEPA 2-AC predictor's transformer blocks so they can
participate in fastwam's MoT (Mixture of Transformers) joint attention, while
preserving the V-JEPA-AC predictor's pretrained attn/mlp weights and 3D RoPE
behavior verbatim.

Design (W-1 with frozen-identity modulation):
  * Each block exposes Wan-style fields (modulation, norm1/2/3, self_attn,
    cross_attn, ffn, gate) so MoT's `_split_modulation`, `_apply_expert_post_block`,
    `_build_expert_attention_io` work unchanged.
  * `block.modulation` is initialised to identity ``[0, 0, 1, 0, 0, 1] x D``
    (shift_msa=0, scale_msa=0, gate_msa=1, shift_mlp=0, scale_mlp=0, gate_mlp=1)
    and **frozen** (`requires_grad=False`). Combined with a zeroed `t_mod`
    from the video expert's `pre_dit`, this makes the modulate+gate path an
    identity, exactly matching the V-JEPA predictor's original
    LayerNorm + attn + residual structure.
  * `block.cross_attn` is added (Wan only — V-JEPA had no text); its
    output projection ``o`` is zero-initialised so the cross-attn branch
    is initially a no-op (residual is x + 0).
  * `block.self_attn` reuses Wan's `SelfAttention` class for module
    interface compatibility, but the RoPE applied by MoT must be the
    V-JEPA 3D RoPE (not Wan's complex-number RoPE) to match what the
    pretrained weights expect.

Text conditioning enters two ways (mirroring how V-JEPA-AC fed action+state
prefix tokens):
  1. As a per-timestep prefix of `text_pool_queries` tokens (default 3,
     matching V-JEPA-AC's [action, state, extrinsics] prefix slot count),
     concatenated in front of the H*W spatial patch tokens of each latent
     timestep. These prefix tokens participate in self-attention with the
     video patch tokens (3D RoPE: prefix tokens get only the temporal axis).
  2. As cross-attention context (Wan style), giving the model a redundant
     but globally-pooled-from-text conditioning channel. With cross-attn
     ``o`` zero-init, this is initially silent and is learnable.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .text_attention_pool import TextAttentionPool
from .vjepa_rope_attention import vjepa_apply_rope_to_qk
from .wan_video_dit import (
    CrossAttention,
    GateModule,
    SelfAttention,
    flash_attention,
    modulate,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Block: V-JEPA predictor block wrapped in Wan-style interface
# ---------------------------------------------------------------------------


class VJEPAWrappedBlock(nn.Module):
    """Wan-DiT-shaped block that hosts V-JEPA 2-AC predictor pretrained weights.

    Module field names match `wan_video_dit.DiTBlock` so MoT's helpers
    (`_split_modulation`, `_apply_expert_post_block`, etc.) can drive it.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Self-attn / cross-attn: reuse Wan modules so MoT sees the expected
        # `q/k/v/o/norm_q/norm_k` interface. We will load V-JEPA pretrained
        # `qkv.weight + qkv.bias` into q/k/v as concatenated slices; the
        # 3D-RoPE is *not* baked into these modules — MoT will apply it
        # externally via `vjepa_apply_rope_to_qk`.
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps=eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps=eps)

        # Wan-style three norms. `norm1`/`norm2` are non-affine because
        # `modulate(.)` provides their affine on the fly. `norm3` is the
        # standard Wan affine layer-norm fed into cross-attn.
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)

        # Wan-style FFN with GELU(tanh) activation. V-JEPA uses GELU (no tanh),
        # so we use plain GELU here for closer numerical match to pretrained mlp.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),  # NOTE: V-JEPA's MLP uses default torch.nn.GELU (no 'tanh').
            nn.Linear(ffn_dim, hidden_dim),
        )

        # Modulation: same shape Wan expects, but **identity-initialised**
        # and **frozen** so the video expert's residual path equals
        # V-JEPA's original block (no AdaLN-zero) regardless of training.
        # Layout per Wan: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        identity_mod = torch.zeros(1, 6, hidden_dim)
        identity_mod[0, 2, :] = 1.0  # gate_msa = 1
        identity_mod[0, 5, :] = 1.0  # gate_mlp = 1
        self.modulation = nn.Parameter(identity_mod, requires_grad=False)

        self.gate = GateModule()

        # Zero-init the cross-attn output projection so it is an identity
        # branch at init (residual path: x + 0).
        nn.init.zeros_(self.cross_attn.o.weight)
        if self.cross_attn.o.bias is not None:
            nn.init.zeros_(self.cross_attn.o.bias)

    # ----------------------------------------------------------------- #
    # Loaders
    # ----------------------------------------------------------------- #

    @torch.no_grad()
    def load_vjepa_block_state(
        self,
        vjepa_block_sd: Dict[str, torch.Tensor],
        prefix: str = "",
    ) -> None:
        """Load V-JEPA-AC predictor block weights into this wrapped block.

        Expected vjepa keys (relative to ``prefix``)::

            attn.qkv.weight  [3D, D]
            attn.qkv.bias    [3D]
            attn.proj.weight [D, D]
            attn.proj.bias   [D]
            mlp.fc1.weight   [4D, D]
            mlp.fc1.bias     [4D]
            mlp.fc2.weight   [D, 4D]
            mlp.fc2.bias     [D]
            norm1.weight     [D]
            norm1.bias       [D]
            norm2.weight     [D]
            norm2.bias       [D]

        The V-JEPA `norm1`/`norm2` per-channel affine (γ, β) cannot be loaded
        directly because we use elementwise_affine=False (Wan style with
        external modulate). We *fold* them into the q/k/v/fc1 input weights
        instead, which is mathematically equivalent for LayerNorm fed
        directly into a Linear (the Linear absorbs the per-channel scale γ
        and the bias absorbs Linear @ β):

            y = Linear(γ * z + β)
              = (Linear.weight * γ.broadcast) @ z + (Linear.weight @ β + Linear.bias)
        """
        D = self.hidden_dim
        get = lambda k: vjepa_block_sd[prefix + k]

        # ---- norm1 absorption: fold (γ1, β1) into self_attn q/k/v ----
        gamma1 = get("norm1.weight").to(self.self_attn.q.weight.dtype)
        beta1 = get("norm1.bias").to(self.self_attn.q.weight.dtype)

        qkv_w = get("attn.qkv.weight").to(self.self_attn.q.weight.dtype)  # [3D, D]
        qkv_b = get("attn.qkv.bias").to(self.self_attn.q.weight.dtype)    # [3D]

        # split fused qkv into three [D, D] / [D] slices
        q_w, k_w, v_w = qkv_w[:D], qkv_w[D : 2 * D], qkv_w[2 * D :]
        q_b, k_b, v_b = qkv_b[:D], qkv_b[D : 2 * D], qkv_b[2 * D :]

        # absorb γ1 (per-input-channel scale) and β1 (per-input-channel shift)
        # weight: scale *columns* by γ1; bias: add Linear(β1) to old bias.
        for w, b in [(q_w, q_b), (k_w, k_b), (v_w, v_b)]:
            w_new = w * gamma1.unsqueeze(0)  # [D, D]
            b_new = b + (w @ beta1)
            yield_target = None  # placeholder
        # Re-run for assignment (need separate variables per branch).
        q_w_new = q_w * gamma1.unsqueeze(0)
        k_w_new = k_w * gamma1.unsqueeze(0)
        v_w_new = v_w * gamma1.unsqueeze(0)
        q_b_new = q_b + (q_w @ beta1)
        k_b_new = k_b + (k_w @ beta1)
        v_b_new = v_b + (v_w @ beta1)

        self.self_attn.q.weight.copy_(q_w_new)
        self.self_attn.q.bias.copy_(q_b_new)
        self.self_attn.k.weight.copy_(k_w_new)
        self.self_attn.k.bias.copy_(k_b_new)
        self.self_attn.v.weight.copy_(v_w_new)
        self.self_attn.v.bias.copy_(v_b_new)

        # output proj
        self.self_attn.o.weight.copy_(get("attn.proj.weight").to(self.self_attn.o.weight.dtype))
        self.self_attn.o.bias.copy_(get("attn.proj.bias").to(self.self_attn.o.bias.dtype))

        # ---- norm2 absorption: fold (γ2, β2) into ffn fc1 ----
        gamma2 = get("norm2.weight").to(self.ffn[0].weight.dtype)
        beta2 = get("norm2.bias").to(self.ffn[0].weight.dtype)

        fc1_w = get("mlp.fc1.weight").to(self.ffn[0].weight.dtype)  # [4D, D]
        fc1_b = get("mlp.fc1.bias").to(self.ffn[0].weight.dtype)    # [4D]
        fc1_w_new = fc1_w * gamma2.unsqueeze(0)
        fc1_b_new = fc1_b + (fc1_w @ beta2)
        self.ffn[0].weight.copy_(fc1_w_new)
        self.ffn[0].bias.copy_(fc1_b_new)
        # fc2 is post-activation, no norm to absorb
        self.ffn[2].weight.copy_(get("mlp.fc2.weight").to(self.ffn[2].weight.dtype))
        self.ffn[2].bias.copy_(get("mlp.fc2.bias").to(self.ffn[2].weight.dtype))

        # IMPORTANT: must zero the RMSNorm scale weight on q/k since V-JEPA
        # has *no* such norm — but Wan's SelfAttention sandwiches RMSNorm
        # between (qkv linear) and (RoPE+attn). To make our path equivalent
        # to V-JEPA's "qkv -> RoPE -> attn", we set the RMSNorm scale weight
        # to a constant such that RMSNorm becomes identity.
        # Actually, RMSNorm is a normalization (scales by 1/||x||) — it's
        # NOT identity even with weight=1. To bypass it, we'd need
        # weight = ||x|| per token, which is not feasible.
        # Instead: we replace the RMSNorm with nn.Identity here so the
        # forward exactly matches V-JEPA.
        self.self_attn.norm_q = nn.Identity()
        self.self_attn.norm_k = nn.Identity()


# ---------------------------------------------------------------------------
# Video expert
# ---------------------------------------------------------------------------


class VJEPAVideoExpert(nn.Module):
    """V-JEPA 2-AC predictor as a fastwam video expert.

    pre_dit/post_dit interface mirrors `WanVideoDiT` so it slots into
    `fastwam.MoT` cleanly.
    """

    # Constants identifying this expert's RoPE flavor / interface kind so
    # MoT can route it through `vjepa_apply_rope_to_qk` rather than
    # Wan's `rope_apply` (the two are NOT numerically equivalent).
    EXPERT_ATTN_KIND = "vjepa_3d_rope"

    def __init__(
        self,
        in_dim: int = 1408,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 16,
        attn_head_dim: int = 64,
        num_layers: int = 24,
        text_dim: int = 4096,
        text_pool_queries: int = 3,
        rope_grid_size: int = 16,
        eps: float = 1e-6,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if hidden_dim != num_heads * attn_head_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal num_heads*attn_head_dim "
                f"({num_heads}*{attn_head_dim} = {num_heads * attn_head_dim})"
            )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.text_dim = text_dim
        self.text_pool_queries = text_pool_queries
        self.rope_grid_size = rope_grid_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Loadable from V-JEPA-AC predictor.predictor_embed
        self.predictor_embed = nn.Linear(in_dim, hidden_dim)

        # Loadable from predictor.predictor_norm + predictor_proj
        self.predictor_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.predictor_proj = nn.Linear(hidden_dim, in_dim)

        # Text pool: T5 cache 4096d -> k=3 tokens × hidden_dim
        self.text_pool = TextAttentionPool(
            num_queries=text_pool_queries,
            in_dim=text_dim,
            out_dim=hidden_dim,
        )
        # Wan-style text embedding for cross-attn
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            VJEPAWrappedBlock(
                hidden_dim=hidden_dim,
                attn_head_dim=attn_head_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

    # --------------------------------------------------------------- #
    # MoT-compatible interface
    # --------------------------------------------------------------- #

    def pre_dit(
        self,
        x: torch.Tensor,                     # [B, in_dim, T, H, W]   (clean V-JEPA encoder features)
        context: torch.Tensor,               # [B, L_text, text_dim]
        context_mask: Optional[torch.Tensor] = None,  # [B, L_text]
        timestep: Optional[torch.Tensor] = None,      # ignored (deterministic)
        action: Optional[torch.Tensor] = None,        # ignored (no action conditioning here)
        fuse_vae_embedding_in_latents: bool = False,  # ignored
    ) -> Dict[str, Any]:
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B, C, T, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"`x.shape[1]` must be in_dim={self.in_dim}, got {x.shape[1]}")
        B, _, T, H, W = x.shape

        # 1. predictor_embed: [B, in_dim, T, H, W] -> [B, T, H, W, hidden_dim]
        x_t = x.permute(0, 2, 3, 4, 1).contiguous()                 # [B, T, H, W, in_dim]
        x_emb = self.predictor_embed(x_t)                            # [B, T, H, W, hidden]
        x_emb = x_emb.view(B, T, H * W, self.hidden_dim)             # [B, T, H*W, hidden]

        # 2. text_pool -> k=3 tokens
        if context_mask is None:
            context_mask = torch.ones(
                (B, context.shape[1]), dtype=torch.bool, device=context.device
            )
        text_tok = self.text_pool(context, context_mask)             # [B, k, hidden]

        # 3. broadcast text prefix per timestep, prepend to spatial patches
        text_per_t = text_tok.unsqueeze(1).expand(B, T, -1, -1)      # [B, T, k, hidden]
        combined = torch.cat([text_per_t, x_emb], dim=2)             # [B, T, k+H*W, hidden]
        tokens = combined.flatten(1, 2)                              # [B, T*(k+H*W), hidden]

        # 4. cross-attn context (Wan-style)
        context_emb = self.text_embedding(context)                   # [B, L_text, hidden]
        # MoT expects [B, S, L_text]
        S = tokens.shape[1]
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, S, -1)

        # 5. t_mod = ZERO (deterministic; no timestep signal)
        # Match Wan-DiT t_mod shape: [B, S, 6, hidden_dim]
        t_mod = tokens.new_zeros(B, S, 6, self.hidden_dim)

        # 6. freqs placeholder. We are NOT using Wan's complex-RoPE here;
        # MoT will detect EXPERT_ATTN_KIND and apply V-JEPA RoPE via
        # vjepa_apply_rope_to_qk(...). We still pass an attribute-free
        # sentinel so the MoT signature is satisfied.
        freqs = None

        meta = {
            "B": B,
            "T": T,
            "H": H,
            "W": W,
            "k_text": self.text_pool_queries,
            "hidden_dim": self.hidden_dim,
            "tokens_per_frame": self.text_pool_queries + H * W,
            "expert_attn_kind": self.EXPERT_ATTN_KIND,
            "rope_grid_size": self.rope_grid_size,
            "in_dim": self.in_dim,
        }
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": None,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": meta,
        }

    def post_dit(self, tokens_out: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        meta = pre_state["meta"]
        B, T, H, W, k = meta["B"], meta["T"], meta["H"], meta["W"], meta["k_text"]
        # tokens_out: [B, T*(k+H*W), hidden]
        x = tokens_out.view(B, T, k + H * W, self.hidden_dim)
        # strip text prefix
        x = x[:, :, k:, :]                                        # [B, T, H*W, hidden]
        x = x.view(B, T, H, W, self.hidden_dim)                   # [B, T, H, W, hidden]
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)                                # [B, T, H, W, in_dim]
        x = x.permute(0, 4, 1, 2, 3).contiguous()                 # [B, in_dim, T, H, W]
        return x

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Bidirectional self-attention over the entire video token block.

        Matches V-JEPA-AC's behaviour (attn_mask=None inside ACRoPEAttention),
        which lets every video/text-prefix token attend to every other one.
        We accept all the standard MoT signature args for parity with
        ``WanVideoDiT.build_video_to_video_mask``.
        """
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Loader: V-JEPA 2-AC predictor checkpoint -> VJEPAVideoExpert
# ---------------------------------------------------------------------------


@torch.no_grad()
def load_vjepa2ac_predictor_into_expert(
    expert: VJEPAVideoExpert,
    ckpt_path: str,
    strict_block_count: bool = True,
) -> Dict[str, int]:
    """Load V-JEPA 2-AC predictor weights from a torch.hub-style ckpt.

    The full checkpoint dict has keys::

        ['encoder', 'predictor', 'opt', 'scaler', 'target_encoder', ...]

    We only consume ``predictor``. Within it:
      * ``predictor_embed`` (Linear in_dim->hidden) -> expert.predictor_embed
      * ``predictor_norm``  (LayerNorm) -> expert.predictor_norm
      * ``predictor_proj``  (Linear hidden->in_dim) -> expert.predictor_proj
      * ``predictor_blocks.{i}.*`` -> expert.blocks[i].load_vjepa_block_state(...)

    Skipped (replaced by `text_pool` in our design):
      * ``action_encoder``, ``state_encoder``, ``extrinsics_encoder``

    Args:
        expert: An initialised :class:`VJEPAVideoExpert`.
        ckpt_path: Path to ``vjepa2-ac-vitg.pt``.
        strict_block_count: If True, raise if predictor's number of blocks
            differs from len(expert.blocks).

    Returns:
        A dict summarising counts (loaded_blocks, skipped_keys, etc.).
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "predictor" not in blob:
        raise KeyError(f"Checkpoint at {ckpt_path} has no 'predictor' key. Keys: {list(blob.keys())}")
    sd = blob["predictor"]

    # Strip "module." prefix.
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # --- count blocks in pretrained predictor ---
    block_indices = sorted({
        int(k.split(".")[1]) for k in sd.keys() if k.startswith("predictor_blocks.")
    })
    n_pretrained_blocks = len(block_indices)

    if strict_block_count and n_pretrained_blocks != len(expert.blocks):
        raise ValueError(
            f"V-JEPA-AC predictor has {n_pretrained_blocks} blocks but "
            f"expert.blocks has {len(expert.blocks)}. Set num_layers correctly "
            f"or pass strict_block_count=False."
        )
    n_loaded_blocks = min(n_pretrained_blocks, len(expert.blocks))

    # --- predictor_embed ---
    if "predictor_embed.weight" in sd and "predictor_embed.bias" in sd:
        expert.predictor_embed.weight.copy_(
            sd["predictor_embed.weight"].to(expert.predictor_embed.weight.dtype)
        )
        expert.predictor_embed.bias.copy_(
            sd["predictor_embed.bias"].to(expert.predictor_embed.bias.dtype)
        )
    else:
        logger.warning("V-JEPA ckpt missing predictor_embed.{weight,bias}; leaving random init.")

    # --- predictor_norm ---
    if "predictor_norm.weight" in sd and "predictor_norm.bias" in sd:
        expert.predictor_norm.weight.copy_(
            sd["predictor_norm.weight"].to(expert.predictor_norm.weight.dtype)
        )
        expert.predictor_norm.bias.copy_(
            sd["predictor_norm.bias"].to(expert.predictor_norm.bias.dtype)
        )
    else:
        logger.warning("V-JEPA ckpt missing predictor_norm.{weight,bias}; leaving random init.")

    # --- predictor_proj ---
    if "predictor_proj.weight" in sd and "predictor_proj.bias" in sd:
        expert.predictor_proj.weight.copy_(
            sd["predictor_proj.weight"].to(expert.predictor_proj.weight.dtype)
        )
        expert.predictor_proj.bias.copy_(
            sd["predictor_proj.bias"].to(expert.predictor_proj.bias.dtype)
        )
    else:
        logger.warning("V-JEPA ckpt missing predictor_proj.{weight,bias}; leaving random init.")

    # --- per-block loading ---
    for i in range(n_loaded_blocks):
        prefix = f"predictor_blocks.{i}."
        block_keys = {k[len(prefix):]: sd[k] for k in sd if k.startswith(prefix)}
        expert.blocks[i].load_vjepa_block_state(block_keys, prefix="")

    skipped_top_level = sorted({
        k.split(".")[0] for k in sd.keys()
        if not k.startswith("predictor_blocks.")
        and not k.startswith("predictor_embed")
        and not k.startswith("predictor_norm")
        and not k.startswith("predictor_proj")
    })

    summary = {
        "n_loaded_blocks": n_loaded_blocks,
        "n_pretrained_blocks": n_pretrained_blocks,
        "n_expert_blocks": len(expert.blocks),
        "skipped_top_level": skipped_top_level,
    }
    logger.info(
        "Loaded V-JEPA 2-AC predictor into VJEPAVideoExpert: %s",
        summary,
    )
    return summary
