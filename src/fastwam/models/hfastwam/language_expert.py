"""Language Expert for H-FastWAM.

A 30-layer causal-LM-style transformer whose block structure is
**identical** to :class:`fastwam.models.wan22.wan_video_dit.DiTBlock`,
so it can plug straight into :class:`fastwam.models.wan22.mot.MoT`
alongside the video and action experts and share their self-attention.

Why reuse DiTBlock directly?
----------------------------
:class:`MoT._build_expert_attention_io` calls
``block.norm1 / block.self_attn / block.modulation`` — duck-typing these
is error prone, so we pay the small cost of carrying a :class:`DiTBlock`
per layer. t_mod is not meaningful for a language expert, so the caller
passes **zeros** and ``block.modulation`` becomes a static bias (which
we initialise to zero at construction). Cross-attention is skipped by
passing ``context_payload=None`` from the caller; in that case
:func:`MoT._apply_expert_post_block` does not touch ``cross_attn``, so
the cross-attn parameters stay a dead weight set (minor cost).

Visual grounding
----------------
The language expert does **not** have its own image encoder. Instead, it
sees the current observation through **shared MoT self-attention**: the
attention mask allows language queries to attend to the video expert's
**first-frame tokens** (clean, t=0).  This design:

  - Eliminates a redundant vision encoder (was SigLIP, now removed).
  - Forces language to condition on the same visual features that drive
    video generation — better alignment.
  - Simplifies the architecture: one DINO encoder, one visual path.

Inputs
------
The expert consumes a single concatenated token stream per step:

    [task_token_ids ‖ subtask_token_ids]

- **Task / subtask tokens** come from a shared ``nn.Embedding``. 1D RoPE
  handles position.

Outputs
-------
After the final MoT layer, :meth:`post_dit` applies a final LayerNorm
and projects each subtask position to vocab logits via ``lm_head``. The
CE loss is computed only on the subtask-id slice of the sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.wan_video_dit import DiTBlock, precompute_freqs_cis

logger = logging.getLogger(__name__)


CROSS_ENTROPY_IGNORE_INDEX: int = -100


@dataclass
class LanguageExpertOutput:
    """Structured return value of :meth:`LanguageExpert.post_dit`."""
    logits: torch.Tensor          # [B, L_sub, vocab_size]
    hidden_states: torch.Tensor   # [B, S_total, hidden_dim] after final norm


class LanguageExpert(nn.Module):
    """Language expert that shares MoT self-attention with video/action.

    Visual grounding comes from attending to the video expert's first-frame
    tokens through the shared MoT self-attention mask (no separate image
    encoder needed).

    Attributes exposed for MoT compatibility:
        blocks (nn.ModuleList[DiTBlock]): per-layer transformer blocks.
        num_heads, attn_head_dim: heads configuration (must match peers).
        use_gradient_checkpointing (bool): forwarded to MoT post-block
            checkpointing path.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        vocab_size: int,
        max_task_len: int = 128,
        max_subtask_len: int = 128,
        eps: float = 1e-6,
        use_gradient_checkpointing: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0 or attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be a positive even number (RoPE), got {attn_head_dim}"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.vocab_size = int(vocab_size)
        self.max_task_len = int(max_task_len)
        self.max_subtask_len = int(max_subtask_len)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        # ---- Token embeddings -------------------------------------------- #
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim).to(dtype=dtype)

        # Learned type embeddings — gives the model an explicit signal for
        # "this is a task token vs. subtask token". Two segments.
        self.segment_embedding = nn.Embedding(2, hidden_dim).to(dtype=dtype)

        # Final norm + LM head (tied to token_embedding for parameter sharing)
        self.final_norm = nn.LayerNorm(hidden_dim, eps=eps).to(dtype=dtype)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False).to(dtype=dtype)
        self.lm_head.weight = self.token_embedding.weight

        # ---- DiT-compatible blocks --------------------------------------- #
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        # DiTBlock's modulation starts from a random init; for language we
        # want it to act as a *static* no-op bias (no timestep semantics),
        # and we want the initial forward to be a vanilla transformer.
        #
        # ``_split_modulation`` chunks ``(modulation + t_mod)`` into
        # ``[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]``.
        # The block then does:
        #     x = x + gate_msa * self_attn(modulate(norm1(x), shift_msa, scale_msa))
        #     x = x + gate_mlp * ffn(modulate(norm2(x), shift_mlp, scale_mlp))
        # with ``modulate(y, shift, scale) = y * (1 + scale) + shift``.
        #
        # For a plain transformer block (no time conditioning) we want:
        #   - shift_*  = 0  (no bias)
        #   - scale_*  = 0  (modulate() collapses to identity)
        #   - gate_*   = 1  (attention / FFN outputs pass straight through)
        #
        # So we init ``modulation`` so that gate rows are 1 and the rest are 0.
        # The caller then passes ``t_mod = 0``, preserving these values.
        for block in self.blocks:
            with torch.no_grad():
                block.modulation.zero_()
                # Indices 2 (gate_msa) and 5 (gate_mlp) in the second dim.
                block.modulation[:, 2, :].fill_(1.0)
                block.modulation[:, 5, :].fill_(1.0)

        # ---- 1D causal RoPE frequencies ---------------------------------- #
        max_total_len = max_task_len + max_subtask_len
        self.register_buffer(
            "freqs",
            precompute_freqs_cis(attn_head_dim, end=max(max_total_len, 1024)),
            persistent=False,
        )

        logger.info(
            "LanguageExpert: hidden=%d, heads=%d×%d=%d, layers=%d, vocab=%d, "
            "max_seq=%d (no image tokens — visual grounding via MoT attention to video)",
            hidden_dim, num_heads, attn_head_dim, num_heads * attn_head_dim,
            num_layers, vocab_size, max_total_len,
        )

    # ------------------------------------------------------------------ #
    # Token assembly
    # ------------------------------------------------------------------ #
    def build_input_tokens(
        self,
        task_token_ids: torch.Tensor,
        subtask_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """Assemble ``[task ‖ subtask]`` token stream.

        Args:
            task_token_ids: ``[B, L_task]`` task instruction token ids.
                Must not contain ``-100``.
            subtask_token_ids: ``[B, L_sub]`` ground-truth subtask ids,
                may contain ``-100`` at ignored positions.

        Returns:
            (tokens, freqs, segments) where:
              - tokens: ``[B, S, hidden_dim]`` ready for MoT
              - freqs: ``[S, 1, attn_head_dim/2]`` 1D RoPE frequencies
              - segments: dict with ``task_len / subtask_len``
                so the caller can slice the MoT output back out.
        """
        B = task_token_ids.shape[0]
        if subtask_token_ids.shape[0] != B:
            raise ValueError("Batch sizes must match across task/subtask.")

        L_task = int(task_token_ids.shape[1])
        L_sub = int(subtask_token_ids.shape[1])
        S = L_task + L_sub
        if S > int(self.freqs.shape[0]):
            raise ValueError(
                f"Total token length {S} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        dtype = self.token_embedding.weight.dtype
        device = task_token_ids.device

        # Task tokens: regular embedding + segment-0.
        task_tokens = self.token_embedding(task_token_ids.to(device))
        task_seg = self.segment_embedding(
            torch.zeros(L_task, dtype=torch.long, device=device)
        )
        task_tokens = task_tokens + task_seg.unsqueeze(0)

        # Subtask tokens: -100 sentinel replaced by 0 so nn.Embedding does
        # not blow up; the label tensor is still used for the CE loss.
        safe_subtask = torch.where(
            subtask_token_ids == CROSS_ENTROPY_IGNORE_INDEX,
            torch.zeros_like(subtask_token_ids),
            subtask_token_ids,
        ).to(device)
        subtask_tokens = self.token_embedding(safe_subtask)
        sub_seg = self.segment_embedding(
            torch.ones(L_sub, dtype=torch.long, device=device)
        )
        subtask_tokens = subtask_tokens + sub_seg.unsqueeze(0)

        tokens = torch.cat([task_tokens, subtask_tokens], dim=1)

        freqs = self.freqs[:S].view(S, 1, -1).to(device)

        return tokens, freqs, {"task_len": L_task, "subtask_len": L_sub}

    # ------------------------------------------------------------------ #
    # MoT pre-/post-hook
    # ------------------------------------------------------------------ #
    def pre_dit(
        self,
        task_token_ids: torch.Tensor,
        subtask_token_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build the inputs that ``MoT.forward`` expects for this expert.

        Returns:
            Dict with keys::

                tokens:      [B, S, hidden_dim]
                freqs:       [S, 1, attn_head_dim/2]
                t_mod:       zeros of shape [1, 6, hidden_dim] so that
                             MoT._split_modulation cleanly maps to
                             ``block.modulation`` acting as static bias.
                segments:    dict with task_len / subtask_len
        """
        tokens, freqs, segments = self.build_input_tokens(
            task_token_ids, subtask_token_ids,
        )
        dtype = tokens.dtype
        device = tokens.device
        # ``DiTBlock.forward`` expects t_mod shape ``[B, 6, hidden_dim]`` or a
        # broadcastable ``[1, 6, hidden_dim]``. For language we want a no-op
        # *timestep contribution*, not a no-op block, so t_mod = 0 combined
        # with ``block.modulation`` (initialised so gate rows = 1, shift /
        # scale rows = 0 — see ``__init__``) yields a vanilla transformer
        # block: ``x = x + self_attn(...)`` then ``x = x + ffn(...)``.
        t_mod = torch.zeros(
            (1, 6, self.hidden_dim), dtype=dtype, device=device,
        )
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "segments": segments,
        }

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state: Dict[str, Any],
    ) -> LanguageExpertOutput:
        """Project the subtask-slice of MoT output to vocab logits.

        The MoT output has shape ``[B, S, hidden_dim]`` with
        ``S = task_len + subtask_len``. We run a final
        LayerNorm over the whole sequence and extract the
        ``[task_len : ]`` slice for LM-head projection.
        """
        segments = pre_state["segments"]
        task_len = int(segments["task_len"])

        hidden = self.final_norm(tokens)
        subtask_hidden = hidden[:, task_len:, :]
        logits = self.lm_head(subtask_hidden)
        return LanguageExpertOutput(logits=logits, hidden_states=hidden)

    # ------------------------------------------------------------------ #
    # Language loss
    # ------------------------------------------------------------------ #
    @staticmethod
    def language_loss(
        logits: torch.Tensor,
        subtask_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Standard teacher-forced next-token CE.

        ``logits[:, :-1]`` predicts ``subtask_token_ids[:, 1:]``, with
        ``ignore_index=-100`` on the label side.
        """
        if logits.ndim != 3:
            raise ValueError(f"logits must be [B, L, V], got {tuple(logits.shape)}")
        if subtask_token_ids.ndim != 2:
            raise ValueError(
                f"subtask_token_ids must be [B, L], got {tuple(subtask_token_ids.shape)}"
            )
        if logits.shape[1] != subtask_token_ids.shape[1]:
            raise ValueError(
                "logits / subtask_token_ids length mismatch: "
                f"{logits.shape[1]} vs {subtask_token_ids.shape[1]}"
            )

        vocab_size = logits.shape[-1]
        shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
        shift_labels = subtask_token_ids[:, 1:].contiguous().view(-1)
        return F.cross_entropy(
            shift_logits.float(),
            shift_labels.to(device=shift_logits.device),
            ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
        )

    # ------------------------------------------------------------------ #
    # Autoregressive generation (inference path)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_logits(
        self,
        tokens_after_mot: torch.Tensor,
        task_len: int,
    ) -> torch.Tensor:
        """Compute next-token logits from the *last* position of the MoT output.

        Used during autoregressive inference: run MoT with the current
        ``[task ‖ partial_subtask]`` stream, grab the last position's
        hidden state, and project to vocab.
        """
        hidden = self.final_norm(tokens_after_mot)
        last = hidden[:, -1:, :]
        return self.lm_head(last)

    # ------------------------------------------------------------------ #
    # Attention mask builder (language rows only)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_language_rows(
        task_len: int,
        subtask_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Boolean mask for the language-query rows inside the full H-FastWAM
        attention mask. Shape: ``[task_len + subtask_len, S_lang]``.

        Attention rules (within the language block):
          - task tokens: causal over task text.
          - subtask tokens: causal over their own positions, full over task.
        """
        S_lang = task_len + subtask_len
        rows = torch.zeros(S_lang, S_lang, dtype=torch.bool, device=device)
        task_s, task_e = 0, task_len
        sub_s, sub_e = task_len, S_lang

        # task: causal next-token prediction.
        if task_len > 0:
            rows[task_s:task_e, task_s:task_e] = torch.tril(
                torch.ones((task_len, task_len), dtype=torch.bool, device=device)
            )
        # subtask: causal within, sees all task
        rows[sub_s:sub_e, task_s:task_e] = True
        for i in range(subtask_len):
            rows[sub_s + i, sub_s:sub_s + i + 1] = True

        return rows
