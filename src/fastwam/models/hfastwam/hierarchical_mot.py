"""Deprecated — Hierarchical-MoT is no longer used.

The original plan was to extend ``MoT`` with a third (language) expert
and hand-crafted structured attention. That implementation had several
bugs and was never actually invoked by :class:`HFastWAM` (the model
bypassed ``hmot`` entirely and ran video/action experts independently).

The current :class:`HFastWAM` reuses :class:`fastwam.models.wan22.mot.MoT`
with ``{"video": ..., "action": ...}`` and drops the language branch
from the self-attention loop. The VLM's decoder hidden states are fed
into the DiTs via cross-attention (Knowledge-Insulation: detached).

This module is kept only so that ``import`` statements from older code
fail loudly rather than silently binding to a broken implementation.
"""

from __future__ import annotations


_REMOVAL_NOTE = (
    "HierarchicalMoT / build_hierarchical_attention_mask have been removed. "
    "H-FastWAM now reuses fastwam.models.wan22.mot.MoT with {'video', 'action'}. "
    "See fastwam.models.hfastwam.hfastwam for the current training/inference path."
)


def build_hierarchical_attention_mask(*_args, **_kwargs):  # pragma: no cover - deprecated
    raise NotImplementedError(_REMOVAL_NOTE)


class HierarchicalMoT:  # pragma: no cover - deprecated
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(_REMOVAL_NOTE)


__all__ = ["HierarchicalMoT", "build_hierarchical_attention_mask"]
