"""Deprecated — replaced by the in-MoT :class:`LanguageExpert`.

The original ``VLMPolicy`` was a standalone VLM that fed its decoder
hidden states into the DiTs via cross-attention. Under the 3-expert MoT
design, the language branch lives **inside** the shared self-attention
pool as :class:`fastwam.models.hfastwam.language_expert.LanguageExpert`
with its K/V selectively detached for knowledge insulation. There is no
separate VLM module any more.

This file is retained only to make stale imports fail loudly.
"""

from __future__ import annotations


_REMOVAL_NOTE = (
    "VLMPolicy has been removed. The language branch is now a first-class "
    "expert inside MoT — see fastwam.models.hfastwam.language_expert.LanguageExpert."
)


class VLMPolicy:  # pragma: no cover - deprecated
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(_REMOVAL_NOTE)


__all__ = ["VLMPolicy"]
